import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
from sklearn.cluster import SpectralClustering

# ============================================================
# ▶ 1. 子函数：单个高维圆锥方向采样（ConMeZO 核心思想）
# ============================================================
def sample_cone_direction(center, angle, device):
    """
    在一个以 center 为轴的圆锥中采样一个高维单位向量。
    angle: 弧度
    """
    center_norm = torch.norm(center)
    if center_norm < 1e-8:
        # 动量太小时 → 退化为随机方向
        v = torch.randn_like(center, device=device)
        return v / torch.norm(v)

    center_unit = center / center_norm

    # 生成一个与 center 尽量正交的随机方向 u
    u = torch.randn_like(center, device=device)
    u = u - torch.dot(u, center_unit) * center_unit
    u_norm = torch.norm(u)

    if u_norm < 1e-8:
        u = torch.randn_like(center, device=device)
        u = u - torch.dot(u, center_unit) * center_unit
        u_norm = torch.norm(u)

    u = u / u_norm

    cos_t = torch.cos(angle)
    sin_t = torch.sin(angle)

    dir_vec = cos_t * center_unit + sin_t * u
    return dir_vec / torch.norm(dir_vec)


# ============================================================
# ▶ 2. 子函数：从圆锥中采样 P 的 d 列（高维 ConMeZO 版本）
# ============================================================
def sample_cone_matrix(center, angle, D, d, device):
    """
    生成一个 D×d 的矩阵 P，每一列都是圆锥采样得到的单位向量。
    center 是 D 维 vector（full-momentum）
    """
    center_norm = torch.norm(center)
    if center_norm < 1e-8:
        # 训练初期无动量方向 ⇒ 完全随机子空间
        P = torch.randn(D, d, device=device)
        P = P / torch.norm(P, dim=0, keepdim=True)
        return P

    cols = []
    for _ in range(d):
        col = sample_cone_direction(center, angle, device)
        cols.append(col.view(-1, 1))

    P = torch.cat(cols, dim=1)   # D × d
    return P


@torch.no_grad()
def _flatten_params(params):
    return torch.cat([p.detach().view(-1) for p in params])


def estimate_task_curvature_single(
    model,
    param_list,
    forward_fn,
    num_probes=5,
    sigma=1e-3,
    device=None,
):
    """
    在当前 θ 下，用随机探针方向近似一个任务的平均方向曲率:
        κ ≈ E_v [ v^T H v ]

    - model: 当前模型
    - param_list: 要估计曲率的参数列表（例如 shared_params）
    - forward_fn(model) -> scalar loss: 指定任务 loss
    - num_probes: 随机方向数量
    - sigma: 有限差分步长，用于梯度差分估计 H v ≈ (g(θ+σv)-g(θ))/σ
    """
    if device is None:
        device = next(model.parameters()).device

    was_training = model.training
    model.eval()

    # ---------- 1) g(θ) ----------
    model.zero_grad(set_to_none=True)
    loss0 = forward_fn(model)

    grads0 = torch.autograd.grad(
        loss0,
        param_list,
        create_graph=False,
        allow_unused=True,  # 有的参数可能没用到
    )

    g0_list = []
    for g, p in zip(grads0, param_list):
        if g is None:
            g0_list.append(torch.zeros_like(p).reshape(-1))
        else:
            g0_list.append(g.reshape(-1))
    g0 = torch.cat(g0_list).to(device)

    D = g0.numel()

    curvature_sum = 0.0

    # ---------- 2) 多个随机 probe ----------
    for _ in range(num_probes):
        # 随机单位向量 v
        v = torch.randn(D, device=device)
        v = v / (v.norm() + 1e-12)

        # θ ← θ + σ v（注意用 no_grad，避免 leaf inplace 报错）
        idx = 0
        for p in param_list:
            n = p.numel()
            delta = (sigma * v[idx: idx + n]).view_as(p)
            with torch.no_grad():
                p.add_(delta)
            idx += n

        # g(θ + σv)
        model.zero_grad(set_to_none=True)
        loss1 = forward_fn(model)
        grads1 = torch.autograd.grad(
            loss1,
            param_list,
            create_graph=False,
            allow_unused=True,
        )
        g1_list = []
        for g, p in zip(grads1, param_list):
            if g is None:
                g1_list.append(torch.zeros_like(p).reshape(-1))
            else:
                g1_list.append(g.reshape(-1))
        g1 = torch.cat(g1_list)

        # θ 恢复
        idx = 0
        for p in param_list:
            n = p.numel()
            delta = (sigma * v[idx: idx + n]).view_as(p)
            with torch.no_grad():
                p.sub_(delta)
            idx += n

        # H v ≈ (g1 - g0) / σ，曲率样本 v^T H v
        hvp = (g1 - g0) / sigma
        curv_sample = torch.dot(v, hvp).item()
        curvature_sum += curv_sample

    if was_training:
        model.train()
    else:
        model.eval()

    return curvature_sum / num_probes


def rbd_init_multitask_coneP_curvature(
    model,
    init_data_loader,
    device,
    d=32,
    steps=50,
    lr=1e-3,
    task_weights=None,
    beta=0.9,
    cone_angle_deg=30,
    init_epochs=1,
    # === 曲率相关参数 ===
    curv_maximize_weight=0.1,  # 控制最大化曲率的力度 (lambda)
    grad_est_eps=1.0,  # c 空间有限差分步长
    num_curv_probes=1,  # 估算曲率值时用的 probe 数
    curv_sigma=1e-3,  # θ 空间梯度差分步长
    margin=0.1,
):
    """
    RBD + ConeP + Curvature Maximization (在子空间 c 里做 0 阶曲率升）

    目标：寻找一个 θ，使得
      - 多任务 Loss 较小
      - 同时总体曲率 K(θ) = Σ_t E_v[v^T H_t(θ) v] 尽可能大
    """
    model.train()

    # 只对 shared_parameters 做 RBD
    shared_params = list(model.shared_parameters())
    shared_ids = {id(p) for p in shared_params}
    named_params = [
        (name, p) for name, p in model.named_parameters() if id(p) in shared_ids
    ]
    params = [p for _, p in named_params]

    # 初始扁平参数 θ0
    flat_params = _flatten_params(params).to(device)
    theta_start = flat_params.clone()
    D = flat_params.numel()

    n_tasks = 40  # CelebA 有 40 个任务
    if task_weights is None:
        task_weights = [1.0] * n_tasks

    # 动量 buffer & 圆锥角
    m_full = torch.zeros(D, device=device)
    angle = torch.deg2rad(torch.tensor(cone_angle_deg, device=device))

    print(
        f"✨ RBD: ConeP + Maximize Curvature | d={d}, lr={lr}, "
        f"lambda={curv_maximize_weight}, probes={num_curv_probes}, eps={grad_est_eps}"
    )

    def write_flat_to_model(theta_flat: torch.Tensor):
        """把扁平向量写回 shared 参数。"""
        offset = 0
        with torch.no_grad():
            for _, p in named_params:
                n = p.numel()
                p.copy_(theta_flat[offset: offset + n].view_as(p))
                offset += n

    loss_fn = torch.nn.BCELoss()

    for epoch in range(init_epochs):
        print(f"⚙️  Epoch {epoch+1}/{init_epochs}")

        for step, batch in enumerate(init_data_loader):
            if steps > 0 and step >= steps:
                break

            x, y = batch
            x = x.to(device)
            y = [y_.to(device) for y_ in y]

            # 当前基准 θ0
            theta0 = flat_params.detach()

            # 采样 Cone 子空间 P: (D, d)
            P = sample_cone_matrix(m_full, angle, D, d, device)

            # --- 定义每个任务的 forward_fn ---
            def make_forward_fn(task_idx):
                def _forward_task(m):
                    y_pred = m(x)
                    return loss_fn(y_pred[task_idx], y[task_idx])
                return _forward_task

            forward_funcs = [make_forward_fn(t) for t in range(n_tasks)]

            # =====================================================
            # A. Task Gradient in c-space (多任务 loss 的下降方向)
            # =====================================================
            g_c_task_total = torch.zeros(d, device=device)
            loss_logs = []

            for t_idx in range(n_tasks):
                # 恢复到基准 θ0
                write_flat_to_model(theta0)

                model.zero_grad(set_to_none=True)
                loss = forward_funcs[t_idx](model)
                loss_logs.append(loss.item())
                loss.backward()

                flat_grad = torch.cat([
                    (p.grad.view(-1) if p.grad is not None else torch.zeros_like(p).view(-1))
                    for _, p in named_params
                ])

                g_c = P.t() @ flat_grad  # (d,)

                g_c_task_total += task_weights[t_idx] * g_c

            # =====================================================
            # B. Curvature Gradient in c-space
            # =====================================================
            g_c_curv_total = torch.zeros(d, device=device)

            if curv_maximize_weight > 0.0:
                # 在 c 空间采一个方向 h
                h = torch.randn(d, device=device)
                h = h / (h.norm() + 1e-12)

                # θ_pos / θ_neg = θ0 ± P (ε h)
                delta_theta = P @ (grad_est_eps * h)

                # --- K_pos ---
                theta_pos = theta0 + delta_theta
                write_flat_to_model(theta_pos)

                k_pos = []
                for fn in forward_funcs:
                    k_pos.append(
                        estimate_task_curvature_single(
                            model,
                            params,
                            fn,
                            num_probes=num_curv_probes,
                            sigma=curv_sigma,
                            device=device,
                        )
                    )

                # --- K_neg ---
                theta_neg = theta0 - delta_theta
                write_flat_to_model(theta_neg)

                k_neg = []
                for fn in forward_funcs:
                    k_neg.append(
                        estimate_task_curvature_single(
                            model,
                            params,
                            fn,
                            num_probes=num_curv_probes,
                            sigma=curv_sigma,
                            device=device,
                        )
                    )

                # 回到 θ0（下一步 Task / update 用）
                write_flat_to_model(theta0)

                # ===== 提前终止条件：估计 θ0 处的任务曲率 =====
                # K_t(θ0) ≈ (K_pos_t + K_neg_t) / 2
                K_theta0 = [(k_pos[t] + k_neg[t]) / 2.0 for t in range(n_tasks)]

                # 检查80%的任务曲率 > margin
                num_above_margin = sum(1 for K in K_theta0 if K > margin)
                threshold_ratio = 0.8
                if num_above_margin >= int(n_tasks * threshold_ratio):
                    print(
                        f"🎯 {num_above_margin}/{n_tasks} ({num_above_margin/n_tasks*100:.1f}%) tasks curvature > margin at "
                        f"epoch {epoch+1}, step {step+1} → stop init"
                    )
                    print(
                        f"[Curv Anchor] Min K={min(K_theta0):.4f}, "
                        f"Max K={max(K_theta0):.4f}, "
                        f"Mean K={sum(K_theta0)/len(K_theta0):.4f}"
                    )
                    return model
                # ===== 提前终止条件结束 =====

                # 0阶梯度估计: ∂K/∂c ≈ (K_pos - K_neg)/(2ε) * h
                curv_slope = (min(k_pos) - min(k_neg)) / (2 * grad_est_eps)
                g_c_curv_total = curv_slope * h

            # =====================================================
            # C. 合成方向并在 θ-space 更新
            # =====================================================
            #   c_dir = -∇_c L_task + λ * ∇_c K
            final_c_dir = -g_c_task_total + curv_maximize_weight * g_c_curv_total

            # 映射回 θ 空间
            step_dir = P @ final_c_dir  # (D,)

            target_norm = theta0.norm()
            current_norm = step_dir.norm()
            scale_factor = target_norm / (current_norm + 1e-12)
            step_dir = step_dir * scale_factor

            # 动量
            m_full = beta * m_full + (1.0 - beta) * step_dir

            # 更新 θ
            flat_params = theta0 + lr * m_full

            if (step + 1) % 10 == 0:
                print(
                    f"[Epoch {epoch+1}] [Step {step+1}] "
                    f"Avg Loss={sum(loss_logs)/len(loss_logs):.4f} | "
                    f"||g_c_task||={g_c_task_total.norm():.3f} "
                    f"||g_c_curv||={g_c_curv_total.norm():.3f}"
                )

    # 最后把 θ 写回模型
    write_flat_to_model(flat_params)
    print("🎉 RBD Init (ConeP + Curvature Optimized) Done!")
    return model


def rbd_init_multitask_curvature_fullspace(
    model,
    init_data_loader,
    device,
    steps=50,
    lr=1e-3,
    task_weights=None,
    beta=0.9,
    init_epochs=1,
    curv_maximize_weight=1,   # ⭐唯一超参⭐
    grad_est_eps=1,
    num_curv_probes=5,
    curv_sigma=1e-4,  # 曲率估计的有限差分步长（越小越精确，但需数值稳定）
):
    """
    极简曲率初始化（带 K 正负惩罚）- CelebA 版本（40 任务）：
    - 基础仍是：通过有限差分估计 ∂K/∂θ
    - 但不再无脑增大 K，而是：
        * K 为正 → 往小压（避免曲率过大太尖）
        * K 为负 → 往上拉（减少负曲率）
        * |K| 越大，惩罚越大；负数接近 0 也会有惩罚
    - 仍然对每个任务做梯度正交投影，保持一阶 loss 不变
    - 只保留 1 个曲率相关超参 curv_maximize_weight（整体步长）
    """

    model.train()

    # ---------- collect shared params ----------
    shared_params = list(model.shared_parameters())
    shared_ids = {id(p) for p in shared_params}
    named_params = [(n, p) for (n, p) in model.named_parameters() if id(p) in shared_ids]
    params = [p for _, p in named_params]

    flat_params = torch.cat([p.detach().reshape(-1) for p in params]).to(device)
    theta_start = flat_params.clone()
    D = flat_params.numel()

    n_tasks = 40  # CelebA 有 40 个任务
    if task_weights is None:
        task_weights = [1.0] * n_tasks

    # momentum buffer
    m_full = torch.zeros(D, device=device)

    def write_flat_to_model(theta_flat):
        idx = 0
        with torch.no_grad():
            for _, p in named_params:
                n = p.numel()
                p.copy_(theta_flat[idx:idx+n].view_as(p))
                idx += n

    loss_fn = torch.nn.BCELoss()

    print(f"✨ RBD-init | curvature regularized | step={curv_maximize_weight}")

    for epoch in range(init_epochs):
        print(f"⚙️ Epoch {epoch+1}/{init_epochs}")

        for step, batch in enumerate(init_data_loader):
            if steps > 0 and step >= steps:
                break

            x, y = batch
            x = x.to(device)
            y = [y_.to(device) for y_ in y]

            theta0 = flat_params.detach()

            # ---------- task forward functions ----------
            def make_forward_fn(task_idx):
                def _forward_task(m):
                    y_pred = m(x)
                    return loss_fn(y_pred[task_idx], y[task_idx])
                return _forward_task

            forward_funcs = [make_forward_fn(t) for t in range(n_tasks)]

            # ---------- 1) compute per-task gradients ----------
            per_task_grads = []
            g_task_total = torch.zeros(D, device=device)
            losses = []

            for t_idx in range(n_tasks):
                write_flat_to_model(theta0)

                loss = forward_funcs[t_idx](model)
                losses.append(loss.item())

                model.zero_grad(set_to_none=True)
                loss.backward()

                flat_grad = torch.cat([
                    (p.grad.view(-1) if p.grad is not None else torch.zeros_like(p).view(-1))
                    for _, p in named_params
                ])

                per_task_grads.append(flat_grad.clone())
                g_task_total += task_weights[t_idx] * flat_grad

            # ---------- 2) curvature slope estimate ----------
            h = torch.randn(D, device=device)
            h /= h.norm() + 1e-12

            theta_pos = theta0 + grad_est_eps * h
            theta_neg = theta0 - grad_est_eps * h

            # curvature at theta + eps*P*h
            write_flat_to_model(theta_pos)
            k_pos = [
                estimate_task_curvature_single(
                    model, params, fn,
                    num_probes=num_curv_probes, sigma=curv_sigma, device=device)
                for fn in forward_funcs
            ]

            # curvature at theta - eps*P*h
            write_flat_to_model(theta_neg)
            k_neg = [
                estimate_task_curvature_single(
                    model, params, fn,
                    num_probes=num_curv_probes, sigma=curv_sigma, device=device)
                for fn in forward_funcs
            ]

            # restore θ
            write_flat_to_model(theta0)

            # 当前点的曲率估计（中点）和沿 h 的曲率 slope
            K_theta0 = [(k_pos[t] + k_neg[t]) / 2.0 for t in range(n_tasks)]
            per_task_curv_slope = [(k_pos[t] - k_neg[t]) / (2 * grad_est_eps)
                                   for t in range(n_tasks)]

            # ---------- 3) 曲率"惩罚"：K 离 0 越远，惩罚越大 ----------
            # 思路：
            #   α_t = - sign(K_t) * (1 + |K_t|)
            #   - K_t > 0: α_t < 0 → 沿 -∇K 走，压小正曲率；|K| 越大步子越大
            #   - K_t < 0: α_t > 0 → 沿 +∇K 走，减小负曲率；|K| 越大步子越大
            #   - K_t 接近 0 的负数也会有 α_t>0（不会被放过）
            per_task_g_theta_curv = []

            base_penalty = 1
            ratio = 0.7
            for t in range(n_tasks):
                K_t = K_theta0[t]
                slope_t = per_task_curv_slope[t]

                delta_mag = abs(K_t)
                dir_sign = 1.0 if K_t < 0 else -1.0

                if K_t < 0:
                    alpha_t = ratio * base_penalty * (1 + delta_mag)
                else:
                    alpha_t = (1-ratio) * base_penalty * (1 + delta_mag)

                g_theta_curv_t = dir_sign * alpha_t * slope_t * h.clone()
                per_task_g_theta_curv.append(g_theta_curv_t)

            # ---------- 4) 任务正交投影（保持 loss 一阶不变） ----------
            for t in range(n_tasks):
                coeff = (per_task_g_theta_curv[t] @ per_task_grads[t]) / (
                    per_task_grads[t].norm()**2 + 1e-12
                )
                per_task_g_theta_curv[t] -= coeff * per_task_grads[t]

            # ---------- 5) scale by single hyperparam ----------
            step_dir = curv_maximize_weight * sum(per_task_g_theta_curv)

            # ---------- momentum ----------
            m_full = beta * m_full + (1.0 - beta) * step_dir

            # apply update
            flat_params = theta0 + lr * m_full

            avg_loss = sum(losses) / len(losses)
            min_K = min(K_theta0)
            max_K = max(K_theta0)
            mean_K = sum(K_theta0) / len(K_theta0)
            min_slope = min(per_task_curv_slope)
            max_slope = max(per_task_curv_slope)
            mean_slope = sum(per_task_curv_slope) / len(per_task_curv_slope)
            print(
                f"[Epoch {epoch+1}] [Step {step+1}] "
                f"Avg Loss={avg_loss:.4f} | "
                f"K_min={min_K:.4f} K_max={max_K:.4f} K_mean={mean_K:.4f} | "
                f"curv_slope_min={min_slope:.4f} curv_slope_max={max_slope:.4f} curv_slope_mean={mean_slope:.4f} | "
                f"step_dir={step_dir.norm():.4f}"
            )

    # apply final result
    write_flat_to_model(flat_params)

    delta = flat_params - theta_start
    print(f"Δθ norm = {delta.norm().item():.4e}, ratio = {ratio}, base_penalty = {base_penalty}, power = None")
    print("🎉 RBD-init (curvature-regularized ascent) Done!")
    return model






def rbd_init_multitask_curvature_spectral_once(
    model,
    init_data_loader,
    device,
    steps=50,
    lr=1e-3,
    beta=0.9,
    init_epochs=1,
    curv_maximize_weight=1.0,
    grad_est_eps=1.0,
    num_curv_probes=5,
    curv_sigma=1e-4,
    num_clusters=3,
    warmup_steps=10,
    ratio=0.7,              # ← 你原来就有的
    base_penalty=1.0        # ← 你原来就有的
):
    """
    RBD init with:
    - one-time gradient spectral clustering
    - fixed clusters
    - curvature penalty EXACTLY following original design:
        * K > 0 : weak suppression
        * K < 0 : stronger pull-back
    """

    model.train()

    # ---------- collect shared params ----------
    shared_params = list(model.shared_parameters())
    shared_ids = {id(p) for p in shared_params}
    named_params = [(n, p) for n, p in model.named_parameters()
                    if id(p) in shared_ids]
    params = [p for _, p in named_params]

    flat_params = torch.cat(
        [p.detach().reshape(-1) for p in params]
    ).to(device)

    theta_start = flat_params.clone()
    D = flat_params.numel()

    n_tasks = 40
    loss_fn = torch.nn.BCELoss()

    m_full = torch.zeros(D, device=device)

    def write_flat(theta):
        idx = 0
        with torch.no_grad():
            for _, p in named_params:
                n = p.numel()
                p.copy_(theta[idx:idx+n].view_as(p))
                idx += n

    print(f"✨ RBD-init | spectral once | ratio={ratio}")

    # ---------- warmup buffers ----------
    grad_accum = [torch.zeros(D, device=device) for _ in range(n_tasks)]
    clusters = None
    clustered = False

    # ---------- early stopping based on proportion of K>0 ----------
    # 在每个 step 里，统计 K>0 的任务比例；在最近 early_stop_window 个 step 中，
    # 如果有超过 early_stop_ratio 的 step 满足「K>0 的任务比例 > pos_ratio_threshold」，则提前停止。
    early_stop_window = 5
    early_stop_ratio = 0.5   # 最近 step 中，满足条件的 step 比例阈值
    pos_ratio_threshold = 0.6  # 单个 step 内，K>0 的任务比例阈值
    pos_ratio_window = []
    early_stop = False


    for epoch in range(init_epochs):
        for step, batch in enumerate(init_data_loader):
            if steps > 0 and step >= steps:
                break

            x, y = batch
            x = x.to(device)
            y = [yy.to(device) for yy in y]

            theta0 = flat_params.detach()

            # ---------- forward fns ----------
            def make_forward(t):
                def _f(m):
                    out = m(x)
                    return loss_fn(out[t], y[t])
                return _f

            forward_fns = [make_forward(t) for t in range(n_tasks)]

            # ---------- per-task gradients ----------
            per_task_grads = []
            losses = []

            for t in range(n_tasks):
                write_flat(theta0)
                loss = forward_fns[t](model)
                losses.append(loss.item())

                model.zero_grad(set_to_none=True)
                loss.backward()

                g = torch.cat([
                    (p.grad.view(-1) if p.grad is not None
                     else torch.zeros_like(p).view(-1))
                    for _, p in named_params
                ])

                per_task_grads.append(g.clone())

                if not clustered and step < warmup_steps:
                    grad_accum[t] += g.detach()
            
            # ---------- one-time spectral clustering ----------
            if (not clustered) and step == warmup_steps:
                with torch.no_grad():
                    G = torch.stack([grad_accum[t]/(grad_accum[t].norm() + 1e-12) for t in range(n_tasks)])

                    S = (G @ G.T).cpu().numpy()
                    S = np.clip(S, 0.0, None)

                    sc = SpectralClustering(
                        n_clusters=num_clusters,
                        affinity="precomputed",
                        assign_labels="kmeans",
                        random_state=0
                    )
                    labels = sc.fit_predict(S)

                    clusters = defaultdict(list)
                    for t, c in enumerate(labels):
                        clusters[int(c)].append(t)

                    clustered = True
                    print("✅ fixed clusters:")
                    for c, ts in clusters.items():
                        print(f"  cluster {c}: {ts}")

            if not clustered:
                continue

            # ---------- curvature slope ----------
            h = torch.randn(D, device=device)
            h /= h.norm() + 1e-12

            write_flat(theta0 + grad_est_eps * h)
            k_pos = [
                estimate_task_curvature_single(
                    model, params, fn,
                    num_probes=num_curv_probes,
                    sigma=curv_sigma,
                    device=device
                )
                for fn in forward_fns
            ]

            write_flat(theta0 - grad_est_eps * h)
            k_neg = [
                estimate_task_curvature_single(
                    model, params, fn,
                    num_probes=num_curv_probes,
                    sigma=curv_sigma,
                    device=device
                )
                for fn in forward_fns
            ]

            write_flat(theta0)

            K = [(k_pos[t] + k_neg[t]) / 2 for t in range(n_tasks)]
            slope = [(k_pos[t] - k_neg[t]) / (2 * grad_est_eps)
                     for t in range(n_tasks)]

            # ---------- update positive-K ratio window for early stopping ----------
            K_mean_val = float(sum(K) / len(K))
            num_pos_tasks = sum(1 for v in K if v > 0.0)
            pos_ratio = num_pos_tasks / float(n_tasks)

            pos_ratio_window.append(pos_ratio)
            if len(pos_ratio_window) > early_stop_window:
                pos_ratio_window.pop(0)

            num_steps_high = sum(1 for r in pos_ratio_window if r >= pos_ratio_threshold)
            if (len(pos_ratio_window) == early_stop_window and
                    num_steps_high >= early_stop_window * early_stop_ratio):
                print(
                    f"⏹ Early stopping triggered at epoch {epoch+1}, step {step+1}: "
                    f"{num_steps_high}/{early_stop_window} recent steps have "
                    f'K>0 ratio > {pos_ratio_threshold:.2f} '
                    f"(current step ratio={pos_ratio:.3f})."
                )
                early_stop = True

            # ---------- cluster-level aggregation + projection (cluster acts like a task) ----------
            cluster_updates = []
            eps = 1e-12

            for c, tasks in clusters.items():
                # 1) 聚合标量：K 与 slope，以及梯度（在一次循环中完成）
                K_sum = 0.0
                slope_sum = 0.0
                grad_c = torch.zeros_like(per_task_grads[0])
                
                for t in tasks:
                    K_sum += K[t]
                    slope_sum += slope[t]
                    grad_c += per_task_grads[t]
                
                n_c = max(len(tasks), 1)
                # n_c = 1
                K_c = K_sum / n_c
                slope_c = slope_sum / n_c
                grad_c /= n_c

                # 2) 完全沿用你原本的 penalty 逻辑（只把 K_t -> K_c）
                delta_mag = abs(K_c)
                dir_sign = 1.0 if K_c < 0 else -1.0

                if K_c < 0:
                    alpha_c = ratio * base_penalty * delta_mag
                else:
                    alpha_c = (1.0 - ratio) * base_penalty * delta_mag

                # 3) 生成"cluster-curvature gradient"（现在等价于 3 个任务的 curvature update）
                #    注意：slope_c 是标量，h 是单位方向向量
                g_c = (dir_sign * alpha_c * slope_c) * h

                # 4) cluster-level projection：用 cluster 的"平均梯度方向"做正交化

                coeff = (g_c @ grad_c) / (grad_c.norm()**2 + eps)
                g_c_proj = g_c - coeff * grad_c

                cluster_updates.append(g_c_proj)

            # ---------- update ----------
            if early_stop:
                break

            step_dir = curv_maximize_weight * sum(cluster_updates)
            m_full = beta * m_full + (1.0 - beta) * step_dir
            flat_params = theta0 + lr * m_full

            print(
                f"[E{epoch+1} S{step+1}] "
                f"loss={sum(losses)/len(losses):.4f} | "
                f"K_min={min(K):.4f} K_max={max(K):.4f} K_mean={K_mean_val:.4f} | "
                f"curv_slope_min={min(slope):.4f} curv_slope_max={max(slope):.4f} curv_slope_mean={sum(slope)/len(slope):.4f} | "
                f"step_norm={step_dir.norm():.3e}"
            )

        if early_stop:
            break

    write_flat(flat_params)

    # --------- report parameter change (absolute & relative) ----------
    delta = flat_params - theta_start
    delta_norm = delta.norm().item()
    theta_norm = theta_start.norm().item()
    rel_change = delta_norm / (theta_norm + 1e-12)
    max_abs_change = delta.abs().max().item()
    print(
        f"Δθ L2 = {delta_norm:.4e} "
        f"(rel={rel_change:.4e} wrt ||θ0||={theta_norm:.4e}, "
        f"max|Δθ_i|={max_abs_change:.4e}), "
        f"ratio={ratio}, base_penalty={base_penalty}, power=None, num_clusters={num_clusters}, beta={beta}, pos_ratio_threshold={pos_ratio_threshold},early_stop_ratio={early_stop_ratio}"
        f"early_stop_window={early_stop_window}"
    )
    print("🎉 RBD-init DONE")

    return model
