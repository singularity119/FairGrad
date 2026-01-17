import torch
import torch.nn as nn
import torch.nn.functional as F


def calc_loss(x_pred, x_output, task_type):
    device = x_pred.device

    # binary mark to mask out undefined pixel space
    binary_mask = (torch.sum(x_output, dim=1) != 0).float().unsqueeze(1).to(device)

    if task_type == "semantic":
        # semantic loss: depth-wise cross entropy
        loss = F.nll_loss(x_pred, x_output, ignore_index=-1)

    if task_type == "depth":
        # depth loss: l1 norm
        loss = torch.sum(torch.abs(x_pred - x_output) * binary_mask) / torch.nonzero(
            binary_mask, as_tuple=False
        ).size(0)

    return loss

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
    grad_est_eps=1.0,
    num_curv_probes=5,
    curv_sigma=1e-4,  # 曲率估计的有限差分步长（越小越精确，但需数值稳定）
    ratio=0.9,              # ← 你原来就有的
    base_penalty=1.0,        # ← 你原来就有的
):
    """
    极简曲率初始化（带 K 正负惩罚）：
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

    n_tasks = 2
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

    print(f"✨ RBD-init | curvature regularized | ratio={ratio}, base_penalty={base_penalty}, curv_maximize_weight={curv_maximize_weight}, grad_est_eps={grad_est_eps}, num_curv_probes={num_curv_probes}, curv_sigma={curv_sigma}")

    # ---------- early stopping based on proportion of K>0 ----------
    # 在每个 step 里，统计 K>0 的任务比例；在最近 early_stop_window 个 step 中，
    # 如果有超过 early_stop_ratio 的 step 满足「K>0 的任务比例 > pos_ratio_threshold」，则提前停止。
    early_stop_window = 10
    early_stop_ratio = 0.6   # 最近 step 中，满足条件的 step 比例阈值
    pos_ratio_threshold = 0.6  # 单个 step 内，K>0 的任务比例阈值
    pos_ratio_window = []
    early_stop = False

    for epoch in range(init_epochs):
        print(f"⚙️ Epoch {epoch+1}/{init_epochs}")

        for step, batch in enumerate(init_data_loader):
            if steps > 0 and step >= steps:
                break

            train_data, train_label, train_depth = batch
            train_data = train_data.to(device)
            train_label = train_label.long().to(device)
            train_depth = train_depth.to(device)

            theta0 = flat_params.detach()

            # ---------- task forward functions ----------
            def _forward_sem(m):
                preds, _ = m(train_data, return_representation=True)
                return calc_loss(preds[0], train_label, "semantic")

            def _forward_dep(m):
                preds, _ = m(train_data, return_representation=True)
                return calc_loss(preds[1], train_depth, "depth")

            forward_funcs = [_forward_sem, _forward_dep]

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
            # h = torch.randn(D, device=device)
            # h /= h.norm() + 1e-12

            h = sample_cone_direction(m_full, 0.1, device)

            theta_pos = theta0 + grad_est_eps * h
            theta_neg = theta0 - grad_est_eps * h

            # curvature at theta + eps*h
            write_flat_to_model(theta_pos)
            k_pos = [estimate_task_curvature_single(
                        model, params, fn,
                        num_probes=num_curv_probes, sigma=curv_sigma, device=device)
                     for fn in forward_funcs]

            # curvature at theta - eps*h
            write_flat_to_model(theta_neg)
            k_neg = [estimate_task_curvature_single(
                        model, params, fn,
                        num_probes=num_curv_probes, sigma=curv_sigma, device=device)
                     for fn in forward_funcs]

            # restore θ
            write_flat_to_model(theta0)

            # 当前点的曲率估计（中点）和沿 h 的曲率 slope
            K_theta0 = [(k_pos[t] + k_neg[t]) / 2.0 for t in range(n_tasks)]
            per_task_curv_slope = [(k_pos[t] - k_neg[t]) / (2 * grad_est_eps)
                                   for t in range(n_tasks)]

            # ---------- update positive-K ratio window for early stopping ----------
            num_pos_tasks = sum(1 for v in K_theta0 if v > 0.0)
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

            # ---------- 3) 曲率"惩罚"：K 离 0 越远，惩罚越大 ----------
            per_task_g_theta_curv = []

            for t in range(n_tasks):
                K_t = K_theta0[t]
                slope_t = per_task_curv_slope[t]


                delta_mag= abs(K_t)
                dir_sign = 1.0 if K_t < 0 else - 1.0

                if K_t < 0:
                    alpha_t = ratio * base_penalty *  delta_mag
                else:
                    alpha_t = (1-ratio) * base_penalty * delta_mag
                

                g_theta_curv_t =  dir_sign * alpha_t * slope_t * h.clone()

                per_task_g_theta_curv.append(g_theta_curv_t)

            # ---------- 4) 任务正交投影（保持 loss 一阶不变） ----------
            for t in range(n_tasks):
                coeff = (per_task_g_theta_curv[t] @ per_task_grads[t]) / (
                    per_task_grads[t].norm()**2 + 1e-12
                )
                per_task_g_theta_curv[t] -= coeff * per_task_grads[t]

            # ---------- 5) scale by single hyperparam ----------
            if early_stop:
                break

            step_dir = curv_maximize_weight * sum(per_task_g_theta_curv)

            # ---------- momentum ----------
            m_full = beta * m_full + (1 - beta) * step_dir

            # apply update
            flat_params = theta0 + lr * m_full

            print(
                f"[Epoch {epoch+1}] [Step {step+1}] "
                f"L_sem={losses[0]:.3f} L_dep={losses[1]:.3f} | "
                f"K_sem={K_theta0[0]:.4f} K_dep={K_theta0[1]:.4f} | "
                f"curv_slope_sem={per_task_curv_slope[0]:.4f} curv_slope_dep={per_task_curv_slope[1]:.4f} | "
                f"pos_ratio={pos_ratio:.3f} | "
                f"step_dir={step_dir.norm():.4f} "
            )

        if early_stop:
            break

    # apply final result
    write_flat_to_model(flat_params)

    delta = flat_params - theta_start
    rel_change = delta.norm().item() / (theta_start.norm().item() + 1e-12)
    print(f"Δθ norm = {delta.norm().item():.4e}, rel_change = {rel_change:.4e}, ratio = {ratio}, base_penalty = {base_penalty}, power = None, curv_maximize_weight = {curv_maximize_weight}, grad_est_eps = {grad_est_eps}, num_curv_probes = {num_curv_probes}, curv_sigma = {curv_sigma}")
    print("🎉 RBD-init (curvature-regularized ascent) Done!")
    return model
