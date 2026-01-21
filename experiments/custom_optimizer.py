import torch
from torch.optim import Optimizer
import math

# adaptive beta1 adam
class AdaptiveBetaAdam(Optimizer):
    def __init__(self, params, lr=1e-3, 
                 betas=(0.9, 0.999), # 这里 betas[0] 是默认的初始 beta1
                 beta_range=(0.1, 0.5), # (beta_min, beta_max)
                 eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        self.beta_min, self.beta_max = beta_range
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdaptiveBetaAdam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # 获取基础参数
            beta1_init, beta2 = group['betas']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # 当前梯度
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdaptiveBetaAdam does not support sparse gradients')

                state = self.state[p]

                # 状态初始化
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format) # m_t
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format) # v_t
                    state['prev_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format) # g_{t-1}

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                prev_grad = state['prev_grad']
                state['step'] += 1

                # ============================================================
                # 核心逻辑：计算 Adaptive Beta
                # ============================================================
                
                # 1. 计算 Cosine Similarity (Layer-wise / Tensor-wise)
                # 我们对整个张量计算一个标量相似度，这比 element-wise 更稳定
                # 展平计算点积
                g_cur_flat = grad.view(-1)
                g_prev_flat = prev_grad.view(-1)
                
                if state['step'] == 1:
                    # 第一步没有历史，使用默认 beta
                    beta1 = beta1_init
                else:
                    # 计算 cos_sim
                    dot_product = torch.dot(g_cur_flat, g_prev_flat)
                    norm_cur = torch.norm(g_cur_flat)
                    norm_prev = torch.norm(g_prev_flat)
                    
                    # 防止除以零
                    if norm_cur > 0 and norm_prev > 0:
                        cos_sim = dot_product / (norm_cur * norm_prev)
                    else:
                        cos_sim = torch.tensor(0.0, device=grad.device)
                    # print('Adaject Gradient Sim: ', cos_sim.item())
                    # 2. 映射到 [beta_min, beta_max]
                    # 公式：beta = min + (max - min) * (cos_sim + 1) / 2
                    # 这样 -1 -> min, 1 -> max
                    
                    # factor = (cos_sim + 1.0) / 2.0
                    factor = cos_sim

                    factor = torch.clamp(factor, 0.0, 1.0) # 截断保险
                    beta1 = self.beta_min + (self.beta_max - self.beta_min) * factor.item()
                
                # 记录当前梯度供下一步使用
                prev_grad.copy_(grad)
                
                # ============================================================
                # 标准 Adam 更新逻辑 (使用动态 beta1)
                # ============================================================
                
                # Weight Decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Update Momentum (m_t) using adaptive beta1
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update Velocity (v_t) using fixed beta2 (通常不需要自适应 beta2)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step'] 
                # 注意：这里简单的 bias correction 可能不完全准确，因为 beta1 是动态的
                # 但在工程实现中，通常直接用当前的 beta1 做近似，或者忽略这个极其微小的误差
                bias_correction2 = 1 - beta2 ** state['step']

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


# adaptive beta1 adamw
class AdaptiveBetaAdamW(Optimizer):
    """
    AdaptiveBetaAdamW: 基于曲率动态调整 Beta1 的 AdamW 优化器。
    
    Args:
        params (iterable): 待优化参数
        lr (float): 学习率
        betas (Tuple[float, float]): (初始 beta1, beta2)
        eps (float): 数值稳定性项
        weight_decay (float): 权重衰减系数 (Decoupled)
        beta_range (Tuple[float, float]): Beta1 的动态范围 (min, max)
        use_momentum_proxy (bool): 【显存优化选项】
            - True: 使用动量 m_{t-1} 代替 g_{t-1} 来计算相似度。节省显存，不需额外存 prev_grad。
            - False: 严格存储上一步梯度 g_{t-1}。显存占用更高，但符合严格定义的“梯度曲率”。
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, beta_range=(0.5, 0.95), 
                 use_momentum_proxy=False):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
            
        self.beta_min, self.beta_max = beta_range
        self.use_momentum_proxy = use_momentum_proxy
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdaptiveBetaAdamW, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1_init, beta2 = group['betas']
            
            for p in group['params']:
                if p.grad is None:
                    continue

                # AdamW 特性 1: Perform stepweight decay (Decoupled Weight Decay)
                # 先做 Weight Decay，再做梯度计算，这是 AdamW 的核心
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdaptiveBetaAdamW does not support sparse gradients')

                state = self.state[p]

                # State Initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format) # m_t
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format) # v_t
                    if not self.use_momentum_proxy:
                        state['prev_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format) # g_{t-1}

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                # ============================================================
                # 核心逻辑：Adaptive Beta1 计算
                # ============================================================
                beta1 = beta1_init # 默认值
                
                if state['step'] > 1:
                    # 获取“参考向量”：要么是上一步梯度，要么是动量缓冲
                    if self.use_momentum_proxy:
                        reference_vec = exp_avg # 使用 m_{t-1} 近似历史方向
                    else:
                        reference_vec = state['prev_grad'] # 使用真实的 g_{t-1}

                    # 计算 Cosine Similarity (Layer-wise)
                    g_flat = grad.view(-1)
                    ref_flat = reference_vec.view(-1)
                    
                    dot = torch.dot(g_flat, ref_flat)
                    norm_g = torch.norm(g_flat)
                    norm_ref = torch.norm(ref_flat)
                    
                    if norm_g > 0 and norm_ref > 0:
                        cos_sim = dot / (norm_g * norm_ref)
                    else:
                        cos_sim = torch.tensor(0.0, device=grad.device)
                    
                    # 线性映射: [-1, 1] -> [beta_min, beta_max]
                    # cos=1 (一致) -> max_beta (加速)
                    # cos=-1 (震荡) -> min_beta (减速/转弯)
                    factor = (cos_sim + 1.0) / 2.0
                    factor = torch.clamp(factor, 0.0, 1.0)
                    beta1 = self.beta_min + (self.beta_max - self.beta_min) * factor.item()

                # 如果不使用 proxy，需要保存当前梯度给下一步用
                if not self.use_momentum_proxy:
                    state['prev_grad'].copy_(grad)

                # ============================================================
                # 标准 Adam 更新逻辑
                # ============================================================
                
                # Update Momentum (m_t) with dynamic beta1
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update Velocity (v_t) with fixed beta2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                denom = (exp_avg_sq.sqrt() / math.sqrt(1 - beta2 ** state['step'])).add_(group['eps'])
                
                # Bias correction for m_t (Approximate using current beta1)
                step_size = group['lr'] / (1 - beta1 ** state['step'])
                
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


# both designs for adaptive beta1 and wd
class OmniAdaptiveAdamW(Optimizer):
    """
    OmniAdaptiveAdamW: 双重自适应优化器。
    
    Feature 1 (Direction): 基于梯度历史一致性动态调整 Beta1 (Momentum)。
    Feature 2 (Scale): 基于梯度信噪比 (GSNR) 动态调整 Weight Decay。
    
    Args:
        params (iterable): 待优化参数
        lr (float): 学习率
        betas (Tuple[float, float]): (初始 beta1, beta2)
        eps (float): 数值稳定性项
        weight_decay (float): 基础权重衰减系数 (建议设为 0.05 或更高，作为"强正则"的基准)
        
        # Adaptive Beta1 Config
        beta_range (Tuple[float, float]): Beta1 的动态范围 (min, max)
        use_momentum_proxy (bool): 是否使用动量 m_{t-1} 代替 g_{t-1} 计算方向一致性 (省显存)
        
        # GSNR WD Config
        gsnr_alpha (float): GSNR 调节敏感度。值越大，信号强时 WD 衰减越厉害。推荐 2.0 - 5.0。
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, beta_range=(0.1, 0.95), 
                 use_momentum_proxy=False, gsnr_alpha=2.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
            
        self.beta_min, self.beta_max = beta_range
        self.use_momentum_proxy = use_momentum_proxy
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        gsnr_alpha=gsnr_alpha)
        super(OmniAdaptiveAdamW, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1_init, beta2 = group['betas']
            gsnr_alpha = group['gsnr_alpha']
            
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('OmniAdaptiveAdamW does not support sparse gradients')

                state = self.state[p]

                # --- 1. State Initialization ---
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format) # m_t
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format) # v_t
                    if not self.use_momentum_proxy:
                        state['prev_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format) # g_{t-1}

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                step = state['step']

                # ============================================================
                # Feature 1: Adaptive Beta1 (基于曲率/方向)
                # ============================================================
                beta1 = beta1_init # Default fallback
                
                if step > 1:
                    # 获取参考向量
                    if self.use_momentum_proxy:
                        reference_vec = exp_avg # m_{t-1}
                    else:
                        reference_vec = state['prev_grad'] # g_{t-1}

                    # 计算 Cosine Similarity (Flattened)
                    g_flat = grad.view(-1)
                    ref_flat = reference_vec.view(-1)
                    
                    dot = torch.dot(g_flat, ref_flat)
                    norm_g = torch.norm(g_flat)
                    norm_ref = torch.norm(ref_flat)
                    
                    if norm_g > 0 and norm_ref > 0:
                        cos_sim = dot / (norm_g * norm_ref)
                    else:
                        cos_sim = torch.tensor(0.0, device=grad.device)
                    
                    # Mapping: cos=1 -> beta_max, cos=-1 -> beta_min
                    factor = (cos_sim + 1.0) / 2.0
                    factor = torch.clamp(factor, 0.0, 1.0)
                    beta1 = self.beta_min + (self.beta_max - self.beta_min) * factor.item()

                # 保存当前梯度供下一步 Feature 1 使用
                if not self.use_momentum_proxy:
                    state['prev_grad'].copy_(grad)

                # ============================================================
                # Standard Adam Momentum Update (Using Dynamic Beta1)
                # ============================================================
                # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # ============================================================
                # Feature 2: GSNR-Aware Weight Decay (基于信噪比)
                # ============================================================
                # 计算偏差修正后的矩估计
                # 注意：对于动态 beta1，严格的 bias correction 比较复杂，
                # 这里使用当前 beta1 做近似，这在 Adaptive 算法中是常见的做法。
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                
                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2
                
                # 计算方差 Var(g) = E[g^2] - (E[g])^2
                # Clamp min=0 防止数值误差导致负数
                variance = (v_hat - m_hat.pow(2)).clamp(min=1e-10)
                
                # GSNR = ||Mean|| / ||Std|| (Layer-wise)
                signal_strength = m_hat.norm()
                noise_strength = variance.sqrt().norm()
                gsnr = signal_strength / (noise_strength + 1e-10)
                
                # 动态 WD 因子: GSNR 越大 (信号好)，WD 越小
                gsnr_factor = 1.0 / (1.0 + gsnr_alpha * torch.log(1.0 + gsnr))
                
                # 应用 Weight Decay
                if group['weight_decay'] != 0:
                    effective_wd = group['weight_decay'] * gsnr_factor
                    p.data.mul_(1 - group['lr'] * effective_wd)
                    
                    # (Optional) Debug log
                    # if step % 1000 == 0 and step > 1:
                    #     print(f"Step {step} | Beta1: {beta1:.3f} | GSNR: {gsnr:.2f} | WD_Factor: {gsnr_factor:.3f}")

                # ============================================================
                # Final Parameter Update
                # ============================================================
                denom = (v_hat.sqrt().add_(group['eps']))
                p.addcdiv_(m_hat, denom, value=-group['lr'])

        return loss