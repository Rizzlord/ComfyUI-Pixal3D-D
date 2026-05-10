"""
Adaptive Gradient Clipping utilities for training.
"""
import numpy as np
import torch
from typing import Optional, Iterable, Union


class AdaptiveGradClipper:
    """
    Adaptive gradient clipping for training.
    """
    def __init__(
        self,
        max_norm=None,
        clip_percentile=95.0,
        buffer_size=1000,
        skip_mode=False,
        max_skipped_steps=500,
        use_buffer=True,
    ):
        self.max_norm = max_norm
        self.clip_percentile = clip_percentile
        self.buffer_size = buffer_size
        self.skip_mode = skip_mode  # 如果True，超过阈值时跳过更新；如果False，进行梯度裁剪
        
        self._grad_norm = np.zeros(buffer_size, dtype=np.float32)
        self._max_norm = max_norm
        self._buffer_ptr = 0
        self._buffer_length = 0
        self._skipped_steps = 0  # 记录跳过的步数
        self._skipped_steps_list = np.zeros(buffer_size, dtype=np.int32)
        self._skipped_steps_ptr = 0
        self.max_skipped_steps = max_skipped_steps
        self.use_buffer = use_buffer
    def __repr__(self):
        mode_str = "skip" if self.skip_mode else "clip"
        return f'AdaptiveGradClipper(max_norm={self.max_norm}, clip_percentile={self.clip_percentile}, mode={mode_str})'
        
    def state_dict(self):
        return {
            'grad_norm': self._grad_norm,
            'max_norm': self._max_norm,
            'buffer_ptr': self._buffer_ptr,
            'buffer_length': self._buffer_length,
            'skipped_steps': self._skipped_steps,
            'skipped_steps_list': self._skipped_steps_list,
            'skipped_steps_ptr': self._skipped_steps_ptr,
        }

    def load_state_dict(self, state_dict):
        self._grad_norm = state_dict['grad_norm']
        self._max_norm = state_dict['max_norm']
        self._buffer_ptr = state_dict['buffer_ptr']
        self._buffer_length = state_dict['buffer_length']
        self._skipped_steps = state_dict.get('skipped_steps', 0)  # 兼容旧版本
        # self._skipped_steps_list = state_dict.get('skipped_steps_list', np.zeros(self.buffer_size, dtype=np.int32))
        self._skipped_steps_ptr = state_dict.get('skipped_steps_ptr', 0)

    def log(self):
        return {
            'max_norm': self._max_norm,
            'skipped_steps': self._skipped_steps,
            'skipped_steps_list': self._skipped_steps_list,
            'skipped_steps_ptr': self._skipped_steps_ptr,
        }

    def __call__(self, parameters, norm_type=2.0, error_if_nonfinite=False, foreach=None,optimizer=None):
        """Clip or skip gradients based on their norm with two-tier threshold system.

        The norm is computed over all gradients together, as if they were
        concatenated into a single vector.

        Two-tier threshold logic:
        1. If grad_norm > initial_max_norm (constructor param): 
           - skip_mode=True: SKIP the update (zero gradients)  
           - skip_mode=False: CLIP to adaptive threshold
        2. If adaptive_max_norm < grad_norm <= initial_max_norm:
           - Both modes: CLIP to adaptive threshold  
        3. If grad_norm <= adaptive_max_norm:
           - Both modes: No action (normal update)

        Args:
            parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
                single Tensor that will have gradients normalized
            norm_type (float): type of the used p-norm. Can be ``'inf'`` for
                infinity norm.
            error_if_nonfinite (bool): if True, an error is thrown if the total
                norm of the gradients from :attr:`parameters` is ``nan``,
                ``inf``, or ``-inf``. Default: False (will switch to True in the future)
            foreach (bool): use the faster foreach-based implementation.
                If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
                fall back to the slow implementation for other device types.
                Default: ``None``

        Returns:
            tuple: (grad_norm, should_skip) - grad_norm is the original gradient norm,
                   should_skip indicates whether this step should be skipped
        """
        # 使用初始max_norm作为skip阈值，自适应_max_norm作为clip阈值
        initial_max_norm = self.max_norm if self.max_norm is not None else float('inf')
        adaptive_max_norm = self._max_norm if self._max_norm is not None else float('inf')
        should_skip = False
        
        # 一次调用：获取原始梯度范数并裁剪到adaptive_max_norm
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters, max_norm=adaptive_max_norm, norm_type=norm_type,
            error_if_nonfinite=error_if_nonfinite, foreach=foreach
        )
        if not self.use_buffer:
            return grad_norm, should_skip
        if torch.isfinite(grad_norm):
            grad_norm_value = grad_norm.item()
            
            if self.skip_mode and grad_norm_value > initial_max_norm:
                # Skip模式：如果原始梯度超过初始max_norm，跳过本次更新
                if isinstance(parameters, torch.Tensor):
                    parameters = [parameters]
                for p in parameters:
                    if p.grad is not None:
                        p.grad.zero_()
                should_skip = True
                self._skipped_steps += 1
                self._skipped_steps_list[self._skipped_steps_ptr] = 1
                self._skipped_steps_ptr = (self._skipped_steps_ptr + 1) % self.buffer_size
                if optimizer is not None:
                    optimizer.zero_grad()
                print(f"[AdaptiveGradClipper] Skipping step due to large gradient norm: {grad_norm_value:.6f} > {initial_max_norm:.6f} (initial_max_norm)")
                # Skip时不更新缓冲区，因为异常梯度不应该影响自适应阈值计算
                
            else:
                # 正常情况：使用已经被裁剪到adaptive_max_norm的梯度，更新缓冲区
                self._grad_norm[self._buffer_ptr] = grad_norm_value
                self._buffer_ptr = (self._buffer_ptr + 1) % self.buffer_size
                self._skipped_steps_list[self._skipped_steps_ptr] = 0   
                self._buffer_length = min(self._buffer_length + 1, self.buffer_size)
                self._skipped_steps_ptr = (self._skipped_steps_ptr + 1) % self.buffer_size
                
                if grad_norm_value > adaptive_max_norm:
                    if self.skip_mode:
                        print(f"[AdaptiveGradClipper] Clipping gradient norm: {grad_norm_value:.6f} -> {adaptive_max_norm:.6f} (adaptive_max_norm)")
            
            # 重新计算自适应阈值（只要不skip就可以更新阈值）
            if not should_skip and self._buffer_length == self.buffer_size:
                self._max_norm = np.percentile(self._grad_norm, self.clip_percentile)
                self._max_norm = min(self._max_norm, self.max_norm) if self.max_norm is not None else self._max_norm
        if self._skipped_steps_list.sum() > self.max_skipped_steps:
            raise Exception("Too many skipped steps, something is wrong")
        return grad_norm, should_skip

