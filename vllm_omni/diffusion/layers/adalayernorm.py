from importlib.util import find_spec

import torch
import torch.nn as nn
from vllm.logger import init_logger

from vllm_omni.diffusion.layers.custom_op import CustomOp
from vllm_omni.diffusion.layers.norm import LayerNorm

logger = init_logger(__name__)

_HAS_MINDIESD = find_spec("mindiesd") is not None


class AdaLayerNorm(CustomOp):
    """
    AdaLayerNorm:
        out = layernorm(x) * (1 + scale) + shift
    """

    def __init__(self, hidden_size: int, elementwise_affine: bool = False, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.hidden_size = hidden_size
        self.layernorm = LayerNorm(self.hidden_size, elementwise_affine=self.elementwise_affine, eps=self.eps)

    def forward_cuda(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_native(x, scale, shift)

    def forward_hip(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_native(x, scale, shift)

    def forward_npu(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        if _HAS_MINDIESD:
            try:
                from mindiesd import layernorm_scale_shift

                output = layernorm_scale_shift(self.layernorm, x, scale, shift, fused=True)

                return output
            except ImportError as e:
                logger.warning_once(f"mindiesd import failed, falling back to torch_npu: {e}")

        import torch_npu

        output = (
            torch_npu.npu_layer_norm_eval(x, normalized_shape=[self.hidden_size], eps=self.eps) * (1 + scale) + shift
        )

        return output

    def forward_xpu(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_native(x, scale, shift)

    def forward_native(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        return self.layernorm(x) * (1 + scale) + shift
