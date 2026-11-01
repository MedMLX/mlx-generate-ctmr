"""Static native tensor and module contracts; importing this module stays host-only."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import NotRequired, Protocol, TypedDict

    import mlx.core as mx
    import numpy as np

    type Array = mx.array
    type Dtype = mx.Dtype
    type HostArray = np.ndarray[tuple[int, ...], np.dtype[np.generic]]
    type FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float32]]
    type BoolArray = np.ndarray[tuple[int, ...], np.dtype[np.bool_]]
    type IntArray = np.ndarray[tuple[int, ...], np.dtype[np.int16]]
    type ArrayValue = bool | int | float | complex | list[ArrayValue] | tuple[ArrayValue, ...]

    class ArrayConstructor(Protocol):
        def __call__(
            self, value: HostArray | Array | ArrayValue, /, dtype: mx.Dtype | None = None
        ) -> Array:
            """Construct a native array with an optional dtype."""
            ...

    class ArrayFactory[Tensor, Dtype](Protocol):
        @property
        def float32(self) -> Dtype:
            """Return the runtime's float32 dtype."""
            ...

        def array(self, value: HostArray, /, dtype: Dtype | None = None) -> Tensor:
            """Construct a runtime array from host values."""
            ...

    class ClassFactory[Tensor, Dtype](Protocol):
        @property
        def int32(self) -> Dtype:
            """Return the native int32 dtype."""
            ...
        def full(self, shape: Sequence[int], vals: int, *, dtype: Dtype) -> Tensor:
            """Create a native array filled with an integer value."""
            ...

    class MetalKernel(Protocol):
        def __call__(
            self,
            *,
            inputs: Sequence[Array],
            template: Sequence[tuple[str, int | bool | mx.Dtype]] = (),
            grid: tuple[int, int, int],
            threadgroup: tuple[int, int, int],
            output_shapes: Sequence[Sequence[int]],
            output_dtypes: Sequence[mx.Dtype],
        ) -> list[Array]:
            """Launch a Metal kernel and return its output arrays."""
            ...

    from medmlx_core.typing import FastRuntime as CoreFastRuntime
    from medmlx_core.typing import MlxRuntime as CoreMlxRuntime

    class Fast(CoreFastRuntime, Protocol):
        scaled_dot_product_attention = staticmethod(mx.fast.scaled_dot_product_attention)

        def metal_kernel(
            self,
            name: str,
            input_names: Sequence[str],
            output_names: Sequence[str],
            source: str,
            header: str = "",
            ensure_row_contiguous: bool = True,
            atomic_outputs: bool = False,
            compile_options: object | None = None,
        ) -> MetalKernel:
            """Compile a named Metal kernel with the supplied source."""
            ...

    class Random(Protocol):
        normal = staticmethod(mx.random.normal)
        seed = staticmethod(mx.random.seed)

    class Metal(Protocol):
        is_available = staticmethod(mx.metal.is_available)

    class Mlx(CoreMlxRuntime, Protocol):
        """Native MLX operations used by the generation runtime."""

        @property
        def fast(self) -> Fast:
            """Return native fast operations and Metal-kernel construction."""
            ...

        @property
        def random(self) -> Random:
            """Return native random generation operations."""
            ...

        all = staticmethod(mx.all)
        argmax = staticmethod(mx.argmax)
        bitwise_and = staticmethod(mx.bitwise_and)
        clip = staticmethod(mx.clip)
        contiguous = staticmethod(mx.contiguous)
        cos = staticmethod(mx.cos)
        device_info = staticmethod(mx.device_info)
        exp = staticmethod(mx.exp)
        expand_dims = staticmethod(mx.expand_dims)
        full = staticmethod(mx.full)
        isfinite = staticmethod(mx.isfinite)
        left_shift = staticmethod(mx.left_shift)
        logical_or = staticmethod(mx.logical_or)
        ones = staticmethod(mx.ones)
        ones_like = staticmethod(mx.ones_like)
        sin = staticmethod(mx.sin)
        softmax = staticmethod(mx.softmax)
        split = staticmethod(mx.split)
        sqrt = staticmethod(mx.sqrt)
        squeeze = staticmethod(mx.squeeze)
        zeros_like = staticmethod(mx.zeros_like)

    class ConstructorOptions(TypedDict, total=False):
        spatial_dims: int
        include_fc: bool
        use_flash_attention: bool
        use_combined_linear: bool

    class DiffusionOptions(ConstructorOptions, total=False):
        with_conditioning: bool
        resblock_updown: bool
        transformer_num_layers: int
        cross_attention_dim: int | None
        upcast_attention: bool

    class DiffusionConstructor(DiffusionOptions):
        in_channels: int
        num_channels: Sequence[int]
        num_res_blocks: int | Sequence[int]
        attention_levels: Sequence[bool]
        norm_num_groups: int
        norm_eps: float
        num_head_channels: int | Sequence[int]
        num_class_embeds: int | None

    class RflowConstructor(DiffusionConstructor):
        out_channels: NotRequired[int]
        include_top_region_index_input: NotRequired[bool]
        include_bottom_region_index_input: NotRequired[bool]
        include_spacing_input: NotRequired[bool]
        dropout_cattn: NotRequired[float]

    class ControlnetConstructor(DiffusionConstructor):
        conditioning_embedding_in_channels: int
        conditioning_embedding_num_channels: Sequence[int]
        use_checkpointing: NotRequired[bool]

    class VaeOptions(ConstructorOptions, total=False):
        with_encoder_nonlocal_attn: bool
        with_decoder_nonlocal_attn: bool
        use_convtranspose: bool
        use_checkpointing: bool
        norm_float16: bool
        print_info: bool
        save_mem: bool
        latent_channels: int

    class VaeConstructor(VaeOptions):
        in_channels: int
        out_channels: int
        num_channels: Sequence[int]
        num_res_blocks: Sequence[int]
        attention_levels: Sequence[bool]
        norm_num_groups: int
        norm_eps: float
        dim_split: int
        num_splits: int

    class MaskUnetConstructor(ConstructorOptions):
        in_channels: int
        num_channels: NotRequired[Sequence[int]]
        channels: NotRequired[Sequence[int]]
        out_channels: NotRequired[int]
        num_res_blocks: int | Sequence[int]
        attention_levels: Sequence[bool]
        norm_num_groups: int
        norm_eps: float
        num_head_channels: int | Sequence[int]
        cross_attention_dim: int
        num_class_embeds: int | None
        transformer_num_layers: NotRequired[int]
        with_conditioning: NotRequired[bool]
        resblock_updown: NotRequired[bool]
        dropout_cattn: NotRequired[float]
        upcast_attention: NotRequired[bool]

    class Activations(Protocol):
        def gelu(self, x: Array) -> Array:
            """Apply the GELU activation to a native array."""
            ...

    class IndexedUpdate(Protocol):
        def add(self, value: Array) -> Array:
            """Add values at the selected array indices."""
            ...

    class IndexedUpdates(Protocol):
        def __getitem__(self, key: tuple[slice, ...]) -> IndexedUpdate:
            """Select indices for an array update."""
            ...

    class MlpArithmetic(Protocol):
        def linear(self, array: Array, weight: Array, bias: Array | None, mx: Mlx) -> Array:
            """Apply a linear projection and optional bias."""
            ...
        def silu(self, array: Array, mx: Mlx) -> Array:
            """Apply the SiLU activation."""
            ...

    class ConvArithmetic(Protocol):
        def silu(self, array: Array, mx: Mlx) -> Array:
            """Apply the SiLU activation."""
            ...
        def group_norm_ncdhw(
            self,
            array: Array,
            weight: Array,
            bias: Array,
            *,
            num_groups: int,
            eps: float,
            mx: Mlx,
        ) -> Array:
            """Normalize NCDHW values by channel groups."""
            ...
        def conv3d_ncdhw(
            self,
            array: Array,
            weight: Array,
            bias: Array | None,
            *,
            padding: int,
            mx: Mlx,
            stride: int = 1,
        ) -> Array:
            """Apply a 3D convolution to NCDHW values."""
            ...

    class VaeArithmetic(ConvArithmetic, Protocol):
        def upsample_trilinear_ncdhw(self, array: Array, *, mx: Mlx, scale: int = 2) -> Array:
            """Upsample an NCDHW array with trilinear interpolation."""
            ...

    class ResidualArithmetic(MlpArithmetic, ConvArithmetic, Protocol):
        def avg_pool3d_ncdhw(self, array: Array, *, mx: Mlx) -> Array:
            """Average-pool spatial dimensions of an NCDHW array."""
            ...
        def upsample_nearest_ncdhw(self, array: Array, *, mx: Mlx) -> Array:
            """Upsample an NCDHW array by nearest neighbor."""
            ...

    class AttentionArithmetic(MlpArithmetic, Protocol):
        def attention_projection(
            self, array: Array, weight: Array, bias: Array, mx: Mlx
        ) -> Array:
            """Project attention outputs with an optional bias."""
            ...
        def attention(
            self, query: Array, key: Array, value: Array, *, scale: float, mx: Mlx
        ) -> Array:
            """Compute scaled attention over query, key, and value arrays."""
            ...

    class UnetArithmetic(ResidualArithmetic, AttentionArithmetic, Protocol):
        def sinusoidal_timestep_embedding(
            self, timesteps: Array, embedding_dim: int, *, mx: Mlx, max_period: int = 10000
        ) -> Array:
            """Encode diffusion timesteps as sinusoidal features."""
            ...
