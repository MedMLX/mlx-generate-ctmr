"""Static native tensor and module contracts; importing this module stays host-only."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import NotRequired, Protocol, TypedDict

    import mlx.core as mx
    import numpy as np
    from medmlx_core.typing import MlxRuntime

    type Array = mx.array
    type Dtype = mx.Dtype
    type HostArray = np.ndarray[tuple[int, ...], np.dtype[np.generic]]
    type IntArray = np.ndarray[tuple[int, ...], np.dtype[np.int16]]
    type ArrayValue = bool | int | float | complex | list[ArrayValue] | tuple[ArrayValue, ...]

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

    class MlpArithmetic(Protocol):
        def linear(self, array: Array, weight: Array, bias: Array | None, mx: MlxRuntime) -> Array:
            """Apply a linear projection and optional bias."""
            ...

        def silu(self, array: Array, mx: MlxRuntime) -> Array:
            """Apply the SiLU activation."""
            ...

    class ConvArithmetic(Protocol):
        def silu(self, array: Array, mx: MlxRuntime) -> Array:
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
            mx: MlxRuntime,
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
            mx: MlxRuntime,
            stride: int = 1,
        ) -> Array:
            """Apply a 3D convolution to NCDHW values."""
            ...

    class VaeArithmetic(ConvArithmetic, Protocol):
        def upsample_trilinear_ncdhw(
            self, array: Array, *, mx: MlxRuntime, scale: int = 2
        ) -> Array:
            """Upsample an NCDHW array with trilinear interpolation."""
            ...

    class ResidualArithmetic(MlpArithmetic, ConvArithmetic, Protocol):
        def avg_pool3d_ncdhw(self, array: Array, *, mx: MlxRuntime) -> Array:
            """Average-pool spatial dimensions of an NCDHW array."""
            ...

        def upsample_nearest_ncdhw(self, array: Array, *, mx: MlxRuntime) -> Array:
            """Upsample an NCDHW array by nearest neighbor."""
            ...

    class AttentionArithmetic(MlpArithmetic, Protocol):
        def attention_projection(
            self, array: Array, weight: Array, bias: Array, mx: MlxRuntime
        ) -> Array:
            """Project attention outputs with an optional bias."""
            ...

        def attention(
            self, query: Array, key: Array, value: Array, *, scale: float, mx: MlxRuntime
        ) -> Array:
            """Compute scaled attention over query, key, and value arrays."""
            ...

    class UnetArithmetic(ResidualArithmetic, AttentionArithmetic, Protocol):
        def sinusoidal_timestep_embedding(
            self, timesteps: Array, embedding_dim: int, *, mx: MlxRuntime, max_period: int = 10000
        ) -> Array:
            """Encode diffusion timesteps as sinusoidal features."""
            ...
