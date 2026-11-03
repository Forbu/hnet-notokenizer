from dataclasses import dataclass, field
from typing import List


@dataclass
class HNetConfig:
    """
    Configuration for a 1-stage or 2-stage Byte-Level H-Net Language Model.
    """

    vocab_size: int = 256
    # Hierarchy dimensions per stage (outermost -> innermost)
    # Example 1-stage: [512, 768]
    # Example 2-stage: [384, 512, 768]
    d_model: List[int] = field(default_factory=lambda: [384, 512, 768])

    # Number of layers in each Encoder and Decoder at outer stages, and Main Transformer at innermost stage
    n_encoder_layers: List[int] = field(default_factory=lambda: [4, 4])
    n_decoder_layers: List[int] = field(default_factory=lambda: [4, 4])
    n_main_layers: int = 12

    # Attention head dimension for Transformer layers
    head_dim: int = 64

    # Target downsampling ratios N_s for each outer stage (e.g. [4.5] for 1-stage, [4.0, 2.0] for 2-stage)
    target_compression: List[float] = field(default_factory=lambda: [3.0, 2.0])

    # Weight on the auxiliary ratio load-balancing loss
    aux_loss_weight: float = 0.03

    # Per-stage learning rate multipliers (outer stages see more tokens -> higher effective batch size)
    lr_multipliers: List[float] = field(default_factory=lambda: [2.5, 1.5, 1.0])

    # Conv kernel size for outer SSM/Conv encoder-decoder blocks
    conv_kernel_size: int = 4

    @property
    def num_stages(self) -> int:
        """Number of dynamic chunking stages (1 or 2)."""
        return len(self.d_model) - 1

    @classmethod
    def small_1stage(cls) -> "HNetConfig":
        """Fast ~35M param 1-stage byte H-Net for quick debugging & local/single-GPU tests."""
        return cls(
            d_model=[384, 512],
            n_encoder_layers=[4],
            n_decoder_layers=[4],
            n_main_layers=8,
            head_dim=64,
            target_compression=[4.5],
            lr_multipliers=[2.0, 1.0],
        )

    @classmethod
    def medium_2stage_h100(cls) -> "HNetConfig":
        """~150M param 2-stage byte H-Net tailored for single H100 FineWeb experiments."""
        return cls(
            d_model=[384, 512, 768],
            n_encoder_layers=[4, 4],
            n_decoder_layers=[4, 4],
            n_main_layers=16,
            head_dim=64,
            target_compression=[4.0, 2.0],
            lr_multipliers=[3.0, 1.7, 1.0],
        )

    @classmethod
    def large_2stage_h100(cls) -> "HNetConfig":
        """~420M param 2-stage byte H-Net for longer H100 runs."""
        return cls(
            d_model=[512, 768, 1024],
            n_encoder_layers=[4, 4],
            n_decoder_layers=[4, 4],
            n_main_layers=20,
            head_dim=64,
            target_compression=[4.0, 2.0],
            lr_multipliers=[3.0, 1.7, 1.0],
        )
