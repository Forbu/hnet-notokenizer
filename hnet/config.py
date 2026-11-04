from dataclasses import dataclass, field
from typing import List


@dataclass
class HNetConfig:
    """
    Configuration for a 1-stage or 2-stage Byte-Level H-Net Language Model.
    Matches the design of `goombalab/hnet` and Cartesia's "From Tokens to Bytes" Part 1:
      - Outer Encoder/Decoder use pure SSM blocks ("m4", no FFN)
      - Inner Main Network uses Causal Transformer blocks ("T", with SwiGLU 3.6x D rounded to 128)
      - Partial RoPE: rotary_emb_dim = head_dim // 2 (50% of head channels rotated)
      - Sliding window attention support for outer hybrid stages (`window_size`)
      - Depth-scaled residual initialization (0.02 / sqrt(n_residuals))
    """

    vocab_size: int = 256
    # Hierarchy dimensions per stage (outermost -> innermost)
    # Example 1-stage: [512, 1024]
    d_model: List[int] = field(default_factory=lambda: [512, 1024])

    # Layer layout strings per stage using goombalab/hnet notation:
    #   'm' = SSM mixer only (no FFN), 'M' = SSM + SwiGLU
    #   't' = Causal Attn only (no FFN), 'T' = Causal Attn + SwiGLU
    encoder_layout: List[str] = field(default_factory=lambda: ["m4"])
    decoder_layout: List[str] = field(default_factory=lambda: ["m4"])
    main_layout: str = "T12"

    # Attention head dimension per stage (128 in Cartesia Part 1 scaling ladder, or 64/96)
    head_dim: List[int] = field(default_factory=lambda: [64, 128])

    # Partial RoPE dimension per stage (default: 50% of head_dim, matching hnet_1stage_L.json)
    rotary_emb_dim: List[int] = field(default_factory=lambda: [32, 64])

    # Attention window size per stage (-1 = full causal attention, 1023 = local sliding window)
    window_size: List[int] = field(default_factory=lambda: [1023, -1])

    # SwiGLU intermediate expansion factor (3.6 * D rounded to 128, per Part 1 Table 1)
    mlp_expansion: float = 3.6

    # Target downsampling ratios N_s for each outer stage (4.57 matches BPE in Part 1)
    target_compression: List[float] = field(default_factory=lambda: [4.57])

    # Weight on the auxiliary ratio load-balancing loss
    aux_loss_weight: float = 0.20

    # Per-stage learning rate multipliers (outer stages see ~4.57x more tokens per step)
    lr_multipliers: List[float] = field(default_factory=lambda: [2.5, 1.0])

    # Conv kernel size for outer SSM encoder-decoder blocks
    conv_kernel_size: int = 4

    @property
    def num_stages(self) -> int:
        """Number of dynamic chunking stages (1 or 2)."""
        return len(self.d_model) - 1

    @classmethod
    def small_1stage(cls) -> "HNetConfig":
        """Fast ~35M param 1-stage byte H-Net (layout: ['m4', ['T8'], 'm4'])."""
        return cls(
            d_model=[384, 512],
            encoder_layout=["m4"],
            decoder_layout=["m4"],
            main_layout="T8",
            head_dim=[64, 64],
            rotary_emb_dim=[32, 32],
            window_size=[1023, -1],
            target_compression=[4.57],
            lr_multipliers=[2.0, 1.0],
        )

    @classmethod
    def hnet_1stage_l12(cls) -> "HNetConfig":
        """
        Part-1 L12-matched 1-stage Byte H-Net (~155M params):
          - Outer Stage 0: 'm4' encoder & 'm4' decoder (d=512, pure SSM, no FFN)
          - Inner Main Net: 'T12' Transformer (d=896, head_dim=128, 50% RoPE=64, FFN=3200)
          - Target compression: 4.57 bytes/chunk
        """
        return cls(
            d_model=[512, 896],
            encoder_layout=["m4"],
            decoder_layout=["m4"],
            main_layout="T12",
            head_dim=[64, 128],
            rotary_emb_dim=[32, 64],
            window_size=[1023, -1],
            target_compression=[4.57],
            lr_multipliers=[2.5, 1.0],
        )

    @classmethod
    def hnet_1stage_l16(cls) -> "HNetConfig":
        """
        Part-1 L16-matched 1-stage Byte H-Net (~345M params):
          - Outer Stage 0: 'm4' encoder & 'm4' decoder (d=768, pure SSM, no FFN)
          - Inner Main Net: 'T16' Transformer (d=1152, head_dim=128, 50% RoPE=64, FFN=4096)
          - Target compression: 4.57 bytes/chunk
        """
        return cls(
            d_model=[768, 1152],
            encoder_layout=["m4"],
            decoder_layout=["m4"],
            main_layout="T16",
            head_dim=[64, 128],
            rotary_emb_dim=[32, 64],
            window_size=[1023, -1],
            target_compression=[4.57],
            lr_multipliers=[2.5, 1.0],
        )

    @classmethod
    def medium_2stage_h100(cls) -> "HNetConfig":
        """
        2-stage Byte H-Net matching goombalab/hnet's ['m4', ['T1m4', ['T16'], 'm4T1'], 'm4'] layout:
          - Stage 0: 'm4' encoder & 'm4' decoder (4x compression)
          - Stage 1: 'T1m4' encoder & 'm4T1' decoder with sliding-window attention (2x compression)
          - Stage 2 (Core): 'T16' full-attention Transformer (~8x total compression)
        """
        return cls(
            d_model=[384, 512, 768],
            encoder_layout=["m4", "T1m4"],
            decoder_layout=["m4", "m4T1"],
            main_layout="T16",
            head_dim=[64, 64, 64],
            rotary_emb_dim=[32, 32, 32],
            window_size=[1023, 1023, -1],
            target_compression=[4.0, 2.0],
            lr_multipliers=[3.0, 1.7, 1.0],
        )
