"""GRT models.

??? tip "Use `TransformerEncoder` as the GRT encoder"

    The GRT encoder is just a vanilla transformer encoder! If you're using
    hydra, use the following:
    ```yaml
    encoder:
      _target_: torch.nn.TransformerEncoder
      encoder_layer:
        _target_: torch.nn.TransformerEncoderLayer
        d_model: 512
        nhead: 8
        dim_feedforward: 2048
        dropout: 0.1
        activation: gelu
        layer_norm_eps: 1e-5
        batch_first: true
        norm_first: true
        bias: true
      num_layers: 4
      enable_nested_tensor: false
    ```
    To match the settings used by GRT, after selecting the appropriate
    `d_model` and `num_layers`:

    - Set `dim_feedforward` to `4.0 * d_model`
    - Set `n_head` to `d_model // 64`
"""

from nrdk._typecheck import typechecker

with typechecker("nrdk.models.grt"):
    from .decoders import MLPVectorDecoder, TransformerTensorDecoder
    from .tokenizers import SpectrumAngularMLPTokenizer, SpectrumTokenizer

__all__ = [
    "SpectrumTokenizer",
    "SpectrumAngularMLPTokenizer",
    "TransformerTensorDecoder",
    "MLPVectorDecoder",
]
