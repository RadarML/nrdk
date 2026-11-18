"""GRTv2 models.

GRTv2 uses a single shared decoder for all tasks: each task supplies a set of
output queries (positions to predict at), which are positionally encoded and
decoded against the encoded tokens, followed by a lightweight per-task head.

- [`Encoder`][nrdk.models.grtv2.Encoder]: a
    [`Tokenizer`][nrdk.models.grtv2.Tokenizer], an
    [`InputPositionalEncoder`][nrdk.models.grtv2.InputPositionalEncoder], and
    a transformer stem (nominally a
    [`nrdk.modules.AATransformerEncoder`][nrdk.modules.AATransformerEncoder]).
- [`OutputPositionalEncoder`][nrdk.models.grtv2.OutputPositionalEncoder]:
    positional encoding for [`Query`][nrdk.models.grtv2.Query] positions.
- [`LinearHead`][nrdk.models.grtv2.LinearHead]: per-task output head.
"""

from nrdk._typecheck import typechecker

with typechecker("nrdk.models.grtv2"):
    from .backbone import Encoder, Tokenizer
    from .heads import LinearHead
    from .positions import InputPositionalEncoder, OutputPositionalEncoder
    from .query import Query

__all__ = [
    "Tokenizer", "Encoder",
    "InputPositionalEncoder", "OutputPositionalEncoder",
    "Query",
    "LinearHead",
]
