"""MK-SH-V: decision model over text, documents and screenshots.

Reuses the MK-SH-1 decision contract and its calibration/policy/evaluation
code by importing `mokka_decisions` (pinned git dependency); this package
only adds the visual side: the NeoMME marker-scorer architecture (A1-V), the
synthetic document generator and template-keyed splitting.
"""

from mokka_decisions.contracts import (  # re-exported contract surface
    NONE_ID,
    SPLITS,
    ContractError,
    DecisionCase,
    DecisionResponse,
    Option,
)

__version__ = "0.1.0"

__all__ = [
    "NONE_ID",
    "SPLITS",
    "ContractError",
    "DecisionCase",
    "DecisionResponse",
    "Option",
    "__version__",
]
