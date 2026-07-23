"""Compatibility import for the experiment4 inference scripts."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.exp4_prompts import (  # noqa: E402,F401
    IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE,
    IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE_MULTITURN,
)
