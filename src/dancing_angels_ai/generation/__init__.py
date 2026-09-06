"""Animation-generation interfaces and implementations."""

from dancing_angels_ai.generation.base import AnimationBackend
from dancing_angels_ai.generation.mlx_video import (
    ACTION_PROMPTS,
    MlxVideoBackend,
    MlxVideoConfig,
    MlxVideoGenerationError,
    prompt_for_action,
)
from dancing_angels_ai.generation.wan import (
    WanAnimateBackend,
    WanGenerationConfig,
    WanGenerationError,
)

__all__ = [
    "ACTION_PROMPTS",
    "AnimationBackend",
    "MlxVideoBackend",
    "MlxVideoConfig",
    "MlxVideoGenerationError",
    "WanAnimateBackend",
    "WanGenerationConfig",
    "WanGenerationError",
    "prompt_for_action",
]
