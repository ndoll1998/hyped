from .text_cls import TextClassificationHead, TextClassificationHeadConfig

from transformers import (
    AutoConfig,
    AutoModel
)

__all__ = [
    "TextClassificationHead",
    "TextClassificationHeadConfig",
]

# register all head configs
AutoConfig.register(TextClassificationHeadConfig.model_type, TextClassificationHeadConfig)
AutoModel.register(TextClassificationHeadConfig, TextClassificationHead)
