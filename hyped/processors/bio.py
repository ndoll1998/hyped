from .processor import DataProcessor, DataProcessorConfig
from dataclasses import dataclass
from typing import Any, Literal

from transformers import PreTrainedTokenizer
from datasets.tasks import TextClassification

@dataclass
class BioTaggingProcessorConfig(DataProcessorConfig):
    type:Literal['bio-tagging-processor'] = 'bio-tagging-processor'
    # dataset columns of interest
    tokens_column:str = "tokens"
    tags_column:str ="ner_tags"
    # tag prefixes
    begin_tag_prefix:str = "B-"
    in_tag_prefix:str = "I-"
    # preprocessing parameters
    max_length:int =256

class BioTaggingProcessor(DataProcessor):
    pass
