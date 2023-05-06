import hyped
import datasets
import transformers
import pydantic
import dataclasses
from datetime import datetime
from typing_extensions import Annotated
from typing import Optional

import warnings
# ignore warning of _n_gpu field of TrainingArguments
# dataclass when converted to pydantic model
warnings.filterwarnings("ignore",
    category=RuntimeWarning,
    message="fields may not start with an underscore, ignoring \"_n_gpu\""
)

class DataConfig(pydantic.BaseModel):
    """Data Configuration Model"""
    # dataset config
    dataset:str
    splits:list[str] = [datasets.Split.TRAIN, datasets.Split.TEST]

    # preprocessing pipeline
    pipeline:list[hyped.AnyProcessorConfig]
    filters:list[hyped.AnyFilterConfig]

    # columns to keep
    columns:dict[str, str]

    #pipeline:list[
    #    Annotated[
    #        hyped.AnyProcessorConfig,
    #        pydantic.Field(..., discriminator='processor_type')
    #    ]
    #]
    # data filters
    #filters:list[
    #    Annotated[
    #        hyped.AnyFilterConfig,
    #        pydantic.Field(..., discriminator='filter_type')
    #    ]
    #]

    @pydantic.validator('dataset')
    def validate_dataset(cls, v):
        if v is None:
            raise ValueError("No Dataset provided by configuration!")
        try:
            # try to load dataset builder
            builder = datasets.load_dataset_builder(v)
            return v
        except FileNotFoundError as e:
            # raise exception if dataset builder cannot be found
            raise ValueError("Dataset not found: %s" % v) from e

    @property
    def info(self) -> datasets.DatasetInfo:
        return datasets.load_dataset_builder(self.dataset)._info()

class ModelConfig(pydantic.BaseModel):
    """Model Configuration Model"""
    pretrained_ckpt:str

    @pydantic.validator('pretrained_ckpt')
    def _check_pretrained_ckpt(cls, value):
        try:
            # check if model is valid by loading config
            transformers.AutoConfig.from_pretrained(value)
        except OSError as e:
            # handle model invalid
            raise ValueError("Unkown pretrained checkpoint: %s" % value) from e

        return value

    @property
    def tokenizer(self) -> transformers.PreTrainedTokenizer:
        return transformers.AutoTokenizer.from_pretrained(
            self.pretrained_ckpt,
            use_fast=True,
            add_prefix_space=True
        )

@pydantic.dataclasses.dataclass
@dataclasses.dataclass
class TrainerConfig(transformers.TrainingArguments):
    """ Trainer Configuration """
    # passed from experiment config and needed for output directory
    name:str =None
    # create default for output directory
    run_name:str ="{name}-{timestamp}"
    output_dir:str ="output/{name}-{timestamp}"
    overwrite_output_dir:bool =True
    # early stopping setup
    early_stopping_patience:Optional[int] =1
    early_stopping_threshold:Optional[float] =0.0
    # incremental training setup, i.e. continue training of
    # model in each AL iteration instead of resetting
    incremental:bool =True
    # checkpointing
    load_best_model_at_end:bool =True
    metric_for_best_model:str ='eval_loss'
    greater_is_better:bool =False
    # minimum steps between evaluations
    # overwrites epoch-based evaluation behavior
    min_epoch_length:Optional[int] =15
    # overwrite some default values
    do_train:bool =True
    do_eval:bool =True
    evaluation_strategy:transformers.trainer_utils.IntervalStrategy ="epoch"
    save_strategy:transformers.trainer_utils.IntervalStrategy ="epoch"
    eval_accumulation_steps:Optional[int] =1
    save_total_limit:Optional[int] =3
    label_names:list[str] =dataclasses.field(default_factory=lambda: ['labels'])
    report_to:Optional[list[str]] =dataclasses.field(default_factory=list)
    log_level:Optional[str] ='warning'
    # fields with incomplete types in Training Arguments
    # set type to avoid error in pydantic validation
    debug:str|list[transformers.debug_utils.DebugOption]               =""
    sharded_ddp:str|list[transformers.trainer_utils.ShardedDDPOption]  =""
    fsdp:str|list[transformers.trainer_utils.FSDPOption]               =""
    fsdp_config:Optional[str|dict]                                     =None

    # use pytorch implementation of AdamW optimizer
    # to avoid deprecation warning
    optim="adamw_torch"

    @pydantic.root_validator()
    def _format_output_directory(cls, values):
        # get timestamp
        timestamp=datetime.now().isoformat()
        # format all values depending on output directory
        return values | {
            'output_dir': values.get('output_dir').format(
                name=values.get('name'),
                timestamp=datetime.now().isoformat()
            ),
            'logging_dir': values.get('logging_dir').format(
                name=values.get('name'),
                timestamp=datetime.now().isoformat()
            ),
            'run_name': values.get('run_name').format(
                name=values.get('name'),
                timestamp=datetime.now().isoformat()
            ),
        }
