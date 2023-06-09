import pydantic
import dataclasses
import transformers
from datetime import datetime
from typing import Optional

import warnings
# ignore warning of _n_gpu field of TrainingArguments
# dataclass when converted to pydantic model
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="fields may not start with an underscore, ignoring \"_n_gpu\""
)

@pydantic.dataclasses.dataclass
@dataclasses.dataclass
class TrainerConfig(transformers.TrainingArguments):
    """ Trainer Configuration """
    # passed fromi run config and needed for output directory
    name:str =None
    # create default for output directory
    run_name:str ="{name}-{timestamp}"
    output_dir:str ="output/{name}-{timestamp}"
    overwrite_output_dir:bool =True
    # early stopping setup
    early_stopping_patience:Optional[int] =1
    early_stopping_threshold:Optional[float] =0.0
    # checkpointing
    load_best_model_at_end:bool =True
    metric_for_best_model:str ='eval_loss'
    greater_is_better:bool =False
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
    deepspeed:Optional[str|dict]                                       =None
    # don't do that because we use args and kwargs in the
    # model's forward function which confuses the trainer
    remove_unused_columns:bool =False

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
