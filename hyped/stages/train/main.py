from __future__ import annotations

import os
import json
import datasets
import transformers
import pydantic
import dataclasses
import logging
# utils
from copy import copy
from datetime import datetime
from functools import partial
from itertools import chain, product
from typing import Any, Optional, Literal
from typing_extensions import Annotated
# hyped
from . import modeling
from .metrics import HypedAutoMetric
from .metrics.metrics import AnyHypedMetricConfig

import warnings
# ignore warning of _n_gpu field of TrainingArguments
# dataclass when converted to pydantic model
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="fields may not start with an underscore, ignoring \"_n_gpu\""
)

# TODO: log more stuff
logger = logging.getLogger(__name__)

class TransformerModelConfig(pydantic.BaseModel):
    """Transformer Model Configuration Model"""
    library:Literal['transformers'] = 'transformers'
    # task and head name
    task:str
    head_name:str
    label_column:str = "labels"
    # base model
    pretrained_ckpt:str
    kwargs:dict ={}

    @pydantic.validator('pretrained_ckpt', pre=True)
    def _check_pretrained_ckpt(cls, value):
        try:
            # check if model is valid by loading config
            transformers.AutoConfig.from_pretrained(value)
        except OSError as e:
            # handle model invalid
            raise ValueError("Unkown pretrained checkpoint: %s" % value) from e

        return value

    @property
    def problem_type(self) -> str:
        problem_type_lookup = {
            "sequence-classification": "single_label_classification",
            "multi-label-classification": "multi_label_classification",
            "sequence-tagging": "token_classification",
            "causal-language-modeling": "causal-language-modeling"
        }
        # check if task is valid
        if self.task not in problem_type_lookup:
            raise ValueError("Invalid task `%s`, must be one of %s" % (value, list(problem_type_lookup.keys())))
        # return problem type for task
        return problem_type_lookup[self.task]

    @property
    def auto_class(self) -> type:
        auto_class_lookup = {
            "sequence-classification": transformers.AutoModelForSequenceClassification,
            "multi-label-classification": transformers.AutoModelForSequenceClassification,
            "sequence-tagging": transformers.AutoModelForTokenClassification,
            "causal-language-modeling": transformers.AutoModelForCausalLM
        }
        # check if task is valid
        if self.task not in auto_class_lookup:
            raise ValueError("Invalid task `%s`, must be one of %s" % (value, list(auto_class_lookup.keys())))
        # return problem type for task
        return auto_class_lookup[self.task]

    @property
    def head_config_class(self) -> type:
        head_config_lookup = {
            "sequence-classification": modeling.heads.HypedClsHeadConfig,
            "multi-label-classification": modeling.heads.HypedMlcHeadConfig,
            "sequence-tagging": modeling.heads.HypedTaggingHeadConfig,
            "causal-language-modeling": modeling.heads.HypedCausalLMHeadConfig
        }
        # check if task is valid
        if self.task not in head_config_lookup:
            raise ValueError("Invalid task `%s`, must be one of %s" % (value, list(head_config_lookup.keys())))
        # return problem type for task
        return head_config_lookup[self.task]

    def build(self, info:datasets.DatasetInfo) -> transformers.PreTrainedModel:

        # load pretrained config
        config, kwargs = transformers.AutoConfig.from_pretrained(self.pretrained_ckpt, **self.kwargs, return_unused_kwargs=True)

        # create a head config with the correct labels
        # only used for generating the label space
        head_config = self.head_config_class(label_column=self.label_column)
        head_config.check_and_prepare(info.features)
        # update num labels and label2id mapping
        if head_config.num_labels is not None:
            config.num_labels = head_config.num_labels
        if head_config.id2label is not None:
            config.id2label = head_config.id2label
        # set the problem type, especially important for sequence classification
        # tells the model whether to solve a single- or multi-label sequence classification task
        config.problem_type = self.problem_type

        # load pretrained model and wrap it
        model = self.auto_class.from_pretrained(self.pretrained_ckpt, config=config, **kwargs)
        model = modeling.TransformerModelWrapper(
            model,
            head_name=self.head_name,
            head_config=head_config
        )
        # return wrapped model instance
        return model

    def build_tokenizer(self) -> transformers.PreTrainedTokenizer:
        return transformers.AutoTokenizer.from_pretrained(self.pretrained_ckpt, use_fast=True)

    @property
    def trainer_t(self) -> type[transformers.Trainer]:
        # use the default transformers trainer
        return transformers.Trainer

class AdapterTransformerModelConfig(pydantic.BaseModel):
    """Adapter Transformer Model Configuration Model"""
    library:Literal['adapter-transformers'] = 'adapter-transformers'
    # base model
    pretrained_ckpt:str
    kwargs:dict ={}
    # adapter setup
    adapter_name:None|str = None # defaults to dataset name
    adapter:None|transformers.adapters.AdapterArguments = None
    # prediction heads
    heads:dict[
        str,
        Annotated[
            modeling.heads.AnyHypedHeadConfig,
            pydantic.Field(..., discriminator='head_type')
        ]
    ]

    def check_and_prepare(self, features:datasets.Features) -> None:
        [hconfig.check_and_prepare(features) for hconfig in self.heads.values()]

    def build(self, info:datasets.DatasetInfo) -> transformers.PreTrainedModel:
        # set default adapter name and prepare model for data
        self.adapter_name = self.adapter_name or info.builder_name
        self.check_and_prepare(info.features)

        # load pretrained configuration and wrap for adapter model
        config, kwargs = transformers.AutoConfig.from_pretrained(self.pretrained_ckpt, **self.kwargs, return_unused_kwargs=True)
        config = transformers.adapters.wrappers.configuration.wrap_config(config)
        # add prediction head configs
        config.prediction_heads = {hname: dataclasses.asdict(hconfig) for hname, hconfig in self.heads.items()}

        # build the model
        model = modeling.HypedAutoAdapterModel.from_pretrained(
            self.pretrained_ckpt,
            config=config,
            **kwargs
        )
        # activate all heads
        model.active_head = list(model.heads.keys())
        # set up adapter
        if self.adapter is not None:
            # check if name is set
            if self.adapter_name is None:
                raise ValueError("`adapter_name` in model configuration not set!")
            # set up adapter
            transformers.adapters.training.setup_adapter_training(
                model=model,
                adapter_args=dataclasses.replace(
                    self.adapter,
                    train_adapter=True
                ),
                adapter_name=self.adapter_name
            )

            # unfreeze model parameters when not only 
            # training adapter weights 
            if not self.adapter.train_adapter:
                model.freeze_model(False)

        # return model instance
        return model

    def build_tokenizer(self) -> transformers.PreTrainedTokenizer:
        return transformers.AutoTokenizer.from_pretrained(self.pretrained_ckpt, use_fast=True)

    @property
    def trainer_t(self) -> type[transformers.Trainer]:
        use_adapter_trainer = (self.adapter is not None) and self.adapter.train_adapter
        return modeling.MultiHeadAdapterTrainer if use_adapter_trainer else \
            modeling.MultiHeadTrainer

    @pydantic.validator('pretrained_ckpt', pre=True)
    def _check_pretrained_ckpt(cls, value):
        try:
            # check if model is valid by loading config
            transformers.AutoConfig.from_pretrained(value)
        except OSError as e:
            # handle model invalid
            raise ValueError("Unkown pretrained checkpoint: %s" % value) from e

        return value

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

class RunConfig(pydantic.BaseModel):
    """Run Configuration Model"""
    # run name
    name:str
    # model and trainer configuration
    model:TransformerModelConfig|AdapterTransformerModelConfig = pydantic.Field(..., discriminator='library')
    trainer:TrainerConfig
    metrics:dict[
        str,
        list[
            Annotated[
                AnyHypedMetricConfig,
                pydantic.Field(..., discriminator='metric_type')
            ]
        ]
    ]

    @pydantic.validator('model', pre=True)
    def _infer_model_library(cls, value):
        if 'library' not in value:

            if ('heads' in value) and ('task' in value):
                raise ValueError("Could not infer library from model config, both `heads` and `task` field specified!")

            if 'heads' in value:
                # if heads are present then this is an adapter model
                value['library'] = "adapter-transformers"

            elif 'task' in value:
                # if task is specified then this is a pure transformer model
                value['library'] = "transformers"

            else:
                raise ValueError("Could not infer library from model config, neither `heads` nor `task` field specified!")

        return value

    @pydantic.validator('trainer', pre=True)
    def _pass_name_to_trainer_config(cls, v, values):
        assert 'name' in values
        if isinstance(v, pydantic.BaseModel):
            return v.copy(update={'name': values.get('name')})
        elif isinstance(v, dict):
            return v | {'name': values.get('name')}

def get_format_info(data:datasets.Dataset) -> datasets.Features:
    return dataclasses.replace(
        data.info,
        task_templates=[],
        features=data.info.features.copy() if data.format['columns'] is None else \
            datasets.Features({n: data.info.features[n] for n in data.format['columns']})
    )

def load_data_split(path:str, split:str) -> datasets.Dataset:
    # check if specific dataset split exists
    dpath = os.path.join(path, str(split))
    if not os.path.isdir(dpath):
        raise FileNotFoundError(dpath)
    # load split
    in_memory = os.environ.get("HF_DATASETS_FORCE_IN_MEMORY", None)
    data = datasets.load_from_disk(dpath, keep_in_memory=in_memory)
    logger.debug("Loaded data from `%s`" % dpath)
    # return loaded dataset
    return data

def combine_infos(infos:list[datasets.DatasetInfo]):

    first = copy(infos[0])
    # check if features match up
    for info in infos[1:]:
        if info.features == first.features:
            raise ValueError("Dataset features for `%s` and `%s` don't match up." % (first.builder_name, info.builder_name))
    # build full name
    first.builder_name = '_'.join([info.builder_name for info in infos])
    return first

def collect_data(
    data_dumps:list[str],
    splits:list[str] = [
        datasets.Split.TRAIN,
        datasets.Split.VALIDATION,
        datasets.Split.TEST
    ],
    in_memory:bool =False
) -> datasets.DatasetDict:

    ds = {split: [] for split in splits}
    # load dataset splits of interest
    for path, split in product(data_dumps, splits):
        try:
            # try to load data split
            data = load_data_split(path, split)
            ds[split].append(data)
        except FileNotFoundError:
            pass

    # concatenate datasets
    return datasets.DatasetDict({
        split: datasets.concatenate_datasets(data, info=combine_infos([d.info for d in data]), split=split)
        for split, data in ds.items()
        if len(data) > 0
    })

def build_trainer(
    trainer_t:type[transformers.Trainer],
    info:datasets.DatasetInfo,
    tokenizer:transformers.PreTrainedTokenizer,
    model:transformers.PreTrainedModel,
    args:transformers.TrainingArguments,
    metric_configs:dict[str, AnyHypedMetricConfig],
    local_rank:int =-1
) -> transformers.Trainer:
    """Create trainer instance ensuring correct interfacing between trainer and metrics"""

    # create fixed order over label names for all model heads
    label_names = chain.from_iterable(h.get_label_names() for h in model.heads.values())
    label_names = list(set(list(label_names)))
    # set label names order in arguments
    args.label_names = label_names
    # update local rank in trainer configuration
    args.local_rank = local_rank

    # create metrics
    metrics = HypedAutoMetric.from_model(
        model=model,
        metric_configs=metric_configs,
        label_order=args.label_names
    )

    # create data collator
    collator = modeling.HypedDataCollator(
        tokenizer=tokenizer,
        heads=model.heads.values(),
        features=info.features
    )

    # create trainer instance
    trainer = trainer_t(
        model=model,
        args=args,
        # datasets need to be set manually
        train_dataset=None,
        eval_dataset=None,
        # data collator
        data_collator=collator,
        # compute metrics
        preprocess_logits_for_metrics=metrics.preprocess,
        compute_metrics=metrics.compute
    )
    # add early stopping callback
    trainer.add_callback(
        transformers.EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold
        )
    )
    # return trainer instance
    return trainer

def train(
    config:RunConfig,
    ds:datasets.DatasetDict,
    output_dir:str = None,
    local_rank:int = -1,
    disable_tqdm:bool = False
) -> transformers.Trainer:

    # check for train and validation datasets
    if datasets.Split.TRAIN not in ds:
        raise KeyError("No train dataset found, got %s!" % list(ds.keys()))
    if datasets.Split.VALIDATION not in ds:
        raise KeyError("No validation dataset found, got %s!" % list(ds.keys()))

    # update trainer arguments
    config.trainer.output_dir = output_dir or args.output_dir
    config.trainer.disable_tqdm = disable_tqdm

    # get dataset info but replace features with restricted features
    data = next(iter(ds.values()))
    info = get_format_info(data)
    # build model and trainer
    trainer = build_trainer(
        trainer_t=config.model.trainer_t,
        info=info,
        tokenizer=config.model.build_tokenizer(),
        model=config.model.build(info),
        args=config.trainer,
        metric_configs=config.metrics,
        local_rank=local_rank
    )
    # set datasets
    trainer.train_dataset = ds[datasets.Split.TRAIN]
    trainer.eval_dataset = ds[datasets.Split.VALIDATION]

    # run trainer
    trainer.train()

    return trainer

def main(
    config:str,
    data:list[str],
    out_dir:str,
    local_rank:int =-1
) -> None:
    # check if config exists
    if not os.path.isfile(config):
        raise FileNotFoundError(config)

    # load config
    logger.info("Loading run configuration from %s" % config)
    config = RunConfig.parse_file(config)

    # run training
    splits = [datasets.Split.TRAIN, datasets.Split.VALIDATION]
    trainer = train(config, collect_data(data, splits), out_dir, local_rank)

    # save trainer model in output directory if given
    if out_dir is not None:
        trainer.save_model(os.path.join(out_dir, "best-model"))
