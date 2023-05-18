import os
import json
import hyped
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
from typing import Any, Optional
from typing_extensions import Annotated

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

class ModelConfig(pydantic.BaseModel):
    """Model Configuration Model"""
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
            hyped.modeling.heads.AnyHypedHeadConfig,
            pydantic.Field(..., discriminator='head_type')
        ]
    ]

    def check_and_prepare(self, features:datasets.Features) -> None:
        [hconfig.check_and_prepare(features) for hconfig in self.heads.values()]

    @property
    def pretrained_config(self) -> transformers.PretrainedConfig:
        # load pretrained configuration and wrap for adapter model
        config = transformers.AutoConfig.from_pretrained(self.pretrained_ckpt)
        config = transformers.adapters.wrappers.configuration.wrap_config(config)
        # add prediction head configs
        config.prediction_heads = {hname: dataclasses.asdict(hconfig) for hname, hconfig in self.heads.items()}
        # add adapter configs if needed
        if self.adapter is not None:
            if self.adapter_name is None:
                raise ValueError("`adapter_name` in model configuration not set!")
            if self.adapter_name not in config.adapters:
                adapter_config = transformers.adapters.AdapterConfig.load(self.adapter.adapter_config)
                config.adapters.add(self.adapter_name, config=adapter_config)
        # return config
        return config

    @property
    def trainer_t(self) -> type[transformers.Trainer]:
        use_adapter_trainer = (self.adapter is not None) and self.adapter.train_adapter
        return hyped.modeling.MultiHeadAdapterTrainer if use_adapter_trainer else hyped.modeling.MultiHeadTrainer

    @pydantic.validator('pretrained_ckpt')
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
    model:ModelConfig
    trainer:TrainerConfig
    metrics:dict[
        str,
        list[
            Annotated[
                hyped.evaluate.metrics.AnyHypedMetricConfig,
                pydantic.Field(..., discriminator='metric_type')
            ]
        ]
    ]

    @pydantic.validator('trainer', pre=True)
    def _pass_name_to_trainer_config(cls, v, values):
        assert 'name' in values
        if isinstance(v, pydantic.BaseModel):
            return v.copy(update={'name': values.get('name')})
        elif isinstance(v, dict):
            return v | {'name': values.get('name')}


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
    model:transformers.PreTrainedModel,
    args:transformers.TrainingArguments,
    features:datasets.Features,
    metric_configs:dict[str, hyped.evaluate.metrics.AnyHypedMetricConfig],
    output_dir:str = None,
    disable_tqdm:bool =False
) -> transformers.Trainer:
    """Create trainer instance ensuring correct interfacing between trainer and metrics"""

    # create fixed order over label names
    label_names = chain.from_iterable(h.get_label_names() for h in model.heads.values())
    label_names = list(set(list(label_names)))
    # specify label columns and overwrite output directory if given
    args.label_names = label_names
    args.output_dir = output_dir or args.output_dir
    # disable tqdm
    args.disable_tqdm = disable_tqdm

    # create metrics
    metrics = hyped.evaluate.HypedAutoMetric.from_model(
        model=model,
        metric_configs=metric_configs,
        label_order=args.label_names
    )

    # create data collator
    collator = hyped.modeling.HypedDataCollator(
        heads=model.heads.values(),
        features=features
    )

    # create trainer instance
    return trainer_t(
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

def train(
    config:RunConfig,
    ds:datasets.DatasetDict,
    output_dir:str = None,
    disable_tqdm:bool = False
) -> transformers.Trainer:

    # check for train and validation datasets
    if datasets.Split.TRAIN not in ds:
        raise KeyError("No train dataset found, got %s!" % list(ds.keys()))
    if datasets.Split.VALIDATION not in ds:
        raise KeyError("No validation dataset found, got %s!" % list(ds.keys()))

    # set default adapter name
    config.model.adapter_name = config.model.adapter_name or \
        ds[datasets.Split.TRAIN].info.builder_name

    # prepare model for data
    features = ds[datasets.Split.TRAIN].info.features
    config.model.check_and_prepare(features)
    # build the model
    model = hyped.modeling.HypedAutoAdapterModel.from_pretrained(
        config.model.pretrained_ckpt,
        config=config.model.pretrained_config,
        **config.model.kwargs
    )
    # activate all heads
    model.active_head = list(model.heads.keys())
    # set up adapter
    if config.model.adapter is not None:
        transformers.adapters.training.setup_adapter_training(
            model=model,
            adapter_args=config.model.adapter,
            adapter_name=config.model.adapter_name
        )

    trainer = build_trainer(
        trainer_t=config.model.trainer_t,
        model=model,
        args=config.trainer,
        features=features,
        metric_configs=config.metrics,
        output_dir=output_dir,
        disable_tqdm=disable_tqdm
    )
    # set train and validation datasets
    trainer.train_dataset=ds[datasets.Split.TRAIN]
    trainer.eval_dataset=ds[datasets.Split.VALIDATION]
    # add early stopping callback
    trainer.add_callback(
        transformers.EarlyStoppingCallback(
            early_stopping_patience=config.trainer.early_stopping_patience,
            early_stopping_threshold=config.trainer.early_stopping_threshold
        )
    )

    # run trainer
    trainer.train()

    return trainer

def main():
    from argparse import ArgumentParser
    # build argument parser
    parser = ArgumentParser(description="Train Transformer model on prepared datasets")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to run configuration file in .json format")
    parser.add_argument("-d", "--data", type=str, nargs='+', required=True, help="Paths to prepared data dumps")
    parser.add_argument("-o", "--out-dir", type=str, default=None, help="Output directory, by default uses directoy specified in config")
    # parse arguments
    args = parser.parse_args()

    # check if config exists
    if not os.path.isfile(args.config):
        raise FileNotFoundError(args.config)
    # load config
    logger.info("Loading run configuration from %s" % args.config)
    config = RunConfig.parse_file(args.config)

    # run training
    splits = [datasets.Split.TRAIN, datasets.Split.VALIDATION]
    trainer = train(config, collect_data(args.data, splits), args.out_dir)

    # save trainer model in output directory if given
    if args.out_dir is not None:
        trainer.save_model(os.path.join(args.out_dir, "best-model"))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
