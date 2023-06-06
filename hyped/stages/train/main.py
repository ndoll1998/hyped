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
from hyped import modeling
from hyped.metrics import AutoHypedMetric
# config
from .configs.run import RunConfig

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
    model:hyped.modeling.HypedModelWrapper,
    args:transformers.TrainingArguments,
    metric_configs:dict[str, AnyHypedMetricConfig],
    local_rank:int =-1
) -> transformers.Trainer:
    """Create trainer instance ensuring correct interfacing between trainer and metrics"""
    # create fixed order over label names for all model heads
    label_names = chain.from_iterable(h_config.label_columns for h_config in model.head_configs)
    label_names = list(set(list(label_names)))
    # set label names order in arguments
    args.label_names = label_names
    # update local rank in trainer configuration
    args.local_rank = local_rank

    # create metrics
    metrics = AutoHypedMetric.from_model(
        model=model,
        metric_configs=metric_configs,
        label_order=args.label_names
    )

    # create data collator
    collator = modeling.HypedDataCollator(
        tokenizer=tokenizer,
        h_configs=model.head_configs,
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
        tokenizer=config.model.tokenizer,
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
