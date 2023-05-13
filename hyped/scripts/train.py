import os
import json
import torch
import hyped
import datasets
import evaluate
import transformers
import numpy as np
import logging
# utils
from typing import Any
from itertools import chain
from functools import partial
from hyped.scripts.utils.data import DataDump
from hyped.scripts.utils.configs import RunConfig

# TODO: log more stuff
logger = logging.getLogger(__name__)

def collect_data(
    data_dumps:list[str],
    splits:list[str] = [
        datasets.Split.TRAIN,
        datasets.Split.VALIDATION,
        datasets.Split.TEST
    ]
) -> DataDump:

    data = {split: [] for split in splits}
    features = None
    # load data dumps
    for dpath in data_dumps:
        # check if data dump file exists
        if not os.path.isfile(dpath):
            raise FileNotFoundError(dpath)
        # load data
        dump = torch.load(dpath)
        # set features at first iteration
        features = features or dump.features
        # check feature compatibility
        if dump.features != features:
            raise ValueError("Features of dataset %s don't align with those of %s." % (
                dpath, data_dumps[0]))
        # add to total data
        for s, d in dump.datasets.items():
            if s in data:
                logger.debug("Loaded `%s` split of data dump %s." % (s, dpath))
                data[s].append(d)

    # create data dump object for collected datasets
    return DataDump(
        name="combined",
        features=features,
        datasets={
            k: torch.utils.data.ConcatDataset(ds)
            for k, ds in data.items()
            if len(ds) > 0
        }
    )

def build_trainer(
    model:transformers.PreTrainedModel,
    args:transformers.TrainingArguments,
    metrics_kwargs:dict ={},
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
    metrics = hyped.evaluate.HypedAutoMetrics.from_model(
        model=model,
        metrics_kwargs=metrics_kwargs,
        label_order=args.label_names
    )

    # create trainer instance
    return hyped.modeling.MultiHeadTrainer(
        model=model,
        args=args,
        # datasets need to be set manually
        train_dataset=None,
        eval_dataset=None,
        # compute metrics
        preprocess_logits_for_metrics=metrics.preprocess,
        compute_metrics=metrics
    )

def train(
    config:RunConfig,
    data_dump:DataDump,
    output_dir:str = None,
    disable_tqdm:bool = False
) -> transformers.Trainer:

    # check for train and validation datasets
    if datasets.Split.TRAIN not in data_dump.datasets:
        raise KeyError("No train dataset found, got %s!" % list(data_dump.datasets.keys()))
    if datasets.Split.VALIDATION not in data_dump.datasets:
        raise KeyError("No validation dataset found, got %s!" % list(data_dump.datasets.keys()))

    # prepare model for data
    config.model.check_and_prepare(data_dump.features)
    # build the model
    model = hyped.modeling.HypedAutoAdapterModel.from_pretrained(
        config.model.pretrained_ckpt,
        config=config.model.pretrained_config,
        **config.model.kwargs
    )
    # activate all heads
    model.active_head = list(model.heads.keys())

    trainer = build_trainer(
        model=model,
        args=config.trainer,
        metrics_kwargs=config.metrics,
        output_dir=output_dir,
        disable_tqdm=disable_tqdm
    )
    # set train and validation datasets
    trainer.train_dataset=data_dump.datasets[datasets.Split.TRAIN]
    trainer.eval_dataset=data_dump.datasets[datasets.Split.VALIDATION]
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
