import os
import torch
import hyped
import datasets
import evaluate
import transformers
import numpy as np
import logging
# utils
import json
from typing import Any
from functools import partial
from hyped.scripts.utils.data import DataDump
from hyped.scripts.utils.configs import RunConfig

# TODO: log more stuff
logger = logging.getLogger(__name__)

class Metrics(object):

    def __init__(
        self,
        label_names:list[str],
        head_configs:dict[str,hyped.modeling.PredictionHeadConfig],
        metrics_per_head:dict[str,dict[str,Any]]
    ) -> None:
        # save label names and head configs
        self.label_names = label_names
        self.head_configs = head_configs
        # also create fixed order over heads
        self.head_names = list(head_configs.keys())
        # build metrics per head
        self.metrics_per_head = {
            head: [
                partial(evaluate.load(name).compute, **kwargs)
                for name, kwargs in metrics.items()
            ]
            for head, metrics in metrics_per_head.items()
        }

    def __call__(self, eval_pred:transformers.EvalPrediction):
        # unpack and build label lookups
        # note that labels are ordered by `trainer.config.label_names`
        preds, labels = eval_pred
        labels = (labels,) if isinstance(labels, np.ndarray) else labels
        labels = dict(zip(self.label_names, labels))

        # compute metrics
        return {
            "%s_%s" % (name, key): val
            for name, head in self.head_configs.items()
            for metric in self.metrics_per_head[name]
            for key, val in metric(
                predictions=preds[name],
                references=labels[head.label_column]
            ).items()
        }

    def preprocess_fn(self, logits, labels):
        """Preprocessing function taking arg-max of logits"""
        return {name: logits[name].argmax(dim=-1) for name in self.head_names}

def train(
    config:RunConfig,
    data_dumps:list[str],
    output_dir:None|str =None,
    disable_tqdm:bool =False
) -> transformers.Trainer:

    # check if data dump files exist
    for dpath in data_dumps:
        if not os.path.isfile(dpath):
            raise FileNotFoundError(dpath)

    data = {
        datasets.Split.TRAIN: [],
        datasets.Split.VALIDATION: [],
        datasets.Split.TEST: []
    }
    features = None
    # load data dumps
    for dpath in data_dumps:
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
                data[s].append(d)
            else:
                logger.warning("Discarding `%s` split of data dump %s." % (s, dpath))

    # build combined train and validation datasets
    train_data = torch.utils.data.ConcatDataset(data[datasets.Split.TRAIN])
    val_data = torch.utils.data.ConcatDataset(data[datasets.Split.VALIDATION])
    test_data = torch.utils.data.ConcatDataset(data[datasets.Split.TEST])

    # set label space
    config.model.check_and_prepare(features)
    # build the model
    model = hyped.modeling.ArbitraryEncoderWithHeads.from_pretrained_encoder(
        config.model.encoder_pretrained_ckpt,
        heads=config.model.heads,
        **config.model.kwargs
    )

    # specify label columns and overwrite output directory if given
    config.trainer.label_names = [h.label_column for h in config.model.heads.values()]
    config.trainer.output_dir = output_dir or config.trainer.output_dir
    # disable tqdm
    config.trainer.disable_tqdm = disable_tqdm

    # create metrics instance
    metrics = Metrics(
        label_names=config.trainer.label_names,
        head_configs=config.model.heads,
        metrics_per_head=config.metrics
    )

    # create trainer instance
    trainer = transformers.Trainer(
        model=model,
        args=config.trainer,
        # set datasets
        train_dataset=train_data,
        eval_dataset=val_data,
        # add early stopping callback
        callbacks=[
            transformers.EarlyStoppingCallback(
                early_stopping_patience=config.trainer.early_stopping_patience,
                early_stopping_threshold=config.trainer.early_stopping_threshold
            )
        ],
        # compute metrics
        preprocess_logits_for_metrics=metrics.preprocess_fn,
        compute_metrics=metrics
    )

    # train and test model
    trainer.train()
    test_metrics = trainer.evaluate(
        eval_dataset=test_data,
        metric_key_prefix="test"
    )

    # save test metrics to output directory
    with open(os.path.join(config.trainer.output_dir, "test_scores.json"), 'w+') as f:
        f.write(json.dumps(test_metrics, indent=4))

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
    train(config, args.data, args.out_dir, disable_tqdm=False)

if __name__ == '__main__':
    main()
