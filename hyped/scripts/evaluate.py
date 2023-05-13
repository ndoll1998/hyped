import os
import json
import torch
import hyped
import datasets
import logging
# utils
from hyped.scripts.train import build_trainer
from hyped.scripts.utils.configs import RunConfig

logger = logging.getLogger(__name__)

def main():
    from argparse import ArgumentParser
    # build argument parser
    parser = ArgumentParser(description="Train Transformer model on prepared datasets")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to run configuration file in .json format")
    parser.add_argument("-m", "--model-ckpt", type=str, required=True, help="Path to fine-tuned model checkpoint")
    parser.add_argument("-d", "--data", type=str, nargs='+', required=True, help="Paths to prepared data dumps")
    parser.add_argument("-s", "--splits", type=str, nargs='*', default=[datasets.Split.TEST], help="Subset of data splits to prepare, defaults to test split")
    parser.add_argument("-o", "--out-dir", type=str, default=None, help="Output directory, by default saves metrics in checkpoint")
    # parse arguments
    args = parser.parse_args()

    # check if config exists
    if not os.path.isfile(args.config):
        raise FileNotFoundError(args.config)
    # load config
    logger.info("Loading run configuration from %s" % args.config)
    config = RunConfig.parse_file(args.config)

    # prepare config for evaluation
    config.trainer.save_strategy = 'no'
    # not used but created and there is no way around i guess
    config.trainer.output_dir = os.path.join("/tmp", config.output_dir)

    # load model and activate all heads
    model = hyped.modeling.HypedAutoAdapterModel.from_pretrained(args.model_ckpt)
    model.active_heads = list(model.heads.keys())
    # build trainer but we're only using it for evaluation
    trainer = build_trainer(
        model=model,
        args=config.trainer,
        metrics_kwargs=config.metrics,
        disable_tqdm=False
    )

    # create directory to save metrics in
    fpath = os.path.join(args.model_ckpt, "metrics")
    fpath = args.out_dir if args.out_dir is not None else fpath
    os.makedirs(fpath, exist_ok=True)

    for dpath in args.data:
        # load data dump
        dump = torch.load(dpath)
        name = dump.name
        # log dataset to evaluate
        logger.info("Evaluating dataset %s" % name)

        for split in args.splits:
            if split in dump.datasets:
                # evaluate model on dataset
                metrics = trainer.evaluate(dump.datasets[split], metric_key_prefix=split)
                logger.info(metrics)
                # save metrics in checkpoint directory
                with open(os.path.join(fpath, "%s-%s.json" % (name, split)), 'w+') as f:
                    f.write(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
