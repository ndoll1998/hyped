import os
import json
import hyped
import datasets
import logging
# utils
from itertools import product
from hyped.scripts.train import (
    RunConfig,
    build_trainer,
    load_data_split
)

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
    config.trainer.output_dir = os.path.join("/tmp", config.trainer.output_dir)

    # load model and activate first adapter and all heads
    model = hyped.modeling.HypedAutoAdapterModel.from_pretrained(args.model_ckpt)
    # ativate adapter
    if config.model.adapter is not None:
        # fallback to first adapter in model
        adapter = config.model.adapter_name or next(iter(model.config.adapters))
        model.active_adapters = adapter
    # activate all prediciton heads
    model.active_heads = list(config.model.heads.keys())

    # trainer but we're only using it for evaluation
    trainer = None

    # create directory to save metrics in
    fpath = os.path.join(args.model_ckpt, "metrics")
    fpath = args.out_dir if args.out_dir is not None else fpath
    os.makedirs(fpath, exist_ok=True)

    for path, split in product(args.data, args.splits):
        # load dataset
        data = load_data_split(path, split)
        name = data.info.builder_name

        # build trainer on first iteration
        trainer = trainer or build_trainer(
            trainer_t=config.model.trainer_t,
            model=model,
            args=config.trainer,
            features=data.features,
            metric_configs=config.metrics,
            disable_tqdm=False
        )
        # log dataset to evaluate
        logger.info("Evaluating dataset %s" % name)

        # evaluate model on dataset
        metrics = trainer.evaluate(data, metric_key_prefix=split)
        logger.info(metrics)
        # save metrics in checkpoint directory
        with open(os.path.join(fpath, "%s-%s.json" % (name, split)), 'w+') as f:
            f.write(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
