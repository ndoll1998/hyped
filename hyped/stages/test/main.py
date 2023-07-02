import os
import json
import logging
# utils
from itertools import product
from hyped import modeling
from hyped.stages.train.main import (
    ExperimentConfig,
    get_format_info,
    build_trainer,
    load_data_split
)

logger = logging.getLogger(__name__)

def main(
    config:str,
    model_ckpt:str,
    data:list[str],
    splits:list[str],
    out_dir:str,
    local_rank:int =-1
) -> None:

    # check if config exists
    if not os.path.isfile(config):
        raise FileNotFoundError(config)
    # load config
    logger.info("Loading run configuration from %s" % config)
    config = ExperimentConfig.parse_file(config)

    # prepare config for evaluation
    config.trainer.save_strategy = 'no'
    # not used but created and there is no way around i guess
    config.trainer.output_dir = os.path.join("/tmp", config.trainer.output_dir)

    # load model from checkpoint
    model = config.model.load(model_ckpt)

    # trainer but we're only using it for evaluation
    trainer = None

    # create directory to save metrics in
    fpath = os.path.join(model_ckpt, "metrics")
    fpath = out_dir if out_dir is not None else fpath
    os.makedirs(fpath, exist_ok=True)

    for path, split in product(data, splits):
        # load dataset
        data = load_data_split(path, split)
        name = data.info.builder_name

        # build trainer on first iteration
        trainer = trainer or build_trainer(
            trainer_t=config.model.trainer_t,
            info=get_format_info(data),
            tokenizer=config.model.tokenizer,
            model=model,
            args=config.trainer,
            metric_configs=config.metrics,
            local_rank=local_rank
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
