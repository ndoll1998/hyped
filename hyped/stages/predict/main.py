import os
import json
import logging
import datasets
# utils
from itertools import product
from hyped.stages.train import modeling
from hyped.stages.train.main import (
    RunConfig,
    get_format_info,
    build_trainer,
    load_data_split
)

logger = logging.getLogger(__name__)

def main(
    config:str,
    model_ckpt:str,
    data:str,
    splits:list[str],
    out_dir:str,
    local_rank:int =-1
) -> None:

    # check if config exists
    if not os.path.isfile(config):
        raise FileNotFoundError(config)
    # load config
    logger.info("Loading run configuration from %s" % config)
    config = RunConfig.parse_file(config)

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

    data_path = data
    for split in splits:
        # load dataset
        ds = load_data_split(data_path, split)
        name = ds.info.builder_name

        # build trainer on first iteration
        trainer = trainer or build_trainer(
            trainer_t=config.model.trainer_t,
            info=get_format_info(ds),
            tokenizer=config.model.build_tokenizer(),
            model=model,
            args=config.trainer,
            metric_configs=config.metrics,
            local_rank=local_rank
        )
        # log dataset to evaluate
        logger.info("Predicting %s split of dataset %s" % (split, name))

        # evaluate model on dataset
        preds = trainer.predict(ds).predictions
        preds_ds = [p.convert_to_dataset(o) for p, o in preds.items()]
        assert all(len(ds) == len(p) for p in preds_ds)
        # concatenate all prediction datasets with the source dataset
        out_ds = datasets.concatenate_datasets([ds] + preds_ds, axis=1)

        # save dataset to disk
        os.makedirs(os.path.join(out_dir, split), exist_ok=True)
        out_ds.save_to_disk(os.path.join(out_dir, split))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
