import os
import json
import logging
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
    data:list[str],
    splits:list[str],
    out_dir:str
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

    # load model
    if config.model.library == 'transformers':
        # load pretrained model and wrap
        model = config.model.auto_class.from_pretrained(model_ckpt)
        model = modeling.TransformerModelWrapper(model, head_name=config.model.head_name)

    elif config.model.library == 'adapter-transformers':
        # load model and activate first adapter and all heads
        model = modeling.HypedAutoAdapterModel.from_pretrained(model_ckpt)
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
            tokenizer=config.model.build_tokenizer(),
            model=model,
            args=config.trainer,
            metric_configs=config.metrics
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