import os
import torch
import hyped
import datasets
import transformers
import logging
# utils
from hyped.scripts.utils.data import NamedTensorDataset
from hyped.scripts.utils.configs import RunConfig

logger = logging.getLogger(__name__)

def train(
    config:RunConfig,
    data_dumps:list[str],
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
        ds = torch.load(dpath)
        f = ds.pop('__features')
        # set features at first iteration
        features = features or f
        # check feature compatibility
        if f != features:
            raise ValueError("Features of dataset %s don't align with those of %s." % (
                dpath, data_dumps[0]))
        # add to total data
        for s, d in ds.items():
            if s in data:
                data[s].append(d)
            else:
                logger.warning("Discarding `%s` split of data dump %s." % (s, dpath))

    # build combined train and validation datasets
    train_data = torch.utils.data.ConcatDataset(data[datasets.Split.TRAIN])
    val_data = torch.utils.data.ConcatDataset(data[datasets.Split.VALIDATION])

    # set label space
    config.model.set_label_space_from_features(features)
    # build the model
    model = hyped.modeling.ArbitraryEncoderWithHeads.from_pretrained_encoder(
        config.model.encoder_pretrained_ckpt,
        heads=config.model.heads,
        **config.model.kwargs
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
        # TODO: metrics
        preprocess_logits_for_metrics=lambda logits, _: [],
        compute_metrics=None
    )

    # train model
    trainer.train()

def main():
    from argparse import ArgumentParser
    # build argument parser
    parser = ArgumentParser(description="Train Transformer model on prepared datasets")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to run configuration file in .json format")
    parser.add_argument("-d", "--data", type=str, nargs='+', required=True, help="Paths to prepared data dumps")
    parser.add_argument("-o", "--out-dir", type=str, required=True, help="Output directory to dump checkpoints and metrics in")
    # parse arguments
    args = parser.parse_args()

    # check if config exists
    if not os.path.isfile(args.config):
        raise FileNotFoundError(args.config)
    # load config
    logger.info("Loading run configuration from %s" % args.config)
    config = RunConfig.parse_file(args.config)

    # run training
    train(config, args.data, disable_tqdm=False)

if __name__ == '__main__':
    main()
