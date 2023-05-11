"""Script to prepare a dataset.

Dataset Preparation consists of the following steps:
    0. download raw data (if necessary)
    1. apply data processor
    2. filter dataset
    3. convert to faster data format (NamedTensorDataset)
"""
import os
import hyped
import torch
import datasets
import transformers
import numpy as np
import logging
# utils
from hyped.scripts.utils.data import NamedTensorDataset, DataDump
from hyped.scripts.utils.configs import PrepareConfig

logger = logging.getLogger(__name__)

def prepare_dataset(
    ds:datasets.DatasetDict,
    config:PrepareConfig,
    max_size:int | None =None,
) -> DataDump:

    # create pipeline
    pipe = hyped.pipeline.Pipeline(
        processors=config.pipeline,
        filters=config.filters
    )

    # reduce datasets if they are too large
    for s, d in ds.items():
        if (max_size is not None) and (len(d) > max_size):
            logger.info("Sampling %s/%s data points from %s split" % (max_size, len(d), s))
            idx = np.random.choice(len(d), max_size, replace=False)
            ds[s] = d.select(idx)

    # pass features through pipeline
    features = pipe.map_features(config.data.info.features)
    # pass datasets through pipeline
    ds = pipe(ds)

    # rename columns
    for t, s in config.columns.items():
        if t != s:
            ds = ds.rename_column(s, t)
    # set data format to torch
    ds.set_format(type='torch', columns=list(config.columns.keys()))

    # get data schema after pipeline, column renaming and formatting
    features = datasets.Features({t: features[s] for t, s in config.columns.items()})
    logger.debug("Dataset Features: %s" % str(features))

    # check if all features are stackable
    for n, f in features.items():
        if isinstance(f, datasets.Sequence) and (f.length == -1):
            logger.info("Feature %s has undefined length, cannot converting to Tensor Dataset!" % n)
            break
    else:
        # convert to tensor dataset as all sequences have fixed length
        return DataDump(
            features=features,
            datasets={s: NamedTensorDataset.from_dataset(d) for s, d in ds.items()}
        )

    # return as is
    return DataDump(features=features, datasets=ds)


def main():
    from argparse import ArgumentParser
    # build argument parser
    parser = ArgumentParser(description="Prepare dataset for training")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to run configuration file in .json format")
    parser.add_argument("-n", "--max-size", type=int, default=None, help="Maximum number of data points per split")
    parser.add_argument("-o", "--out-file", type=str, required=True, help="File to store prepared dataset in")
    # parse arguments
    args = parser.parse_args()

    # check if config exists
    if not os.path.isfile(args.config):
        raise FileNotFoundError(args.config)

    # load config
    logger.info("Loading data configuration from %s" % args.config)
    config = PrepareConfig.parse_file(args.config)
    # load dataset splits
    logger.info("Downloading/Loading dataset splits")
    ds = datasets.load_dataset(
        config.data.dataset,
        split=config.data.splits,
        **config.data.kwargs
    )
    # prepare dataset
    logger.info("Prepareing dataset splits")
    ds = prepare_dataset(ds, config, max_size=args.max_size)

    # save data splits to file
    logger.info("Saving dataset to %s" % args.out_file)
    # create output directory
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    torch.save(ds, args.out_file)

if __name__ == '__main__':
    main()
