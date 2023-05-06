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
from hyped.scripts.utils.data import NamedTensorDataset
from hyped.scripts.utils.configs import DataConfig

logger = logging.getLogger(__name__)

def prepare_dataset(
    ds:datasets.DatasetDict,
    config:DataConfig,
    max_size:int | None =None,
) -> dict[str, torch.utils.data.Dataset | datasets.Features]:

    # reduce datasets if they are too large
    for s, d in ds.items():
        if (max_size is not None) and (len(d) > max_size):
            logger.info("Sampling %s/%s data points from %s split" % (max_size, len(d), s))
            idx = np.random.choice(len(d), max_size, replace=False)
            ds[s] = d.select(idx)

    # get initial features
    features = config.info.features
    # apply pipeline to dataset
    for p_config in config.pipeline:
        # build processor
        p_type = hyped.pipeline.get_processor_type_from_config(p_config)
        p = p_type(p_config)
        # map features
        features = p.map_features(features)
        # apply processor
        ds = ds.map(
            function=p,
            with_indices=p.requires_index,
            with_rank=p.requires_rank,
            batched=False,
            load_from_cache_file=False,
            desc=p_type.__name__
        )

        # make sure the dataset columns match the updated
        # features at least in terms of naming
        for s, d in ds.items():
            assert set(d.column_names) == set(features), "Mismatch between %s dataset columns (%s) and features (%s)" % (s, str(d.column_names), str(list(features.keys())))

    # apply filters to dataset
    for f_config in config.filters:
        # build processor
        f_type = hyped.pipeline.get_filter_type_from_config(f_config)
        f = f_type(f_config)
        # apply processor
        ds = ds.filter(
            function=f,
            with_indices=f.requires_index,
            batched=False,
            load_from_cache_file=False,
            desc=f_type.__name__
        )

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
        return {'__features': features} | \
            {s: NamedTensorDataset.from_dataset(d) for s, d in ds.items()}

    # return as is
    return {'__features': features} | ds


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
    config = DataConfig.parse_file(args.config)
    # load dataset splits
    logger.info("Downloading/Loading dataset splits")
    ds = datasets.load_dataset(config.dataset, split=config.splits)
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
