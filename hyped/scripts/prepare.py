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
) -> dict[str, datasets.arrow_dataset.Dataset]:

    # reduce datasets if they are too large
    for s, d in ds.items():
        if (max_size is not None) and (len(d) > max_size):
            logger.info("Sampling %s/%s data points from %s split" % (max_size, len(d), s))
            idx = np.random.choice(len(d), max_size, replace=False)
            ds[s] = d.select(idx)

    # apply pipeline to dataset
    for p_config in config.pipeline:
        # build processor
        p_type = hyped.get_processor_type_from_config(p_config)
        p = p_type(p_config)
        # apply processor
        ds = ds.map(
            function=p,
            with_indices=p.requires_index,
            with_rank=p.requires_rank,
            batched=False,
            load_from_cache_file=False,
            desc=p_type.__name__
        )

    # apply filters to dataset
    for f_config in config.filters:
        # build processor
        f_type = hyped.get_filter_type_from_config(f_config)
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
    # convert dataset to named tensor dataset
    logger.info("Converting data format")
    return {s: NamedTensorDataset.from_dataset(d) for s, d in ds.items()}

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
    ds = datasets.DatasetDict(zip(config.splits, ds))
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
