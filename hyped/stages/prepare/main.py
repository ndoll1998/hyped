import os
import datasets
import transformers
import numpy as np
import pydantic
import logging
# hyped
from .pipeline import Pipeline
from .processors import AnyProcessorConfig
from .filters import AnyFilterConfig
# utils
from typing_extensions import Annotated


logger = logging.getLogger(__name__)

class DataConfig(pydantic.BaseModel):
    """Data Configuration Model"""
    dataset:str
    splits:dict[str, str] = {
        datasets.Split.TRAIN: datasets.Split.TRAIN,
        datasets.Split.VALIDATION: datasets.Split.VALIDATION,
        datasets.Split.TEST: datasets.Split.TEST
    }
    kwargs:dict = {}

    @pydantic.root_validator(pre=False)
    def _check_dataset(cls, v):

        if v.get('dataset', None) is None:
            raise ValueError("No Dataset provided by configuration!")
        try:
            # try to load dataset builder
            builder = datasets.load_dataset_builder(v['dataset'], **v['kwargs'])
            return v
        except FileNotFoundError as e:
            # raise exception if dataset builder cannot be found
            raise ValueError("Dataset not found: %s" % v) from e

    @pydantic.validator('kwargs')
    def _prepare_kwargs(cls, v):
        if 'data_files' in v:
            data_files = v['data_files']
            # make data files absolut paths
            if isinstance(data_files, str):
                data_files = os.path.abspath(data_files)
            elif isinstance(data_files, (tuple, list)):
                data_files = [os.path.abspath(f) for f in data_files]
            elif isinstance(data_files, dict):
                data_files = {k: os.path.abspath(f) for k,f in data_files.items()}
            # update data files
            v['data_files'] = data_files
        return v

class PrepareConfig(pydantic.BaseModel):
    """Data Configuration Model"""
    # dataset config
    data:DataConfig
    # preprocessing pipeline
    pipeline:list[
        Annotated[
            AnyProcessorConfig,
            pydantic.Field(..., discriminator='processor_type')
        ]
    ]
    filters:list[AnyFilterConfig]
    # columns to keep
    columns:dict[str, str]

    # data filters
    #filters:list[
    #    Annotated[
    #        AnyFilterConfig,
    #        pydantic.Field(..., discriminator='filter_type')
    #    ]
    #]

def prepare_dataset(
    ds:datasets.DatasetDict,
    config:PrepareConfig,
    max_size:int | None =None,
) -> datasets.DatasetDict:

    # get dataset info
    info = next(iter(ds.values())).info

    # create pipeline
    pipe = Pipeline(
        processors=config.pipeline,
        filters=config.filters
    )

    # reduce datasets if they are too large
    for s, d in ds.items():
        if (max_size is not None) and (len(d) > max_size):
            logger.info("Sampling %s/%s data points from %s split" % (max_size, len(d), s))
            idx = np.random.choice(len(d), max_size, replace=False)
            ds[s] = d.select(idx)

    # prepare pipeline and pass datasets through
    features = pipe.prepare(info.features)
    ds = pipe(ds)
    # check features
    assert features == next(iter(ds.values())).features

    # rename columns
    for t, s in config.columns.items():
        if t != s:
            ds = ds.rename_column(s, t)

    # set data format to torch
    ds.set_format(type='torch', columns=list(config.columns.keys()))

    # get data schema after pipeline, column renaming and formatting
    features = datasets.Features({t: features[s] for t, s in config.columns.items()})
    logger.debug("Dataset Features: %s" % str(features))

    # log some info
    logger.info("Data Preprocessing Complete.")
    for s, d in ds.items():
        logger.info("Generated %s split of %i documents" % (s, len(d)))

    return ds


def main(
    config:str,
    max_size:int,
    splits:list[str],
    out_dir:str,
    local_rank:int =-1 # not used
) -> None:

    # register hyped datasets
    import hyped.datasets

    # check if config exists
    if not os.path.isfile(config):
        raise FileNotFoundError(config)

    # load config
    logger.info("Loading data configuration from %s" % config)
    config = PrepareConfig.parse_file(config)

    # overwrite splits
    for split in splits:
        if split not in config.data.splits:
            raise ValueError("Splits `%s` not specified in configuration %s." % (split, config))
    # only keep splits that are named in arguments
    if len(splits) > 0:
        config.data.splits = {s: config.data.splits[s] for s in splits}

    # load dataset splits
    logger.info("Downloading/Loading dataset splits")
    ds = datasets.load_dataset(
        config.data.dataset,
        split=config.data.splits,
        **config.data.kwargs
    )
    # prepare dataset
    logger.info("Preparing dataset splits")
    ds = prepare_dataset(ds, config, max_size=max_size)

    # convert dataset dict keys to string for save to disk
    ds = datasets.DatasetDict({str(k): d for k, d in ds.items()})
    # save dataset to disk
    logger.info("Saving dataset to %s" % out_dir)
    ds.save_to_disk(out_dir)
