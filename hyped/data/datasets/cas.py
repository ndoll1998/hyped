import os
import cassis
import datasets
import multiprocessing as mp
from itertools import chain
from dataclasses import dataclass
from collections import defaultdict
from typing import Iterator, Sequence

_PRIMITIVE_TYPE_MAP = {
    "uima.cas.Boolean": datasets.Value('bool'),
    "uima.cas.Byte":    datasets.Value('binary'),
    "uima.cas.Short":   datasets.Value('int16'),
    "uima.cas.Integer": datasets.Value('int32'),
    "uima.cas.Long":    datasets.Value('int64'),
    "uima.cas.Float":   datasets.Value('float32'),
    "uima.cas.Double":  datasets.Value('float64'),
    "uima.cas.String":  datasets.Value('string'),
}


def _init_process(typesystem_path: str) -> None:
    """Initialize worker process"""

    # get the current process object
    proc = mp.current_process()

    # load typesystem and store as attribute of the
    # process for easy access in worker function
    with open(typesystem_path, 'rb') as f:
        proc.typesystem = cassis.load_typesystem(f)


def _worker(fpath:str) -> dict:

    # get typesystem from process
    proc = mp.current_process()
    typesystem = proc.typesystem

    with open(fpath, 'rb') as f:
        # load cas from different formats, fallback to xmi by default
        if fpath.endswith('.json'):
            cas = cassis.load_cas_from_json(f, typesystem=typesystem)
        else:
            cas = cassis.load_cas_from_xmi(f, typesystem=typesystem)

    # create features dictionary
    # use default dict to avoid key-errors for features that are
    # present in the typesystem but not used in the specific cas
    features = defaultdict(list)
    features['text'] = cas.sofa_string

    # extract annotation features
    for annotation_type in typesystem.get_types():
        # get all features of interest for the annotation type
        feature_types = [
            f for f in annotation_type.all_features
            if typesystem.is_primitive(f.rangeType)
        ]

        # iterate over all annotations of the current type
        for annotation in cas.select(annotation_type):
            # add features to dict
            for feature_type in feature_types:
                key = '%s:%s' % (annotation_type.name, feature_type.name)
                features[key].append(annotation.get(feature_type.name))

    return features


@dataclass
class CasDatasetConfig(datasets.BuilderConfig):

    # path to the cassis typesystem
    typesystem: str = None
    num_processes: int = mp.cpu_count()


class CasDataset(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = CasDatasetConfig

    @property
    def features(self) -> datasets.Features:

        # load typesystem
        with open(self.config.typesystem, 'rb') as f:
            typesystem = cassis.load_typesystem(f)

        # extract features from typesystem
        return datasets.Features(
            {
                'text': datasets.Value('string')
            } | {
                "%s:%s" % (t.name, f.name): datasets.Sequence(
                    _PRIMITIVE_TYPE_MAP[f.rangeType.name]
                )
                for t in typesystem.get_types()
                for f in t.all_features
                if typesystem.is_primitive(f.rangeType)
            }
        )

    def _info(self):

        # make sure the typesystem exists
        if (self.config.typesystem is not None) and not os.path.isfile(self.config.typesystem):
            raise FileNotFoundError(self.config.typesystem)

        return datasets.DatasetInfo(
            description="",
            features=self.features,
            supervised_keys=None
        )

    def _split_generators(self, dl_manager):

        # check data files argument
        if self.config.data_files is None:
            raise ValueError(
                "No data files specified. Please specify `data_files` in "
                "call to `datasets.load_dataset`."
            )
        if not isinstance(self.config.data_files, dict):
            raise ValueError(
                "Expected `data_files` to be a dictionary mapping splits "
                "to files, got %s" % type(data_files).__name__
            )

        # prepare data files
        data_files = dl_manager.download_and_extract(self.config.data_files)
        assert isinstance(data_files, dict), "Expected dict but got %s" % type(data_files).__name__

        splits = []
        # generate data split generators
        for split_name, files in data_files.items():
            # generate split generator
            files = [dl_manager.iter_files(file) for file in files]
            split = datasets.SplitGenerator(
                name=split_name,
                gen_kwargs=dict(files=files),
            )
            split.split_info.num_examples = len(files)
            # add to splits
            splits.append(split)

        return splits

    def _generate_examples(self, files:list[Iterator[str]]):

        # clamp number of processes between 1 and cpu-count
        num_processes = min(max(self.config.num_processes, 1), mp.cpu_count())
        # create worker pool with access to cas typesystem
        with mp.Pool(num_processes, initializer=_init_process, initargs=(self.config.typesystem,)) as pool:
            # process all files
            yield from enumerate(pool.map(_worker, chain.from_iterable(files)))
