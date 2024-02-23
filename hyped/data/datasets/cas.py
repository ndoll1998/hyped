from __future__ import annotations
import os
import cassis
import datasets
import logging
import multiprocessing as mp
from itertools import chain
from dataclasses import dataclass
from collections import defaultdict
from typing import Iterator, Any

logger = logging.getLogger(__name__)

_PRIMITIVE_TYPE_MAP = {
    "uima.cas.Boolean": datasets.Value("bool"),
    "uima.cas.Byte": datasets.Value("binary"),
    "uima.cas.Short": datasets.Value("int16"),
    "uima.cas.Integer": datasets.Value("int32"),
    "uima.cas.Long": datasets.Value("int64"),
    "uima.cas.Float": datasets.Value("float32"),
    "uima.cas.Double": datasets.Value("float64"),
    "uima.cas.String": datasets.Value("string"),
}


def _get_types_from_typesystem(
    typesystem: cassis.TypeSystem, type_names: None | list[str] = None
) -> list[cassis.typesystem.Type]:
    """Get list of types from typesystem

    Arguments:
        typesystem (cassis.TypeSystem): the typesystem containing the types
        type_names (None|list[str]): list of types to get from the typesystem

    Returns:
        types (list[cassis.typesystem.Type]): requested type objects
    """
    # fallback to all types in typesystem
    if type_names is None:
        return typesystem.get_types()

    # ensure that all requested types are present in the typesystem
    for type_name in type_names:
        if not typesystem.contains_type(type_name):
            raise TypeError(
                "Annotation Type `%s` not found in typesystem" % type_name
            )

    # get requested types from typesystem
    return list(map(typesystem.get_type, type_names))


def _init_process(config: CasDatasetConfig) -> None:
    """Initialize worker process"""

    # get the current process object
    proc = mp.current_process()

    # store config in process
    proc.config = config
    # load typesystem and store as attribute of the
    # process for easy access in worker function
    with open(config.typesystem, "rb") as f:
        proc.typesystem = cassis.load_typesystem(f)


def _worker(fpath: str) -> None | dict[str, Any]:
    """Worker process loading a cas object from the given file and convert
    it into a dictionary

    This function assumes that the executing process executed `_init_process`
    beforehand to have access to the config (`mp.current_process().config`)
    and typesystem (`mp.current_process().typesystem`).

    Arguments:
        fpath (str):
            path to the file containing a valid cas in .xmi or .json format

    Returns:
        features (None|dict[str, Any]):
            the feature dictionary with features matching the dataset
            `features` attribute (see `CasDataset.features`)
    """
    # get current process objects storing
    # the dataset configuration and typesystem
    proc = mp.current_process()

    try:
        with open(fpath, "rb") as f:
            # load cas from different formats, fallback to xmi by default
            if fpath.endswith(".json"):
                cas = cassis.load_cas_from_json(f, typesystem=proc.typesystem)
            else:
                cas = cassis.load_cas_from_xmi(f, typesystem=proc.typesystem)
    except Exception as e:
        # log error
        logger.error(e)
        return None

    annotation_types = list(
        _get_types_from_typesystem(
            proc.typesystem, proc.config.annotation_types
        )
    )
    # collect all annotations and create a fixed ordering over
    # the annotations of each type
    annotations = {
        annotation_type.name: [a.xmiID for a in cas.select(annotation_type)]
        for annotation_type in annotation_types
    }
    # check ids
    assert all(
        xmi_id is not None
        for xmi_id in chain.from_iterable(annotations.values())
    )

    # create features dictionary
    # use default dict to avoid key-errors for features that are
    # present in the typesystem but not used in the specific cas
    features = defaultdict(list)
    features["sofa"] = cas.sofa_string
    features["meta"] = {
        "file_path": fpath
    }

    # extract annotation features
    for annotation_type in annotation_types:
        # get all features of interest for the annotation type
        primitive_feature_types = [
            f
            for f in annotation_type.all_features
            if proc.typesystem.is_primitive(f.rangeType)
        ]
        nested_feature_types = [
            f
            for f in annotation_type.all_features
            if f.rangeType.name in annotations.keys()
        ]

        # iterate over all annotations of the current type
        for annotation in cas.select(annotation_type):
            # add primitive features to dict
            for feature_type in primitive_feature_types:
                key = "%s:%s" % (annotation_type.name, feature_type.name)
                features[key].append(annotation.get(feature_type.name))
            # add nested features to dict
            for feature_type in nested_feature_types:
                key = "%s:%s" % (annotation_type.name, feature_type.name)
                features[key].append(
                    annotations[feature_type.rangeType.name].index(
                        annotation.get(feature_type.name).xmiID
                    )
                )

    return features


@dataclass
class CasDatasetConfig(datasets.BuilderConfig):
    """Cas Dataset Configuration

    The attributes of the configuration are typically set by providing
    them as keyword arguments to the `datasets.load_dataset` function.

    Attributes:
        typesystem (str): path to a file containing the cas typesystem to use
        num_processes (int):
            the number of processes to spawn for processing cas objects
        annotation_types (None|list[str]):
            the set of annotation types to extract from the cas objects.
            Defaults to all types present in the typesystem.
    """

    # cas type management
    typesystem: str = None
    annotation_types: None | list[str] = None
    # number of processes to use
    num_processes: int = mp.cpu_count()


class CasDataset(datasets.GeneratorBasedBuilder):
    """Cas Dataset

    Typically used by call to `datasets.load_dataset with appropriate
    keyword arguments (see `CasDatasetConfig` for defails)

    ```
    datasets.load_dataset('hyped.data.datasets.cas', **kwargs)
    ```
    """

    BUILDER_CONFIG_CLASS = CasDatasetConfig

    @property
    def features(self) -> datasets.Features:
        # load typesystem
        with open(self.config.typesystem, "rb") as f:
            typesystem = cassis.load_typesystem(f)

        # get all requested types
        all_types = typesystem.get_types()
        req_types = list(
            _get_types_from_typesystem(
                typesystem, self.config.annotation_types
            )
        )

        all_type_names = {_type.name for _type in all_types}
        req_type_names = {_type.name for _type in req_types}

        # check if all required types are present
        for t in req_types:
            for f in t.all_features:
                # check if the feature refers to other annotations
                if (f.rangeType.name in all_type_names) and (
                    f.rangeType.name not in req_type_names
                ):
                    raise RuntimeError(
                        "Annotation type `%s` requires type `%s`"
                        % (t.name, f.rangeType.name)
                    )

        primitive_features = {
            # all primitive features of all requested types
            "%s:%s"
            % (t.name, f.name): datasets.Sequence(
                _PRIMITIVE_TYPE_MAP[f.rangeType.name]
            )
            for t in _get_types_from_typesystem(
                typesystem, self.config.annotation_types
            )
            for f in t.all_features
            if typesystem.is_primitive(f.rangeType)
        }

        nested_features = {
            # all nested features that point to other annotations
            "%s:%s"
            % (t.name, f.name): datasets.Sequence(datasets.Value("int32"))
            for t in req_types
            for f in t.all_features
            if f.rangeType.name in req_type_names
        }

        # extract features from typesystem
        return datasets.Features(
            {
                "sofa": datasets.Value("string"),
                "meta": datasets.Features(
                    {"file_path": datasets.Value("string")}
                )
            }
            | primitive_features
            | nested_features
        )

    def _info(self):
        # make sure the typesystem exists
        if (self.config.typesystem is not None) and not os.path.isfile(
            self.config.typesystem
        ):
            raise FileNotFoundError(self.config.typesystem)

        return datasets.DatasetInfo(
            description="", features=self.features, supervised_keys=None
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
                "to files, got %s" % type(self.config.data_files).__name__
            )

        # prepare data files
        data_files = dl_manager.download_and_extract(self.config.data_files)
        assert isinstance(self.config.data_files, dict), (
            "Expected dict but got %s" % type(data_files).__name__
        )

        splits = []
        # generate data split generators
        for split_name, files in self.config.data_files.items():
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

    def _generate_examples(self, files: list[Iterator[str]]):
        # clamp number of processes between 1 and cpu-count
        num_processes = min(max(self.config.num_processes, 1), mp.cpu_count())
        # create worker pool with access to cas typesystem
        with mp.Pool(
            num_processes, initializer=_init_process, initargs=(self.config,)
        ) as pool:
            # process all files
            yield from enumerate(
                filter(
                    lambda x: x is not None,
                    pool.imap_unordered(_worker, chain.from_iterable(files)),
                )
            )
