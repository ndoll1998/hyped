import json
import inspect
from datasets import Features
from hyped.data.pipe import DataPipe
from datasets import Dataset
from hyped.utils.feature_access import FeatureKey
from hyped.data.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
)
from dataclasses import fields, _MISSING_TYPE
from typing import Any


# TODO: move this functionality into the data pipe memory class
class DataPipeFormatter(object):
    def __init__(self, sample_ds: Dataset) -> None:
        self.sample_ds = sample_ds

    @property
    def with_samples(self) -> bool:
        return len(self.sample_ds) > 0

    def _build_data_processor_doc(self) -> str:
        doc = {
            "index": "index of the processor in the data pipeline",
            "type_id": "the type identifier of the data processor",
            "config": "the configuration of the data processor",
            "features": "the output features of the data processor",
        }

        if self.with_samples:
            doc["samples"] = (
                "a list of sample outputs of the data processor "
                "in json format"
            )

        return json.dumps(doc, indent=2)

    def build_doc(self) -> str:
        return (
            "The data pipeline is represented as a list of dictionaries. "
            "Each dictionary in the list describes one processor in the "
            "pipeline. The dictionaries have the following structure: %s"
            % self._build_data_processor_doc()
        )

    def build_desc(self, pipe: DataPipe) -> str:
        # prepare the data pipeline for the dataset
        pipe.prepare(self.sample_ds.features)

        # build descriptions of all processors in current pipeline
        processor_descriptions = (
            [
                self.build_processor_dict_with_samples(i, p, samples)
                for i, (p, samples) in enumerate(
                    zip(
                        pipe,
                        pipe.iter_batch_process(
                            examples=self.sample_ds.to_dict(),
                            index=list(range(len(self.sample_ds))),
                            rank=0,
                        ),
                    )
                )
            ]
            if self.with_samples
            else [self.build_processor_dict(i, p) for i, p in enumerate(pipe)]
        )

        # json serialize processor descriptions
        return json.dumps(processor_descriptions, indent=2)

    def build_processor_dict(self, i: int, p: BaseDataProcessor) -> str:
        # convert configuration to dictionary and remove the type hash
        config = p.config.to_dict()
        config.pop("__type_hash__")

        return {
            "index": i,
            "type_id": config.pop("t"),
            "config": config,
            "features": self.format_features(p.out_features),
        }

    def build_processor_dict_with_samples(
        self, i: int, p: BaseDataProcessor, samples: dict[str, list[Any]]
    ) -> str:
        return self.build_processor_dict(i, p) | {
            "samples": self.format_samples(samples)
        }

    def format_features(self, features: Features) -> str:
        return str(features)

    def format_samples(self, samples: dict[str, list[Any]]) -> str:
        # unpack samples, i.e dict of lists -> list of dicts
        num_samples = len(next(iter(samples.values())))
        samples = [
            {k: samples[k][i] for k in samples.keys()}
            for i in range(num_samples)
        ]
        # serialize samples
        return json.dumps(samples)


# TODO: move this functionality to the data processor retriever class
class DataProcessorTypeFormatter(object):
    @classmethod
    def build_doc(cls) -> str:
        return (
            "%s\n"
            "Note that arguments of the type %s can either take on a string "
            "value or a tuple value describing a path to the feature in case "
            "the feature is nested. The path tuple can contain strings to "
            "index dictionaries and integers to index sequences."
        ) % (
            json.dumps(
                {
                    "name": "the name of the data processor",
                    "type_id": "the type identifier of the data processor",
                    "documentation": (
                        "documentation obout the data processor and how to "
                        "configure it"
                    ),
                    "args": (
                        "list of configuration arguments and their default "
                        "values if present"
                    ),
                },
                indent=2,
            ),
            str(FeatureKey),
        )

    @classmethod
    def build_desc(cls, processor_t: BaseDataProcessor) -> str:
        return json.dumps(
            {
                "name": processor_t.__name__,
                "type_id": processor_t.config_type.t,
                "config_docs": inspect.getdoc(processor_t.config_type),
                "config_args": cls.build_config_args_doc(
                    processor_t.config_type
                ),
            }
        )

    @classmethod
    def build_config_args_doc(
        cls, config_type: type[BaseDataProcessorConfig]
    ) -> dict[str, str]:
        def _build_arg_doc(f):
            doc = {"type": str(f.type)}

            if not isinstance(f.default, _MISSING_TYPE):
                doc["default"] = f.default

            return doc

        return {
            f.name: _build_arg_doc(f)
            for f in fields(config_type)
            if f.name not in ("t", "output_format")
        }
