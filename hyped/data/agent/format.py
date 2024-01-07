import json
import inspect
from datasets import Features
from hyped.data.pipe import DataPipe
from hyped.data.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
)
from dataclasses import fields, _MISSING_TYPE


class DataPipeFormatter(object):
    @classmethod
    def build_doc(cls) -> str:
        return (
            "A list of dictionaries describing the current data pipeline "
            "in json format. Each dictionary in the list describes one "
            "processor in the pipeline. The dictionaries have the following "
            "structure: %s"
            % json.dumps(
                {
                    "index": "index of the processor in the data pipeline",
                    "type_id": "the type identifier of the data processor",
                    "config": "the configuration of the data processor",
                },
                indent=2,
            )
        )

    @classmethod
    def build_desc(cls, pipe: DataPipe) -> str:
        # build descriptions of all processors in current pipeline
        processor_descriptions = [
            cls.build_processor_dict(i, p) for i, p in enumerate(pipe)
        ]
        # json serialize processor descriptions
        return json.dumps(processor_descriptions, indent=2)

    @classmethod
    def build_processor_dict(cls, i: int, p: BaseDataProcessor) -> str:
        # convert configuration to dictionary and remove the type hash
        config = p.config.to_dict()
        config.pop("__type_hash__")

        return {"index": i, "type_id": config.pop("t"), "config": config}


class DataProcessorTypeFormatter(object):
    @classmethod
    def build_doc(cls) -> str:
        return json.dumps(
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
            },
            indent=2,
        )

    @classmethod
    def build_config_args_doc(
        cls, config_type: type[BaseDataProcessorConfig]
    ) -> dict[str, str]:
        return {
            f.name: (
                {
                    "type": str(f.type),
                }
                if isinstance(f.default, _MISSING_TYPE)
                else {"type": str(f.type), "default": f.default}
            )
            for f in fields(config_type)
            if f.name not in ("t", "output_format")
        }


# TODO: format features
class FeaturesFormatter(object):
    @classmethod
    def build_doc(cls) -> str:
        return "description of dataset features and their types"

    @classmethod
    def build_desc(self, features: Features) -> str:
        return str(features)
