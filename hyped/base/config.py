from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from typing import Literal, Any
from .registry import RegisterTypes


@dataclass
class BaseConfig(RegisterTypes):
    t: Literal["base.config"] = "base.config"

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration object to dictionary"""
        return asdict(self) | {"__type_hash__": hash(type(self))}

    @classmethod
    def from_dict(cls, dct: dict[str, Any]) -> BaseConfig:
        """Convert dict to configuration object of appropriate type

        The type is inferred by the following prioritization:

        1. based on the `__type_hash__` if present in the dictionary
        2. based on the type identifier `t` if present in the dictionary
        3. use the root class, i.e. the class on which the function is called

        Arguments:
            dct (dict[str, Any]): dictionary to be converted

        Returns:
            config (BaseConfig): the constructed configuration object
        """
        if "__type_hash__" in dct:
            # create a copy to avoid removing entries from the original
            # pop hash from dict as its meta data and not a valid field
            dct = dct.copy()
            h = dct.pop("__type_hash__")
            # get type from registry
            T = cls.type_registry.get_type_by_hash(h)

        elif "t" in dct:
            # similar to hash above, pop type id from dict as its
            # a class variable and not a field of the config
            dct = dct.copy()
            t = dct.pop("t")
            # get type from type id
            T = cls.type_registry.get_type_by_t(t)

        else:
            # fallback to class on which the function is called
            T = cls

        # create instance
        return T(**dct)

    def serialize(self, **kwargs) -> str:
        """Serialize config object into json format

        Arguments:
            **kwargs: arguments forwarded to `json.dumps`

        Returns:
            serialized_config (str): the serialized configuration string
        """
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def deserialize(cls, serialized: str) -> BaseConfig:
        """Deserialize a json string into a configuration object of
        appropriate type

        The type is inferred by the following prioritization:

        1. based on the `__type_hash__` if present in the json string
        2. based on the type identifier `t` if present in the json string
        3. use the root class, i.e. the class on which the function is called

        Arguments:
            serialized (str): the serialized string in json format

        Returns:
            config (BaseConfig): the constructed configuration object
        """
        return cls.from_dict(json.loads(serialized))
