from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from typing import Literal, Any
from .registry import RegisterTypes
from .auto import BaseAutoClass


@dataclass
class BaseConfig(RegisterTypes):
    t: Literal["base.config"] = "base.config"

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration object to dictionary"""
        return asdict(self) | {"__type_hash__": hash(type(self))}

    def to_json(self, **kwargs) -> str:
        """Serialize config object into json format

        Arguments:
            **kwargs: arguments forwarded to `json.dumps`

        Returns:
            serialized_config (str): the serialized configuration string
        """
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, dct: dict[str, Any]) -> BaseConfig:
        """Convert dict to configuration instance

        Arguments:
            dct (dict[str, Any]): dictionary to be converted

        Returns:
            config (BaseConfig): the constructed configuration object
        """

        dct = dct.copy()
        # pop type hash and type identifier as they are meta
        # information and not actual fields needing to be set
        h = dct.pop("__type_hash__", None)
        t = dct.pop("t", None)

        # make sure hashes match up
        if (h is not None) and (h != hash(cls)):
            raise ValueError(
                "Type hash in dict doesn't match type hash of config"
            )
        # make sure type identifiers match up
        if (t is not None) and (t != cls.t):
            raise ValueError(
                "Type identifier in dict doesn't match type identifier"
                "of config: %s != %s" % (t, cls.t)
            )
        # instantiate config
        return cls(**dct)

    @classmethod
    def from_json(cls, serialized: str) -> BaseConfig:
        """Deserialize a json string into a config

        Arguments:
            serialized (str): the serialized string in json format

        Returns:
            config (BaseConfig): the constructed configuration object
        """
        return cls.from_dict(json.loads(serialized))


class AutoConfig(BaseAutoClass[BaseConfig]):
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
            # get type from registry
            h = dct.get("__type_hash__")
            T = cls.type_registry.get_type_by_hash(h)

        elif "t" in dct:
            # get type from type id
            t = dct.get("t")
            T = cls.type_registry.get_type_by_t(t)

        else:
            # fallback to class on which the function is called
            T = cls

        # create instance
        return T.from_dict(dct)

    @classmethod
    def from_json(cls, serialized: str) -> BaseConfig:
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
