from __future__ import annotations
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Literal, Any, TypeVar, Generic
from .registry import RegisterTypes
from .generic import solve_typevar
from .auto import BaseAutoClass


@dataclass
class BaseConfig(RegisterTypes):
    t: Literal["hyped.base.config.config"] = "hyped.base.config.config"

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration object to dictionary"""
        return asdict(self) | {"__type_hash__": type(self).type_hash}

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
        if (h is not None) and (h != cls.type_hash):
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
            raise TypeError(
                "Unable to resolve type of config: `%s`" % str(dct)
            )

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


U = TypeVar("U", bound=BaseConfig)


class BaseConfigurable(Generic[U], RegisterTypes, ABC):
    """Base class for configurable types

    Configurable types define a `from_config` classmethod.
    Sub-types must implement this function.
    """

    CONFIG_TYPE: None | type[BaseConfig] = None

    @classmethod
    @property
    def generic_config_type(cls) -> type[BaseConfig]:
        """Get the generic configuration type of the configurable specified
        by the type variable `U`
        """
        # get config class
        t = solve_typevar(cls, U)
        # check type
        if (t is not None) and not issubclass(t, BaseConfig):
            raise TypeError(
                "Configurable config type `%s` doesn't inherit from `%s`"
                % (str(t), str(BaseConfig))
            )
        return t or BaseConfig

    @classmethod
    @property
    def config_type(cls) -> type[BaseConfig]:
        """Get the (final) configuration type of the configurable

        The final configuration type is specified by the `CONFIG_TYPE`
        class attribute. Falls back to the generic config type if the
        class attribute is not specified. Also checks that the concrete
        configuration type is valid, i.e. inherits the generic configuration
        type.
        """
        generic_t = cls.generic_config_type
        # concrete config type must inherit generic config type
        if (cls.CONFIG_TYPE is not None) and not issubclass(
            cls.CONFIG_TYPE, generic_t
        ):
            raise TypeError(
                "Concrete config type `%s` specified by `CONFIG_TYPE` must "
                "inherit from generic config type `%s`"
                % (cls.CONFIG_TYPE, generic_t)
            )
        # return final config type and fallback to generic
        # type if not specified
        return cls.CONFIG_TYPE or generic_t

    @classmethod
    @property
    def t(cls) -> str:
        """Type identifier used in type registry. Identifier is build
        from configuration type identifier by appending `.impl`.
        """
        # specify registry type identifier based on config type identifier
        return "%s.impl" % cls.config_type.t

    @classmethod
    @abstractmethod
    def from_config(self, config: U) -> BaseConfigurable:
        """Abstract construction method, must be implemented by sub-types

        Arguments:
            config (T): configuration to construct the instance from

        Returns:
            inst (Configurable): instance
        """
        ...


V = TypeVar("V", bound=BaseConfigurable)


class BaseAutoConfigurable(BaseAutoClass[V]):
    """Base Auto Class for configurable types"""

    @classmethod
    def from_config(cls, config: BaseConfig) -> V:
        # build type identifier of configurable corresponding
        # to the config
        t = "%s.impl" % config.t
        T = cls.type_registry.get_type_by_t(t)
        # create instance
        return T.from_config(config)
