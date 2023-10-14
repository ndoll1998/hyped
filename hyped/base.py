from __future__ import annotations
import json
from abc import ABC, ABCMeta
from dataclasses import dataclass, asdict
from types import MappingProxyType
from typing import Literal, Any


class __TypeRegistryMeta(ABCMeta):
    _global_hash_register = dict()  # maps type-id to type-hash
    _global_type_register = dict()  # maps type-hash to type

    _hash_tree = dict()  # maps type-hash to parent type-hashes

    def register_type(cls, T, bases):
        h = hash(T)
        # update registries
        cls._global_hash_register[T.t] = h
        cls._global_type_register[h] = T
        # add type hash to all base nodes of the type
        for b in map(hash, bases):
            if b in cls._hash_tree:
                cls._hash_tree[b].add(h)
        # add node for type in hash tree
        cls._hash_tree[h] = set()

    def hash_tree_bfs(cls, root):
        # breadth-first search through hash tree
        seen, nodes = set(), [root]
        while len(nodes) > 0:
            node = nodes.pop()
            seen.add(node)
            # update node list
            new_nodes = cls._hash_tree.get(node, set()) - seen
            nodes.extend(new_nodes)
            # yield current node
            yield node

    @property
    def hash_register(cls):
        # build inverted hash register mapping hash to type-id
        inv_hash_register = {h: t for t, h in cls._global_hash_register.items()}
        # build up-to-date sub-tree hash register
        subtree = list(cls.hash_tree_bfs(root=hash(cls)))
        return {cls._global_type_register[h].t: h for h in subtree} | {
            inv_hash_register[h]: h
            for h in filter(inv_hash_register.__contains__, subtree)
        }

    @property
    def type_register(cls):
        return {
            h: cls._global_type_register[h] for h in cls.hash_tree_bfs(root=hash(cls))
        }

    def __new__(cls, name, bases, attrs) -> None:
        # create new type and register it
        T = super().__new__(cls, name, bases, attrs)
        cls.register_type(cls, T, bases)
        # return new type
        return T


class TypeRegistry(ABC, metaclass=__TypeRegistryMeta):
    """Type Registry Base Class

    Tracks all types that inherit from a class.

    Attributes:
        t (ClassVar[str]): type identifier
    """

    t: Literal["base"] = "base"

    @classmethod
    @property
    def hash_register(cls) -> dict[int, type]:
        """Immutable hash register mapping type id to the
        corresponding type hash
        """
        return MappingProxyType(cls.hash_register)

    @classmethod
    @property
    def type_register(cls) -> dict[int, type]:
        """Immutable type register mapping type hash to the
        corresponding type
        """
        return MappingProxyType(cls.type_register)

    @classmethod
    @property
    def type_ids(cls) -> list[type]:
        """List of all registered type identifiers"""
        return list(cls.hash_register.keys())

    @classmethod
    @property
    def types(cls) -> list[type]:
        """List of all registered types"""
        return list(cls.type_register.values())

    @classmethod
    def get_type_by_t(cls, t: str) -> type:
        """Get registered type by type id

        Arguments:
            t (int): type identifier

        Returns:
            T (type): type corresponding to `t`
        """
        # check if type id is present in register
        if t not in cls.hash_register:
            raise ValueError(
                "Type id '%s' not registered, registered type ids: %s"
                % (t, ", ".join(cls.type_ids))
            )
        # get type corresponding to id
        return cls.get_type_by_hash(cls.hash_register[t])

    @classmethod
    def get_type_by_hash(cls, h: int) -> type:
        """Get registered type by type hash

        Arguments:
            h (int): type hash

        Returns:
            T (type): registered type corresponding to `h`
        """
        # check if hash is present in register
        if h not in cls.type_register:
            raise TypeError(
                "No type found matching hash %s, registered types: %s"
                % (str(h), ", ".join(list(map(str, cls.types))))
            )
        # get type corresponding to hash
        return cls.type_register[h]


@dataclass
class BaseConfig(TypeRegistry):
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
            T = cls.get_type_by_hash(h)

        elif "t" in dct:
            # similar to hash above, pop type id from dict as its
            # a class variable and not a field of the config
            dct = dct.copy()
            t = dct.pop("t")
            # get type from type id
            T = cls.get_type_by_t(t)

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
        """Deserialize a json string into a configuration object of appropriate type

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
