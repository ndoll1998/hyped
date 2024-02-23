from abc import ABC, ABCMeta
from types import MappingProxyType
from typing import ClassVar, Iterator
from datasets.packaged_modules import _hash_python_lines
import inspect


class Registrable(ABC):
    """Base Class for Registrable Types

    Class Attributes:
        t (str): type identifier
    """

    t: ClassVar[str] = "hyped.base.register.registrable"

    @classmethod
    @property
    def type_hash(cls) -> str:
        """Get type hash

        The computed hash-code is based on the source code of the
        type and *not* runtime specific. This allows for type
        resolving by hash accross python runtimes.

        When the source code of the class is not available, the
        hash of the class module and name is computed instead.
        This is the case for pre-compiled classes or classes
        defined in jupyter notebooks.

        Returns:
            h (str): hash code of type
        """
        try:
            src = inspect.getsource(cls)
            src_lines = src.splitlines()
        except OSError:
            src_lines = [cls.__module__, cls.__name__]
        return _hash_python_lines(src_lines)


class TypeRegistry(object):
    """Type Registry

    Stores registrable types and holds functionality to get all
    registered sub-types of a given root type.

    Registrable types must inherit from the `Registrable` type.
    """

    def __init__(self):
        self.global_hash_register: dict[str, str] = dict()
        self.global_type_register: dict[str, type] = dict()
        self.hash_tree: dict[str, list[str]] = dict()

    def register_type(self, T: type, bases: tuple[type]):
        """Register a type

        Arguments:
            T (type): the type to register, must be a subclass of `Registrable`
            bases (tuple[type]): the base types of the type `T`
        """
        # check type
        if not issubclass(T, Registrable):
            raise TypeError(
                "Registrable types must inherit from `%s`" % str(Registrable)
            )

        h = T.type_hash
        # update registers
        self.global_hash_register[T.t] = h
        self.global_type_register[h] = T
        # add type hash to all base nodes of the type
        for b in [b.type_hash for b in bases if issubclass(b, Registrable)]:
            if b in self.hash_tree:
                self.hash_tree[b].add(h)
        # add node for type in hash tree
        self.hash_tree[h] = set()

    def hash_tree_bfs(self, root: str) -> Iterator[str]:
        """Breadth-Frist Search through inheritance tree rooted at given type

        Arguments:
            root (str): hash of the root type

        Returns:
            node_iter (Iterator[str]):
                iterator over all hashes of sub-types of the given root type
        """
        # breadth-first search through hash tree
        seen, nodes = set(), [root]
        while len(nodes) > 0:
            node = nodes.pop()
            seen.add(node)
            # update node list
            new_nodes = self.hash_tree.get(node, set()) - seen
            nodes.extend(new_nodes)
            # yield current node
            yield node

    def get_hash_register(self, root: type) -> dict[str, str]:
        """Get the hash register for sub-types of a given root type

        Arguments:
            root (type): root type

        Returns:
            hash_register (dict[str, str]):
                the hash register mapping type identifiers to type hashes for
                types that inherit the root type
        """
        # build inverted hash register mapping hash to type-id
        inv_hash_register = {
            h: t for t, h in self.global_hash_register.items()
        }
        # build up-to-date sub-tree hash register
        subtree = list(self.hash_tree_bfs(root=root.type_hash))
        return {self.global_type_register[h].t: h for h in subtree} | {
            inv_hash_register[h]: h
            for h in filter(inv_hash_register.__contains__, subtree)
        }

    def get_type_register(self, root: type) -> dict[str, type]:
        """Get the type register for sub-types of a given root type

        Arguments:
            root (type): root type

        Returns:
            type_register (dict[str, type]):
                the type register mapping type hashes to types for
                types that inherit the root type
        """
        return {
            h: self.global_type_register[h]
            for h in self.hash_tree_bfs(root=root.type_hash)
        }


class RootedTypeRegistryView(object):
    """Rooted Type Registry View

    Only has access to registered types that inherit the specified root.

    Arguments:
        root (type): root type
        registry (TypeRegistry): type registry
    """

    def __init__(self, root: type, registry: TypeRegistry) -> None:
        self.root = root
        self.registry = registry

    @property
    def hash_register(self) -> dict[str, str]:
        """Immutable hash register mapping type id to the
        corresponding type hash
        """
        return MappingProxyType(self.registry.get_hash_register(self.root))

    @property
    def type_register(self) -> dict[str, type]:
        """Immutable type register mapping type hash to the
        corresponding type
        """
        return MappingProxyType(self.registry.get_type_register(self.root))

    @property
    def type_ids(self) -> list[type]:
        """List of all registered type identifiers"""
        return list(self.hash_register.keys())

    @property
    def types(self) -> list[type]:
        """List of all registered types"""
        return list(self.type_register.values())
    
    @property
    def concrete_types(self) -> list[type]:
        """List of all concrete (i.e. non-abstract) registered types"""
        return [t for t in self.types if not inspect.isabstract(t)]

    def get_type_by_t(self, t: str) -> type:
        """Get registered type by type id

        Arguments:
            t (int): type identifier

        Returns:
            T (type): type corresponding to `t`
        """
        # check if type id is present in register
        if t not in self.hash_register:
            raise ValueError(
                "Type id '%s' not registered, registered type ids: %s"
                % (t, ", ".join(self.type_ids))
            )
        # get type corresponding to id
        return self.get_type_by_hash(self.hash_register[t])

    def get_type_by_hash(self, h: str) -> type:
        """Get registered type by type hash

        Arguments:
            h (str): type hash

        Returns:
            T (type): registered type corresponding to `h`
        """
        # check if hash is present in register
        if h not in self.type_register:
            raise TypeError(
                "No type found matching hash %s, registered types: %s"
                % (str(h), ", ".join(list(map(str, self.types))))
            )
        # get type corresponding to hash
        return self.type_register[h]


# create default type registry
default_registry = TypeRegistry()


class register_types(ABCMeta):
    """meta-class to automatically register sub-types of a specific
    type in a type registry
    """

    _registry: ClassVar[TypeRegistry] = default_registry

    def __new__(cls, name, bases, attrs) -> None:
        # create new type and register it
        T = super().__new__(cls, name, bases, attrs)
        cls._registry.register_type(T, bases)
        # return new type
        return T

    @property
    def type_registry(cls) -> RootedTypeRegistryView:
        return RootedTypeRegistryView(root=cls, registry=cls._registry)


class RegisterTypes(Registrable, metaclass=register_types):
    """Base class that automatically registers sub-types to the
    default type registry
    """
