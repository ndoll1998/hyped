from .registry import (
    Registrable,
    TypeRegistry,
    RootedTypeRegistryView,
    default_registry,
)
from typing import Generic, TypeVar, get_args

T = TypeVar("T")


class BaseAutoClass(Generic[T]):
    _registry: TypeRegistry = default_registry

    def __init__(self):
        raise EnvironmentError(
            "%s is designed to be instantiated using the"
            "`%s.from_config(config)` method"
            % (type(self).__name__, type(self).__name__)
        )

    @classmethod
    @property
    def type_registry(cls) -> RootedTypeRegistryView:
        """Type registry of base type"""
        # resolve generic type
        # TODO: this assumes that the auto-class directly and only
        # inherits from the `BaseAutoClass`
        t = get_args(cls.__orig_bases__[0])[0]
        # check type
        if not issubclass(t, Registrable):
            raise TypeError(
                "Autoclass generic types must be registrable, got %s" % t
            )
        # build rooted view on type registry
        return RootedTypeRegistryView(root=t, registry=cls._registry)
