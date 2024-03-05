from typing import Generic, TypeVar

from .generic import solve_typevar
from .registry import (
    Registrable,
    RootedTypeRegistryView,
    TypeRegistry,
    default_registry,
)

T = TypeVar("T", bound=Registrable)


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
        t = solve_typevar(cls, T)
        # check type
        if not issubclass(t, Registrable):
            raise TypeError(
                "Autoclass generic types must be registrable, got %s" % t
            )
        # build rooted view on type registry
        return RootedTypeRegistryView(root=t, registry=cls._registry)
