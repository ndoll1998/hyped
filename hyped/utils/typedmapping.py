from .typedlist import check_type
from typing import Generic, TypeVar, Any, get_args

K = TypeVar("K")
V = TypeVar("V")

class typedmapping(Generic[K, V], dict):

    def __init__(self):
        super(typedmapping, self).__init__()

    @property
    def _K(self) -> type:
        return get_args(self.__orig_class__)[0]

    @property
    def _V(self) -> type:
        return get_args(self.__orig_class__)[1]

    def handle_key_type_conflict(self, key:Any) -> K:
        return key

    def handle_val_type_conflict(self, val:Any) -> V:
        return val

    def check_key_type(self, key:Any) -> K:
        # handle type conflict if value has incorrect type
        if not check_type(key, self._K):
            key = self.handle_key_type_conflict(key)
            # check again
            if not check_type(key, self._K):
                raise TypeError("Expected key of type %s, got %s." % (self._K, type(key)))
        # otherwise all fine
        return key

    def check_val_type(self, val:Any) -> V:
        # handle type conflict if value has incorrect type
        if not check_type(val, self._V):
            val = self.handle_key_type_conflict(val)
            # check again
            if not check_type(val, self._V):
                raise TypeError("Expected value of type %s, got %s." % (self._V, type(val)))
        # otherwise all fine
        return val

    def update(self, other:dict) -> None:
        # convert to dict if necessary
        other = other if isinstance(other, dict) else dict(other)
        # check types
        other = {
            self.check_key_type(k): self.check_val_type(v)
            for k, v in other.items()
        }
        # update dict
        return super(typedmapping, self).update(other)

    def __setitem__(self, key:K, val:V) -> None:
        return super(typedmapping, self).__setitem__(
            self.check_key_type(key),
            self.check_val_type(val)
        )
