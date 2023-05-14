from __future__ import annotations
from typing import Generic, GenericAlias, TypeVar, Any, get_args, get_origin

T = TypeVar("T")

def check_type(obj:Any, t:type) -> bool:
    # check if type is a generic type
    if isinstance(t, GenericAlias) and issubclass(get_origin(t), type):
        # in this case we want the object to be a class/type
        # instead of an instance of a specific class
        return issubclass(obj, get_args(t))
    # check if object is an instance of the requested type
    return isinstance(obj, t)

class typedlist(Generic[T], list):

    def __init__(self):
        super(typedlist, self).__init__()

    @property
    def _T(self) -> type:
        return get_args(self.__orig_class__)[0]

    def handle_type_conflict(self, val:Any) -> T:
        return val

    def check_type(self, val:Any) -> T:
        # handle type conflict if value has incorrect type
        if not isinstance(val, self._T):
            val = self.handle_type_conflict(val)
            # check again
            if not isinstance(val, self._T):
                raise TypeError("Expected instance of type %s, got %s." % (self._T, type(val)))

        # otherwise all fine
        return val

    def append(self, val:T) -> None:
        val = self.check_type(val)
        super(typedlist, self).append(val)

    def extend(self, vals:list[T]) -> None:
        vals = [self.check_type(v) for v in vals]
        super(typedlist, self).extend(vals)

    def __add__(self, val:T) -> typedlist[T]:
        val = self.check_type(val)
        return super(typedlist, self).__add__(val)

    def __iadd__(self, vals:list[T]) -> typedlist[T]:
        vals = [self.check_type(v) for v in vals]
        return super(typedlist, self).__iadd__(vals)
