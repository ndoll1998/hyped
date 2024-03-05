from typing import Generic, TypeVar, _GenericAlias, get_args, get_origin


def _get_typevar_index(t: type, T: TypeVar) -> None | int:
    # trivial case
    if not hasattr(t, "__orig_bases__"):
        return None
    # search for typevar in base types
    for b in t.__orig_bases__:
        if isinstance(b, _GenericAlias):
            # check if base type is a generic alias
            a = get_args(b)
            if T in a:
                return a.index(T)

        else:
            o, a = get_origin(b), get_args(b)
            # return index of typevar if present
            if (o == Generic) and (T in a):
                return a.index(T)
    # typevar not found
    return None


def solve_typevar(t: type, T: TypeVar) -> type | None:
    """Resolve type variable from inheritance tree of given type

    Arguments:
        t (type): type to analyse for specification of typevar
        T (TypeVar): type variable to resolve

    Returns:
        tt (type|None):
            type if it can be resolved from the given type, None otherwise
    """

    def _solve(t, T):
        if not hasattr(t, "__orig_bases__"):
            return None

        for b in t.__orig_bases__:
            # check if the base is a generic type
            if isinstance(b, _GenericAlias):
                # get origin and arguments of generic type
                origin = get_origin(b)
                args = get_args(b)
                # search for typevar in origin
                index = _get_typevar_index(origin, T)

                if index is not None:
                    return args[index]

                # recurse to base types
                candidate = _solve(origin, T)
                if candidate is not None:
                    return candidate

            else:
                # not a generic type so just check the bases
                candidate = _solve(b, T)
                if candidate is not None:
                    return candidate

    # follow typevar chain to specification
    while isinstance(T, TypeVar):
        T = _solve(t, T)

    return T
