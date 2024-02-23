from typing import Generic, TypeVar

from hyped.base.generic import _get_typevar_index, solve_typevar


class TestResolveTypeVar:
    def test_get_typevar_index(self):
        class A:
            pass

        class B:
            pass

        T = TypeVar("T")
        U = TypeVar("U")
        V = TypeVar("V")

        class C(A, Generic[T, U, V], B):
            pass

        assert _get_typevar_index(C, T) == 0
        assert _get_typevar_index(C, U) == 1
        assert _get_typevar_index(C, V) == 2

    def test_solve_typevar_unset(self):
        T = TypeVar("T")

        class A(Generic[T]):
            pass

        assert solve_typevar(A, T) is None

    def test_solve_typevar_easy(self):
        T = TypeVar("T")

        class A:
            pass

        class B(Generic[T]):
            pass

        class C(B[A]):
            pass

        assert solve_typevar(C, T) == A

    def test_solve_typevar_deep(self):
        T = TypeVar("T")
        U = TypeVar("U")
        V = TypeVar("V")

        class A:
            pass

        class B:
            pass

        class C(Generic[T, U]):
            pass

        class D(C[A, B]):
            pass

        class E(D):
            pass

        class F(E, Generic[V]):
            pass

        class G:
            pass

        class H(F[G]):
            pass

        # resolve T
        assert solve_typevar(D, T) == A
        assert solve_typevar(E, T) == A
        assert solve_typevar(F, T) == A
        assert solve_typevar(H, T) == A
        # resolve U
        assert solve_typevar(D, U) == B
        assert solve_typevar(E, U) == B
        assert solve_typevar(F, U) == B
        assert solve_typevar(H, U) == B
        # resilve V
        assert solve_typevar(H, V) == G

    def test_solve_typevar_chain(self):
        T = TypeVar("T")
        U = TypeVar("U")
        V = TypeVar("V")

        # typevar chain of different typevars

        class A:
            pass

        class B(Generic[T]):
            pass

        class C(B[U]):
            pass

        class D(C[V]):
            pass

        class E(D[A]):
            pass

        assert solve_typevar(E, T) == A
        assert solve_typevar(E, U) == A
        assert solve_typevar(E, V) == A

        # typevar chain of the same typevar

        class A:
            pass

        class B(Generic[T]):
            pass

        class C(B[T]):
            pass

        class D(C[T]):
            pass

        class E(D[A]):
            pass

        assert solve_typevar(E, T) == A
