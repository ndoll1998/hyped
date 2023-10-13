from hyped.base import TypeRegister


class TestTypeRegister:
    def test_registers(self):
        types = set(TypeRegister.types)
        type_ids = set(TypeRegister.type_ids)

        class A(TypeRegister):
            t = "A"

        # check simple case
        new_types = set(TypeRegister.types) - types
        new_type_ids = set(TypeRegister.type_ids) - type_ids
        assert {A} == new_types
        assert {"A"} == new_type_ids

        # set up complex case
        class B(TypeRegister):
            t = "B"

        class C(B):
            t = "C"

        class D(C, B):
            t = "D"

        # check complex case
        new_types = set(TypeRegister.types) - types
        new_type_ids = set(TypeRegister.type_ids) - type_ids
        assert {A, B, C, D} == new_types
        assert {"A", "B", "C", "D"} == new_type_ids

        # test overwriting registered type ids
        class C1(D):
            t = "C"

        # should have a new type but the type id is overwritten
        new_types = set(TypeRegister.types) - types
        new_type_ids = set(TypeRegister.type_ids) - type_ids
        assert {A, B, C, D, C1} == new_types
        assert {"A", "B", "C", "D"} == new_type_ids

    def test_get_type_by_hash(self):
        class A(TypeRegister):
            t = "A"

        class B(TypeRegister):
            t = "B"

        class C(B):
            t = "C"

        assert A == TypeRegister.get_type_by_hash(hash(A))
        assert B == TypeRegister.get_type_by_hash(hash(B))
        assert C == TypeRegister.get_type_by_hash(hash(C))

    def test_get_type_by_t(self):
        class A(TypeRegister):
            t = "A"

        class B(TypeRegister):
            t = "B"

        class C(B):
            t = "C"

        assert A == TypeRegister.get_type_by_t(A.t)
        assert B == TypeRegister.get_type_by_t(B.t)
        assert C == TypeRegister.get_type_by_t(C.t)
