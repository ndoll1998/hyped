import json
from dataclasses import dataclass
from hyped.base import TypeRegister, BaseConfig


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


class TestBaseConfig:
    def test_dict_conversion(self):
        @dataclass
        class A(BaseConfig):
            t: str = "A"
            x: str = ""
            y: str = ""

        @dataclass
        class B(A):
            t: str = "B"
            z: str = ""

        a = A(x="x", y="y")
        b = B(x="a", y="b", z="c")
        # convert to dictionaties
        a_dict = a.to_dict()
        b_dict = b.to_dict()

        # test reconstruction from type hash
        assert a == BaseConfig.from_dict(a_dict)
        assert b == BaseConfig.from_dict(b_dict)

        # test reconstruction from type identifier
        a_dict.pop("__type_hash__")
        b_dict.pop("__type_hash__")
        assert a == BaseConfig.from_dict(a_dict)
        assert b == BaseConfig.from_dict(b_dict)

        # test reconstruction by explicit class
        a_dict.pop("t")
        b_dict.pop("t")
        assert a == A.from_dict(a_dict)
        assert b == B.from_dict(b_dict)

    def test_serialization(self):
        @dataclass
        class A(BaseConfig):
            t: str = "A"
            x: str = ""
            y: str = ""

        @dataclass
        class B(A):
            t: str = "B"
            z: str = ""

        a = A(x="x", y="y")
        b = B(x="a", y="b", z="c")
        # convert to dictionaties
        a_json = a.serialize()
        b_json = b.serialize()

        # test reconstruction from type hash
        assert a == BaseConfig.deserialize(a_json)
        assert b == BaseConfig.deserialize(b_json)

        # test reconstruction from type identifier
        a_dict = json.loads(a_json)
        b_dict = json.loads(b_json)
        a_dict.pop("__type_hash__")
        b_dict.pop("__type_hash__")
        a_json = json.dumps(a_dict)
        b_json = json.dumps(b_dict)
        assert a == BaseConfig.deserialize(a_json)
        assert b == BaseConfig.deserialize(b_json)

        # test reconstruction by explicit class
        a_dict.pop("t")
        b_dict.pop("t")
        a_json = json.dumps(a_dict)
        b_json = json.dumps(b_dict)
        assert a == A.from_dict(a_dict)
        assert b == B.from_dict(b_dict)
