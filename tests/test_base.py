import json
from dataclasses import dataclass
from hyped.base import TypeRegistry, BaseConfig
from copy import deepcopy
import pytest


@pytest.fixture(autouse=True)
def _reset_register():
    # get registry state before test execution
    global_hash_register = type(TypeRegistry)._global_hash_register.copy()
    global_type_register = type(TypeRegistry)._global_type_register.copy()
    hash_tree = deepcopy(type(TypeRegistry)._hash_tree)

    # execute test
    yield

    # recover registry state
    type(TypeRegistry)._global_hash_register = global_hash_register
    type(TypeRegistry)._global_type_register = global_type_register
    type(TypeRegistry)._hash_tree = hash_tree


class TestTypeRegistry:
    def test_registers(self):
        types = set(TypeRegistry.types)
        type_ids = set(TypeRegistry.type_ids)

        class A(TypeRegistry):
            t = "A"

        # check simple case
        assert {A} == set(TypeRegistry.types) - types
        assert {"A"} == set(TypeRegistry.type_ids) - type_ids

        # set up complex case
        class B(TypeRegistry):
            t = "B"

        class C(B):
            t = "C"

        class D(C, B):
            t = "D"

        # check complex case
        assert {A, B, C, D} == set(TypeRegistry.types) - types
        assert {"A", "B", "C", "D"} == set(TypeRegistry.type_ids) - type_ids

        # test overwriting registered type ids
        class C1(D):
            t = "C"

        # should have a new type but the type id is overwritten
        assert {A, B, C, D, C1} == set(TypeRegistry.types) - types
        assert {"A", "B", "C", "D"} == set(TypeRegistry.type_ids) - type_ids

    def test_subtype_registers(self):
        types = set(TypeRegistry.types)
        type_ids = set(TypeRegistry.type_ids)

        class A(TypeRegistry):
            t = "A"

        class B(A):
            t = "B"

        class C(A):
            t = "C"

        class D(B):
            t = "D"

        class D2(C):
            t = "D"

        # check types
        assert {A, B, C, D, D2} == set(TypeRegistry.types) - types
        assert {A, B, C, D, D2} == set(A.types)
        assert {B, D} == set(B.types)
        assert {C, D2} == set(C.types)
        # check type ids
        assert {"A", "B", "C", "D"} == set(TypeRegistry.type_ids) - type_ids
        assert {"A", "B", "C", "D"} == set(A.type_ids)
        assert {"B", "D"} == set(B.type_ids)
        assert {"C", "D"} == set(C.type_ids)

    def test_get_type_by_hash(self):
        class A(TypeRegistry):
            t = "A"

        class B(TypeRegistry):
            t = "B"

        class C(B):
            t = "C"

        assert A == TypeRegistry.get_type_by_hash(hash(A))
        assert B == TypeRegistry.get_type_by_hash(hash(B))
        assert C == TypeRegistry.get_type_by_hash(hash(C))

    def test_get_type_by_t(self):
        class A(TypeRegistry):
            t = "A"

        class B(TypeRegistry):
            t = "B"

        class C(B):
            t = "C"

        assert A == TypeRegistry.get_type_by_t(A.t)
        assert B == TypeRegistry.get_type_by_t(B.t)
        assert C == TypeRegistry.get_type_by_t(C.t)


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
