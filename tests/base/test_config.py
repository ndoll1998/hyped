import json
from dataclasses import dataclass
from hyped.base.config import (
    BaseConfig,
    AutoConfig,
    BaseConfigurable,
    BaseAutoConfigurable,
)
from hyped.base.registry import default_registry
from copy import deepcopy
from typing import TypeVar
import pytest

T = TypeVar("T")


class Configurable(BaseConfigurable[T]):
    """Non-abstract configurable type for easier testing"""

    def __init__(self, config):
        self.c = config

    @classmethod
    def from_config(cls, config):
        return cls(config)


@pytest.fixture(autouse=True)
def _reset_registry():
    # get registry state before test execution
    global_hash_register = default_registry.global_hash_register.copy()
    global_type_register = default_registry.global_type_register.copy()
    hash_tree = deepcopy(default_registry.hash_tree)

    # execute test
    yield

    # recover registry state
    default_registry.global_hash_register = global_hash_register
    default_registry.global_type_register = global_type_register
    default_registry.hash_tree = hash_tree


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
        assert a == AutoConfig.from_dict(a_dict)
        assert b == AutoConfig.from_dict(b_dict)

        # test reconstruction from type identifier
        a_dict.pop("__type_hash__")
        b_dict.pop("__type_hash__")
        assert a == AutoConfig.from_dict(a_dict)
        assert b == AutoConfig.from_dict(b_dict)

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
        a_json = a.to_json()
        b_json = b.to_json()

        # test reconstruction from type hash
        assert a == AutoConfig.from_json(a_json)
        assert b == AutoConfig.from_json(b_json)

        # test reconstruction from type identifier
        a_dict = json.loads(a_json)
        b_dict = json.loads(b_json)
        a_dict.pop("__type_hash__")
        b_dict.pop("__type_hash__")
        a_json = json.dumps(a_dict)
        b_json = json.dumps(b_dict)
        assert a == AutoConfig.from_json(a_json)
        assert b == AutoConfig.from_json(b_json)

        # test reconstruction by explicit class
        a_dict.pop("t")
        b_dict.pop("t")
        a_json = json.dumps(a_dict)
        b_json = json.dumps(b_dict)
        assert a == A.from_json(a_json)
        assert b == B.from_json(b_json)


class TestBaseConfigurable:
    def test_config_type(self):
        @dataclass
        class aConfig(BaseConfig):
            t: str = "a.config"

        @dataclass
        class bConfig(BaseConfig):
            t: str = "b.config"

        class A(Configurable[aConfig]):
            pass

        class B(Configurable[bConfig]):
            pass

        assert A.config_type == aConfig
        assert B.config_type == bConfig

        assert A.t.startswith(aConfig.t)
        assert B.t.startswith(bConfig.t)

    def test_auto_from_config(self):
        @dataclass
        class aConfig(BaseConfig):
            t: str = "a.config"

        @dataclass
        class bConfig(BaseConfig):
            t: str = "b.config"

        class A(Configurable[aConfig]):
            pass

        class B(Configurable[bConfig]):
            pass

        class AutoConfigurable(BaseAutoConfigurable[Configurable]):
            pass

        class AutoA(BaseAutoConfigurable[A]):
            pass

        class AutoB(BaseAutoConfigurable[B]):
            pass

        # shared auto class
        assert isinstance(AutoConfigurable.from_config(aConfig()), A)
        assert isinstance(AutoConfigurable.from_config(bConfig()), B)
        # non-shared auto class
        assert isinstance(AutoA.from_config(aConfig()), A)
        assert isinstance(AutoB.from_config(bConfig()), B)
        # test target type out of scope
        with pytest.raises(ValueError):
            AutoB.from_config(aConfig())

        with pytest.raises(ValueError):
            AutoA.from_config(bConfig())
