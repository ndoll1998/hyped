import datasets
from hyped.data.processors.base import (
    BaseDataProcessorConfig,
    BaseDataProcessor,
)
from datasets import Features, Value
from dataclasses import dataclass


@dataclass
class ConstantDataProcessorConfig(BaseDataProcessorConfig):
    """Configuration for `ConstantDataProcessor`

    Attributes:
        name (str): name of the feature to be added
        value (str): value of the feature to be added
    """

    name: str = "A"
    value: str = "B"


class ConstantDataProcessor(BaseDataProcessor[ConstantDataProcessorConfig]):
    """Data Processor that adds a constant string feature to every example"""

    def map_features(self, features):
        return {self.config.name: datasets.Value("string")}

    def process(self, example, *args, **kwargs):
        return example | {self.config.name: self.config.value}


class TestDataProcessor:
    def test_feature_management(self):
        y = Features({"A": Value("string")})

        # create processor instance
        c = ConstantDataProcessorConfig(name="A", value="B")
        p = ConstantDataProcessor(c)
        assert not p.is_prepared

        # easy case
        x = Features({"X": Value("int32")})
        p.prepare(x)
        assert p.is_prepared
        assert p.in_features == x
        assert p.new_features == y
        assert p.out_features == Features(x | y)

        # conflict between input and new features
        x = Features({"A": Value("int32")})
        p.prepare(x)
        assert p.is_prepared
        assert p.in_features == x
        assert p.new_features == y
        assert p.out_features == Features(x | y)

    def test_batch_processing(self):
        c = ConstantDataProcessorConfig(name="A", value="B")
        p = ConstantDataProcessor(c)
        assert not p.is_prepared

        x = Features({"X": Value("int32")})
        p.prepare(x)
        # create batch of examples and pass through processor
        batch = {"X": ["example %i" % i for i in range(10)]}
        batch = p.batch_process(batch, index=range(10), rank=0)

        # check processor output
        assert ("X" in batch) and ("A" in batch)
        assert all(x == ("example %i" % i) for i, x in enumerate(batch["X"]))
        assert all(a == "B" for a in batch["A"])
