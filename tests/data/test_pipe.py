from datasets import Features, Value
from hyped.data.pipe import DataPipe
from tests.data.processors.test_base import (
    ConstantDataProcessor,
    ConstantDataProcessorConfig,
)


class TestDataPipe:
    def test_preparation_logic(self):
        # create data processor configs
        c1 = ConstantDataProcessorConfig(name="A", value="1")
        c2 = ConstantDataProcessorConfig(name="B", value="2")
        c3 = ConstantDataProcessorConfig(name="C", value="3")
        # create data processors
        p1 = ConstantDataProcessor(c1)
        p2 = ConstantDataProcessor(c2)
        p3 = ConstantDataProcessor(c3)
        # create data pipe
        p = DataPipe([p1, p2, p3])
        assert not p.is_prepared

        # create different input features
        x = Features({"X": Value("int32")})
        y = Features({"Y": Value("int32")})

        # prepare pipeline with X
        p.prepare(x)
        assert p.is_prepared

        # prepare any processor with Y
        # this should break the feature pipe
        p2.prepare(y)
        assert p2.is_prepared
        assert not p.is_prepared

        # preparing the pipe again should fix the issue
        p.prepare(x)
        assert p.is_prepared

    def test_feature_management(self):
        # create data processor configs
        c1 = ConstantDataProcessorConfig(name="A", value="1")
        c2 = ConstantDataProcessorConfig(name="B", value="2")
        c3 = ConstantDataProcessorConfig(name="C", value="3")
        # create data processors
        p1 = ConstantDataProcessor(c1)
        p2 = ConstantDataProcessor(c2)
        p3 = ConstantDataProcessor(c3)
        # create data pipe
        p = DataPipe([p1, p2, p3])
        # create input and expected output features
        x = Features({"X": Value("int32")})
        y = Features({k: Value("string") for k in "ABC"})

        # prepare pipe
        p.prepare(x)
        # check features
        assert p.is_prepared
        assert p.in_features == x
        assert p.new_features == y
        assert p.out_features == Features(x | y)

    def test_batch_processing(self):
        # create data processor configs
        c1 = ConstantDataProcessorConfig(name="A", value="1")
        c2 = ConstantDataProcessorConfig(name="B", value="2")
        c3 = ConstantDataProcessorConfig(name="C", value="3")
        # create data processors
        p1 = ConstantDataProcessor(c1)
        p2 = ConstantDataProcessor(c2)
        p3 = ConstantDataProcessor(c3)
        # create data pipe
        p = DataPipe([p1, p2, p3])

        # create input batch and corresponding features
        x = Features({"X": Value("int32")})
        batch = {"X": ["example %i" % i for i in range(10)]}
        # apply pipe
        p.prepare(x)
        batch = p.batch_process(batch, index=list(range(10)), rank=0)
        # check processor output
        assert all(k in batch for k in "XABC")
        assert all(x == ("example %i" % i) for i, x in enumerate(batch["X"]))
        assert all(a == "1" for a in batch["A"])
        assert all(a == "2" for a in batch["B"])
        assert all(a == "3" for a in batch["C"])
        # prepare pipeline with X
        p.prepare(x)
