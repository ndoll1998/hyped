from hyped.data.processors.base import DataProcessorConfig, DataProcessor
from datasets import Features, Value


class TestDataProcessor:
    def test_feature_management(self):
        y = Features({"A": Value("int32"), "B": Value("int32")})

        class DummyDataProcessor(DataProcessor):
            def process(self, *args, **kwargs):
                pass

            def map_features(self, features):
                return y

        # create processor instance
        p = DummyDataProcessor(DataProcessorConfig())
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
