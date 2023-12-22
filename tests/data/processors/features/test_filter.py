from tests.data.processors.base import BaseTestDataProcessor
from hyped.data.processors.features.filter import (
    FilterFeatures,
    FilterFeaturesConfig,
)
from itertools import chain, product
from datasets import Features, Sequence, Value
import pytest

FEATURES = Features(
    {
        "X": Value("int32"),
        "Y": Sequence(Value("int32"), length=3),
        "A": {
            "x": Value("int32"),
            "y": Sequence(Value("int32"), length=2),
            "z": Value("int32"),
        },
        "B": Sequence(
            Features(
                {
                    "x": Value("int32"),
                    "y": Sequence(Value("int32"), length=4),
                }
            ),
            length=2,
        ),
    }
)

BATCH = {
    "X": list(range(0, 12)),
    "Y": [[i + 1, i + 2, i + 3] for i in range(0, 12)],
    "A": [{"x": 2 * i, "y": [3 * i, 4 * i], "z": 5 * i} for i in range(0, 12)],
    "B": [
        [
            {"x": i, "y": [i, i, i, i]},
            {"x": i * 2, "y": [i * 2, i * 2, i * 2, i * 2]},
        ]
        for i in range(0, 12)
    ],
}


class BaseTestFilterFeatures(BaseTestDataProcessor):
    @pytest.fixture()
    def kwargs(self):
        raise NotImplementedError

    @pytest.fixture()
    def in_features(self):
        return FEATURES

    @pytest.fixture()
    def in_batch(self):
        return BATCH

    @pytest.fixture
    def processor(self, kwargs):
        return FilterFeatures(kwargs["config"])

    @pytest.fixture
    def expected_out_features(self, kwargs):
        return kwargs["out_features"]

    @pytest.fixture
    def expected_out_batch(self, kwargs):
        return kwargs["out_batch"]


class TestKeepFeatures(BaseTestFilterFeatures):
    @pytest.fixture(
        params=chain(
            [
                {
                    "config": FilterFeaturesConfig(keep=key),
                    "out_features": {key: FEATURES[key]},
                    "out_batch": {key: BATCH[key]},
                }
                for key in "XYAB"
            ],
            [
                {
                    "config": FilterFeaturesConfig(keep=list(keys)),
                    "out_features": {k: FEATURES[k] for k in keys},
                    "out_batch": {k: BATCH[k] for k in keys},
                }
                for n in range(1, 4)
                for keys in product("XYAB", repeat=n)
            ],
            # TODO: these need to be implemented
            [
                # {
                #    "config": FilterFeaturesConfig(keep=("A", "x")),
                #    "out_features": {"A": {"x": FEATURES["A"]["x"]}},
                #    "out_batch": {
                #        "A": [{"x": item["x"]} for item in BATCH["A"]]
                #    },
                # },
                # {
                #    "config": FilterFeaturesConfig(keep=("A", "y")),
                #    "out_features": {"A": {"y": FEATURES["A"]["y"]}},
                #    "out_batch": {
                #        "A": [{"y": item["y"]} for item in BATCH["A"]]
                #    },
                # },
                # {
                #    "config": FilterFeaturesConfig(keep=("A", "z")),
                #    "out_features": {"A": {"z": FEATURES["A"]["z"]}},
                #    "out_batch": {
                #        "A": [{"z": item["z"]} for item in BATCH["A"]]
                #    },
                # },
                # {
                #    "config": FilterFeaturesConfig(
                #        keep=[("A", "x"), ("A", "y")]
                #    ),
                #    "out_features": {
                #        "A": {
                #            "x": FEATURES["A"]["x"], "y": FEATURES["A"]["y"]
                #        }
                #    },
                #    "out_batch": {
                #        "A": [
                #            {"x": item["x"], "y": item["y"]}
                #            for item in BATCH["A"]
                #         ]
                #     },
                # },
                # {
                #     "config": FilterFeaturesConfig(
                #         keep=[("A", "y"), ("A", "z")]
                #     ),
                #     "out_features": {
                #         "A": {
                #             "y": FEATURES["A"]["y"], "z": FEATURES["A"]["z"]
                #         }
                #     },
                #     "out_batch": {
                #         "A": [
                #             {"y": item["y"], "z": item["z"]}
                #             for item in BATCH["A"]
                #         ]
                #     },
                # },
                # {
                #     "config": FilterFeaturesConfig(
                #         keep=[("A", "x"), ("A", "z")]
                #     ),
                #     "out_features": {
                #         "A": {
                #             "x": FEATURES["A"]["x"], "z": FEATURES["A"]["z"]
                #         }
                #     },
                #     "out_batch": {
                #         "A": [
                #             {"x": item["x"], "z": item["z"]}
                #             for item in BATCH["A"]
                #         ]
                #     },
                # },
                # {
                #    "config": FilterFeaturesConfig(
                #        keep=[("A", "x"), ("A", "y"), ("A", "z")]
                #    ),
                #    "out_features": {"A": FEATURES["A"]},
                #    "out_batch": {"A": BATCH["A"]},
                # },
                # {
                #    "config": FilterFeaturesConfig(
                #        keep=("B", slice(-1), "x")
                #    ),
                #    "out_features": {
                #        "B": Sequence(
                #            FEATURES["B"].feature["x"],
                #            length=FEATURES["B"].length,
                #        )
                #    },
                #    "out_batch": {
                #        "B": [
                #            [{"x": item["x"]} for item in seq]
                #            for seq in BATCH["B"]
                #        ]
                #    },
                # },
            ],
        )
    )
    def kwargs(self, request):
        return request.param


class TestRemoveFeatures(BaseTestFilterFeatures):
    @pytest.fixture(
        params=chain(
            [
                {
                    "config": FilterFeaturesConfig(keep=key),
                    "out_features": {key: FEATURES[key]},
                    "out_batch": {key: BATCH[key]},
                }
                for key in "XYAB"
            ],
            [
                {
                    "config": FilterFeaturesConfig(remove=list(keys)),
                    "out_features": {
                        k: FEATURES[k] for k in "XYAB" if k not in keys
                    },
                    "out_batch": {
                        k: BATCH[k] for k in "XYAB" if k not in keys
                    },
                }
                for n in range(1, 4)
                for keys in product("XYAB", repeat=n)
            ],
            # TODO: include tests for non-string keys once implemented
        )
    )
    def kwargs(self, request):
        return request.param


class TestErrorOnInvalidConfig(BaseTestDataProcessor):
    @pytest.fixture
    def in_features(self):
        return FEATURES

    @pytest.fixture(
        params=[
            FilterFeaturesConfig(),
            FilterFeaturesConfig(keep=["X"], remove=["Y"]),
        ]
    )
    def processor(self, request):
        return FilterFeatures(request.param)

    @pytest.fixture
    def expected_err_on_prepare(self):
        return ValueError
