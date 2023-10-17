import os
import cassis
import datasets
import pytest
import hyped.data.datasets  # noqa: F401
from cassis.typesystem import TYPE_NAME_STRING


def build_typesystem(path):
    # create sample typesystem
    typesystem = cassis.TypeSystem(add_document_annotation_type=False)
    # add label annotation
    label = typesystem.create_type(
        name="cassis.Label", supertypeName="uima.cas.TOP"
    )
    typesystem.create_feature(
        domainType=label, name="label", rangeType=TYPE_NAME_STRING
    )
    # add entity annotation
    entity = typesystem.create_type(name="cassis.Entity")
    typesystem.create_feature(
        domainType=entity, name="entityType", rangeType=TYPE_NAME_STRING
    )
    # save typesystem
    typesystem.to_xml(os.path.join(path, "typesystem.test.xml"))


def build_examples(path):
    # load test typesystem
    with open(os.path.join(path, "typesystem.test.xml"), "rb") as f:
        typesystem = cassis.load_typesystem(f)
    # get annotation types
    Entity = typesystem.get_type("cassis.Entity")
    Label = typesystem.get_type("cassis.Label")
    # create cas object
    cas = cassis.Cas(typesystem=typesystem)
    cas.sofa_string = "U.N. official Ekeus heads for Baghdad."
    # add annotations
    cas.add_all(
        [
            Entity(begin=0, end=4, entityType="ORG"),
            Entity(begin=30, end=37, entityType="LOC"),
            Label(label="Document"),
        ]
    )
    # save in json and xmi format
    cas.to_json(os.path.join(path, "cas.test.json"))
    cas.to_xmi(os.path.join(path, "cas.test.xmi"))


class TestCasDataset:
    @pytest.fixture(autouse=True)
    def _create_resources(self, tmpdir):
        # create resource files
        build_typesystem(tmpdir)
        build_examples(tmpdir)
        # run test
        yield

    def test_load_data(self, tmpdir):
        # load dataset
        ds = datasets.load_dataset(
            "hyped.data.datasets.cas",
            typesystem=os.path.join(tmpdir, "typesystem.test.xml"),
            data_files={"train": os.path.join(tmpdir, "cas.test.*")},
        )

        # check dataset length
        assert len(ds["train"]) == 2
        # check dataset features
        assert "text" in ds["train"].features
        assert "cassis.Entity:begin" in ds["train"].features
        assert "cassis.Entity:end" in ds["train"].features
        assert "cassis.Entity:entityType" in ds["train"].features
        assert "cassis.Label:label" in ds["train"].features

        # check annotations
        for example in ds["train"]:
            text = example["text"]
            # test label annotation
            assert example["cassis.Label:label"] == ["Document"]
            # test entity annotation features
            assert len(example["cassis.Entity:entityType"]) == len(
                example["cassis.Entity:begin"]
            )
            assert len(example["cassis.Entity:entityType"]) == len(
                example["cassis.Entity:end"]
            )

            # test annotation content
            for eType, begin, end in zip(
                example["cassis.Entity:entityType"],
                example["cassis.Entity:begin"],
                example["cassis.Entity:end"],
            ):
                assert eType in {"ORG", "LOC"}
                # test content
                if eType == "ORG":
                    assert text[begin:end] == "U.N."
                if eType == "LOC":
                    assert text[begin:end] == "Baghdad"

    def test_load_specific_types_only(self, tmpdir):
        # load dataset
        ds = datasets.load_dataset(
            "hyped.data.datasets.cas",
            typesystem=os.path.join(tmpdir, "typesystem.test.xml"),
            data_files={"train": os.path.join(tmpdir, "cas.test.*")},
            annotation_types=["cassis.Label"],
        )

        # check dataset length
        assert len(ds["train"]) == 2
        assert "text" in ds["train"].features
        # label should be included
        assert "cassis.Label:label" in ds["train"].features
        # entity should be excluded
        assert "cassis.Entity:begin" not in ds["train"].features
        assert "cassis.Entity:end" not in ds["train"].features
        assert "cassis.Entity:entityType" not in ds["train"].features
