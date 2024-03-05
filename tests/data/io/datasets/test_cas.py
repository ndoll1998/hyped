import os

import cassis
import datasets
import pytest
from cassis.typesystem import TYPE_NAME_STRING

import hyped.data.io.datasets  # noqa: F401


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
    # add bi-relation entity
    relation = typesystem.create_type(
        name="cassis.Relation", supertypeName="uima.cas.TOP"
    )
    typesystem.create_feature(
        domainType=relation, name="source", rangeType=entity
    )
    typesystem.create_feature(
        domainType=relation, name="target", rangeType=entity
    )
    # save typesystem
    typesystem.to_xml(os.path.join(path, "typesystem.test.xml"))


def build_examples(path):
    # load test typesystem
    with open(os.path.join(path, "typesystem.test.xml"), "rb") as f:
        typesystem = cassis.load_typesystem(f)
    # get annotation types
    Entity = typesystem.get_type("cassis.Entity")
    Relation = typesystem.get_type("cassis.Relation")
    Label = typesystem.get_type("cassis.Label")
    # create cas object
    cas = cassis.Cas(typesystem=typesystem)
    cas.sofa_string = "U.N. official Ekeus heads for Baghdad."
    # create entities
    org = Entity(begin=0, end=4, entityType="ORG")
    loc = Entity(begin=30, end=37, entityType="LOC")
    # add annotations
    cas.add_all(
        [org, loc, Relation(source=org, target=loc), Label(label="Document")]
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
            "hyped.data.io.datasets.cas",
            typesystem=os.path.join(tmpdir, "typesystem.test.xml"),
            data_files={"train": os.path.join(tmpdir, "cas.test.*")},
            cache_dir=os.path.join(tmpdir, "cache"),
        )

        # check dataset length
        assert len(ds["train"]) == 2
        # check features
        assert "sofa" in ds["train"].features
        # label features
        assert "cassis.Label:label" in ds["train"].features
        # entity features
        assert "cassis.Entity:begin" in ds["train"].features
        assert "cassis.Entity:end" in ds["train"].features
        assert "cassis.Entity:entityType" in ds["train"].features
        # relation features
        assert "cassis.Relation:source" in ds["train"].features
        assert "cassis.Relation:target" in ds["train"].features

        # check annotations
        for example in ds["train"]:
            text = example["sofa"]
            # test label annotation
            assert example["cassis.Label:label"] == ["Document"]
            # test entity annotation features
            assert len(example["cassis.Entity:entityType"]) == 2
            assert len(example["cassis.Entity:entityType"]) == len(
                example["cassis.Entity:begin"]
            )
            assert len(example["cassis.Entity:entityType"]) == len(
                example["cassis.Entity:end"]
            )
            # test relation annotation features
            assert len(example["cassis.Relation:source"]) == 1
            assert len(example["cassis.Relation:source"]) == len(
                example["cassis.Relation:target"]
            )

            # test entity content
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

            # test relation content
            for src, tgt in zip(
                example["cassis.Relation:source"],
                example["cassis.Relation:target"],
            ):
                assert example["cassis.Entity:entityType"][src] == "ORG"
                assert example["cassis.Entity:entityType"][tgt] == "LOC"

    def test_load_specific_types_only(self, tmpdir):
        # load dataset
        ds = datasets.load_dataset(
            "hyped.data.io.datasets.cas",
            typesystem=os.path.join(tmpdir, "typesystem.test.xml"),
            data_files={"train": os.path.join(tmpdir, "cas.test.*")},
            annotation_types=["cassis.Label"],
            cache_dir=os.path.join(tmpdir, "cache"),
        )

        # check dataset length
        assert len(ds["train"]) == 2
        assert "sofa" in ds["train"].features
        # label should be included
        assert "cassis.Label:label" in ds["train"].features
        # entity should be excluded
        assert "cassis.Entity:begin" not in ds["train"].features
        assert "cassis.Entity:end" not in ds["train"].features
        assert "cassis.Entity:entityType" not in ds["train"].features
        # relation should be excluded
        assert "cassis.Relation:source" not in ds["train"].features
        assert "cassis.Relation:target" not in ds["train"].features

    def test_error_on_required_type(self, tmpdir):
        with pytest.raises(RuntimeError):
            # load dataset
            datasets.load_dataset(
                "hyped.data.io.datasets.cas",
                typesystem=os.path.join(tmpdir, "typesystem.test.xml"),
                data_files={"train": os.path.join(tmpdir, "cas.test.*")},
                annotation_types=[
                    "cassis.Label",
                    "cassis.Relation",  # relation require entities
                ],
                cache_dir=os.path.join(tmpdir, "cache"),
            )
