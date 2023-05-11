import os
import cassis
import datasets
# helpers
from itertools import chain
from functools import cached_property
from dataclasses import dataclass
from typing import Optional

@dataclass
class CasConfig(datasets.BuilderConfig):
    name:str = "cas"

    typesystem: str = None
    # token annotations
    token_type: Optional[str] = None

    # label annotations
    # referse to document-level annotations
    label_type: Optional[str] = None
    label_attr: Optional[str] = None
    label_names: Optional[list[str]|set[str]] = None

    # named entity annotations
    # requires token annotations
    # dataset provides BIO tags for entities
    entity_type: Optional[str] = None
    entity_attr: Optional[str] = None
    entity_names: Optional[list[str]|set[str]] = None

    # bounding-box annotations
    # requires token annotations
    # dataset provides bounding box for each token
    bbox_type: Optional[str] = None
    bbox_x_attr: Optional[str] = None # x-position
    bbox_y_attr: Optional[str] = None # y-position
    bbox_w_attr: Optional[str] = None # width
    bbox_h_attr: Optional[str] = None # height

    # bi-relation annotations
    relation_type: Optional[str] = None
    relation_source: Optional[str] = None
    relation_target: Optional[str] = None
    relation_names: Optional[list[str]|set[str]] = None


class Cas(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = CasConfig

    @cached_property
    def typesystem(self) -> cassis.TypeSystem:
        # load typesystem from file
        with open(self.config.typesystem, 'rb') as f:
            return cassis.load_typesystem(f)

    @property
    def tokens_feature(self):
        # check if token type is specified
        if self.config.token_type is None:
            return None
        # check if type is valid
        if not self.typesystem.contains_type(self.config.token_type):
            raise ValueError("Token type `%s` not found in typesystem." % self.config.token_type)

        # build feature
        return datasets.Sequence(datasets.Value('string'))

    @property
    def labels_feature(self):
        # check if label type is provided
        if self.config.label_type is None:
            return None
        # check if type is valid
        if not self.typesystem.contains_type(self.config.label_type):
            raise ValueError("Label type `%s` not found in typesystem." % self.config.label_type)

        # check if label attribute is set
        if self.config.label_attr is None:
            raise ValueError("Dataset requires valid label attribute. Please specify `label_attr` in call to `datasets.load_dataset`.")
        # check if label attribute is valid
        label_type = self.typesystem.get_type(self.config.label_type)
        if label_type.get_feature(self.config.label_attr) is None:
            raise ValueError("Label attribute `%s` not a valid feature of type `%s`" % (self.config.label_attr, self.config.label_type))

        # check if label names is set
        if not isinstance(self.config.label_names, (list, tuple)):
            raise ValueError("Dataset requires valid list of label names, got `label_names=%s`" % str(self.config.label_names))
        # build feature
        return datasets.Sequence(datasets.ClassLabel(names=list(self.config.label_names)))

    @property
    def entities_feature(self):
        # check if entity type is specified
        if self.config.entity_type is None:
            return None

        # entity annotations require token annotations
        if self.config.token_type is None:
            raise ValueError("Entity annotations require token annotations!")

        # check if type is valid
        if not self.typesystem.contains_type(self.config.entity_type):
            raise ValueError("Entity type `%s` not found in typesystem." % self.config.entity_type)

        # check if token attribute is set
        if self.config.entity_attr is None:
            raise ValueError("Dataset requires valid entity attribute. Please specify `entity_attr` in call to `datasets.load_dataset`.")
        # check if entity attribute is valid
        entity_type = self.typesystem.get_type(self.config.entity_type)
        if entity_type.get_feature(self.config.entity_attr) is None:
            raise ValueError("Entity attribute `%s` not a valid feature of type `%s`" % (self.config.entity_attr, self.config.entity_type))

        # check if entity names is set
        if not isinstance(self.config.entity_names, (list, tuple)):
            raise ValueError("Dataset requires valid list of entity names, got `entity_names=%s`" % str(self.config.label_names))

        # build entity feature
        return datasets.Features({
            'begin': datasets.Sequence(datasets.Value('int32')),
            'end': datasets.Sequence(datasets.Value('int32')),
            'type': datasets.Sequence(
                datasets.ClassLabel(
                    names=list(self.config.entity_names)
                )
            )
        })

    @property
    def positions_feature(self):
        return None

    @property
    def relations_feature(self):
        return None

    def extract_tokens(self, cas:cassis.Cas, tokens:Optional[list]):
        # check if tokens are requested
        if self.config.token_type is None:
            return None
        # extract tokens from cas and sort them
        assert tokens is not None
        return [t.get_covered_text() for t in tokens]

    def extract_labels(self, cas:cassis.Cas):
        # check if label is requested
        if self.config.label_type is None:
            return None
        # get label annotation from cas and get label from it
        labels = [
            label.get(self.config.label_attr)
            for label in cas.select(self.config.label_type)
        ]
        # filter out only those that are in the list of provided labels
        return [l for l in labels if l in self.config.label_names]

    def extract_entities(self, cas:cassis.Cas, tokens:Optional[list]):
        # check if label is requested
        if self.config.entity_type is None:
            return None

        # build token to index lookup table
        token2idx = {t:i for i, t in enumerate(tokens)}

        entities = {
            'begin': [],
            'end': [],
            'type': []
        }
        # get all entity annotations
        for e in cas.select(self.config.entity_type):
            # check if entity type is listed in entity names
            if e.get(self.config.entity_attr) not in self.config.entity_names:
                continue
            # get all tokens that are covered by the entity annotations
            covered = cas.select_covered(self.config.token_type, e)
            covered_idx = [token2idx[t] for t in covered]
            # build item
            entities['begin'].append(min(covered_idx))
            entities['end'].append(max(covered_idx)+1)
            entities['type'].append(e.get(self.config.entity_attr))

        assert len(entities['begin']) == len(entities['end']) == len(entities['type'])
        # return bio labels
        return entities

    def extract_positions(self, cas:cassis.Cas, tokens:Optional[list]):
        return None

    def extract_relations(self, cas:cassis.Cas, tokens:Optional[list]):
        return None

    def _info(self):

        # build dict of all features, including invalid ones
        features = {
            "text": datasets.Value("string"),
            "tokens": self.tokens_feature,
            "labels": self.labels_feature,
            "entities": self.entities_feature,
            "positions": self.positions_feature,
            "relations": self.relations_feature
        }
        # remove all invalid features
        features = datasets.Features({k:f for k, f in features.items() if f is not None})

        return datasets.DatasetInfo(
            description="Dataset loaded from CAS",
            features=features,
            supervised_keys=None
        )

    def _split_generators(self, dl_manager):

        # check data files argument
        if self.config.data_files is None:
            raise ValueError("No data files specified. Please specify `data_files` in call to `datasets.load_dataset`.")
        if not isinstance(self.config.data_files, dict):
            raise ValueError("Expected `data_files` to be a dictionary mapping splits to files, got %s" % type(data_files).__name__)

        # prepare data files
        data_files = dl_manager.download_and_extract(self.config.data_files)
        assert isinstance(data_files, dict), "Expected dict but got %s" % type(data_files).__name__

        splits = []
        # generate data split generators
        for split_name, files in data_files.items():
            # prepare files
            files = [dl_manager.iter_files(file) for file in files]
            # generate split generator
            split = datasets.SplitGenerator(
                name=split_name,
                gen_kwargs=dict(files=files),
            )
            split.split_info.num_examples = len(files)
            # add to splits
            splits.append(split)

        return splits

    def _generate_examples(self, files:list[str]):
        # iterate over given files
        for idx, file in enumerate(chain(*files)):
            # load cas from file
            with open(file, 'rb') as f:
                cas = cassis.load_cas_from_xmi(f, typesystem=self.typesystem)

            # the tokens are required for other annotations and
            # their order matters for preprocessing
            tokens = sorted(cas.select(self.config.token_type), key=lambda t: t.begin) \
                if self.config.token_type is not None else None
            # build data item
            item = {
                'text': cas.sofa_string,
                'labels': self.extract_labels(cas),
                'tokens': self.extract_tokens(cas, tokens),
                'entities': self.extract_entities(cas, tokens),
                'positions': self.extract_positions(cas, tokens),
                'relations': self.extract_relations(cas, tokens)
            }
            # remove all invalid features and yield item
            item = {k: v for k, v in item.items() if v is not None}
            yield idx, item
