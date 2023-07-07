import os
import cassis
import datasets
import tokenizers
import numpy as np
import warnings
import logging
import multiprocessing as mp
import queue
# helpers
from itertools import chain
from functools import cached_property, partial
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator, Optional, Any

# get logger
logger = logging.getLogger(__name__)

class CasGenerator:

    def __init__(
        self,
        typesystem: cassis.TypeSystem,
        split_into: Optional[str] = None,
        split_filter_attr: Optional[str] = None,
        split_filter_space: Optional[list[str]] = None
    ) -> None:
        # save typesystem and arguments
        self.typesystem = typesystem
        self.split_into = split_into
        self.filter_attr = split_filter_attr
        self.filter_space = split_filter_space

        # check if type to split into is set and valid
        if self.should_split and not self.typesystem.contains_type(self.split_into):
            raise ValueError(
                "Type `%s` to split cas into not found in typesystem." % self.split_into
            )

        if self.should_filter and not isinstance(split_filter_space, (list, tuple, set)):
            raise ValueError(
                "Invalid filter space, expected a list but got %s" % str(split_filter_space)
            )
        elif self.should_filter:
            # convert filter space to set for faster lookups
            self.filter_space = set(split_filter_space)

    @property
    def should_split(self) -> bool:
        return self.split_into is not None

    @property
    def should_filter(self) -> bool:
        return self.should_split and (self.filter_attr is not None)

    def split_cas(self, cas: cassis.Cas) -> Iterator[cassis.Cas]:

        for split in cas.select(self.split_into):

            if self.should_filter and (split.get(self.filter_attr) not in self.filter_space):
                # split is filtered out
                continue

            # create empty cas for split
            split_cas = cassis.Cas(typesystem=self.typesystem)
            split_cas.sofa_string = split.get_covered_text()

            # add all annotations
            for T in self.typesystem.get_types():
                # collect features of the type
                features = [
                    f.name for f in T.all_features if f.name not in ('id', 'sofa')
                ]

                for a in cas.select_covered(T, split):
                    # collect kwargs and update positions
                    kwargs = {f: a.get(f) for f in features}
                    if 'begin' in features:
                        kwargs['begin'] -= split.begin
                        kwargs['end'] -= split.begin
                    # create new annotation
                    new_a = T(**kwargs)
                    split_cas.add_annotation(new_a)
                    # make sure the annotations match
                    assert a.get_covered_text() == new_a.get_covered_text()

            yield split_cas

    def generate(self, fpath: str) -> Iterator[cassis.Cas]:
        # load cas from file
        with open(fpath, 'rb') as f:
            cas = cassis.load_cas_from_xmi(f, typesystem=self.typesystem)

        if not self.should_split:
            # yield full cas and thats it :)
            yield cas

        else:
            # split cas
            yield from self.split_cas(cas)
        

class PreTokenizer:

    def __init__(self, typesystem: cassis.TypeSystem, token_type: str) -> None:
        self.typesystem = typesystem
        self.token_type = token_type
        # create pretokenizer
        self.tokenizer = tokenizers.pre_tokenizers.Sequence([
            tokenizers.pre_tokenizers.Whitespace(),
            tokenizers.pre_tokenizers.Punctuation(),
            tokenizers.pre_tokenizers.UnicodeScripts(),
        ])

    def tokenize(self, cas: cassis.Cas) -> cassis.Cas:
        # remove all existing token annotations
        for token in cas.select(self.token_type):
            cas.remove_annotation(token)

        # apply pretokenizer
        tokens = self.tokenizer.pre_tokenize_str(cas.sofa_string)
        # add annotations
        Token = self.typesystem.get_type(self.token_type)
        cas.add_annotations([
            Token(begin=b, end=e) for _, (b, e) in tokens
        ])
        # return cas with new token annotations
        return cas


class FeatureExtractor:

    def __init__(
        self, typesystem: cassis.TypeSystem, required_types_and_attrs:dict[str, list[str]] ={}
    ) -> None:
        self.typesystem = typesystem
        self.required_types_and_attrs = required_types_and_attrs

        # check if all required types and their attributes are valid
        for t, attrs in self.required_types_and_attrs.items():
            # make sure type is valid
            if not self.typesystem.contains_type(t):
                raise ValueError("Type `%s` not found in typesystem." % t)

            T = self.typesystem.get_type(t)
            for attr in attrs:
                assert attr is not None
                if T.get_feature(attr) is None:
                    raise ValueError("Type `%s` has not feature `%s`" % (t, attr))

    @property
    def required_types(self):
        return self.required_types_and_attrs.keys()

    @property
    def feature(self):
        raise NotImplementedError()

    def extract(self, cas: cassis.Cas):
        raise NotImplementedError()


class TextFeatureExtractor(FeatureExtractor):

    @property
    def feature(self):
        return datasets.Value('string')

    def extract(self, cas: cassis.Cas) -> str:
        return cas.sofa_string


class TokensFeatureExtractor(FeatureExtractor):

    def __init__(self, typesystem: cassis.TypeSystem, token_type: str) -> None:
        super(TokensFeatureExtractor, self).__init__(
            typesystem, {token_type: []}
        )
        self.token_type = token_type

    @property
    def feature(self):
        return datasets.Sequence(datasets.Value('string'))

    def extract(self, cas: cassis.Cas) -> None:
        return [
            t.get_covered_text() for t in sorted(
                cas.select(self.token_type),
                key=lambda e: e.begin
            )
        ]


class LabelsFeatureExtractor(FeatureExtractor):

    def __init__(
        self,
        typesystem: cassis.TypeSystem,
        label_type: str,
        label_attr: str,
        label_names: list[str]
    ) -> None:
        
        # make sure label attribute is set
        if self.entity_attr is None:
            raise ValueError(
                "Dataset requires valid label attribute. Please specify `label_attr` "
                "in call to `datasets.load_dataset`."
            )
        
        # initialize
        super(LabelsFeatureExtractor, self).__init__(
            typesystem, {label_type: [label_attr]}
        )
        self.label_type = label_type
        self.label_attr = label_attr
        self.label_names = label_names

        # check if label names is set
        if not isinstance(self.label_names, (set, list, tuple)):
            raise ValueError(
                "Dataset requires valid list of label names, "
                "got `label_names=%s`" % str(self.label_names)
            )

        # convert to set
        self.label_names = set(self.label_names)

    @property
    def feature(self):
        return datasets.Sequence(datasets.ClassLabel(names=list(self.label_names)))

    def extract(self, cas: cassis.Cas) -> None:
        # get label annotation from cas and get label from it
        labels = [
            label.get(self.label_attr)
            for label in cas.select(self.label_type)
        ]
        # filter out only those that are in the list of provided labels
        return [l for l in labels if l in self.label_names]


class EntitiesFeatureExtractor(FeatureExtractor):

    def __init__(
        self,
        typesystem: cassis.TypeSystem,
        token_type: str,
        entity_type: str,
        entity_attr: str,
        entity_names: list[str]|set[str]
    ) -> None:

        # make sure entity attribute is set
        if entity_attr is None:
            raise ValueError(
                "Dataset requires valid entity attribute. Please specify `entity_attr` "
                "in call to `datasets.load_dataset`."
            )

        # initialize
        super(EntitiesFeatureExtractor, self).__init__(
            typesystem, {token_type: [], entity_type: [entity_attr]}
        )
        self.token_type = token_type
        self.entity_type = entity_type
        self.entity_attr = entity_attr
        self.entity_names = entity_names

        # check if entity names is set
        if not isinstance(self.entity_names, (list, tuple)):
            raise ValueError(
                "Dataset requires valid list of entity names, "
                "got `entity_names=%s`" % str(self.label_names)
            )

        # convert to set
        self.entity_names = set(self.entity_names)

    @property
    def feature(self):
        return datasets.Features({
            'begin': datasets.Sequence(datasets.Value('int32')),
            'end': datasets.Sequence(datasets.Value('int32')),
            'type': datasets.Sequence(
                datasets.ClassLabel(
                    names=list(self.entity_names)
                )
            )
        })

    def extract(self, cas: cassis.Cas) -> None:
        # get tokens in order and create mapping from token to token index
        tokens = sorted(cas.select(self.token_type), key=lambda e: e.begin)
        token2idx = {t:i for i, t in enumerate(tokens)}
        
        entities = {'begin': [], 'end': [], 'type': []}
        # get all entity annotations
        for e in cas.select(self.entity_type):
            # check if entity type is listed in entity names
            if e.get(self.entity_attr) not in self.entity_names:
                continue
            # get all tokens that are covered by the entity annotations
            covered = cas.select_covered(self.token_type, e)
            covered_idx = [token2idx[t] for t in covered]

            if len(covered_idx) > 0:
                # build item
                entities['begin'].append(min(covered_idx))
                entities['end'].append(max(covered_idx)+1)
                entities['type'].append(e.get(self.entity_attr))
            else:
                warnings.warn(
                    "No tokens found for entity `%s` of type `%s`" % (
                        e.get_covered_text(), e.get(self.entity_attr)
                    ),
                    UserWarning
                )

        return entities


class BoundingBoxFeatureExtractor(FeatureExtractor):

    def __init__(
        self,
        typesystem: cassis.TypeSystem,
        token_type: str,
        bbox_type: str,
        bbox_x_attr: str,
        bbox_y_attr: str,
        bbox_w_attr: str,
        bbox_h_attr: str,
        normalize: bool,
        bbox_x_scale: int,
        bbox_y_scale: int
    ) -> None:
        
        if bbox_x_attr is None:
            raise ValueError(
                "Dataset requires valid bounding box x-position attribute. Please specify "
                "`bbox_x_attr` in call to `datasets.load_dataset`."
            )
        if bbox_y_attr is None:
            raise ValueError(
                "Dataset requires valid bounding box y-position attribute. Please specify "
                "`bbox_y_attr` in call to `datasets.load_dataset`."
            )
        if bbox_w_attr is None:
            raise ValueError(
                "Dataset requires valid bounding box width attribute. Please specify "
                "`bbox_w_attr` in call to `datasets.load_dataset`."
            )
        if bbox_h_attr is None:
            raise ValueError(
                "Dataset requires valid bounding box height attribute. Please specify "
                "`bbox_h_attr` in call to `datasets.load_dataset`."
            )
        
        # initialize
        super(BoundingBoxFeatureExtractor, self).__init__(
            typesystem, {
                token_type: [],
                bbox_type: [bbox_x_attr, bbox_y_attr, bbox_w_attr, bbox_h_attr]
            }
        )
        self.token_type = token_type
        self.bbox_type = bbox_type
        self.x_attr = bbox_x_attr
        self.y_attr = bbox_y_attr
        self.w_attr = bbox_w_attr
        self.h_attr = bbox_h_attr
        # normalize bounding boxes
        self.normalize = normalize
        self.x_scale = bbox_x_scale
        self.y_scale = bbox_y_scale

    @property
    def feature(self):
        # matrix of shape (n, 4) where n is the number of tokens
        # each row is in the format of (x0, y0, x1, y1)
        return datasets.Sequence(
            datasets.Sequence(
                datasets.Value('int32' if self.normalize else 'float32'),
                length=4
            )
        )

    def extract(self, cas: cassis.Cas):

        # extract all relevant information from cas
        tokens = sorted(cas.select(self.token_type), key=lambda e: e.begin)
        bbox_annotations = np.asarray(list(cas.select(self.bbox_type)))
        bbox_spans = np.asarray([(a.begin, a.end) for a in bbox_annotations])

        if len(bbox_annotations) == 0:
            raise ValueError("No bounding box annotations found!")

        bboxes = np.empty((len(tokens), 4))

        for i, token in enumerate(tokens):
            # find bboxes overlapping begin and end of the token
            begin_overlap_mask = (bbox_spans[:, 0] <= token.begin) & (token.begin <= bbox_spans[:, 1])
            end_overlap_mask = (bbox_spans[:, 0] <= token.end) & (token.end <= bbox_spans[:, 1])
            # make sure there are overlaps
            assert begin_overlap_mask.any()
            assert end_overlap_mask.any()
            # get overlapping bounding box annotations
            begin_bbox = bbox_annotations[begin_overlap_mask][0]
            end_bbox = bbox_annotations[end_overlap_mask][0]
            # get bounding boxes
            bx, by, bw, bh = begin_bbox.get(self.x_attr), begin_bbox.get(self.y_attr), begin_bbox.get(self.w_attr), begin_bbox.get(self.h_attr)
            ex, ey, ew, eh = end_bbox.get(self.x_attr), end_bbox.get(self.y_attr), end_bbox.get(self.w_attr), end_bbox.get(self.h_attr)
            # make sure bounding boxes are top-left anchored
            bx, bw = (bx, bw) if bw > 0 else (bx+bw, -bw)
            by, bh = (by, bh) if bh > 0 else (by+bh, -bh)
            ex, ew = (ex, ew) if ew > 0 else (ex+ew, -ew)
            ey, eh = (ey, eh) if eh > 0 else (ey+eh, -eh)
            # compute x-offsets of token to begin/end-bbox anntotations by linear interpolation on character positions
            x0_off = (token.begin - begin_bbox.begin) / (begin_bbox.end - begin_bbox.begin) * bw
            x1_off = (token.end - end_bbox.begin) / (end_bbox.end - end_bbox.begin) * ew 
            # add to bounding boxes
            bboxes[i, :] = (
                bx + x0_off,
                by,
                ex + x1_off,
                by + bh
            )

        if self.normalize:
            # find maximum value
            width = bboxes[:, 2].max()
            height = bboxes[:, 3].max()
            # normalize all values
            bboxes[:, 0] *= self.x_scale / width
            bboxes[:, 1] *= self.y_scale / height
            bboxes[:, 2] *= self.x_scale / width
            bboxes[:, 3] *= self.y_scale / height
            # convert to int
            bboxes = bboxes.astype(np.int32)
            assert (bboxes[:, (0, 2)] <= self.x_scale).all()
            assert (bboxes[:, (1, 3)] <= self.y_scale).all()

        return bboxes.tolist()


class BioLabelsFeatureExtractor(FeatureExtractor):

    def __init__(
        self,
        typesystem: cassis.TypeSystem,
        # bio label scheme
        out_tag: str,
        begin_tag_prefix: str,
        in_tag_prefix: str,
        # source entities
        token_type: str,
        entity_type: str,
        entity_attr: str,
        entity_names: list[str]|set[str]
    ) -> None:
        # initialize
        super(BioLabelsFeatureExtractor, self).__init__(
            typesystem, {token_type: [], entity_type: [entity_attr]}
        )

        # run type same type checking as for entities
        EntitiesFeatureExtractor(
            typesystem=typesystem,
            token_type=token_type,
            entity_type=entity_type,
            entity_attr=entity_attr,
            entity_names=entity_names
        )

        self.out_tag = out_tag
        self.begin_tag_prefix = begin_tag_prefix
        self.in_tag_prefix = in_tag_prefix

        self.token_type = token_type
        self.entity_type = entity_type
        self.entity_attr = entity_attr
        self.entity_names = set(entity_names)
        
    @property
    def feature(self):
        return datasets.Sequence(
            datasets.ClassLabel(
                names=[self.out_tag] + [
                    "%s%s" % (prefix, name)
                    for name in self.entity_names
                    for prefix in (
                        self.begin_tag_prefix,
                        self.in_tag_prefix
                    ) 
                ]
            )
        )

    def get_non_overlapping_entities(self, cas: cassis.Cas) -> list:

        entities = filter(
            lambda e: e.get(self.entity_attr) in self.entity_names,
            cas.select(self.entity_type)
        )
        entities = sorted(entities, key=lambda e: e.end - e.begin)

        spans = np.asarray([(e.begin, e.end) for e in entities])
        mask = np.zeros(len(entities), dtype=bool)
        # select non-overlapping entities, prioritize by scoring function
        for i, e in enumerate(entities):
            # find overlaps with already selected entities
            overlap_mask = mask & (
                (e.begin <= spans[:, 0]) & (spans[:, 0] <= e.end) |
                (e.begin <= spans[:, 1]) & (spans[:, 1] <= e.end)
            )

            # handle overlaps
            if overlap_mask.any():
                # if it overlaps with more than one that discard the
                if overlap_mask.sum() >= 2:
                    continue

                # otherwise keep the entity with higher score
                j = int(np.nonzero(overlap_mask)[0])
                other = entities[j]
                # keep the one with the highest score
                if other.score >= e.score:
                    continue
                mask[j] = False
                mask[i] = True
            else:
                # select entity
                mask[i] = True

        # collect all selected entities
        return [e for i, e in enumerate(entities) if mask[i]]

    def extract(self, cas: cassis.Cas) -> list[str]:
        # get tokens in order and create mapping from token to token index
        tokens = sorted(cas.select(self.token_type), key=lambda e: e.begin)
        token2idx = {t:i for i, t in enumerate(tokens)}

        # create initial bio labels
        bio = np.full(len(tokens), fill_value=self.out_tag, dtype=object)

        # get all entity annotations
        for i, e in enumerate(self.get_non_overlapping_entities(cas)):
            # get all tokens that are covered by the entity annotations
            covered = cas.select_covered(self.token_type, e)
            covered_idx = [token2idx[t] for t in covered]
            # double check for overlaps
            assert (bio[covered_idx] == self.out_tag).all()
            assert (bio[covered_idx] == self.out_tag).all()

            if len(covered_idx) > 0:
                bio[covered_idx[0]] = "%s%s" % (self.begin_tag_prefix, e.get(self.entity_attr))
                bio[covered_idx[1:]] = "%s%s" % (self.in_tag_prefix, e.get(self.entity_attr))

        return bio


@dataclass
class CasDatasetConfig(datasets.BuilderConfig):
    name:str = "cas"

    typesystem: str = None

    # split cas into multiple examples of given annotation type
    split_into: Optional[str] = None
    split_filter_attr: Optional[str] = None
    split_filter_space: Optional[list[str]] = None

    # token annotations
    pretokenize: bool = True
    token_type: Optional[str] = None

    # label annotations
    # referse to document-level annotations
    label_type: Optional[str] = None
    label_attr: Optional[str] = None
    label_names: Optional[list[str]|set[str]] = None

    # named entity annotations
    # requires token annotations
    # dataset provides character-spans for entities
    entity_type: Optional[str] = None
    entity_attr: Optional[str] = None
    entity_names: Optional[list[str]|set[str]] = None

    # generate bio labels from entity annotations
    generate_bio_labels: bool = False
    bio_out_tag: str = "O"
    bio_begin_tag_prefix: str = "B-"
    bio_in_tag_prefix: str = "I-"

    # bounding-box annotations
    # requires token annotations
    # dataset provides bounding box for each token
    bbox_type: Optional[str] = None
    bbox_x_attr: Optional[str] = None # x-position
    bbox_y_attr: Optional[str] = None # y-position
    bbox_w_attr: Optional[str] = None # width
    bbox_h_attr: Optional[str] = None # height
    normalize_bbox: Optional[bool] = True
    bbox_x_scale: int = 1000
    bbox_y_scale: int = 1000

    # bi-relation annotations
    relation_type: Optional[str] = None
    relation_source: Optional[str] = None
    relation_target: Optional[str] = None
    relation_names: Optional[list[str]|set[str]] = None

    def __post_init__(self):
        if self.pretokenize and (self.token_type is None):
            raise ValueError("`token_type` is required when `pretokenize=True`")


class WorkerProcess(mp.Process):

    def __init__(
        self,
        worker_id: int,
        config:CasDatasetConfig,
        in_q: mp.Queue,
        out_q: mp.Queue,
        cmd_q: mp.Queue,
        **kwargs
    ) -> None:
        super(WorkerProcess, self).__init__(**kwargs)
        # worker id and config
        self.worker_id = worker_id
        self.config = config
        # save queues
        self.in_q = in_q
        self.out_q = out_q
        self.cmd_q = cmd_q

    @cached_property
    def typesystem(self) -> cassis.TypeSystem:
        # load typesystem from file
        with open(self.config.typesystem, 'rb') as f:
            return cassis.load_typesystem(f)

    @cached_property
    def generator(self) -> CasGenerator:
        return CasGenerator(
            typesystem=self.typesystem,
            split_into=self.config.split_into,
            split_filter_attr=self.config.split_filter_attr,
            split_filter_space=self.config.split_filter_space
        )

    @cached_property
    def pretokenizer(self) -> None|PreTokenizer:
        return PreTokenizer(
            typesystem=self.typesystem,
            token_type=self.config.token_type
        ) if self.config.pretokenize else None

    @cached_property
    def text(self) -> None|TextFeatureExtractor:
        return TextFeatureExtractor(self.typesystem)

    @cached_property
    def tokens(self) -> None|TokensFeatureExtractor:
        return TokensFeatureExtractor(
            typesystem=self.typesystem,
            token_type=self.config.token_type
        ) if self.config.token_type is not None else None

    @cached_property
    def labels(self) -> None|LabelsFeatureExtractor:
        return LabelsFeatureExtractor(
            typesystem=self.typesystem,
            label_type=self.config.label_type,
            label_attr=self.config.label_attr,
            label_names=self.config.label_names
        ) if self.config.label_type is not None else None

    @cached_property
    def entities(self) -> None|EntitiesFeatureExtractor:
        return EntitiesFeatureExtractor(
            typesystem=self.typesystem,
            token_type=self.config.token_type,
            entity_type=self.config.entity_type,
            entity_attr=self.config.entity_attr,
            entity_names=self.config.entity_names
        ) if self.config.entity_type is not None else None

    @cached_property
    def boxes(self) -> None|BoundingBoxFeatureExtractor:
        return BoundingBoxFeatureExtractor(
            typesystem=self.typesystem,
            token_type=self.config.token_type,
            bbox_type=self.config.bbox_type,
            bbox_x_attr=self.config.bbox_x_attr,
            bbox_y_attr=self.config.bbox_y_attr,
            bbox_w_attr=self.config.bbox_w_attr,
            bbox_h_attr=self.config.bbox_h_attr,
            normalize=self.config.normalize_bbox,
            bbox_x_scale=self.config.bbox_x_scale,
            bbox_y_scale=self.config.bbox_y_scale
        ) if self.config.bbox_type is not None else None

    @property
    def bio_labels(self) -> None|BioLabelsFeatureExtractor:
        return BioLabelsFeatureExtractor(
            typesystem=self.typesystem,
            out_tag=self.config.bio_out_tag,
            begin_tag_prefix=self.config.bio_begin_tag_prefix,
            in_tag_prefix=self.config.bio_in_tag_prefix,
            token_type=self.config.token_type,
            entity_type=self.config.entity_type,
            entity_attr=self.config.entity_attr,
            entity_names=self.config.entity_names
        ) if self.config.generate_bio_labels else None

    @property
    def feature_extractors(self) -> dict[str, FeatureExtractor]:
        extractors = {
            "text": self.text,
            "tokens": self.tokens,
            "labels": self.labels,
            "entities": self.entities,
            "boxes": self.boxes,
            "bio": self.bio_labels
        }
        return {k: v for k, v in extractors.items() if v is not None}

    def run(self):

        while True:

            try:
                # get file from input queue
                fpath = self.in_q.get(block=True, timeout=1)

                # generate all cas from fpath
                for cas in self.generator.generate(fpath):

                    # pretokenize cas and extract features
                    cas = self.pretokenizer.tokenize(cas) if (self.pretokenizer is not None) else cas
                    features = {k: e.extract(cas) for k, e in self.feature_extractors.items()}
                    # put features to out queue
                    self.out_q.put(features)

            except queue.Empty:
                # terminate process
                self.cmd_q.put(self.worker_id)
                break

            except Exception as e:
                # log exception and go on
                logger.debug(e, exc_info=True)


class WorkerPool:

    def __init__(self, config: CasDatasetConfig) -> None:
        # save config
        self.config = config
        # create in- and out-queue
        self.in_q = mp.Queue()
        self.out_q = mp.Queue()
        self.cmd_q = mp.Queue()

    def run(self, fpaths: list[str]) -> Iterator[dict[str, Any]]:
        # add all file paths to the in queue
        list(map(self.in_q.put_nowait, fpaths))
        # create all worker processes
        processes = [
            WorkerProcess(
                i, self.config, self.in_q, self.out_q, self.cmd_q,
                name="CasDataset.WorkerProcess.%i" % i, daemon=True
            )
            for i in range(os.cpu_count())
        ]
        # start all processes
        for p in processes:
            p.start()

        # keep going as long as any process is still alive
        while any(map(WorkerProcess.is_alive, processes)):

            try:
                # yield all items from output queue
                yield from iter(partial(self.out_q.get, block=True, timeout=5), None)

            except queue.Empty:
                # add sentinel to command queue
                self.cmd_q.put_nowait(None)
                # join all processes that terminated
                for worker_id in iter(self.cmd_q.get, None):
                    processes[worker_id].join()


class CasDataset(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = CasDatasetConfig

    def _info(self):

        # create a dummy worker process only to check if config is valid
        # note that this process is never started so shouldn't block any cores
        p = WorkerProcess(0, self.config, None, None, None)
        # instantiate all, their init checks the config and throws errors
        p.generator
        p.pretokenizer
        p.feature_extractors

        # all good then return the dataset info
        return datasets.DatasetInfo(
            description="Dataset loaded from CAS",
            features=datasets.Features({
                k: e.feature for k, e in p.feature_extractors.items()
            }),
            supervised_keys=None
        )

    def _split_generators(self, dl_manager):

        # check data files argument
        if self.config.data_files is None:
            raise ValueError(
                "No data files specified. Please specify `data_files` in "
                "call to `datasets.load_dataset`."
            )
        if not isinstance(self.config.data_files, dict):
            raise ValueError(
                "Expected `data_files` to be a dictionary mapping splits "
                "to files, got %s" % type(data_files).__name__
            )

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

    def _generate_examples(self, files:list[Iterator[str]]):
        # create worker pool and execute it
        yield from enumerate(WorkerPool(self.config).run(chain(*files)))
