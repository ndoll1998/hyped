import torch
from transformers import PreTrainedTokenizer
from transformers.data import DefaultDataCollator
from transformers.adapters.heads import (
    PredictionHead,
    TaggingHead,
    MultiLabelClassificationHead,
    CausalLMHead
)
from datasets.features import Features, Sequence
# utils
from abc import ABC, abstractmethod
from typing import Any, Mapping
from itertools import chain
from functools import partial, cmp_to_key
from hyped.utils.typedmapping import typedmapping

def is_sized(f:object) -> bool:

    # recursive check for dict
    if isinstance(f, (Features, dict)):
        return all(map(is_sized, f.values()))

    # recursive check for sequences, note that the sequence length
    # is the only breaking point
    if isinstance(f, Sequence):
        return (f.length >= 0) and is_sized(f.feature)

    # all others are valid
    return True

class LabelsCollator(ABC):

    def __init__(
        self,
        head:PredictionHead,
        tokenizer:PreTrainedTokenizer,
        features:Features,
        return_tensors:str ='pt'
    ) -> None:
        self.head = head
        self.tokenizer = tokenizer
        self.features = features
        self.return_tensors = return_tensors

        # make sure all labels are present in features
        if any(n not in self.features for n in self.head.get_label_names()):
            raise ValueError("Not all labels present in data, required %s but got %s" % (
                list(self.head.get_label_names()), list(self.features.keys())
            ))

    @abstractmethod
    def __call__(self, features:list[dict[str, Any]], enc:dict[str, Any]) -> dict[str, Any]:
        ...

class DefaultLabelsCollator(DefaultDataCollator, LabelsCollator):

    def __init__(
        self,
        head:list[PredictionHead],
        tokenizer:PreTrainedTokenizer,
        features:Features,
        return_tensors:str ='pt'
    ) -> None:
        # initialize base classes
        DefaultDataCollator.__init__(self, return_tensors=return_tensors)
        LabelsCollator.__init__(
            self,
            head=head,
            tokenizer=tokenizer,
            features=features,
            return_tensors=return_tensors
        )
        # check features
        if not is_sized(features):
            raise ValueError("Cannot apply default label collator to non-sized features %s" % (
                list(self.features.keys())
            ))

    def __call__(self, features:list[dict[str, Any]], enc:dict[str, Any]) -> dict[str, Any]:
        return DefaultDataCollator.__call__(self, features)

class TaggingLabelsCollator(LabelsCollator):

    def __init__(
        self,
        head:list[PredictionHead],
        tokenizer:PreTrainedTokenizer,
        features:Features,
        return_tensors:str ='pt'
    ) -> None:
        # initialize collator
        super(TaggingLabelsCollator, self).__init__(
            head=head,
            tokenizer=tokenizer,
            features=features,
            return_tensors=return_tensors
        )
        # main model input feature, typically input_ids
        self.input_feature_name = tokenizer.model_input_names[0]

        if return_tensors != 'pt':
            raise NotImplementedError("Only pytorch tensors supported yet")

    def build_attn_mask(self, enc):
        # get attention mask from encoding if present
        if 'attention_mask' in enc:
            return enc['attention_mask'].bool()
        # build attention mask from input ids
        assert self.input_feature_name in enc
        return enc[self.input_feature_name] != self.tokenizer.pad_token_id

    def pad(self, labels:list[list[int]], mask:torch.Tensor) -> torch.Tensor:
        assert mask.sum(dim=-1).tolist() == list(map(len, labels))
        # create batch tensor filled with padding id and write values
        batch = torch.full_like(mask, fill_value=-100, dtype=torch.long)
        batch[mask] = torch.tensor(list(chain(*labels)))
        # return batch
        return batch

    def __call__(self, features:list[dict[str, Any]], enc:dict[str, Any]) -> dict[str, Any]:
        # get attention mask
        mask = self.build_attn_mask(enc)
        # pad label features to match sequence length
        return {
            n: self.pad([f[n] for f in features], mask)
            for n in self.head.get_label_names()
        }

class MultiLabelsCollator(LabelsCollator):

    def __init__(
        self,
        head:list[PredictionHead],
        tokenizer:PreTrainedTokenizer,
        features:Features,
        return_tensors:str ='pt'
    ) -> None:
        # initialize collator
        super(MultiLabelsCollator, self).__init__(
            head=head,
            tokenizer=tokenizer,
            features=features,
            return_tensors=return_tensors
        )

        # get the label space size
        assert 'num_labels' in self.head.config
        self.num_labels = self.head.config['num_labels']

        if return_tensors != 'pt':
            raise NotImplementedError("Only pytorch tensors supported yet")

    def binarize(self, labels:list[list[int]]) -> torch.Tensor:

        bin_labels = torch.zeros((len(labels), self.num_labels), dtype=torch.long)
        # binarize labels
        for i, ids in enumerate(labels):
            bin_labels[i, ids] = 1

        return bin_labels

    def __call__(self, features:list[dict[str, Any]], enc:dict[str, Any]) -> dict[str, Any]:
        return {
            n: self.binarize([f[n] for f in features])
            for n in self.head.get_label_names()
        }

class CausalLMLabelsCollator(TaggingLabelsCollator):

    def __init__(
        self,
        head:list[PredictionHead],
        tokenizer:PreTrainedTokenizer,
        features:Features,
        return_tensors:str ='pt'
    ) -> None:
        # initialize collator
        super(CausalLMLabelsCollator, self).__init__(
            head=head,
            tokenizer=tokenizer,
            features=features,
            return_tensors=return_tensors
        )
        # get label name
        assert len(self.head.get_label_names()) == 1
        self.label_name, = self.head.get_label_names()

    def __call__(self, features:list[dict[str, Any]], enc:dict[str, Any]) -> dict[str, Any]:
        # check if the labels are already present in the encoding
        # then it is also already present in the collation output
        return {} if self.label_name in enc else {
            self.label_name: self.pad(
                [f[self.label_name] for f in features],
                self.build_attn_mask(enc)
            )
        }

class HypedDataCollator(object):

    HEAD_COLLATOR_MAPPING = typedmapping[
        type[PredictionHead],
        type[LabelsCollator]
    ]()

    def __init__(
        self,
        tokenizer:PreTrainedTokenizer,
        heads:list[PredictionHead],
        features:Features,
        return_tensors:str ='pt'
    ):
        # save heads
        self.heads = list(heads)

        # set of all label feature names that are not input features
        label_feature_names = {n for h in heads for n in h.get_label_names()}
        # separate features of different type processed by different collators
        self.enc_features = {n: features[n] for n in tokenizer.model_input_names if n in features}
        self.lbl_features = {n: features[n] for n in label_feature_names if n in features}
        self.other_features = {
            n: f for n, f in features.items() if \
            n not in self.enc_features.keys() and \
            n not in self.lbl_features.keys()
        }

        # can't collate other features as they are not sized
        #if not is_sized(self.other_features):
        #    raise ValueError()
        # use default collators for sized features
        self.others_collator = lambda x: {} # DefaultDataCollator(return_tensors)

        # use the tokenizer collator if the encoding features
        # are not sized and the default collator if they are
        self.enc_collator = DefaultDataCollator(return_tensors) if is_sized(self.enc_features) else \
            partial(
                tokenizer.pad,
                return_attention_mask=True,
                return_tensors='pt'
            )

        # set padding token if not set properly
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                # set padding token to end-of-sequence
                tokenizer.pad_token = tokenizer.eos_token
            else:
                raise NotImplementedError


        self.lbls_collators = []
        # build all label collators
        for head in heads:
            # get the label names
            label_names = tuple(head.get_label_names())

            # check if there are label conflicts, i.e. is
            # any label used by multiple heads
            for c in self.lbls_collators:
                for n in c.features.keys():
                    if n in label_names:
                        raise ValueError("Detected label conflict, label `%s` is used by multiple heads." % n)

            if all(n not in features for n in label_names):
                # no labels for this head found, so there is no
                # need for a collator
                break

            if any(n not in features for n in label_names):
                # not all labels present
                raise ValueError()

            # find the correct collator type for the head
            # sort to make sure to get the closest parent head type
            key = cmp_to_key(lambda t, v: 2 * issubclass(v, t) - 1)
            for head_t in sorted(type(self).HEAD_COLLATOR_MAPPING, key=key):
                if isinstance(head, head_t):
                    # create the collator
                    collator_t = type(self).HEAD_COLLATOR_MAPPING[head_t]
                    collator = collator_t(
                        head=head,
                        tokenizer=tokenizer,
                        features=Features({n:f for n,f in features.items() if n in label_names}),
                        return_tensors=return_tensors
                    )
                    # use the collator to process the labels
                    self.lbls_collators.append(collator)
                    break
            else:
                raise ValueError("No collator found for labels of head with type `%s`" % type(head))

        # all label features should be processed by some label collator
        assert set(self.lbl_features.keys()) == {n for c in self.lbls_collators for n in c.features.keys()}


    def __call__(self, features:list[dict[str, Any]]) -> dict[str, Any]:
        # convert features to mapping
        if not isinstance(features[0], Mapping):
            features = [vars(f) for f in features]
        # separate encoding and other features, note that these must all
        # be present as we checked the feature names with the dataset features
        # in the constructor
        encs = [{n: f[n] for n in self.enc_features.keys()} for f in features]
        others = [{n: f[n] for n in self.other_features.keys()} for f in features]
        # collate features
        encs = self.enc_collator(encs)
        others = self.others_collator(others)
        # collate label features
        lbls = {}
        for collator in self.lbls_collators:
            lbl_features = [{n: f.get(n) for n in collator.features.keys()} for f in features]
            lbl_features = collator(lbl_features, encs)
            lbls.update(lbl_features)
        # merge all collated features
        return encs | lbls | others

    @classmethod
    def register_head_collator(
        cls,
        head_t:type[PredictionHead],
        collator_t:type[LabelsCollator]
    ) -> None:
        cls.HEAD_COLLATOR_MAPPING[head_t] = collator_t

# register data collators
HypedDataCollator.register_head_collator(PredictionHead, DefaultLabelsCollator)
HypedDataCollator.register_head_collator(TaggingHead, TaggingLabelsCollator)
HypedDataCollator.register_head_collator(MultiLabelClassificationHead, MultiLabelsCollator)
HypedDataCollator.register_head_collator(CausalLMHead, CausalLMLabelsCollator)

