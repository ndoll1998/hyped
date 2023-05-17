
from .heads.base import HypedPredictionHead

from transformers.data import default_data_collator

import warnings
from datasets.features import Features
from typing import Any, Mapping

class HypedDataCollator(object):

    def __init__(
        self,
        heads:list[HypedPredictionHead],
        features:Features,
        return_tensors:str ='pt'
    ):

        self.heads = list(heads)
        self.features = features
        self.return_tensors = return_tensors

    @property
    def label_columns(self) -> set[str]:
        return set([h.label_column for h in self.heads])

    def __call__(self, features:list[dict[str, Any]]) -> dict[str, Any]:
        # convert features to mapping
        if not isinstance(features[0], Mapping):
            features = [vars(f) for f in features]

        first = features[0]
        # separate label columns from remaining
        label_batch = {
            n: [item.pop(n) for item in features]
            for n in self.label_columns if n in first
        }

        # collate all remaining features
        # TODO: implement input padding here
        batch = default_data_collator(features, self.return_tensors)

        # collate labels of all heads
        for h in self.heads:
            if h.label_column in label_batch:

                if h.label_column in batch:
                    warnings.warn("Label column `%s` already present in batch. This is likely due to multiple heads using the same label column.", UserWarning)
                    continue
                # collate labels for head
                batch[h.label_column] = h.collate_labels(
                    labels=label_batch[h.label_column],
                    return_tensors=self.return_tensors
                )

        return batch

