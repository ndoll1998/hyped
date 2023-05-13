from __future__ import annotations
from tqdm.auto import tqdm
from torch import Tensor
from torch.utils.data import Dataset, TensorDataset, default_collate
from dataclasses import dataclass

class NamedTensorDataset(TensorDataset):
    """ Dataset similar to `TensorDataset` but returns takes named tensors
        as keyword arguments and yields dictionaries instead of tuples.
    """

    def __init__(self, **named_tensors) -> None:
        self.names, tensors = zip(*named_tensors.items())
        super(NamedTensorDataset, self).__init__(*tensors)
    def __getitem__(self, idx) -> dict:
        tensors = super(NamedTensorDataset, self).__getitem__(idx)
        return dict(zip(self.names, tensors))

    @staticmethod
    def from_dataset(dataset:Dataset) -> NamedTensorDataset:
        # convert dataset to list of dicts
        data = [item for item in tqdm(dataset, desc="Converting")]
        assert all((isinstance(item, dict) for item in dataset)), "Dataset items should be dictionaries"
        # collate dataset items to tensors
        data = default_collate(data)
        assert isinstance(data, dict)
        assert all((isinstance(t, Tensor) for t in data.values()))
        # convert to named tensor dataset
        return NamedTensorDataset(**data)

@dataclass
class DataDump(object):
    name:str
    features:datasets.Features
    datasets:dict[str,Dataset]
