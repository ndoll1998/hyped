import os

import datasets

# for easy of use
load_dataset = datasets.load_dataset

_hash_python_lines = datasets.packaged_modules._hash_python_lines
_PACKAGED_DATASETS_MODULES = (
    datasets.packaged_modules._PACKAGED_DATASETS_MODULES
)


def _register_from_file_name(dataset_id: str, file_name: str) -> None:
    # build full path to file
    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, file_name)
    # make sure the file is valid
    assert os.path.isfile(file_path)
    assert file_path.endswith(".py")
    # get only the file name
    file_name = os.path.basename(file_path)
    file_name, _ = os.path.splitext(file_name)

    with open(file_path, "r") as f:
        # register the dataset generator file
        _PACKAGED_DATASETS_MODULES[dataset_id] = (
            dataset_id,
            _hash_python_lines(f.readlines()),
        )


# register datasets
_register_from_file_name("hyped.data.io.datasets.cas", "cas.py")
_register_from_file_name("hyped.data.io.datasets.typed_json", "typed_json.py")
