import inspect
import datasets
from . import cas
from . import xcr

# for easy of use
load_dataset = datasets.load_dataset

_hash_python_lines = datasets.packaged_modules._hash_python_lines
_PACKAGED_DATASETS_MODULES = datasets.packaged_modules._PACKAGED_DATASETS_MODULES
# register datasets
_PACKAGED_DATASETS_MODULES.update({
    'hyped.datasets.cas': (cas.__name__, _hash_python_lines(inspect.getsource(cas).splitlines())),
    'hyped.datasets.xcr': (xcr.__name__, _hash_python_lines(inspect.getsource(xcr).splitlines()))
})
