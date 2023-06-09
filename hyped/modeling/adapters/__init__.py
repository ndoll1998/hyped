# check if adapter-transformers is installed
try:
    import transformers.adapters
except ImportError:
    raise ModuleNotFoundError("adapter-transformers backend requires adapter-transformers to be installed.")

from . import heads, auto
from .wrapper import HypedAdapterModelWrapper
