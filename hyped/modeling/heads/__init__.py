from .cls import HypedClsHead, HypedClsHeadConfig
from .tagging import HypedTaggingHead, HypedTaggingHeadConfig

AnyHypedHeadConfig = \
    HypedClsHeadConfig | \
    HypedTaggingHeadConfig
