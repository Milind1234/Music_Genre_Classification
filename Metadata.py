"""
Backward-compatibility bridge.
Delegates to predictor.Metadata so legacy notebooks and scripts
calling `import Metadata` or `from Metadata import getmetadata` work seamlessly.
"""

from predictor.Metadata import *
from predictor.Metadata import getmetadata
