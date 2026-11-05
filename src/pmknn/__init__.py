"""PM-kNN: portable-metric kNN (formerly ceknn).

Each input is carried to its shape function g_j, the consensus of how its values move the prediction across a
Rashomon set of competitive models; each counts with a weight w_j learned by a diagonal NCA; the prediction is the
vote of the k nearest rows of a case base. The metric (g_j, w_j) can be exported to JSON and used at another site
with that site's own case base.
"""

__version__ = "0.4.0"

from .core import PMKNN, EncodingSettings, KNNSettings, LearnSettings, NCASettings  # noqa: E402
from .models import ModelSpace  # noqa: E402
from .rashomon import ShapeSettings  # noqa: E402
from .schema import Variable  # noqa: E402

__all__ = ["PMKNN", "ModelSpace", "Variable", "EncodingSettings", "ShapeSettings", "NCASettings", "KNNSettings",
           "LearnSettings", "__version__"]
