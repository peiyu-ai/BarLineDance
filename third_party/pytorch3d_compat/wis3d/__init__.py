"""Import-only ``wis3d`` stand-in, same contract as the pytorch3d shim.

GVHMR's extraction import chain reaches ``wis3d`` through
``cv2_utils -> wis3d_utils`` (module-scope ``from wis3d import Wis3D``), but
``Wis3D`` is only *instantiated* by debug visualization helpers the extraction
driver never calls.  The real wis3d drags in a web-server dependency chain
(cherrypy and friends) for a capability this pipeline must never rely on, so
this stub satisfies the import and raises on use instead of silently
pretending a debug visualizer exists.
"""

__version__ = "0.0.0+atomicdance-compat"


class Wis3D:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "wis3d is stubbed by third_party/pytorch3d_compat: the extraction "
            "pipeline never visualizes; install the real wis3d if you truly "
            "need its debug viewer"
        )
