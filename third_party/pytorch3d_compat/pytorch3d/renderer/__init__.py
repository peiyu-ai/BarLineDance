"""Import-only stubs for the renderer symbols GVHMR references.

Same rationale as ``pytorch3d.structures``: importable so that GVHMR's vis
module loads, raising on use so a stubbed renderer can never be mistaken for a
real one.  ``hmr4d/utils/vis/renderer.py`` imports exactly the names below.
"""


def _unimplemented(name):
    """Build a stub that imports fine and refuses to pretend it can render."""

    message = (
        "pytorch3d.renderer.{name} is not implemented by the AtomicDance compat "
        "shim, which covers only the transforms and knn that GVHMR inference "
        "needs. Install the real pytorch3d to render meshes.".format(name=name)
    )

    class _Stub:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError(message)

    _Stub.__name__ = name
    _Stub.__qualname__ = name
    return _Stub


PerspectiveCameras = _unimplemented("PerspectiveCameras")
TexturesVertex = _unimplemented("TexturesVertex")
PointLights = _unimplemented("PointLights")
Materials = _unimplemented("Materials")
RasterizationSettings = _unimplemented("RasterizationSettings")
MeshRenderer = _unimplemented("MeshRenderer")
MeshRasterizer = _unimplemented("MeshRasterizer")
SoftPhongShader = _unimplemented("SoftPhongShader")
FoVPerspectiveCameras = _unimplemented("FoVPerspectiveCameras")
HardPhongShader = _unimplemented("HardPhongShader")
BlendParams = _unimplemented("BlendParams")

from . import cameras  # noqa: E402,F401
from .cameras import look_at_rotation  # noqa: E402,F401

__all__ = [
    "BlendParams",
    "FoVPerspectiveCameras",
    "HardPhongShader",
    "Materials",
    "MeshRasterizer",
    "MeshRenderer",
    "PerspectiveCameras",
    "PointLights",
    "RasterizationSettings",
    "SoftPhongShader",
    "TexturesVertex",
    "cameras",
    "look_at_rotation",
]
