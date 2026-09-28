"""Import-only stubs for the mesh types GVHMR's renderer references.

``tools/demo/demo.py`` imports ``hmr4d.utils.vis.renderer`` at module scope, so
these names must resolve for the module to load at all.  Rendering itself runs
only *after* the predictions are written, and the wild-3D extraction path never
reaches it.

They raise on use rather than returning empty meshes: a renderer that silently
produces nothing would look like a successful run that just happened to have no
output, which is exactly the failure mode the wild-3D pipeline is built to
prevent.
"""

_MESSAGE = (
    "pytorch3d.structures.{name} is not implemented by the AtomicDance compat "
    "shim, which covers only the transforms and knn that GVHMR inference needs. "
    "Install the real pytorch3d to render meshes."
)


class Meshes:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(_MESSAGE.format(name="Meshes"))


class Pointclouds:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(_MESSAGE.format(name="Pointclouds"))


def join_meshes_as_scene(*args, **kwargs):
    raise NotImplementedError(_MESSAGE.format(name="join_meshes_as_scene"))


def join_meshes_as_batch(*args, **kwargs):
    raise NotImplementedError(_MESSAGE.format(name="join_meshes_as_batch"))


from . import meshes  # noqa: E402,F401  (submodule import path used by GVHMR)

__all__ = [
    "Meshes",
    "Pointclouds",
    "join_meshes_as_batch",
    "join_meshes_as_scene",
    "meshes",
]
