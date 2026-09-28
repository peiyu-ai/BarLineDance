"""``pytorch3d.structures.meshes`` submodule path used by GVHMR's renderer.

GVHMR imports ``from pytorch3d.structures.meshes import join_meshes_as_scene``,
so the name has to exist here as well as on the package.
"""

from . import (  # noqa: F401
    Meshes,
    Pointclouds,
    join_meshes_as_batch,
    join_meshes_as_scene,
)

__all__ = ["Meshes", "Pointclouds", "join_meshes_as_batch", "join_meshes_as_scene"]
