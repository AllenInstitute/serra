"""serra — analytical multi-material meshes from voxelized segmentations.

Turns a 3-D array of integer labels into one triangle mesh per label, in a
single pass over the volume.

    >>> import serra_mesh
    >>> mesher = serra_mesh.Mesher(voxel_resolution=[4, 4, 40])
    >>> mesher.mesh(cutout)                    # doctest: +SKIP
    >>> mesh = serra_mesh.Mesher().get(504)         # doctest: +SKIP
"""

from serra_mesh._serra_mesh import __version__
from serra_mesh.dice import dice
from serra_mesh.mesh import Mesh
from serra_mesh.mesher import Mesher
from serra_mesh.stitch import stitch

__all__ = ["Mesher", "Mesh", "dice", "stitch", "__version__"]
