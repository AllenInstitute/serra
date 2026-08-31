"""serra — analytical multi-material meshes from voxelized segmentations.

Turns a 3-D array of integer labels into one triangle mesh per label, in a
single pass over the volume.

    >>> import serra
    >>> mesher = serra.Mesher(voxel_resolution=[4, 4, 40])
    >>> mesher.mesh(cutout)                    # doctest: +SKIP
    >>> mesh = serra.Mesher().get(504)         # doctest: +SKIP
"""

from serra._serra import __version__
from serra.mesh import Mesh
from serra.mesher import Mesher

__all__ = ["Mesher", "Mesh", "__version__"]
