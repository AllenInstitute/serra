## purpose of repo

This repo is a libray for making highly performant, analytical meshes from highly multi-material voxelized segmentations. 

It is written in rust, with python bindings that accept 3d numpy arrays of integer labels, and produces triangular meshes with vertices and faces (indices into the vertices array) as numpy arrays.  

It is designed to be highly performant for extracting all of the meshes of all the unique integers lablled within the 3d array. 

It does this by utilizing a multi-material implementation of a meshing approach where it only loops a single time over the 3d dataset, as there will be hundreds of thousands of potential objects within a chunk of the dataset. 

It is designed to be highly efficient in it's memory management and not use high multiples of the original data size and final result.  

It is capable of also producing efficient results on single material segmentations (such as binary masks of single objects).

It is designed to be run across chunks of data and ensures that when run on arrays with single voxel overlap between those arrays that the resulting vertices at the seams between the arrays align perfectly, so that a vertex deduplication based stitching algorithm can create a graph which is accurate and spans the chunks.  This will allow it to be part of a pipeline that meshes very large arrays by iterating over chunks. 

It is designed to use the best of modern computer vision approaches to produce meshes which are highly accurate to the surface being reconstructed by the voxel segmentation, and that the derivatives on the mesh are accurate and analytically smooth.  

It should produce meshes that are watertight within the volume, but can be open at the edge of the volume if the object extends there.  This is related to stitching in so far as the stitched result should be watertight once the mesh is stitched across the chunk.  This should be explicitly tested in the test suite.  For example, a tube which crosses two chunk boundaries should be watertight after both chunks are meshes and stitched. 

The mesh should not produce any non-manifold vertices or edges. 

The mesh should have tight distributions of triangle areas. 

The api should allow the user to specify the orientation and axis order of each of it's arrays axis, to support XYZ, ZYX and Y going up with index or down, following the various conventions that exist. 

You should research and consult multiple computer graphics literature searches and consult existing libraries for implementation advice about how to do any particular task.  However, we would like to keep the dependancy chain of this code base relatively light and try to limit dependancies to core and well maintained rust libraries, and not import code that is not well maintained.  We can re-use code from open source libraries if their licenses allow it.

The library should have facilities for producing meshes of varying resolutions, and downsampling meshes from their current state to lower counts of vertices and faces.  This will be used to make multi-resolution versions of meshes.  The downsampling procedures should make it possible to export map of the lower resolution mesh vertices to the higher resolution mesh vertices in order to analytically map the meshes between levels. 

## api

Note: the PyPI distribution is `serra-mesh` and the module is `serra_mesh`.
The name `serra` was already taken on PyPI by an unrelated package that ships
its own top-level `serra` module, so sharing the import name risked one
install silently clobbering the other.

The api should be something like

```
import serra_mesh

# assume i have a X,Y,Z array called cutout

mesher=serra_mesh.Mesher(voxel_resolution=[4,4,40])
mesher.mesh(cutout)
mesh=mesher.get(504)
# mesh.vertices and mesh.faces contain the numpy arrays of object 504
```

## repository
The repository is setup with github actions to run tests, which should happen automatically on PR opening.
The repository has an manual action to automatically bump the version of the repository (major, minor, patch)
and then repackage all the code and release it via pypi. 
The repository uses modern python and rust packaging toolchains to facilitating compiling the rust library and it's python bindings. 

## validation and benchmarking
In order to validate and benchmark meshing it should compare an contrast with zmesh. 
zmesh has the right api, but we want this library to be competitive in terms of performance with zmesh
To test this we should download exmaple chunks of data from the precomputed microns volume
cloud path = gs://iarpa_microns/minnie/minnie65/seg_m1300
using cloud-volume at 32 x 32 x 40 nm resolution at a chunk size of 512 x 512 x 512. 
you can use the cloud-volume libary to access and download those chunks in python and then save artifacts in the repository for later testing and direct access in whatever format you think makes sense to load into rust and/or python. 

You should make sure that the resulting code runs with competitive runtimes to zmesh and memory requirements.
The meshes should be similar in the sense of volumetric overlap, but we want our meshes to be watertight and nicer analytically. 

To test this you should use a robust winding number approach to measure whether points in the volume are inside or outside both meshes.  The answer to this should produce pretty similiar (within 5%) volume sizes and descrepancies should be higher closer to the points are to the mesh boundaries of zmesh.  If there are artifacts or mistakes in our meshing procedure they should produce wildly varying volumetric numbers, and points which don't agree whether they are inside or outside the object should be far from the zmesh mesh boundaries. 

Also, there should be a series of tests on meshing objects of known geomteric represetnations, tubes, spheres, etc, where the analytical calculation of the volume, area and normal directions should be known, and the empirically measured volume, area and normal directions should be close to the analytical values. 

## repo organization
the repository should be logically organized in appropriate modules and the code well commented and documented, starting at a high level for users of the library, with more detailed developer guides for how the code is organized.

## working on this code

`cargo fmt` and `ruff format` both run here, and both reflow code — collapsing a
multi-line `use` or condition onto one line, or splitting a call across lines.
That has repeatedly broken scripted edits: a `str.replace` whose pattern no
longer matches writes the file back unchanged and reports success.

So: **never patch a file with a string replace that can silently match nothing.**
Assert the pattern was found, and assert the count when several sites must
change. After running a formatter, re-read any file before matching against it.

The worst case is not a compile error. When `owns_quad`'s ownership gate was
patched at one of two call sites, the result still compiled and still produced
plausible meshes — it just produced a different number of faces depending on the
thread count, which only the determinism test caught.
