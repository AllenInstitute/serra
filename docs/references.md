# References

## The method serra implements

The extractor is **multi-label SurfaceNets**, as described in:

> Frisken, S. F. (2022). *SurfaceNets for Multi-Label Segmentations with
> Preservation of Sharp Boundaries*. Journal of Computer Graphics Techniques,
> 11(1), 34–54.
> [PMC9623606](https://pmc.ncbi.nlm.nih.gov/articles/PMC9623606/) ·
> PMID 36325473

Everything characteristic of serra comes from that paper: one vertex per cell
rather than per crossed edge; labels treated as an index set so a single pass
handles any number of them; quads dual to sign-changing edges; sharp boundaries
preserved where three or more labels meet; and a fairing pass that displaces
vertices by no more than a bounded distance from the cell.

Two deliberate departures are worth naming, since they are the reason serra
exists rather than a reimplementation:

- **Fixed-point vertex positions.** Positions are integers in units of 1/256 of
  a voxel, which makes a seam cell's vertex bit-identical in every chunk that
  contains it. Without that, chunk meshes cannot be welded by exact equality.
- **Pinned seam vertices during fairing.** Frisken's relaxation is global;
  serra holds the outermost cell layer fixed so a chunk's mesh depends only on
  that chunk's own array, whatever the iteration count. See
  [Chunked meshing](chunked.md).

## Lineage

SurfaceNets itself, for single-material binary data:

> Gibson, S. F. F. (1998). *Constrained Elastic Surface Nets: Generating Smooth
> Surfaces from Binary Segmented Data*. MICCAI 1998, LNCS 1496, 888–898.

The wider dual-contouring family, and the manifold criterion serra's repair
pass enforces when a cell's surface is split into more than one sheet:

> Ju, T., Losasso, F., Schaefer, S., & Warren, J. (2002). *Dual Contouring of
> Hermite Data*. ACM Transactions on Graphics, 21(3), 339–346.

> Schaefer, S., Ju, T., & Warren, J. (2007). *Manifold Dual Contouring*. IEEE
> Transactions on Visualization and Computer Graphics, 13(3), 610–619.

Marching cubes, for contrast — it is what `zmesh` and most connectomics
pipelines use, and what the comparisons throughout these docs measure against:

> Lorensen, W. E., & Cline, H. E. (1987). *Marching Cubes: A High Resolution 3D
> Surface Construction Algorithm*. Computer Graphics (SIGGRAPH '87), 21(4),
> 163–169.

## Simplification

The quadric error metric, and the topological test that decides whether a
collapse is legal:

> Garland, M., & Heckbert, P. S. (1997). *Surface Simplification Using Quadric
> Error Metrics*. SIGGRAPH '97, 209–216.

> Dey, T. K., Edelsbrunner, H., Guha, S., & Nekhayev, D. V. (1999). *Topology
> Preserving Edge Contraction*. Publications de l'Institut Mathématique,
> 66(80), 23–45.

## Smoothing

serra's own relaxation is the constrained Laplacian fairing of Frisken (2022)
above. The unconstrained alternative, measured against it in
[Accuracy and smoothing](accuracy.md) and by `bench/taubin.py`:

> Taubin, G. (1995). *A Signal Processing Approach to Fair Surface Design*.
> SIGGRAPH '95, 351–358.

## Validation

Surfaces are checked against `zmesh` and against the voxels with a robust
generalized winding number (`bench/validate_winding.py`, via libigl):

> Barill, G., Dickson, N., Schmidt, R., Levin, D. I. W., & Jacobson, A. (2018).
> *Fast Winding Numbers for Soups and Clouds*. ACM Transactions on Graphics,
> 37(4), 43.

## Citing serra

See [`CITATION.cff`](https://github.com/AllenInstitute/serra/blob/main/CITATION.cff)
in the repository root; GitHub renders it as a "Cite this repository" button.
If you are citing the *method* rather than this implementation, cite Frisken
(2022).
