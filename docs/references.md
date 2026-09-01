# References

## The method serra implements

The extractor is **multi-label SurfaceNets**, as described in:

> Frisken, S. F. (2022). *SurfaceNets for Multi-Label Segmentations with
> Preservation of Sharp Boundaries*. Journal of Computer Graphics Techniques,
> 11(1), 34–54.
> [PMC9623606](https://pmc.ncbi.nlm.nih.gov/articles/PMC9623606/) ·
> PMID 36325473

What serra takes from that paper:

- **One vertex per cell**, rather than one per crossed edge as in marching
  cubes, with quads dual to the edges whose two ends carry different labels.
- **Labels as an index set.** An edge crosses when its endpoints differ, not
  when an isovalue is crossed, so a single pass covers any number of labels and
  adjacent labels share the cell's vertex position exactly
  (`crossing_mask` and `cell_vertex` in `src/place.rs`).
- **Centroid placement.** The vertex sits at the centroid of the crossed edges'
  midpoints — a pure function of the 12-bit crossing pattern, which is why it
  is a table lookup (`build_centroid` in `src/tables.rs`).
- **Fairing with a bounded displacement**, so smoothing cannot drift the
  surface arbitrarily far from the data (`Relaxation::max_deviation`).

Three places serra differs, which matter if you are comparing against the
paper:

- **No sharp-boundary preservation.** The paper's title contribution is
  splitting a cell's vertex into *surface* and *edge* vertices where three or
  more materials meet, so triple junctions stay sharp. serra does not do this:
  `cell_vertex` sees only the crossing mask and has no notion of how many
  materials are present. Its vertex splitting is a different mechanism, for a
  different purpose — keeping each label's own surface 2-manifold where its
  voxel set touches itself only diagonally.
- **Fairing is per object, not per cell.** Frisken relaxes one position per
  cell, shared by every label there. serra relaxes each label's mesh
  independently, so a wall between two touching objects drifts apart. Now
  measured: up to 2.2 voxels, 71 nm, on real neuropil, where 88.6% of vertex
  positions are shared between labels. See [Related work](related-work.md).
- **Fixed-point vertex positions**, integers in units of 1/256 of a voxel, and
  **pinned seam vertices during fairing**. Together these make a seam cell's
  vertex bit-identical in every chunk that contains it and keep a chunk's mesh
  reproducible from that chunk's own array, whatever the iteration count —
  which is what lets chunks be welded by exact equality. See
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

## Mesh conditioning

Not implemented here, but measured against and discussed in
[Related work](related-work.md):

> Yu, Z., Holst, M. J., Cheng, Y., & McCammon, J. A. (2008).
> *Feature-preserving adaptive mesh generation for molecular shape modeling and
> simulation*. Journal of Molecular Graphics and Modelling, 26(8), 1370–1380.

> Lee, C. T., Laughlin, J. G., Angliviel de La Beaumelle, N., Amaro, R. E.,
> McCammon, J. A., Ramamoorthi, R., Holst, M., & Rangamani, P. (2020). *3D mesh
> processing using GAMer 2 to enable reaction-diffusion simulations in realistic
> cellular geometries*. PLoS Computational Biology, 16(4), e1007756.

The local structure tensor from those papers is the one technique that
transfers; the rest of the toolchain repairs marching-cubes slivers that label
data does not produce.

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
