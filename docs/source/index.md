# OneSC (One tool to Simulate Cells) 

### Introduction
OneSC is a 2-in-1 computational tool that does two things:
1. Infer a functional transcription factors circuit that describe cell state transitions in the single-cell expression data
2. Simulate synthetic single-cell expression profiles that mimic cell state transitions using the core transcription factors circuit as regulatory backbone 

We provide two ways of running OneSC (both GRN inference and simulations). 

The first way requires users to run all the necessary steps individually leading up to GRN inference or simulation. This allows users to fine-tune parameters and inspect any results generated during the intermediate steps. 

The second way contains wrapper functions that work with popular single-cell data structure,[AnnData](https://anndata.readthedocs.io/en/latest/index.html#). Different from the first method, many of the intermediate steps are now included in the wrapper functions. This method is also compatible with single-cell classification tool [pySCN](https://github.com/CahanLab/PySingleCellNet) for quick classification of the simulated cells. It is more convenient to use, but does not offer the flexiblity for users to fine tune any of the intermediate steps. 

```{toctree}
:maxdepth: 1
:caption: User guide
Installation <installation.md>
Method 1: A. Infer functional circuit <infer_grn.md>
Method 1: B. Simulation of Synthetic Cells <simulation_syn_cells.md>
Method 2: A. Infer functional circuit - Scanpy object <infer_grn_scanpy.md>
Method 2: B. Simulation of Synthetic Cells - Scanpy object <infer_grn_scanpy.md>
```