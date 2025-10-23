# py-fastglmpca

![Tests](https://github.com/serjisa/fastglmpca/actions/workflows/tests.yml/badge.svg)
![PyPI](https://img.shields.io/pypi/v/fastglmpca)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Python implementation of `fastglmpca` ([Weine et al., Bioinformatics, 2024](https://doi.org/10.1093/bioinformatics/btae494)) algorithm with PyTorch backend.

The main concept of `fastglmpca` is to use a fast iterative algorithm ("Alternative Poisson Regression") to find a low-rank approximation of the input matrix `X` with a Poisson distribution. It might be used for dimensionality reduction of count data matrices (e.g. scRNA-Seq UMI matrices or nearest neighbours count matrices in Skip-Gram like representations).

The original R package is available at [GitHub](https://github.com/stephenslab/fastglmpca), this Python package is **not** an official implementation that was tested in the [paper](https://doi.org/10.1093/bioinformatics/btae494). In contrast to the original implementation, we don't use line search and instead use adaptive learning rate with backtracking.

## Installation

`fastglmpca` might be installed via `pip`:
```bash
pip install fastglmpca
```
or the latest development version can be installed from GitHub using:
```bash
pip install git+https://github.com/serjisa/py-fastglmpca
```

## Quck start

`fastglmpca` works with both sparse and dense matrices. The input matrix `X` should be a 2D array-like object with shape `(n_samples, n_features)`. The output matrix `Z` will have shape `(n_samples, n_components)`, where `n_components` is the number of components to be computed.

```python
import fastglmpca

# Fitting the model
model = fastglmpca.poisson(X, n_pcs=10, return_model=True)
X_PoiPCA = model.U
# Alternatively, you can run
# X_PoiPCA = fastglmpca.poisson(X, n_pcs=10)

# Fitting new data to existing model
Y_PoiPCA = model.project(Y)
```

Examples with scRNA-Seq dataset processing are available in [this](https://github.com/serjisa/fastglmpca/blob/main/examples/scRNA-Seq.ipynb) and [this](https://github.com/serjisa/py-fastglmpca/blob/main/examples/Coordinates_projection.ipynb) notebooks.

## API

Function `fastglmpca.poisson` has the following parameters:

- `X` : np.ndarray or torch.Tensor or scipy.sparse matrix
    Input data matrix of shape `(n_samples, n_features)`.
- `n_pcs` : int, optional
    Number of principal components to compute. Default is 30.
- `max_iter` : int, optional
    Maximum number of iterations for the optimization algorithm. Default is 1000.
- `tol` : float, optional
    Tolerance for convergence of the optimization algorithm. Default is 1e-4.
- `col_size_factor` : bool, optional
    Whether to use column size factor in the model. Default is True.
- `row_intercept` : bool, optional
    Whether to use row intercept in the model. Default is True.
- `verbose` : bool, optional
    Whether to print verbose output during fitting. Default is False.
- `device` : str or None, optional
    Device to use for computation. If None, uses "cuda" if available, otherwise "mps" if available,
    otherwise "cpu". Default is None.
- `progress_bar` : bool, optional
    Whether to show a progress bar during fitting. Default is True.
- `seed` : int or None, optional
    Random seed for reproducibility. Default is 42.
- `return_model` : bool, optional
    Whether to return the fitted model object. Default is False.
- `learning_rate` : float, optional
    Step size used in updates. Default is 0.5.
- `num_ccd_iter` : int, optional
    Number of cyclic coordinate descent iterations per main iteration to refine factors. Default is 3.
- `batch_size_rows` : int or None, optional
    Number of rows for batched computations of expectation terms; tunes memory vs speed. Default uses an adaptive value up to 1024.
- `batch_size_cols` : int or None, optional
    Number of columns for batched computations of expectation terms; tunes memory vs speed. Default uses an adaptive value up to 1024.
- `init` : str, optional
    Initialization method for factor matrices. `'svd'` (default) uses SVD on `log1p(X)` to produce a strong starting point. `'random'` uses small Gaussian noise for LL and FF which can be useful for stress-testing convergence or avoiding SVD costs on extremely large inputs.
- `adaptive_lr` : bool, optional
    Whether to use adaptive learning rate with backtracking. Default is True.
- `lr_decay` : float, optional
    Decay factor for learning rate. Default is 0.5.
- `min_learning_rate` : float, optional
    Minimum learning rate. Default is 1e-5.
- `max_backtracks` : int, optional
    Maximum number of backtracks for line search. Default is 3.
