from .utils import PoissonGLMPCA
import numpy as np

__version__ = "0.0.1"

def poisson(
    X,
    n_pcs: int = 30,
    max_iter: int = 1000,
    tol: float = 1e-4,
    col_size_factor: bool = True,
    row_intercept: bool = True,
    verbose: bool = False,
    device: str | None = None,
    progress_bar: bool = True,
    seed: int | None = 42,
    return_model: bool = False,
) -> np.ndarray | PoissonGLMPCA:
    """
    Fit a Poisson GLM-PCA model to the input data.

    Parameters
    ----------
    X : array-like, torch.Tensor, or scipy.sparse matrix
        Input data matrix of shape (n_samples, n_features).
    n_pcs : int, optional
        Number of principal components to compute. Default is 30.
    max_iter : int, optional
        Maximum number of iterations for the optimization algorithm. Default is 1000.
    tol : float, optional
        Tolerance for convergence of the optimization algorithm. Default is 1e-4.
    col_size_factor : bool, optional
        Whether to use column size factor in the model. Default is True.
    row_intercept : bool, optional
        Whether to use row intercept in the model. Default is True.
    verbose : bool, optional
        Whether to print verbose output during fitting. Default is False.
    device : str or None, optional
        Device to use for computation. If None, uses "cuda" if available, otherwise "mps" if available,
        otherwise "cpu". Default is None.
    progress_bar : bool, optional
        Whether to display a progress bar during fitting. Default is True.
    seed : int or None, optional
        Random seed for reproducibility. Default is 42.
    return_model : bool, optional
        Whether to return the fitted model. If False, returns the principal components (U). Default is True.

    Returns
    -------
    np.ndarray or PoissonGLMPCA
        If return_model is True, returns the fitted Poisson GLM-PCA model.
        If return_model is False, returns the principal components (U) as a numpy array.
    """

    model = PoissonGLMPCA(
        n_pcs=n_pcs,
        max_iter=max_iter,
        tol=tol,
        col_size_factor=col_size_factor,
        row_intercept=row_intercept,
        verbose=verbose,
        device=device,
        progress_bar=progress_bar,
        seed=seed,
    )
    model.fit(X)
    if return_model:
        return model
    else:
        return model.U