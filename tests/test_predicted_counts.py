import numpy as np
import scipy.sparse as sp
import torch

from fastglmpca.utils import PoissonGLMPCA
import pytest


def _synthetic_poisson_data(n=40, m=30, K=3, seed=123):
    rng = np.random.default_rng(seed)
    U0 = rng.normal(0.0, 0.5, size=(n, K)).astype(np.float32)
    V0 = rng.normal(0.0, 0.5, size=(m, K)).astype(np.float32)
    d0 = rng.uniform(1.0, 2.0, size=(K,)).astype(np.float32)
    row_off = np.zeros(n, dtype=np.float32)
    col_off = np.zeros(m, dtype=np.float32)

    Z = U0 @ np.diag(d0) @ V0.T + row_off[:, None] + col_off[None, :]
    Z = np.clip(Z, -2.0, 2.0)
    lam = np.exp(Z) * 10.0
    Y = rng.poisson(lam).astype(np.float32)
    return Y, K


def _predict_counts(model: PoissonGLMPCA) -> np.ndarray:
    return model.reconstruct_counts(clip=20.0)


@pytest.mark.parametrize(
    "col_size_factor,row_intercept",
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
)
def test_predicted_counts_dense_offsets_modes(col_size_factor, row_intercept):
    Y, K = _synthetic_poisson_data(n=50, m=40, K=3, seed=111)
    model = PoissonGLMPCA(
        n_pcs=K,
        max_iter=150,
        tol=1e-7,
        device="cpu",
        progress_bar=False,
        verbose=False,
        seed=111,
        num_ccd_iter=5,
        col_size_factor=col_size_factor,
        row_intercept=row_intercept,
    )
    model.fit(Y, init="svd")

    Y_hat = _predict_counts(model)
    if sp.issparse(Y_hat):
        Y_hat = Y_hat.toarray()
    y = Y.flatten()
    yhat = Y_hat.flatten()
    corr = np.corrcoef(y, yhat)[0, 1]

    assert np.isfinite(corr)
    assert corr >= 0.75


@pytest.mark.parametrize(
    "col_size_factor,row_intercept",
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
)
def test_predicted_counts_sparse_offsets_modes(col_size_factor, row_intercept):
    Y, K = _synthetic_poisson_data(n=50, m=40, K=3, seed=222)
    Y_sp = sp.coo_matrix(Y)
    model = PoissonGLMPCA(
        n_pcs=K,
        max_iter=150,
        tol=1e-7,
        device="cpu",
        progress_bar=False,
        verbose=False,
        seed=222,
        num_ccd_iter=5,
        col_size_factor=col_size_factor,
        row_intercept=row_intercept,
    )
    model.fit(Y_sp, init="svd")

    Y_hat = _predict_counts(model)
    if sp.issparse(Y_hat):
        Y_hat = Y_hat.toarray()
    y = Y_sp.toarray().flatten()
    yhat = Y_hat.flatten()
    corr = np.corrcoef(y, yhat)[0, 1]

    assert np.isfinite(corr)
    assert corr >= 0.75


def test_predicted_counts_dense_deviation():
    Y, K = _synthetic_poisson_data(n=50, m=40, K=3, seed=10)
    model = PoissonGLMPCA(n_pcs=K, max_iter=150, tol=1e-7, device="cpu", progress_bar=False, verbose=False, seed=10, num_ccd_iter=5)
    model.fit(Y, init="svd")

    Y_hat = _predict_counts(model)
    if sp.issparse(Y_hat):
        Y_hat = Y_hat.toarray()
    y = Y.flatten()
    yhat = Y_hat.flatten()
    corr = np.corrcoef(y, yhat)[0, 1]

    assert np.isfinite(corr)
    assert corr >= 0.75


def test_predicted_counts_sparse_deviation():
    Y, K = _synthetic_poisson_data(n=50, m=40, K=3, seed=20)
    Y_sp = sp.coo_matrix(Y)
    model = PoissonGLMPCA(n_pcs=K, max_iter=150, tol=1e-7, device="cpu", progress_bar=False, verbose=False, seed=20, num_ccd_iter=5)
    model.fit(Y_sp, init="svd")

    Y_hat = _predict_counts(model)
    if sp.issparse(Y_hat):
        Y_hat = Y_hat.toarray()
    y = Y_sp.toarray().flatten()
    yhat = Y_hat.flatten()
    corr = np.corrcoef(y, yhat)[0, 1]

    assert np.isfinite(corr)
    assert corr >= 0.75