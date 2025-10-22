import numpy as np
import scipy.sparse as sp
import pytest
import os, sys

# Match import style used in other tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastglmpca import PoissonGLMPCA


def _synthetic_poisson_pair(n=60, m=50, K=3, seed=123, scale=10.0):
    rng = np.random.default_rng(seed)
    U0 = rng.normal(0.0, 0.5, size=(n, K)).astype(np.float32)
    V0 = rng.normal(0.0, 0.5, size=(m, K)).astype(np.float32)
    d0 = rng.uniform(1.0, 2.0, size=(K,)).astype(np.float32)
    row_off = np.zeros(n, dtype=np.float32)
    col_off = np.zeros(m, dtype=np.float32)

    Z = U0 @ np.diag(d0) @ V0.T + row_off[:, None] + col_off[None, :]
    Z = np.clip(Z, -2.0, 2.0)
    lam = np.exp(Z) * scale

    Y_train = rng.poisson(lam).astype(np.float32)
    Y_new = rng.poisson(lam).astype(np.float32)
    return Y_train, Y_new, K


def _row_offset_for_projection(model: PoissonGLMPCA, Y_new):
    epsilon = 1e-8
    if hasattr(model, "col_offset"):
        if hasattr(model.col_offset, "detach"):
            col_off = model.col_offset.detach().cpu().numpy().astype(np.float32)
        else:
            col_off = np.asarray(model.col_offset, dtype=np.float32)
    else:
        col_off = np.zeros(Y_new.shape[1], dtype=np.float32)

    sum_col_means = np.sum(np.exp(col_off))

    if sp.issparse(Y_new):
        row_sums = np.asarray(Y_new.sum(axis=1)).ravel().astype(np.float32)
    else:
        row_sums = Y_new.sum(axis=1).astype(np.float32)

    row_off = np.log(row_sums / (sum_col_means + epsilon) + epsilon).astype(np.float32)
    return row_off, col_off


def _reconstruct_with_projected(model: PoissonGLMPCA, U_new: np.ndarray, row_off: np.ndarray, col_off: np.ndarray, clip=20.0) -> np.ndarray:
    # model.d, model.V are numpy arrays
    d = model.d.astype(np.float32)
    V = model.V.astype(np.float32)
    Z = U_new @ np.diag(d) @ V.T + row_off[:, None] + col_off[None, :]
    if clip is not None:
        low, high = (-float(clip), float(clip)) if not isinstance(clip, tuple) else clip
        Z = np.clip(Z, low, high)
    return np.exp(Z)


@pytest.mark.parametrize(
    "col_size_factor,row_intercept",
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
)
def test_project_dense_outputs_and_correlation(col_size_factor, row_intercept):
    Y_train, Y_new, K = _synthetic_poisson_pair(n=60, m=50, K=3, seed=101)
    model = PoissonGLMPCA(
        n_pcs=K,
        max_iter=120,
        tol=1e-7,
        device="cpu",
        progress_bar=False,
        verbose=False,
        seed=101,
        num_ccd_iter=5,
        col_size_factor=col_size_factor,
        row_intercept=row_intercept,
    )
    model.fit(Y_train, init="svd")

    U_new = model.project(Y_new, max_iter=120, tol=1e-7, progress_bar=False, init="svd")
    assert isinstance(U_new, np.ndarray)
    assert U_new.shape == (Y_new.shape[0], K)
    assert np.isfinite(U_new).all()

    row_off, col_off = _row_offset_for_projection(model, Y_new)
    Y_hat = _reconstruct_with_projected(model, U_new, row_off, col_off, clip=20.0)

    y = Y_new.flatten()
    yhat = Y_hat.flatten()
    corr = np.corrcoef(y, yhat)[0, 1]

    assert np.isfinite(corr)
    assert corr >= 0.70


@pytest.mark.parametrize(
    "col_size_factor,row_intercept",
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
)
def test_project_sparse_outputs_and_correlation(col_size_factor, row_intercept):
    Y_train, Y_new, K = _synthetic_poisson_pair(n=60, m=50, K=3, seed=202)
    Y_train_sp = sp.coo_matrix(Y_train)
    Y_new_sp = sp.coo_matrix(Y_new)

    model = PoissonGLMPCA(
        n_pcs=K,
        max_iter=120,
        tol=1e-7,
        device="cpu",
        progress_bar=False,
        verbose=False,
        seed=202,
        num_ccd_iter=5,
        col_size_factor=col_size_factor,
        row_intercept=row_intercept,
    )
    model.fit(Y_train_sp, init="svd")

    U_new = model.project(Y_new_sp, max_iter=120, tol=1e-7, progress_bar=False, init="svd")
    assert isinstance(U_new, np.ndarray)
    assert U_new.shape == (Y_new_sp.shape[0], K)
    assert np.isfinite(U_new).all()

    row_off, col_off = _row_offset_for_projection(model, Y_new_sp)
    Y_hat = _reconstruct_with_projected(model, U_new, row_off, col_off, clip=20.0)

    y = Y_new_sp.toarray().flatten()
    yhat = Y_hat.flatten()
    corr = np.corrcoef(y, yhat)[0, 1]

    assert np.isfinite(corr)
    assert corr >= 0.70


def test_project_feature_mismatch_raises():
    Y_train, Y_new, K = _synthetic_poisson_pair(n=20, m=10, K=2, seed=303)
    model = PoissonGLMPCA(n_pcs=K, max_iter=50, tol=1e-7, device="cpu", progress_bar=False, verbose=False, seed=303)
    model.fit(Y_train, init="svd")
    # Mismatch in feature dimension
    Y_bad = np.random.poisson(1.0, size=(5, Y_train.shape[1] + 1)).astype(np.float32)
    with pytest.raises(ValueError):
        _ = model.project(Y_bad, max_iter=10, tol=1e-7, progress_bar=False)


def test_project_non_integer_raises_dense_and_sparse():
    Y_train, Y_new, K = _synthetic_poisson_pair(n=20, m=12, K=2, seed=404)
    model = PoissonGLMPCA(n_pcs=K, max_iter=50, tol=1e-7, device="cpu", progress_bar=False, verbose=False, seed=404)
    model.fit(Y_train, init="svd")

    Y_new_dense = Y_new.copy()
    Y_new_dense[0, 0] = 1.5
    with pytest.raises(ValueError):
        _ = model.project(Y_new_dense, max_iter=10, tol=1e-7, progress_bar=False)

    Y_new_sp = Y_new.copy()
    Y_new_sp[1, 2] = 2.7
    Y_new_sp_coo = sp.coo_matrix(Y_new_sp)
    with pytest.raises(ValueError):
        _ = model.project(Y_new_sp_coo, max_iter=10, tol=1e-7, progress_bar=False)


def test_project_recovers_training_coordinates_dense():
    Y_train, Y_new, K = _synthetic_poisson_pair(n=60, m=50, K=3, seed=310)
    model = PoissonGLMPCA(
        n_pcs=K,
        max_iter=160,
        tol=1e-7,
        device="cpu",
        progress_bar=False,
        verbose=False,
        seed=310,
        num_ccd_iter=5,
        col_size_factor=True,
        row_intercept=True,
    )
    model.fit(Y_train, init="svd")

    U_proj_train = model.project(Y_train, max_iter=160, tol=1e-7, progress_bar=False, init="svd")
    U_fit = model.U

    corrs = []
    for k in range(K):
        c = np.corrcoef(U_proj_train[:, k], U_fit[:, k])[0, 1]
        corrs.append(c)
    assert np.all(np.isfinite(corrs))
    assert np.mean(corrs) >= 0.90


def _generate_counts_from_model(model: PoissonGLMPCA, U_new_true: np.ndarray, clip=20.0, seed=999):
    rng = np.random.default_rng(seed)
    d = model.d.astype(np.float32)
    V = model.V.astype(np.float32)

    if hasattr(model, "col_offset"):
        if hasattr(model.col_offset, "detach"):
            col_off = model.col_offset.detach().cpu().numpy().astype(np.float32)
        else:
            col_off = np.asarray(model.col_offset, dtype=np.float32)
    else:
        col_off = np.zeros(V.shape[0], dtype=np.float32)

    Z = U_new_true @ np.diag(d) @ V.T + col_off[None, :]
    low, high = (-float(clip), float(clip)) if not isinstance(clip, tuple) else clip
    Z = np.clip(Z, low, high)
    lam = np.exp(Z)
    return rng.poisson(lam).astype(np.float32)


def test_project_coordinates_match_generated_dense_no_row_intercept():
    Y_train, _, K = _synthetic_poisson_pair(n=80, m=60, K=3, seed=420)
    model = PoissonGLMPCA(
        n_pcs=K,
        max_iter=180,
        tol=1e-7,
        device="cpu",
        progress_bar=False,
        verbose=False,
        seed=420,
        num_ccd_iter=5,
        col_size_factor=True,
        row_intercept=False,
    )
    model.fit(Y_train, init="svd")

    rng = np.random.default_rng(421)
    U_new_true = rng.normal(0.0, 0.5, size=(50, K)).astype(np.float32)
    Y_new = _generate_counts_from_model(model, U_new_true, clip=20.0, seed=422)

    U_new = model.project(Y_new, max_iter=180, tol=1e-7, progress_bar=False, init="svd")

    corrs = []
    for k in range(K):
        c = np.corrcoef(U_new[:, k], U_new_true[:, k])[0, 1]
        corrs.append(c)
    rmse = np.sqrt(np.mean((U_new - U_new_true) ** 2))

    assert np.all(np.isfinite(corrs))
    assert np.mean(corrs) >= 0.90
    assert rmse <= 0.50


def test_project_coordinates_match_generated_sparse_no_row_intercept():
    Y_train, _, K = _synthetic_poisson_pair(n=80, m=60, K=3, seed=520)
    model = PoissonGLMPCA(
        n_pcs=K,
        max_iter=180,
        tol=1e-7,
        device="cpu",
        progress_bar=False,
        verbose=False,
        seed=520,
        num_ccd_iter=5,
        col_size_factor=True,
        row_intercept=False,
    )
    model.fit(sp.coo_matrix(Y_train), init="svd")

    rng = np.random.default_rng(521)
    U_new_true = rng.normal(0.0, 0.5, size=(50, K)).astype(np.float32)
    Y_new_dense = _generate_counts_from_model(model, U_new_true, clip=20.0, seed=522)
    Y_new_sparse = sp.coo_matrix(Y_new_dense)

    U_new = model.project(Y_new_sparse, max_iter=180, tol=1e-7, progress_bar=False, init="svd")
    corrs = []
    for k in range(K):
        c = np.corrcoef(U_new[:, k], U_new_true[:, k])[0, 1]
        corrs.append(c)
    rmse = np.sqrt(np.mean((U_new - U_new_true) ** 2))

    assert np.all(np.isfinite(corrs))
    assert np.mean(corrs) >= 0.90
    assert rmse <= 0.50