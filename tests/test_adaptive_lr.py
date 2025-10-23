import numpy as np
import scipy.sparse as sp
import os, sys

# Ensure package root is importable like other tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastglmpca import PoissonGLMPCA


def _synthetic_poisson(n=80, m=60, K=3, seed=777, clip=2.0, scale=8.0):
    rng = np.random.default_rng(seed)
    U_rand = rng.normal(0.0, 1.0, size=(n, K)).astype(np.float32)
    V_rand = rng.normal(0.0, 1.0, size=(m, K)).astype(np.float32)
    Uq, _ = np.linalg.qr(U_rand)
    Vq, _ = np.linalg.qr(V_rand)
    d_true = rng.uniform(1.0, 2.0, size=(K,)).astype(np.float32)
    Z = Uq @ np.diag(d_true) @ Vq.T
    Z = np.clip(Z, -float(clip), float(clip))
    lam = np.exp(Z) * float(scale)
    Y = rng.poisson(lam).astype(np.float32)
    return Y


def _is_monotonic_non_decreasing(xs, atol=1e-8):
    diffs = np.diff(np.asarray(xs, dtype=np.float64))
    return np.all(diffs >= -float(atol))


def test_adaptive_lr_monotonic_loglik_fit_dense():
    Y = _synthetic_poisson(n=100, m=80, K=4, seed=801)
    model = PoissonGLMPCA(
        n_pcs=4,
        max_iter=100,
        tol=1e-7,
        device="cpu",
        progress_bar=False,
        verbose=False,
        seed=801,
        num_ccd_iter=5,
        learning_rate=5.0,  # deliberately high to trigger adaptation
        adaptive_lr=True,
        lr_decay=0.5,
        min_learning_rate=1e-5,
        max_backtracks=3,
    )
    model.fit(Y, init="svd")
    assert len(model.loglik_history_) >= 2
    assert _is_monotonic_non_decreasing(model.loglik_history_, atol=1e-7)


def test_adaptive_lr_monotonic_loglik_fit_sparse():
    Y_dense = _synthetic_poisson(n=90, m=70, K=3, seed=802)
    Y = sp.coo_matrix(Y_dense)
    model = PoissonGLMPCA(
        n_pcs=3,
        max_iter=100,
        tol=1e-7,
        device="cpu",
        progress_bar=False,
        verbose=False,
        seed=802,
        num_ccd_iter=5,
        learning_rate=5.0,
        adaptive_lr=True,
        lr_decay=0.5,
        min_learning_rate=1e-5,
        max_backtracks=3,
    )
    model.fit(Y, init="svd")
    assert len(model.loglik_history_) >= 2
    assert _is_monotonic_non_decreasing(model.loglik_history_, atol=1e-7)


def test_project_adaptive_lr_stability():
    Y = _synthetic_poisson(n=80, m=60, K=3, seed=803)
    model = PoissonGLMPCA(
        n_pcs=3,
        max_iter=80,
        tol=1e-7,
        device="cpu",
        progress_bar=False,
        verbose=False,
        seed=803,
        num_ccd_iter=5,
        learning_rate=5.0,
        adaptive_lr=True,
        lr_decay=0.5,
        min_learning_rate=1e-5,
        max_backtracks=3,
    )
    model.fit(Y, init="svd")
    U_new = model.project(Y, max_iter=80, tol=1e-7, progress_bar=False, init="svd")
    assert isinstance(U_new, np.ndarray)
    assert U_new.shape == (Y.shape[0], 3)
    assert np.isfinite(U_new).all()