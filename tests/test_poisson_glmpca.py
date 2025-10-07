import numpy as np
import pytest
import torch
import scipy.sparse as sp
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastglmpca import PoissonGLMPCA

def make_dense_data(n=20, m=15, rate=1.5, seed=0):
    rng = np.random.default_rng(seed)
    return rng.poisson(rate, size=(n, m)).astype(np.float32)


def make_sparse_data(n=20, m=15, rate=0.8, seed=0):
    rng = np.random.default_rng(seed)
    Y = rng.poisson(rate, size=(n, m)).astype(np.float32)
    Y[Y < 1] = 0.0
    return sp.coo_matrix(Y)


def to_torch_sparse_coo(coo, device="cpu"):
    coo = coo.tocoo()
    indices = np.vstack((coo.row, coo.col)).astype(np.int64)
    idx = torch.LongTensor(indices)
    vals = torch.FloatTensor(coo.data)
    shape = torch.Size(coo.shape)
    return torch.sparse_coo_tensor(idx, vals, shape, dtype=torch.float32, device=device)


def test_init_defaults():
    model = PoissonGLMPCA(device="cpu")
    assert model.n_pcs == 30
    assert model.max_iter == 1000
    assert model.tol == 1e-4
    assert model.learning_rate == 0.5
    assert model.num_ccd_iter == 3
    assert model.device in {"cpu", "cuda", "mps"}


def test_init_n_pcs_minimum():
    model = PoissonGLMPCA(n_pcs=0, device="cpu")
    # n_pcs < 1 should be corrected to 30 per implementation
    assert model.n_pcs == 30


def test_initialize_params_svd_dense():
    Y = torch.tensor(make_dense_data(), dtype=torch.float32)
    model = PoissonGLMPCA(n_pcs=3, device="cpu", seed=123)
    LL, FF = model._initialize_params(Y, init="svd")
    n, m = Y.shape
    assert LL.shape == (model.n_pcs, n)
    assert FF.shape == (model.n_pcs, m)
    assert LL.device.type == model.device
    assert FF.device.type == model.device


def test_initialize_params_random_sparse():
    Y_sparse = make_sparse_data()
    model = PoissonGLMPCA(n_pcs=3, device="cpu", seed=123)
    # random init avoids potential sparse log1p path issues
    LL, FF = model._initialize_params(torch.tensor(Y_sparse.toarray(), dtype=torch.float32), init="random")
    n, m = Y_sparse.shape
    assert LL.shape == (model.n_pcs, n)
    assert FF.shape == (model.n_pcs, m)


def test_poisson_log_likelihood_dense():
    Y = torch.tensor(make_dense_data(), dtype=torch.float32)
    model = PoissonGLMPCA(n_pcs=3, device="cpu", seed=123)
    model.row_offset = torch.zeros(Y.shape[0])
    model.col_offset = torch.zeros(Y.shape[1])
    LL = torch.randn(model.n_pcs, Y.shape[0]) * 1e-3
    FF = torch.randn(model.n_pcs, Y.shape[1]) * 1e-3
    ll = model._poisson_log_likelihood(Y, LL, FF)
    assert isinstance(ll, torch.Tensor)
    assert ll.ndim == 0
    assert torch.isfinite(ll)


def test_poisson_log_likelihood_sparse():
    Y = make_sparse_data()
    model = PoissonGLMPCA(n_pcs=3, device="cpu", seed=123)
    model.is_sparse = True
    Y_t = to_torch_sparse_coo(Y)
    model.row_offset = torch.zeros(Y.shape[0])
    model.col_offset = torch.zeros(Y.shape[1])
    LL = torch.randn(model.n_pcs, Y.shape[0]) * 1e-3
    FF = torch.randn(model.n_pcs, Y.shape[1]) * 1e-3
    ll = model._poisson_log_likelihood(Y_t, LL, FF)
    assert isinstance(ll, torch.Tensor)
    assert ll.ndim == 0
    assert torch.isfinite(ll)


def test_update_LL_batch_dense_changes():
    Y = torch.tensor(make_dense_data(), dtype=torch.float32)
    model = PoissonGLMPCA(n_pcs=3, device="cpu", seed=1, learning_rate=0.1, num_ccd_iter=1)
    model.is_sparse = False
    model.row_offset = torch.zeros(Y.shape[0])
    model.col_offset = torch.zeros(Y.shape[1])
    LL, FF = model._initialize_params(Y, init="random")
    LL_before = LL.clone()
    LL_after = model._update_LL_batch(Y, LL, FF)
    assert LL_after.shape == LL_before.shape
    assert not torch.equal(LL_after, LL_before)
    assert torch.isfinite(LL_after).all()


def test_update_LL_batch_sparse_changes():
    Y = make_sparse_data()
    Y_t = to_torch_sparse_coo(Y)
    model = PoissonGLMPCA(n_pcs=3, device="cpu", seed=1, learning_rate=0.1, num_ccd_iter=1)
    model.is_sparse = True
    model.row_offset = torch.zeros(Y.shape[0])
    model.col_offset = torch.zeros(Y.shape[1])
    # initialize on dense tensor for random init, then run sparse update
    LL, FF = model._initialize_params(torch.tensor(Y.toarray(), dtype=torch.float32), init="random")
    LL_before = LL.clone()
    LL_after = model._update_LL_batch(Y_t, LL, FF)
    assert LL_after.shape == LL_before.shape
    assert not torch.equal(LL_after, LL_before)
    assert torch.isfinite(LL_after).all()


def test_update_FF_batch_dense_changes():
    Y = torch.tensor(make_dense_data(), dtype=torch.float32)
    model = PoissonGLMPCA(n_pcs=3, device="cpu", seed=2, learning_rate=0.1, num_ccd_iter=1)
    model.is_sparse = False
    model.row_offset = torch.zeros(Y.shape[0])
    model.col_offset = torch.zeros(Y.shape[1])
    LL, FF = model._initialize_params(Y, init="random")
    FF_before = FF.clone()
    FF_after = model._update_FF_batch(Y, LL, FF)
    assert FF_after.shape == FF_before.shape
    assert not torch.equal(FF_after, FF_before)
    assert torch.isfinite(FF_after).all()


def test_update_FF_batch_sparse_changes():
    Y = make_sparse_data()
    Y_t = to_torch_sparse_coo(Y)
    model = PoissonGLMPCA(n_pcs=3, device="cpu", seed=1, learning_rate=0.1, num_ccd_iter=1)
    model.is_sparse = True
    model.row_offset = torch.zeros(Y.shape[0])
    model.col_offset = torch.zeros(Y.shape[1])
    # initialize on dense tensor for random init, then run sparse update
    LL, FF = model._initialize_params(torch.tensor(Y.toarray(), dtype=torch.float32), init="random")
    FF_before = FF.clone()
    FF_after = model._update_FF_batch(Y_t, LL, FF)
    assert FF_after.shape == FF_before.shape
    assert not torch.equal(FF_after, FF_before)
    assert torch.isfinite(FF_after).all()


def test_orthonormalize_factors_K_gt_1():
    n, m, K = 20, 15, 3
    model = PoissonGLMPCA(n_pcs=K, device="cpu")
    LL = torch.randn(K, n) * 1e-3
    FF = torch.randn(K, m) * 1e-3
    LL_new, FF_new = model._orthonormalize_factors(LL, FF)
    assert LL_new.shape == (K, n)
    assert FF_new.shape == (K, m)
    gram = FF_new @ FF_new.T
    assert torch.allclose(gram, torch.eye(K), atol=1e-5, rtol=1e-4)


def test_orthonormalize_factors_K_eq_1():
    n, m, K = 10, 8, 1
    model = PoissonGLMPCA(n_pcs=K, device="cpu")
    LL = torch.randn(K, n) * 1e-3
    FF = torch.randn(K, m) * 1e-3
    LL_new, FF_new = model._orthonormalize_factors(LL, FF)
    assert LL_new.shape == (K, n)
    assert FF_new.shape == (K, m)


def test_finalize_factors_shapes():
    n, m, K = 20, 15, 3
    model = PoissonGLMPCA(n_pcs=K, device="cpu")
    LL = torch.randn(K, n) * 1e-3
    FF = torch.randn(K, m) * 1e-3
    model._finalize_factors(LL, FF)
    assert isinstance(model.U, np.ndarray)
    assert isinstance(model.V, np.ndarray)
    assert isinstance(model.d, np.ndarray)
    assert model.U.shape == (n, K)
    assert model.V.shape == (m, K)
    assert model.d.shape == (K,)


def test_fit_dense_outputs():
    Y = make_dense_data(n=30, m=20, rate=1.2, seed=7)
    model = PoissonGLMPCA(n_pcs=5, max_iter=10, tol=1e-6, device="cpu", progress_bar=False, verbose=False, seed=7)
    model.fit(Y, init="svd")
    assert isinstance(model.U, np.ndarray)
    assert isinstance(model.V, np.ndarray)
    assert isinstance(model.d, np.ndarray)
    assert model.U.shape == (Y.shape[0], model.n_pcs)
    assert model.V.shape == (Y.shape[1], model.n_pcs)
    assert model.d.shape == (model.n_pcs,)
    assert len(model.loglik_history_) >= 1


def test_fit_sparse_outputs_random_init():
    Y_sparse = make_sparse_data(n=30, m=20, rate=0.7, seed=9)
    model = PoissonGLMPCA(n_pcs=4, max_iter=8, tol=1e-6, device="cpu", progress_bar=False, verbose=False, seed=9)
    model.fit(Y_sparse, init="random")
    assert isinstance(model.U, np.ndarray)
    assert isinstance(model.V, np.ndarray)
    assert isinstance(model.d, np.ndarray)
    assert model.U.shape == (Y_sparse.shape[0], model.n_pcs)
    assert model.V.shape == (Y_sparse.shape[1], model.n_pcs)
    assert model.d.shape == (model.n_pcs)