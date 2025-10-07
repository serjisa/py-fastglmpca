import numpy as np
import torch
import scipy.sparse as sp
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastglmpca import PoissonGLMPCA


def to_torch_sparse_coo(coo, device="cpu"):
    coo = coo.tocoo()
    indices = np.vstack((coo.row, coo.col)).astype(np.int64)
    idx = torch.LongTensor(indices)
    vals = torch.FloatTensor(coo.data)
    shape = torch.Size(coo.shape)
    return torch.sparse_coo_tensor(idx, vals, shape, dtype=torch.float32, device=device)


def expected_loglik(Y: np.ndarray, L: np.ndarray, F: np.ndarray, r_off: np.ndarray, c_off: np.ndarray):
    Z = np.outer(L, F) + r_off[:, None] + c_off[None, :]
    Z = np.clip(Z, -20.0, 20.0)
    term1 = (Y * Z).sum()
    term2 = np.exp(Z).sum()
    return term1 - term2


def expected_update_LL(Y: np.ndarray, L: np.ndarray, F: np.ndarray, r_off: np.ndarray, c_off: np.ndarray, lr: float):
    y_ff = Y @ F
    L_new = L.copy()
    for i in range(Y.shape[0]):
        Z_row = L[i] * F + r_off[i] + c_off
        Z_row = np.clip(Z_row, -20.0, 20.0)
        Lambda = np.exp(Z_row)
        exp_grad = (Lambda * F).sum()
        hess_diag = (Lambda * (F ** 2)).sum()
        hess_neg = min(-hess_diag, -1e-8)
        direction = (y_ff[i] - exp_grad) / hess_neg
        L_new[i] = L[i] - lr * direction
    return L_new


def expected_update_FF(Y: np.ndarray, L: np.ndarray, F: np.ndarray, r_off: np.ndarray, c_off: np.ndarray, lr: float):
    yT_ll = Y.T @ L
    F_new = F.copy()
    for j in range(Y.shape[1]):
        Z_col = L * F[j] + r_off + c_off[j]
        Z_col = np.clip(Z_col, -20.0, 20.0)
        Lambda = np.exp(Z_col)
        exp_grad = (Lambda * L).sum()
        hess_diag = (Lambda * (L ** 2)).sum()
        hess_neg = min(-hess_diag, -1e-8)
        direction = (yT_ll[j] - exp_grad) / hess_neg
        F_new[j] = F[j] - lr * direction
    return F_new


def _fixed_small_case():
    Y = np.array([[1.0, 0.0, 2.0],
                  [3.0, 1.0, 0.0]], dtype=np.float32)
    L = np.array([0.2, -0.1], dtype=np.float32)
    F = np.array([0.5, -0.3, 0.1], dtype=np.float32)
    r_off = np.array([0.0, 0.2], dtype=np.float32)
    c_off = np.array([-0.1, 0.0, 0.3], dtype=np.float32)
    return Y, L, F, r_off, c_off


def test_exact_loglik_dense():
    Y, L, F, r_off, c_off = _fixed_small_case()
    model = PoissonGLMPCA(n_pcs=1, device="cpu", seed=0)
    model.is_sparse = False
    model.row_offset = torch.tensor(r_off, dtype=torch.float32)
    model.col_offset = torch.tensor(c_off, dtype=torch.float32)

    LL = torch.tensor(L.reshape(1, -1), dtype=torch.float32)
    FF = torch.tensor(F.reshape(1, -1), dtype=torch.float32)
    Y_t = torch.tensor(Y, dtype=torch.float32)

    ll = model._poisson_log_likelihood(Y_t, LL, FF).item()
    ll_exp = expected_loglik(Y, L, F, r_off, c_off)
    assert np.isclose(ll, ll_exp, atol=1e-7)


def test_exact_loglik_sparse():
    Y, L, F, r_off, c_off = _fixed_small_case()
    model = PoissonGLMPCA(n_pcs=1, device="cpu", seed=0)
    model.is_sparse = True
    model.row_offset = torch.tensor(r_off, dtype=torch.float32)
    model.col_offset = torch.tensor(c_off, dtype=torch.float32)

    Y_sp = sp.coo_matrix(Y)
    Y_ts = to_torch_sparse_coo(Y_sp)
    LL = torch.tensor(L.reshape(1, -1), dtype=torch.float32)
    FF = torch.tensor(F.reshape(1, -1), dtype=torch.float32)

    ll = model._poisson_log_likelihood(Y_ts, LL, FF).item()
    ll_exp = expected_loglik(Y, L, F, r_off, c_off)
    assert np.isclose(ll, ll_exp, atol=1e-7)


def test_exact_update_LL_dense_K1():
    Y, L, F, r_off, c_off = _fixed_small_case()
    lr = 0.5
    model = PoissonGLMPCA(n_pcs=1, device="cpu", seed=0, learning_rate=lr, num_ccd_iter=1)
    model.is_sparse = False
    model.row_offset = torch.tensor(r_off, dtype=torch.float32)
    model.col_offset = torch.tensor(c_off, dtype=torch.float32)

    LL = torch.tensor(L.reshape(1, -1), dtype=torch.float32)
    FF = torch.tensor(F.reshape(1, -1), dtype=torch.float32)
    Y_t = torch.tensor(Y, dtype=torch.float32)

    LL_updated = model._update_LL_batch(Y_t, LL.clone(), FF.clone())[0].numpy()
    LL_expected = expected_update_LL(Y, L, F, r_off, c_off, lr)
    assert np.allclose(LL_updated, LL_expected, atol=1e-7)


def test_exact_update_LL_sparse_K1():
    Y, L, F, r_off, c_off = _fixed_small_case()
    lr = 0.5
    model = PoissonGLMPCA(n_pcs=1, device="cpu", seed=0, learning_rate=lr, num_ccd_iter=1)
    model.is_sparse = True
    model.row_offset = torch.tensor(r_off, dtype=torch.float32)
    model.col_offset = torch.tensor(c_off, dtype=torch.float32)

    Y_sp = sp.coo_matrix(Y)
    Y_ts = to_torch_sparse_coo(Y_sp)
    LL = torch.tensor(L.reshape(1, -1), dtype=torch.float32)
    FF = torch.tensor(F.reshape(1, -1), dtype=torch.float32)

    LL_updated = model._update_LL_batch(Y_ts, LL.clone(), FF.clone())[0].numpy()
    LL_expected = expected_update_LL(Y, L, F, r_off, c_off, lr)
    assert np.allclose(LL_updated, LL_expected, atol=1e-7)


def test_exact_update_FF_dense_K1():
    Y, L, F, r_off, c_off = _fixed_small_case()
    lr = 0.5
    model = PoissonGLMPCA(n_pcs=1, device="cpu", seed=0, learning_rate=lr, num_ccd_iter=1)
    model.is_sparse = False
    model.row_offset = torch.tensor(r_off, dtype=torch.float32)
    model.col_offset = torch.tensor(c_off, dtype=torch.float32)

    LL = torch.tensor(L.reshape(1, -1), dtype=torch.float32)
    FF = torch.tensor(F.reshape(1, -1), dtype=torch.float32)
    Y_t = torch.tensor(Y, dtype=torch.float32)

    FF_updated = model._update_FF_batch(Y_t, LL.clone(), FF.clone())[0].numpy()
    FF_expected = expected_update_FF(Y, L, F, r_off, c_off, lr)
    assert np.allclose(FF_updated, FF_expected, atol=1e-7)


def test_exact_update_FF_sparse_K1():
    Y, L, F, r_off, c_off = _fixed_small_case()
    lr = 0.5
    model = PoissonGLMPCA(n_pcs=1, device="cpu", seed=0, learning_rate=lr, num_ccd_iter=1)
    model.is_sparse = True
    model.row_offset = torch.tensor(r_off, dtype=torch.float32)
    model.col_offset = torch.tensor(c_off, dtype=torch.float32)

    Y_sp = sp.coo_matrix(Y)
    Y_ts = to_torch_sparse_coo(Y_sp)
    LL = torch.tensor(L.reshape(1, -1), dtype=torch.float32)
    FF = torch.tensor(F.reshape(1, -1), dtype=torch.float32)

    FF_updated = model._update_FF_batch(Y_ts, LL.clone(), FF.clone())[0].numpy()
    FF_expected = expected_update_FF(Y, L, F, r_off, c_off, lr)
    assert np.allclose(FF_updated, FF_expected, atol=1e-7)