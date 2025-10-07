"""Utilities for Poisson GLM-PCA with cyclic coordinate descent updates."""
from __future__ import annotations
import torch
import numpy as np
import warnings
from tqdm import tqdm
import scipy.sparse as sp
from scipy.sparse.linalg import svds


class PoissonGLMPCA:
    """
    Poisson GLM-PCA model fitted via cyclic coordinate descent (CCD).

    Estimates low-rank factors under a Poisson likelihood using a constant learning rate.
    """
    def __init__(
        self,
        n_pcs: int = 30,
        max_iter: int = 1000,
        tol: float = 1e-4,
        col_size_factor: bool = True,
        row_intercept: bool = True,
        verbose: bool = False,
        device: str | None = None,
        progress_bar: bool = True,
        seed: int | None = 42,
        batch_size_rows: int | None = None,
        batch_size_cols: int | None = None,
        learning_rate: float = 0.5,
        num_ccd_iter: int = 3,
    ):
        """
        Initialize the Poisson GLM-PCA model.

        Parameters
        ----------
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
        num_ccd_iter : int, optional
            Number of coordinate descent iterations per main iteration. Default is 3.
        batch_size_rows : int or None, optional
            Batch size for row updates. If None, uses max(1, min(n_samples, 1024)). Default is None.
        batch_size_cols : int or None, optional
            Batch size for column updates. If None, uses max(1, min(n_samples, 1024)). Default is None.
        learning_rate : float, optional
            Step size used in updates. Default is 0.5.
        """
        
        if n_pcs < 1:
            warnings.warn("Number of PCs (K) must be 1 or greater. Using K=30 instead.")
            n_pcs = 30

        self.n_pcs = n_pcs
        self.max_iter = max_iter
        self.tol = tol
        self.col_size_factor = col_size_factor
        self.row_intercept = row_intercept
        self.verbose = verbose
        self.progress_bar = progress_bar
        self.seed = seed
        self.learning_rate = learning_rate
        self.batch_size_rows = batch_size_rows
        self.batch_size_cols = batch_size_cols
        self.num_ccd_iter = num_ccd_iter

        
        if self.seed:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
        
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            if device not in ["cuda", "mps", "cpu"]:
                warnings.warn("Device must be 'cuda', 'mps', or 'cpu'. Using 'cpu' instead.")
                device = "cpu"
            if device == "cuda" and not torch.cuda.is_available():
                warnings.warn("CUDA device is not available. Using 'cpu' instead.")
                device = "cpu"
            if device == "mps" and not torch.mps.is_available():
                warnings.warn("MPS device is not available. Using 'cpu' instead.")
                device = "cpu"
            self.device = device

    def _initialize_params(self, Y: torch.Tensor, init: str = "svd") -> tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize factor matrices LL and FF.

        Parameters
        ----------
        Y : torch.Tensor
            Nonnegative count matrix of shape (n_samples, n_features).
        init : {"svd", "random"}
            Initialization method. "svd" uses SVD of log1p(Y) to derive a strong starting point;
            "random" initializes LL and FF with small Gaussian noise.

        Returns
        -------
        tuple of torch.Tensor
            LL with shape (K, n_samples) and FF with shape (K, n_features).
        """
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        if init not in ["svd", "random"]:
            warnings.warn("Initialization method must be 'svd' or 'random'. Using 'svd' instead.")
            init = "svd"

        if self.verbose:
            print(f"Initializing parameters using '{init}' method...")

        n, m = Y.shape

        if init.lower() == "svd":
            if hasattr(self, "is_sparse") and self.is_sparse:
                Y_scipy = sp.coo_matrix(
                    (Y._values().cpu().numpy(), (Y._indices()[0].cpu().numpy(), Y._indices()[1].cpu().numpy())),
                    shape=Y.shape
                )
                Y_log1p_sparse = Y_scipy.log1p()
                k = min(self.n_pcs, n - 1, m - 1)
                U, s, Vh = svds(Y_log1p_sparse, k=k)
                U, s, Vh = U[:, ::-1], s[::-1], Vh[::-1, :]
                
                U_k = torch.from_numpy(U.copy())
                V_k = torch.from_numpy(Vh.T.copy())
                d_k = torch.from_numpy(s.copy())
            else:
                Y_log1p = torch.log1p(Y)
                try:
                    U, S, Vh = torch.linalg.svd(Y_log1p, full_matrices=False)
                    V = Vh.T
                except torch.linalg.LinAlgError:
                    U, S, V = torch.svd(Y_log1p)

                k = self.n_pcs
                U_k = U[:, :k]
                V_k = V[:, :k]
                d_k = S[:k]

            LL = (U_k * torch.sqrt(d_k)).T
            FF = (V_k * torch.sqrt(d_k)).T
        else:
            LL = torch.randn(self.n_pcs, n, device=self.device) * 1e-4
            FF = torch.randn(self.n_pcs, m, device=self.device) * 1e-4
        
        return LL.to(self.device), FF.to(self.device)

    def _poisson_log_likelihood(
        self,
        Y: torch.Tensor,
        LL: torch.Tensor,
        FF: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the Poisson log-likelihood using batched operations.
 
        Parameters
        ----------
        Y : torch.Tensor
            Nonnegative count matrix of shape (n_samples, n_features).
        LL : torch.Tensor
            Left factor of shape (K, n_samples).
        FF : torch.Tensor
            Right factor of shape (K, n_features).
 
        Returns
        -------
        torch.Tensor
            Scalar tensor with the total log-likelihood.
        """
        clamp_min, clamp_max = -20, 20
        n, m = Y.shape

        if hasattr(self, "is_sparse") and self.is_sparse:
            try:
                Y = Y.coalesce()
            except Exception:
                pass
            if hasattr(Y, "indices") and hasattr(Y, "values"):
                idx = Y.indices()
                vals = Y.values()
            else:
                idx = Y._indices()
                vals = Y._values()
            rows = idx[0]
            cols = idx[1]

            LL_rows = LL[:, rows]
            FF_cols = FF[:, cols]
            dot_k = (LL_rows * FF_cols).sum(dim=0)
            log_lambda_nnz = dot_k + self.row_offset[rows] + self.col_offset[cols]
            log_lambda_nnz = torch.clamp(log_lambda_nnz, clamp_min, clamp_max)
            term1 = torch.sum(vals * log_lambda_nnz)

            total = torch.tensor(0.0, device=self.device)
            bsz = self.batch_size_rows or max(1, min(n, 1024))
            for start in range(0, n, bsz):
                end = min(n, start + bsz)
                LL_b = LL[:, start:end]
                Z = LL_b.T @ FF
                Z = Z + self.row_offset[start:end].unsqueeze(1) + self.col_offset.unsqueeze(0)
                Z = torch.clamp(Z, clamp_min, clamp_max)
                total = total + torch.exp(Z).sum()
            term2 = total
            
        else:
            term1_total = torch.tensor(0.0, device=self.device)
            term2_total = torch.tensor(0.0, device=self.device)
            
            bsz = self.batch_size_rows or max(1, min(n, 1024))
            for start in range(0, n, bsz):
                end = min(n, start + bsz)
                Y_batch = Y[start:end, :]
                LL_b = LL[:, start:end]
                
                log_lambda_batch = LL_b.T @ FF
                log_lambda_batch = log_lambda_batch + self.row_offset[start:end].unsqueeze(1) + self.col_offset.unsqueeze(0)
                log_lambda_batch = torch.clamp(log_lambda_batch, clamp_min, clamp_max)
                
                term1_total = term1_total + (Y_batch * log_lambda_batch).sum()
                term2_total = term2_total + torch.exp(log_lambda_batch).sum()
            
            term1 = term1_total
            term2 = term2_total

        return term1 - term2

    def _update_LL_batch(
        self,
        Y: torch.Tensor,
        LL: torch.Tensor,
        FF: torch.Tensor,
    ) -> torch.Tensor:
        """
        Update LL via cyclic coordinate descent (CCD) with a constant learning rate.
 
        Parameters
        ----------
        Y : torch.Tensor
            Nonnegative count matrix.
        LL : torch.Tensor
            Left factor (K x n_samples).
        FF : torch.Tensor
            Right factor (K x n_features).
 
        Returns
        -------
        torch.Tensor
            Updated LL tensor.
        """
        n, m = Y.shape
        clamp_min, clamp_max = -20, 20
        
        if self.is_sparse:
            YFF = torch.sparse.mm(Y, FF.T)
        else:
            YFF = Y @ FF.T
        
        for _ in range(self.num_ccd_iter):
            for k in range(self.n_pcs):
                y_ff_k = YFF[:, k]
                
                exp_grad_k = torch.zeros(n, device=self.device)
                hess_diag_k = torch.zeros(n, device=self.device)
                ff_k = FF[k, :]
                ff_k_sq = ff_k * ff_k
                
                bsz = self.batch_size_rows or max(1, min(n, 1024))
                for start in range(0, n, bsz):
                    end = min(n, start + bsz)
                    LL_b = LL[:, start:end]
                    Z = LL_b.T @ FF
                    Z = Z + self.row_offset[start:end].unsqueeze(1) + self.col_offset.unsqueeze(0)
                    Z = torch.clamp(Z, clamp_min, clamp_max)
                    Lambda_b = torch.exp(Z)
                    exp_grad_k[start:end] = Lambda_b @ ff_k
                    hess_diag_k[start:end] = Lambda_b @ ff_k_sq
                
                grad_k = y_ff_k - exp_grad_k
                hess_diag_k = torch.clamp(-hess_diag_k, max=-1e-8)
                direction = grad_k / hess_diag_k
                
                LL[k, :] = LL[k, :] - self.learning_rate * direction
        
        return LL

    def _update_FF_batch(
        self,
        Y: torch.Tensor,
        LL: torch.Tensor,
        FF: torch.Tensor,
    ) -> torch.Tensor:
        """
        Update FF via cyclic coordinate descent (CCD) with a constant learning rate.
 
        Parameters
        ----------
        Y : torch.Tensor
            Nonnegative count matrix.
        LL : torch.Tensor
            Left factor (K x n_samples).
        FF : torch.Tensor
            Right factor (K x n_features).
 
        Returns
        -------
        torch.Tensor
            Updated FF tensor.
        """
        n, m = Y.shape
        clamp_min, clamp_max = -20, 20
        
        if self.is_sparse:
            YTLL = torch.sparse.mm(Y.T, LL.T)
        else:
            YTLL = Y.T @ LL.T
        
        for _ in range(self.num_ccd_iter):
            for k in range(self.n_pcs):
                yT_ll_k = YTLL[:, k]
                
                exp_grad_k = torch.zeros(m, device=self.device)
                hess_diag_k = torch.zeros(m, device=self.device)
                ll_k = LL[k, :]
                ll_k_sq = ll_k * ll_k
                
                bszc = self.batch_size_cols or max(1, min(m, 1024))
                for start in range(0, m, bszc):
                    end = min(m, start + bszc)
                    FF_b = FF[:, start:end]
                    Z = LL.T @ FF_b
                    Z = Z + self.row_offset.unsqueeze(1) + self.col_offset[start:end].unsqueeze(0)
                    Z = torch.clamp(Z, clamp_min, clamp_max)
                    Lambda_b = torch.exp(Z)
                    exp_grad_k[start:end] = Lambda_b.T @ ll_k
                    hess_diag_k[start:end] = Lambda_b.T @ ll_k_sq
                
                grad_k = yT_ll_k - exp_grad_k
                hess_diag_k = torch.clamp(-hess_diag_k, max=-1e-8)
                direction = grad_k / hess_diag_k
                
                FF[k, :] = FF[k, :] - self.learning_rate * direction
        
        return FF

    def _orthonormalize_factors(self, LL: torch.Tensor, FF: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Orthonormalize factor matrices to stabilize optimization.
 
        Parameters
        ----------
        LL : torch.Tensor
            Left factor (K x n_samples).
        FF : torch.Tensor
            Right factor (K x n_features).
 
        Returns
        -------
        tuple of torch.Tensor
            Orthonormalized (LL, FF) pair.
        """
        K = self.n_pcs
        if K == 1:
            return LL, FF
        
        try:
            out = torch.linalg.svd(FF.T, full_matrices=False)
            FF_new = out.U.T
            LL_new = torch.diag(out.S) @ out.Vh @ LL
            return LL_new, FF_new
        except torch.linalg.LinAlgError:
            warnings.warn("SVD in orthonormalization failed. Skipping step.")
            return LL, FF


    def fit(
        self,
        Y,
        init: str = "svd",
    ) -> PoissonGLMPCA:
        """
        Fit the Poisson GLM-PCA model.
 
        Parameters
        ----------
        Y : array-like, torch.Tensor, or scipy.sparse matrix
            Input count matrix of shape (n_samples, n_features).
        init : {"svd", "random"}
            Initialization method for LL and FF.
 
        Returns
        -------
        PoissonGLMPCA
            The fitted model instance.
        """

        self.is_sparse = False
        self.loglik_history_ = []
        
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        if sp.issparse(Y):
            self.is_sparse = True
            if self.verbose:
                print("Detected sparse input matrix. Using sparse optimizations.")
            coo = Y.tocoo()
                
            indices = torch.LongTensor(np.vstack((coo.row, coo.col)))
            values = torch.FloatTensor(coo.data)
            shape = torch.Size(coo.shape)
            Y_sparse = torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)
            try:
                Y_sparse = Y_sparse.to(self.device)
            except (NotImplementedError, RuntimeError):
                warnings.warn("Sparse tensor conversion to device not supported or failed. Using CPU.")
                self.device = "cpu"
                Y_sparse = Y_sparse.to(self.device)
            Y = Y_sparse
        elif not isinstance(Y, torch.Tensor):
            Y = torch.tensor(Y, dtype=torch.float32, device=self.device)
        else:
            Y = Y.to(self.device)

        # Validate integer-like counts (values may be float but must represent integers)
        tol_int = 1e-6
        if self.is_sparse:
            try:
                Yc = Y.coalesce()
                vals = Yc.values() if hasattr(Yc, "values") else Y._values()
            except Exception:
                vals = Y._values()
            if vals.numel() > 0:
                frac_dev = torch.abs(vals - torch.round(vals))
                if torch.any(frac_dev > tol_int):
                    raise ValueError("Input count matrix must consist of integers; found non-integer values in sparse data.")
        else:
            if Y.numel() > 0:
                frac_dev = torch.abs(Y - torch.round(Y))
                if torch.any(frac_dev > tol_int):
                    raise ValueError("Input count matrix must consist of integers; found non-integer values in dense data.")

        self.row_offset = torch.zeros(Y.shape[0], device=self.device)
        self.col_offset = torch.zeros(Y.shape[1], device=self.device)
        epsilon = 1e-8

        if self.col_size_factor:
            if self.is_sparse:
                col_means = torch.sparse.sum(Y, dim=0).to_dense() / Y.shape[0]
            else:
                col_means = Y.mean(dim=0)
            col_offset = torch.log(col_means + epsilon)
            self.col_offset = self.col_offset + col_offset

        if self.row_intercept:
            if self.is_sparse:
                sum_col_means = (torch.sparse.sum(Y, dim=0).to_dense() / Y.shape[0]).sum()
                row_sums = torch.sparse.sum(Y, dim=1).to_dense()
            else:
                sum_col_means = Y.mean(dim=0).sum()
                row_sums = Y.sum(dim=1)
            row_offset = torch.log(row_sums / (sum_col_means + epsilon) + epsilon)
            self.row_offset = self.row_offset + row_offset

        n, m = Y.shape
        if self.verbose:
            print(f"Fitting GLM-PCA to {n} samples and {m} features on device: '{self.device}'")
            if self.is_sparse:
                density = Y._nnz() / (n * m)
                print(f"Sparse matrix density: {density:.4f}")
    
        LL, FF = self._initialize_params(Y, init=init)
        loglik = self._poisson_log_likelihood(Y, LL, FF)
        self.loglik_history_.append(loglik.item())
        
        if self.verbose:
            print(f"Initial Log-Likelihood: {loglik:.4f}")

        iterator = tqdm(range(self.max_iter), desc="GLM-PCA Iterations") if self.progress_bar else range(self.max_iter)

        for i in iterator:
            prev_loglik = self.loglik_history_[-1]

            LL, FF = self._orthonormalize_factors(LL, FF)
            LL = self._update_LL_batch(Y, LL, FF)
            
            FF, LL = self._orthonormalize_factors(FF, LL)
            FF = self._update_FF_batch(Y, LL, FF)

            loglik = self._poisson_log_likelihood(Y, LL, FF)
            self.loglik_history_.append(loglik.item())

            if torch.isnan(loglik):
                warnings.warn(f"\nLog-likelihood is NaN at iteration {i+1}. Stopping.")
                break

            delta = abs((loglik - prev_loglik) / (abs(prev_loglik) + 1e-6))

            if self.verbose:
                print(f"Iter {i+1:3d} | Log-Likelihood: {loglik:.4f} | Change: {delta:.2e}")
            if self.progress_bar:
                iterator.set_postfix(loglik=f"{loglik:.4f}", delta=f"{delta:.2e}")

            if delta < self.tol:
                if self.verbose or self.progress_bar:
                    print(f"\nConvergence reached after {i+1} iterations.")
                break
        
        if i == self.max_iter - 1 and (self.verbose or self.progress_bar):
            warnings.warn(f"\nMaximum iterations ({self.max_iter}) reached without convergence.")
            
        self._finalize_factors(LL, FF)
        return self

    def _finalize_factors(self, LL: torch.Tensor, FF: torch.Tensor) -> None:
        """
        Finalize model by computing orthogonal U and V and singular values d. Also
        store sample scores X, feature loadings B, and design matrices W and Z.

        Parameters
        ----------
        LL : torch.Tensor
            Final left factor (K x n_samples).
        FF : torch.Tensor
            Final right factor (K x n_features).
        """
        U_hat, V_hat = LL.T, FF.T

        self.X = U_hat.detach().cpu().numpy()
        self.B = FF.T.detach().cpu().numpy()

        if hasattr(self, "row_offset") and isinstance(self.row_offset, torch.Tensor):
            row_off = self.row_offset.detach().cpu().numpy()
        else:
            row_off = np.zeros(U_hat.shape[0], dtype=np.float32)
        if hasattr(self, "col_offset") and isinstance(self.col_offset, torch.Tensor):
            col_off = self.col_offset.detach().cpu().numpy()
        else:
            col_off = np.zeros(V_hat.shape[0], dtype=np.float32)
        self.W = np.column_stack([row_off, np.ones_like(row_off)])
        self.Z = np.column_stack([np.ones_like(col_off), col_off])
        
        QU, RU = torch.linalg.qr(U_hat, mode="reduced")
        QV, RV = torch.linalg.qr(V_hat, mode="reduced")

        M = RU @ RV.T
        try:
            Um, s, Vmh = torch.linalg.svd(M, full_matrices=False)
        except torch.linalg.LinAlgError:
             warnings.warn("Final SVD failed. Factors may not be fully orthogonal.")
             self.U = QU.detach().cpu().numpy()
             self.d = torch.ones(self.n_pcs).detach().cpu().numpy()
             self.V = QV.detach().cpu().numpy()
             return

        sort_indices = torch.argsort(s, descending=True)
        s = s[sort_indices]
        Um = Um[:, sort_indices]
        Vmh = Vmh[sort_indices, :]

        U = QU @ Um
        V = QV @ Vmh.T

        self.U = U.detach().cpu().numpy()
        self.d = s.detach().cpu().numpy()
        self.V = V.detach().cpu().numpy()

    def reconstruct_counts(self, clip: float | tuple[float, float] | None = 20.0) -> np.ndarray:
        """
        Reconstruct the expected counts matrix from the fitted model.

        Computes the Poisson rate matrix using the stored orthonormal factors
        and offsets as:

            exp(U @ diag(d) @ V.T + row_offset[:, None] + col_offset[None, :])

        Parameters
        ----------
        clip : float or tuple of float or None, optional
            Clip the linear predictor before exponentiation to stabilize extremes.
            A single float applies symmetric clipping to (-clip, clip); a
            tuple specifies (low, high) bounds. If None, no clipping is
            applied. Default is 20.0.

        Returns
        -------
        np.ndarray
            Expected counts of shape (n_samples, n_features).

        Raises
        ------
        RuntimeError
            If the model has not been fitted and required attributes are missing.
        """
        if not (hasattr(self, "U") and hasattr(self, "V") and hasattr(self, "d")):
            raise RuntimeError("Model is not fitted or factors are not finalized.")

        U = self.U
        V = self.V
        d = self.d

        if hasattr(self, "row_offset") and isinstance(self.row_offset, torch.Tensor):
            row_off = self.row_offset.detach().cpu().numpy()
        else:
            row_off = np.zeros(U.shape[0], dtype=np.float32)

        if hasattr(self, "col_offset") and isinstance(self.col_offset, torch.Tensor):
            col_off = self.col_offset.detach().cpu().numpy()
        else:
            col_off = np.zeros(V.shape[0], dtype=np.float32)

        n, m = U.shape[0], V.shape[0]

        Z = U @ np.diag(d) @ V.T + row_off[:, None] + col_off[None, :]
        if clip is not None:
            if isinstance(clip, tuple):
                low, high = clip
            else:
                low, high = (-float(clip), float(clip))
            Z = np.clip(Z, low, high)
        return np.exp(Z)