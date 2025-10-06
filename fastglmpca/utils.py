from __future__ import annotations
import torch
import numpy as np
import warnings
from tqdm import tqdm
import scipy.sparse as sp
from scipy.sparse.linalg import svds


class PoissonGLMPCA:
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
        line_search: bool = True,
        ls_beta: float = 0.5,
        ls_max_steps: int = 10,
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
        """
        
        # Step 1: Validate inputs and set defaults
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
        self.batch_size_rows = batch_size_rows
        self.batch_size_cols = batch_size_cols
        self.line_search = line_search
        self.ls_beta = ls_beta
        self.ls_max_steps = ls_max_steps
        
        # Step 2: Seed random generators for reproducibility
        if self.seed:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
        
        # Step 3: Resolve compute device preference and availability
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

    def _initialize_params(self, Y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize the parameters LL and FF using SVD on the log-transformed data.

        Parameters
        ----------
        Y : torch.Tensor
            Input data matrix of shape (n_samples, n_features).

        Returns
        -------
        LL : torch.Tensor
            Initialized left singular vectors of shape (n_pcs, n_samples).
        FF : torch.Tensor
            Initialized right singular vectors of shape (n_pcs, n_features).
        """
        # Step 1: Seed for deterministic initialization
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        # Step 2: Announce initialization
        if self.verbose:
            print("Initializing parameters using SVD...")

        # Step 3: Compute SVD on log1p(Y), using sparse path when needed
        if hasattr(self, "is_sparse") and self.is_sparse:
            Y_scipy = sp.coo_matrix(
                (Y._values().cpu().numpy(), (Y._indices()[0].cpu().numpy(), Y._indices()[1].cpu().numpy())),
                shape=Y.shape
            )
            Y_log1p_sparse = Y_scipy.log1p()
            U, s, Vh = svds(Y_log1p_sparse, k=self.n_pcs)
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

            U_k = U[:, :self.n_pcs]
            V_k = V[:, :self.n_pcs]
            d_k = S[:self.n_pcs]

        # Step 4: Form initial LL, FF from truncated SVD
        LL = (U_k * torch.sqrt(d_k)).T
        FF = (V_k * torch.sqrt(d_k)).T
        
        return LL.to(self.device), FF.to(self.device)

    def _poisson_log_likelihood(
        self,
        Y: torch.Tensor,
        LL: torch.Tensor,
        FF: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the Poisson log-likelihood for the given data and parameters.

        Parameters
        ----------
        Y : torch.Tensor
            Input data matrix of shape (n_samples, n_features).
        LL : torch.Tensor
            Left singular vectors of shape (n_pcs, n_samples).
        FF : torch.Tensor
            Right singular vectors of shape (n_pcs, n_features).

        Returns
        -------
        log_likelihood : torch.Tensor
            Poisson log-likelihood value.
        """
        # Step 1: Set numerical clamps and dimensions
        clamp_min, clamp_max = -20, 20
        n, m = Y.shape

        # Step 2: Evaluate term1 on non-zeros and term2 via batched exp for sparse
        if hasattr(self, "is_sparse") and self.is_sparse:
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
                LL_b = LL[:, start:end]  # K x b
                Z = LL_b.T @ FF
                Z = Z + self.row_offset[start:end].unsqueeze(1) + self.col_offset.unsqueeze(0)
                Z = torch.clamp(Z, clamp_min, clamp_max)
                total = total + torch.exp(Z).sum()
            term2 = total
        else:
            # Step 3: Dense path computes full log_lambda but avoids forming Lambda
            log_lambda_hat = LL.T @ FF
            log_lambda = log_lambda_hat + self.row_offset.view(-1, 1) + self.col_offset.view(1, -1)
            log_lambda = torch.clamp(log_lambda, clamp_min, clamp_max)
            term1 = (Y * log_lambda).sum()
            term2 = torch.exp(log_lambda).sum()

        return term1 - term2

    def _line_search_update(
        self,
        Y: torch.Tensor,
        LL: torch.Tensor,
        FF: torch.Tensor,
        k: int,
        direction: torch.Tensor,
        update_target: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform backtracking line search for updating either LL[k, :] or FF[k, :].

        Parameters
        ----------
        Y : torch.Tensor
            Input data matrix.
        LL : torch.Tensor
            Current left factor [K, n].
        FF : torch.Tensor
            Current right factor [K, m].
        k : int
            Component index to update.
        direction : torch.Tensor
            Proposed Newton direction for the parameter block.
        update_target : str
            Either "LL" or "FF" to indicate which factor to update.

        Returns
        -------
        LL, FF, loglik : tuple
            Updated factors and the resulting log-likelihood.
        """
        # Step 1: Compute current log-likelihood and set initial step size
        alpha = self.learning_rate
        prev_loglik = self._poisson_log_likelihood(Y, LL, FF)
        # Step 2: Apply single-step update if line search is disabled
        if not self.line_search:
            if update_target == "LL":
                LL[k, :] = LL[k, :] - (alpha * direction)
            else:
                FF[k, :] = FF[k, :] - (alpha * direction)
            new_loglik = self._poisson_log_likelihood(Y, LL, FF)
            return LL, FF, new_loglik

        # Step 3: Backtracking loop to find non-decreasing log-likelihood
        ls_steps = 0
        while ls_steps < self.ls_max_steps:
            candLL = LL.clone()
            candFF = FF.clone()
            if update_target == "LL":
                candLL[k, :] = candLL[k, :] - alpha * direction
            else:
                candFF[k, :] = candFF[k, :] - alpha * direction
            cand_loglik = self._poisson_log_likelihood(Y, candLL, candFF)
            if torch.isnan(cand_loglik):
                alpha = alpha * self.ls_beta
                ls_steps += 1
                continue
            if cand_loglik >= prev_loglik:
                LL, FF = candLL, candFF
                return LL, FF, cand_loglik
            alpha = alpha * self.ls_beta
            ls_steps += 1

        # Step 4: Fallback to last reduced step size if no improvement found
        if update_target == "LL":
            LL[k, :] = LL[k, :] - (alpha * direction)
        else:
            FF[k, :] = FF[k, :] - (alpha * direction)
        new_loglik = self._poisson_log_likelihood(Y, LL, FF)
        return LL, FF, new_loglik

    def fit(
        self,
        Y,
        learning_rate: float = 0.5,
        line_search: bool = True,
        batch_size_rows: int | None = None,
        batch_size_cols: int | None = None,
    ) -> PoissonGLMPCA:
        """
        Fit the Poisson GLM-PCA model to the input data. Newton's method is used for optimization, with
        block coordinate descent for updating the parameters. See more in [Weine et al., Bioinformatics, 2024].

        Parameters
        ----------
        Y : array-like, torch.Tensor, or scipy.sparse matrix
            Input data matrix of shape (n_samples, n_features). Can be dense or sparse.
        learning_rate : float, optional
            Learning rate for the optimization algorithm. Default is 0.5.
            
        Returns
        -------
        self : PoissonGLMPCA
            The fitted model.
            
        Attributes
        ----------
        loglik_history_ : list
            History of log-likelihood values during training.
        """
        # Step 1: Record configuration overrides for this fit
        self.learning_rate = learning_rate
        self.line_search = bool(line_search)

        if batch_size_rows is not None:
            self.batch_size_rows = batch_size_rows
        if batch_size_cols is not None:
            self.batch_size_cols = batch_size_cols

        # Step 2: Normalize input and set sparse/dense path
        self.is_sparse = False
        self.loglik_history_ = []
        
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        if sp.issparse(Y):
            self.is_sparse = True
            if self.verbose:
                print("Detected sparse input matrix. Using sparse optimizations.")
            if isinstance(Y, sp.coo_matrix):
                coo = Y
            else:
                coo = Y.tocoo()
                
            indices = torch.LongTensor(np.vstack((coo.row, coo.col)))
            values = torch.FloatTensor(coo.data)
            shape = torch.Size(coo.shape)
            Y_sparse = torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)
            try:
                Y_sparse = Y_sparse.to(self.device)
            except NotImplementedError:
                warnings.warn("Sparse tensor conversion to device not supported. Using CPU instead. If you want to use MPS / CUDA, please convert the input matrix to dense format.")
                self.device = "cpu"
                Y_sparse = Y_sparse.to(self.device)
            Y = Y_sparse
        elif not isinstance(Y, torch.Tensor):
            Y = torch.tensor(Y, dtype=torch.float32)
            Y = Y.to(self.device)

        # Step 3: Compute row/col offsets without forming dense N x M
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
            if self.col_size_factor:
                sum_col_means = col_means.sum()
            else:
                if self.is_sparse:
                    sum_col_means = torch.sparse.sum(Y, dim=0).to_dense().sum() / Y.shape[0]
                else:
                    sum_col_means = Y.mean(dim=0).sum()
            
            if self.is_sparse:
                row_sums = torch.sparse.sum(Y, dim=1).to_dense()
            else:
                row_sums = Y.sum(dim=1)
            row_offset = torch.log(row_sums / (sum_col_means + epsilon) + epsilon)
            self.row_offset = self.row_offset + row_offset

        # Step 4: Initialize LL, FF via SVD and evaluate initial likelihood
        n, m = Y.shape
        if self.verbose:
            print(f"Fitting GLM-PCA to {n} samples and {m} features on device: '{self.device}'")
            if self.is_sparse:
                density = Y_sparse._nnz() / (n * m)
                print(f"Sparse matrix density: {density:.4f}")
    
        LL, FF = self._initialize_params(Y)
        loglik = self._poisson_log_likelihood(Y, LL, FF)
        self.loglik_history_.append(loglik.item())
        
        if self.verbose:
            print(f"Initial Log-Likelihood: {loglik:.4f}")

        if self.progress_bar:
            iterator = tqdm(range(self.max_iter), desc="GLM-PCA Iterations")
        else:
            iterator = range(self.max_iter)

        # Step 5: Optimization loop with block coordinate Newton updates
        for i in iterator:
            n, m = Y.shape
            clamp_min, clamp_max = -20, 20
            for k in range(self.n_pcs):
                if self.is_sparse:
                    y_ff = torch.sparse.mm(Y, FF[k, :].unsqueeze(1)).squeeze()
                else:
                    y_ff = (Y @ FF[k, :])

                exp_grad = torch.zeros(n, device=self.device)
                hess_diag = torch.zeros(n, device=self.device)
                ffk = FF[k, :]
                ffk_sq = ffk * ffk

                bsz = self.batch_size_rows or max(1, min(n, 1024))
                for start in range(0, n, bsz):
                    end = min(n, start + bsz)
                    LL_b = LL[:, start:end]
                    Z = LL_b.T @ FF
                    Z = Z + self.row_offset[start:end].unsqueeze(1) + self.col_offset.unsqueeze(0)
                    Z = torch.clamp(Z, clamp_min, clamp_max)
                    Lambda_b = torch.exp(Z)
                    exp_grad[start:end] = Lambda_b @ ffk
                    hess_diag[start:end] = -(Lambda_b @ ffk_sq)

                hess_diag = -torch.clamp(-hess_diag, min=1e-8)
                grad_k = y_ff - exp_grad
                direction = grad_k / hess_diag

                LL, FF, loglik = self._line_search_update(Y, LL, FF, k, direction, update_target='LL')

            for k in range(self.n_pcs):
                if self.is_sparse:
                    yT_ll = torch.sparse.mm(Y.transpose(0, 1), LL[k, :].unsqueeze(1)).squeeze()
                else:
                    yT_ll = (Y.T @ LL[k, :])

                exp_grad = torch.zeros(m, device=self.device)
                hess_diag = torch.zeros(m, device=self.device)
                llk = LL[k, :]
                llk_sq = llk * llk

                bszc = self.batch_size_cols or max(1, min(m, 1024))
                for start in range(0, m, bszc):
                    end = min(m, start + bszc)
                    FF_b = FF[:, start:end]
                    Z = LL.T @ FF_b
                    Z = Z + self.row_offset.unsqueeze(1) + self.col_offset[start:end].unsqueeze(0)
                    Z = torch.clamp(Z, clamp_min, clamp_max)
                    Lambda_b = torch.exp(Z)
                    exp_grad[start:end] = Lambda_b.T @ llk
                    hess_diag[start:end] = -(Lambda_b.T @ llk_sq)

                hess_diag = -torch.clamp(-hess_diag, min=1e-8)
                grad_k = yT_ll - exp_grad
                direction = grad_k / hess_diag

                LL, FF, loglik = self._line_search_update(Y, LL, FF, k, direction, update_target='FF')

            # Step 5.1: Check convergence by relative change in log-likelihood
            prev_loglik = loglik
            loglik = self._poisson_log_likelihood(Y, LL, FF)
            self.loglik_history_.append(loglik.item())

            if torch.isnan(loglik):
                warnings.warn(f"\nLog-likelihood is NaN at iteration {i+1}. Stopping.")
                break

            delta = abs((loglik - prev_loglik) / (abs(prev_loglik) + 1e-6))

            if self.verbose:
                print(f"Iter {i+1:3d} | Log-Likelihood: {loglik:.4f} | Change: {delta:.2e}")

            if delta < self.tol:
                if self.verbose or self.progress_bar:
                    print(f"\nConvergence reached after {i+1} iterations.")
                break
        
        # Step 6: Warn if maximum iterations reached
        if i == self.max_iter - 1 and self.verbose:
            warnings.warn(f"\nMaximum iterations ({self.max_iter}) reached without convergence.")
            
        # Step 7: Finalize orthonormal factors via QR + small SVD
        self._finalize_factors(LL, FF)

    def _finalize_factors(self, LL: torch.Tensor, FF: torch.Tensor) -> None:
        """
        Compute final orthogonal factors U, V and singular values d without forming dense n x m.

        This method performs QR decompositions of LL^T (n x K) and FF^T (m x K) to compute U, V, and d.

        Parameters
        ----------
        LL : torch.Tensor
            Left singular vectors of shape (n_samples, n_pcs).
        FF : torch.Tensor
            Right singular vectors of shape (n_features, n_pcs).
            
        Returns
        -------
        None
            The computed orthogonal factors U, V and singular values d are stored in the model attributes.
        """
        # Step 1: QR decompositions of LL^T and FF^T (n x K, m x K)
        QL, RL = torch.linalg.qr(LL.T, mode="reduced")
        QF, RF = torch.linalg.qr(FF.T, mode="reduced")

        # Step 2: Small SVD on K x K core matrix
        M = RL @ RF.T
        Us, s, Vsh = torch.linalg.svd(M, full_matrices=False)

        # Step 3: Compose final U, V from Q factors and SVD components
        U = QL @ Us
        V = QF @ Vsh.T

        self.U = U[:, :self.n_pcs].detach().cpu().numpy()
        self.d = s[:self.n_pcs].detach().cpu().numpy()
        self.V = V[:, :self.n_pcs].detach().cpu().numpy()