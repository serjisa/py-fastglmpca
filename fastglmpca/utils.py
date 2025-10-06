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
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        if self.verbose:
            print("Initializing parameters using SVD...")

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
        log_lambda = LL.T @ FF + self.offset
        log_lambda = torch.clamp(log_lambda, -20, 20)
        
        if hasattr(self, "is_sparse") and self.is_sparse:
            non_zero_indices = Y._indices()
            non_zero_values = Y._values()
            
            log_lambda_at_non_zero = log_lambda[non_zero_indices[0], non_zero_indices[1]]
            term1 = torch.sum(non_zero_values * log_lambda_at_non_zero)
        else:
            term1 = (Y * log_lambda).sum()
            
        term2 = torch.exp(log_lambda).sum()
        
        return term1 - term2

    def fit(self, Y, learning_rate: float = 0.5) -> PoissonGLMPCA:
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
        self.learning_rate = learning_rate
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

        self.offset = torch.zeros(Y.shape, device=self.device)
        epsilon = 1e-8

        if self.col_size_factor:
            if self.is_sparse:
                col_means = torch.sparse.sum(Y, dim=0).to_dense() / Y.shape[0]
            else:
                col_means = Y.mean(dim=0)
            col_offset = torch.log(col_means + epsilon)
            self.offset += col_offset

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
            self.offset += row_offset.view(-1, 1)

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

        for i in iterator:
            log_lambda = LL.T @ FF + self.offset
            Lambda = torch.exp(torch.clamp(log_lambda, -20, 20)) 
            
            for k in range(self.n_pcs):
                if self.is_sparse:
                    grad_k = torch.sparse.mm(Y, FF[k, :].unsqueeze(1)).squeeze() - Lambda @ FF[k, :]
                else:
                    grad_k = (Y - Lambda) @ FF[k, :]
                    
                hess_diag_k = -torch.clamp((Lambda * (FF[k, :]**2)).sum(axis=1), min=1e-8)
                step = learning_rate * (grad_k / hess_diag_k)
                LL[k, :] = LL[k, :] - step
                
                log_lambda_update = torch.outer(step, FF[k, :])
                Lambda = torch.exp(torch.clamp(log_lambda - log_lambda_update, -20, 20))
                log_lambda = log_lambda - log_lambda_update
    
            log_lambda = LL.T @ FF + self.offset
            Lambda = torch.exp(torch.clamp(log_lambda, -20, 20))
    
            for k in range(self.n_pcs):
                if self.is_sparse:
                    Y_t_Lk = torch.sparse.mm(Y.transpose(0, 1), LL[k, :].unsqueeze(1)).squeeze()
                    grad_k = Y_t_Lk - Lambda.T @ LL[k, :]
                else:
                    grad_k = (Y.T - Lambda.T) @ LL[k, :]
                    
                hess_diag_k = -torch.clamp((Lambda.T * (LL[k, :]**2)).sum(axis=1), min=1e-8)
                step = learning_rate * (grad_k / hess_diag_k)
                FF[k, :] = FF[k, :] - step
    
                log_lambda_update = torch.outer(LL[k, :], step)
                Lambda = torch.exp(torch.clamp(log_lambda - log_lambda_update, -20, 20))
                log_lambda = log_lambda - log_lambda_update
    
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
        
        if i == self.max_iter - 1 and self.verbose:
            warnings.warn(f"\nMaximum iterations ({self.max_iter}) reached without convergence.")
            
        self._finalize_factors(LL, FF)

    def _finalize_factors(self, LL: torch.Tensor, FF: torch.Tensor) -> None:
        """
        Decompose optimized LL and FF into orthogonal U, V and diagonal d.

        Parameters
        ----------
        LL : torch.Tensor
            Left singular vectors of shape (n_pcs, n_samples).
        FF : torch.Tensor
            Right singular vectors of shape (n_pcs, n_features).
        """
        log_lambda_hat = LL.T @ FF
        
        U, d, Vh = torch.linalg.svd(log_lambda_hat, full_matrices=False)
        
        self.U = U[:, :self.n_pcs].cpu().numpy()
        self.d = d[:self.n_pcs].cpu().numpy()
        self.V = Vh.T[:, :self.n_pcs].cpu().numpy()