import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
from einops import einsum, rearrange
from tqdm import tqdm
import gc
import contextlib, datetime
from scipy.stats import beta
import wandb

# ────────────────────────── DEBUG UTILITIES ────────────────────────────
def _stat(t):
    """Return basic stats of a tensor as a dict."""
    return dict(min=float(t.min()), max=float(t.max()), mean=float(t.mean()), std=float(t.std()))


def _fmt(d):
    return ("min={min:+.2e} max={max:+.2e} mean={mean:+.2e} std={std:+.2e}").format(**d)


def _register_debug_hooks(module: nn.Module, log_every: int = 1):
    """Attach gradient-logging hooks to all parameters in *module*.

    Prints stats every `log_every` backward passes. The owning module must
    maintain an integer attribute `_debug_step` that increments each
    iteration (see integration in `run_training`).
    """

    for name, p in module.named_parameters():
        def _make_hook(pname=name, param=p):
            def _hook(grad):
                step = getattr(module, "_debug_step", 0)
                if step % log_every == 0:
                    vstats = _fmt(_stat(param.data))
                    gstats = _fmt(_stat(grad))
                    finite = torch.isfinite(grad).all().item()
                    print(f"[dbg {step:03d}] {pname:<20} | param {vstats} | grad {gstats} | finite={finite}")
            return _hook

        p.register_hook(_make_hook())
# ───────────────────────────────────────────────────────────────────────

class BetaMask(torch.nn.Module):
    """
    Handles the marginal Beta-based distribution for each neuron's mask.
    """
    def __init__(self, 
                 num_neurons, 
                 lambda_e, 
                 lambda_beta, 
                 lambda_sim,
                 stretch_left=1e-5,
                 stretch_right=(1 - 1e-5),
                 eps = 1e-6,
                 seq_len=None,
                 device="cpu"):
        super().__init__()

        self.device = device
        # Handle token-specific case
        self.seq_len = seq_len
        if seq_len is not None:
            # If we have token-specific masks, the total number of parameters increases
            self.total_params = num_neurons * seq_len
            self.reshape_dims = (seq_len, num_neurons)
        else:
            self.total_params = num_neurons
            self.reshape_dims = (num_neurons,)
    
        
        self.num_neurons = num_neurons
        self.lambda_e    = lambda_e
        self.lambda_beta = lambda_beta
        self.lambda_sim  = lambda_sim
        self.left = stretch_left
        self.right = stretch_right
        if self.left < 0 and self.right < 1:
            raise ValueError("Must have stretch_left < 0 and stretch_right > 1 or stretch_left > 0 and " +
                             "stretch_right  < 1, have stretch_left = " +
                             f"{stretch_left}, stretch_right = {stretch_right}")
        if self.left < 0:
            self.stretch_mode = 'DELTA_MASS'
        else:
            self.stretch_mode = 'GRADIENT'
        self.eps = eps
        
        # Determine parameter shape based on seq_len
        if seq_len is not None:
            # If we have token-specific masks, include seq_len dimension
            self.param_shape = (seq_len, num_neurons)
        else:
            # Otherwise, just use num_neurons
            self.param_shape = (num_neurons,)
        
        # Initialize parameters with proper shape
        self.alpha_raw = nn.Parameter(
            self.softplus_inv(
            torch.clamp(
                torch.randn(*self.param_shape, device=self.device) * 0.05 + lambda_e / 2, 
                self.eps, lambda_e - self.eps
                )
            )
        )
        
        self.beta_raw = nn.Parameter(
            self.softplus_inv(
            torch.clamp(
                torch.randn(*self.param_shape, device=self.device) * 0.05 + lambda_e / 2, 
                self.eps, lambda_e - self.eps
                )
            )
        )
    
    
    def softplus_inv(self, y):
        # works down to y ≈ 1e-6 without overflow
        return torch.where(
            y < 20,
            torch.log(torch.expm1(y)),        # stable for small y
            y + torch.log1p(-torch.exp(-y))   # stable for large y
        )
    

    def alpha_o(self):
        return F.softplus(self.alpha_raw) + self.eps

    def beta_o(self):
        return F.softplus(self.beta_raw) + self.eps


    def forward(self, mode="sample"):
        """
        Just a convenience method. 
        If mode == "sample", do sampling. If mode == "mle", do the MLE approach.
        """
        if mode == "sample":
            return self.sample_mask()
        else:
            return self.get_mask_mle()
    
    
    def get_effective_params(self):
        """
        Calculate effective alpha and beta parameters from optimization parameters.
        
        Returns:
            Tuple of (alpha_e, beta_e) with same shape as self.alpha_o and self.beta_o
        """
        if not hasattr(self, "alpha_o"):
            print("alpha_o missing   *inside get_effective_params*")
            print("current _buffers  :", list(self._buffers.keys())[:5])
            print("current _parameters:", list(self._parameters.keys())[:5])

        alpha_e = self.lambda_e * self.alpha_o() / (self.alpha_o() + self.beta_o())
        beta_e  = self.lambda_e * self.beta_o() / (self.alpha_o() + self.beta_o())
        
        alpha_e = torch.clamp(alpha_e, min=self.eps)
        beta_e = torch.clamp(beta_e, min=self.eps)
        
        return alpha_e, beta_e
    
    
    def sample_mask(self):
        """
        Reparameterized Beta sampling for each neuron (in [0,1]).
        alpha_e = lambda_e * alpha_o / (alpha_o + beta_o)
        beta_e  = lambda_e * beta_o / (alpha_o + beta_o)
        We can stretch and clamp to avoid infite peaks at 0/1, or to create delta masses at 0/1
        """
        dist = self.beta_dist()
        if self.batch_size is None:
            # Single sample
            z = dist.rsample()  # Shape matches alpha_e, beta_e
        else:
            # Batch of samples
            z = dist.rsample((self.batch_size,))

        return self.stretch_and_clamp_z(z)
    
    
    # Just returns a Beta that will return a flat tensor. So need to deal with any shaping for batch or tokens
    # downstream of any call to this
    def beta_dist(self):
        alpha_e, beta_e = self.get_effective_params()
        return torch.distributions.Beta(alpha_e, beta_e)
    
    
    # Note this will be used to compute the penalty/loss on the mask variable values, while the mask variables
    # themselves are used to compute the loss in terms of faithfulness
    def beta_log_density(self, z):
        """
        Compute log Beta distribution PDF
        
        Args:
            z: Input mask variable tensor
            
        Returns:
            Log PDF values for input z (same shape as input)
        """
        alpha_e, beta_e = self.get_effective_params()
        dist = torch.distributions.Beta(alpha_e, beta_e)
        
        # Handle proper broadcasting for different input shapes
        return dist.log_prob(z)
    
    
    def beta_density(self, z):
        """
        Compute log Beta distribution PDF
        
        Args:
            z: Input mask variable tensor
        Returns:
            PDF values for input z
        """
        alpha_e, beta_e = self.get_effective_params()
        dist = torch.distributions.Beta(alpha_e, beta_e)
        
        # Handle proper broadcasting for different input shapes
        return dist.prob(z)
    
    
    def stretch_and_clamp_z(self, z):
        """
        Stretch distribution beyond [0,1] and clamp to delta masses at 0 and 1
        
        Args:
            z: Input tensor from Beta distribution
        Returns:
            Stretched and clamped values
        """
        z_stretched = z * (self.right - self.left) + self.left
        if self.stretch_mode == 'DELTA_MASS':
            # Bring excess density below 0 and above 1 to delta masses at 0 and 1
            return torch.clamp(x_stretched, 0.0, 1.0)
        else: # self.stretch_mode == 'GRADIENT':
            return z_stretched
            
    
    # Do we want to have a mode arg where you either return the mix or just always return the mean?
    def get_mask_mle(self, batch_size=None):
        """
        Return mask values as the MLE or mean from the "posterior" Beta distribution. For neurons
        where alpha > 1 and beta > 1, return the MLE. When alpha < 1 or beta < 1, return the mean
        """
        alpha_e, beta_e = self.get_effective_params()  # each of length self.total_parameters
        # Default is mean, not MLE, for use whe alpha <= 1 or beta <= 1
        mle = alpha_e / (alpha_e + beta_e)
        mode_condition = (alpha_e > 1) & (beta_e > 1)
        mode_values = (alpha_e - 1) / (alpha_e + beta_e - 2)
        mle = torch.where(mode_condition, mode_values, mle)
        mle = self.stretch_and_clamp_z(mle)

        # Handle batch dimension if needed
        if batch_size is not None:
            if self.seq_len is None:
                # Expand to (batch_size, num_neurons)
                mle = mle.unsqueeze(0).expand(batch_size, -1)
            else:
                # Expand to (batch_size, seq_len, num_neurons)
                mle = mle.unsqueeze(0).expand(batch_size, -1, -1)
        
        return mle
    
    
    def complexity_loss(self, z):
        """
        -log Beta related 'posterior' + penalty for alpha_e and beta_e being too close together

        Args:
            z: Input tensor with appropriate dimensions

        Returns:
            Scalar loss: Mean over tokens and/or batch to maintain scale and not require different
                         lambda parameters to be learned
        """
        log_density = self.beta_log_density(z)

        batch_size = None
        # Sum over all dimensions except batch
        if z.dim() == 1:  # (neurons,)
            log_loss = -torch.sum(log_density)
        else: # z.dim() > 1:
            # (seq, neurons) or (batch, neurons) or (batch, seq, neurons)
            log_loss = -torch.sum(log_density, dim=-1).mean()

        # Calculate penalty for Beta parameters being too close together, not pushing enough of the density
        # towards zero and one
        a_e, b_e = self.get_effective_params()
        penalty = torch.where(a_e < (self.lambda_e / 2), a_e, self.lambda_e - a_e)
        return log_loss + self.lambda_e * penalty.sum()
    
    
    def compute_expected_threshold_sparsity(self, batch_size=None, threshold=0.5):
        """
        Compute expect threshold sparsity (mask is < threshold)

        Args:
            threshold = 0.5: Below this z counts as 0
        Returns:
            Expected threshold sparsity value (scalar or per-batch)
        """
        z = self.get_mask_mle(batch_size)
        if len(z.shape) == 1:  # No batch dimension
            return torch.mean(z < threshold)
        elif len(z.shape) == 2:  # Either (batch, neurons) or (seq, neurons)
            if self.seq_len is not None:
                # This is (seq, neurons)
                return torch.mean(z < threshold)
            else:
                # This is (batch, neurons)
                return torch.mean(z < threshold, dim=1)
        else:  # (batch, seq, neurons)
            return torch.mean(z < threshold, dim=2)
    
    
    def compute_expected_exact_sparsity(self, batch_size=None):
        """
        Compute exprected sparsity (mask is exactly zero)

        Args:
            alpha_o: Learned alpha parameter tensor
            beta_o: Learned beta parameter tensor
        Returns:
            Expected sparsity value (scalar or per-batch)
        """
        z = self.get_mask_mle(batch_size)
        if len(z.shape) == 1:  # No batch dimension
            return torch.mean(z == 0)
        elif len(z.shape) == 2:  # Either (batch, neurons) or (seq, neurons)
            if self.seq_len is not None:
                # This is (seq, neurons)
                return torch.mean(z == 0)
            else:
                # This is (batch, neurons)
                return torch.mean(z == 0, dim=1)
        else:  # (batch, seq, neurons)
            return torch.mean(z == 0, dim=2)
    
    
    def compute_threshold_sparsity(self, z, threshold=0.5):
        """
        Fraction of neurons with z < threshold

        Returns scalar if z has no batch dim, tensor of shape (batch_size,) otherwise
        """
        if len(z.shape) == 1:  # No batch dimension
            return torch.mean(z < threshold)
        elif len(z.shape) == 2:  # Either (batch, neurons) or (seq, neurons)
            if self.seq_len is not None:
                # This is (seq, neurons)
                return torch.mean(z < threshold)
            else:
                # This is (batch, neurons)
                return torch.mean(z < threshold, dim=1)
        else:  # (batch, seq, neurons)
            return torch.mean(z < threshold, dim=2)
    


    def compute_exact_sparsity(self, z):
        """
        Fraction of neurons with z == 0

        Returns scalar if z is 1D, tensor of shape (batch_size,) if z is 2D
        """
        if len(z.shape) == 1:  # No batch dimension
            return torch.mean(z == 0)
        elif len(z.shape) == 2:  # Either (batch, neurons) or (seq, neurons)
            if self.seq_len is not None:
                # This is (seq, neurons)
                return torch.mean(z == 0)
            else:
                # This is (batch, neurons)
                return torch.mean(z == 0, dim=1)
        else:  # (batch, seq, neurons)
            return torch.mean(z == 0, dim=2)




class GaussianCopulaMask(BetaMask):
    """
    Extends Beta marginals with correlation matrix using a Gaussian copula approach.
    Uses low-rank representation of the covariance matrix as (εI_n + QQ^T) 
    where Q is of shape (n_neurons, k) with k << n_neurons.
    """
    def __init__(self,
                 num_neurons,
                 lambda_e,
                 lambda_beta, 
                 lambda_sim,
                 rank_k=100,
                 epsilon=1e-4,     # Small constant for εI_n term
                 lambda_diag=0.1,  # Weight for diagonal penalty
                 lambda_Q=0.01,    # Weight for Q sparsity penalty
                 by_token_Q=False, # Whether to use token-specific Q matrices
                 stretch_left=1e-5,
                 stretch_right=(1 - 1e-5),
                 seq_len=None,
                 batch_size=None,
                 device="cpu",
                 binary_threshold=0.5,
                 dtype=torch.bfloat16):
        super().__init__(num_neurons=num_neurons, lambda_e=lambda_e, lambda_beta=lambda_beta, lambda_sim=lambda_sim,
                         stretch_left=stretch_left, stretch_right=stretch_right,
                         eps=epsilon, seq_len=seq_len, device=device)
        
        self.rank_k = rank_k
        # Lots of eps versions so don't recalculate bazillion times each iteration
        self.epsilon = epsilon                  # For εI_n
        self.inv_epsilon = 1 / epsilon          # For εI_n
        self.sqrt_epsilon = math.sqrt(epsilon)  # For εI_n
        self.epsilon_sqrd = epsilon ** 2        # For εI_n
        self.inv_epsilon_sqrd = 1 / (epsilon ** 2)        # For εI_n
        self.lambda_diag = lambda_diag
        self.lambda_Q = lambda_Q
        self.by_token_Q = by_token_Q  # Whether Q varies by token position
        self.batch_size = batch_size
        self.mean_ablations = None
        self.binary_threshold = binary_threshold
        self.dtype = dtype
        
        # Initialize Q with appropriate shape based on by_token_Q
        if self.by_token_Q and self.seq_len is not None:
            # Token-specific Q: (seq_len, num_neurons, rank_k)
            self.Q = nn.Parameter(self.create_initial_q_matrix(
                num_neurons, rank_k, token_dim=self.seq_len))
        elif self.by_token_Q and self.seq_len is None:
            raise ValueError("Can't set by_token_Q = True and seq_len = None")
        else:
            # Global Q: (num_neurons, rank_k)
            self.Q = nn.Parameter(self.create_initial_q_matrix(num_neurons, rank_k))
        
        # Buffers to save memory in each forward pass. Sizes fall back to 1 when batch_size / seq_len is None
        c_shape = (self.batch_size or 1, self.seq_len or 1, self.num_neurons)
        y_shape = (self.batch_size or 1, self.seq_len or 1, self.rank_k)
        self.register_buffer("_c_buf", torch.empty(*c_shape, dtype=self.dtype, device=self.Q.device))
        self.register_buffer("_y_buf", torch.empty(*y_shape, dtype=self.dtype, device=self.Q.device))

        # For use in complexity_loss
        self.epsilon_term = -self.num_neurons * torch.log(torch.tensor(self.epsilon, device=self.Q.device))

    def create_initial_q_matrix(self, num_neurons, rank_k, token_dim=None, dtype=torch.bfloat16):
        """
        Initialize Q matrix to approximate variance of 1 for each neuron mask
        
        Args:
            num_neurons: Number of neurons
            rank_k: Rank of the low-rank approximation
            token_dim: If not None, create a token-specific Q of shape (token_dim, num_neurons, rank_k)
            dtype: Data type for the Q matrix
            
        Returns:
            Q matrix with appropriate shape
        """
        if token_dim is None:
            q_dims = (num_neurons, rank_k)
        else:
            q_dims = (token_dim, num_neurons, rank_k)
            
        # Create token-specific Q matrices: (token_dim, num_neurons, rank_k)
        Q = torch.randn(*q_dims, device=self.device, dtype=dtype)
        row_norms = Q.norm(dim=-1, keepdim=True)  # Compute L2 norm for each row
        Q = Q / row_norms  # Normalize so each row has norm 1
        return Q
    
    
    def forward(self, mode="sample", batch_size=None):
        """
        Forward pass samples the mask or returns MLE estimate.
        
        Args:
            mode: "sample" for random sampling, "mle" for maximum likelihood estimate
            batch_size: Optional batch size for sampling
            
        Returns:
            The mask tensor
        """
        if mode == "sample":
            return self.sample_mask()
        else:
            return self.get_mask_mle()
    
    def _log_memory(self, stage):
        """Helper function to log memory usage at different stages"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            print(f"[{stage}] VRAM - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
            return allocated, reserved
        return 0, 0
    
    
    def sample_mask(self, cache_masks=False, verbose=False):
        """
        Sample correlated Gaussian variables with covariance (εI_n + QQ^T),
        then transform to Beta variables via copula.
        
        Implementation follows the formula:
        z = √ε c + Qy, where c,y ~ Normal(0,1)
        
        Args:
            batch_size: Optional batch size (None = single mask, int = batch of masks)
        
        Returns:
            Mask samples with appropriate shape
        """
        # ------------------------------------------------------------------
        # 1.  Refill the permanent random-number buffers without autograd
        # ------------------------------------------------------------------
        B = self.batch_size or 1
        T = self.seq_len   or 1

        with torch.no_grad():                         # no tape here
            # views – zero alloc, unneeded unless batch size or seq len change per call
            c = self._c_buf[:B, :T]                   # (B, T, N)
            y = self._y_buf[:B, :T]                   # (B, T, k)
            c.normal_()
            y.normal_()

            # partial build of z:  √ε · c   (doesn't involve trainable params)
            z = self.sqrt_epsilon * c                # bf16, same buffer as c

        # ------------------------------------------------------------------
        # 2.  Add the Q-dependent term **with autograd enabled**
        # ------------------------------------------------------------------
        if not self.by_token_Q:                      # Q : (N, k)
            z_shape = z.shape[:-1]                   # All dims but the last (num_neurons). (B, T) currently
            z = ( z.reshape(-1, self.num_neurons) +
                  y.reshape(-1, self.rank_k) @ self.Q.T
                ).reshape(*z_shape, self.num_neurons)
        else:                                        # Q : (T, N, k)
            # z, y :  (B, T, …)  or  (T, …) when batch_size is None
            z = z + torch.einsum("t n k, ... t k -> ... t n", self.Q, y)

        # ------------------------------------------------------------------
        # 3.  Φ  (std-normal CDF)  → uniform, **in-place**
        # ------------------------------------------------------------------
        z.div_(math.sqrt(2)).erf_().mul_(0.5).add_(0.5)   # z now holds u
        u = z                                             # alias for clarity

        if verbose:
            self._log_memory("Created u")

        # ------------------------------------------------------------------
        # 4.  Beta ICDF   (keeps grad for alpha_o / beta_o)
        # ------------------------------------------------------------------
        alpha_e, beta_e = self.get_effective_params()     # on the tape
        if verbose:
            self._log_memory("Got effective params")
        samples = ( self.fast_icdf(u, alpha_e)
                    if hasattr(self, "fast_icdf")
                    else torch.distributions.Beta(alpha_e, beta_e).icdf(u)
                )
        if verbose:
            self._log_memory("Got samples")

        # ------------------------------------------------------------------
        # 5.  Stretch / clamp  +  optional cache
        # ------------------------------------------------------------------
        samples = self.stretch_and_clamp_z(samples)
        if verbose:
            self._log_memory("Stretched and clamped")
        if cache_masks:
            self.current_mask = samples
        return samples


    def sample_mask_hi_mem(self, cache_masks=False, verbose=False):
        """
        Sample correlated Gaussian variables with covariance (εI_n + QQ^T),
        then transform to Beta variables via copula.
        
        Implementation follows the formula:
        z = √ε c + Qy, where c,y ~ Normal(0,1)
        
        Args:
            batch_size: Optional batch size (None = single mask, int = batch of masks)
        
        Returns:
            Mask samples with appropriate shape
        """
        # Force cleanup before starting
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

        # Handle different cases based on batch_size, seq_len, and by_token_Q
        if verbose:
            self._log_memory("Before creating c, y, and z_correlated")
        if not self.by_token_Q:
            # Q is (num_neurons, rank_k)
            if self.batch_size is None and self.seq_len is None:
                # z shape (num_neurons,)
                c = torch.randn(self.num_neurons, device=self.Q.device)
                y = torch.randn(self.rank_k, device=self.Q.device)
                z_correlated = self.sqrt_epsilon * c + self.Q @ y
            elif self.batch_size is None and self.seq_len is not None:
                # z shape (seq_len, num_neurons)
                c = torch.randn(self.seq_len, self.num_neurons, device=self.Q.device)
                y = torch.randn(self.seq_len, self.rank_k, device=self.Q.device)
                z_correlated = self.sqrt_epsilon * c + y @ self.Q.T
            elif self.batch_size is not None and self.seq_len is None:
                # z shape (self.batch_size, num_neurons)
                c = torch.randn(self.batch_size, self.num_neurons, device=self.Q.device)
                y = torch.randn(self.batch_size, self.rank_k, device=self.Q.device)
                z_correlated = self.sqrt_epsilon * c + y @ self.Q.T
            else: # self.batch_size is not None and seq_len is not None:
                # z shape (self.batch_size, seq_len, num_neurons)
                c = torch.randn(self.batch_size, self.seq_len, self.num_neurons, device=self.Q.device)
                y = torch.randn(self.batch_size, self.seq_len, self.rank_k, device=self.Q.device)
                z_correlated = self.sqrt_epsilon * c + y @ self.Q.T
        else: # self.by_token_Q == True   ==>    seq_len is not None  (enforced in constructor)
            # Q is (seq_len, num_neurons, rank_k). 
            if self.batch_size is None:
                # z shape (seq_len, num_neurons)
                c = torch.randn(self.seq_len, self.num_neurons, device=self.Q.device)
                y = torch.randn(self.seq_len, self.rank_k, device=self.Q.device)
                z_correlated = self.sqrt_epsilon * c + einsum(self.Q, y,
                                                         "token n_neurons rank_k, token rank_k -> token n_neurons")
            else:
                # Batch of samples for each token
                c = torch.randn(self.batch_size, self.seq_len, self.num_neurons, device=self.Q.device)
                y = torch.randn(self.batch_size, self.seq_len, self.rank_k, device=self.Q.device)
                z_correlated = (self.sqrt_epsilon * c + 
                                einsum(self.Q, y,
                                       "token n_neurons rank_k, batch token rank_k -> batch token n_neurons"))
        if verbose:
            self._log_memory("Created c, y, and z_correlated")
        
        # Transform to uniform via standard normal CDF
        u = 0.5 * (1 + torch.erf(z_correlated / math.sqrt(2)))
        if verbose:
            self._log_memory("Created u")
        
        # Clean up z_correlated to save memory
        del z_correlated, c, y
        gc.collect()
        torch.cuda.empty_cache()
        if verbose:
            self._log_memory("After cleanup of z_correlated, c, y")
        
        # Transform to Beta via inverse CDF
        alpha_e, beta_e = self.get_effective_params()
        if verbose:
            self._log_memory("Got effective params")

        if hasattr(self, "fast_icdf"):             # ← injected by SCM
            samples = self.fast_icdf(u, alpha_e)   # fast table lookup
        else:
            dist = torch.distributions.Beta(alpha_e, beta_e)
            samples = dist.icdf(u)                 # slow but always correct
        if verbose:
            self._log_memory("Got samples")
        
        # Clean up intermediate tensors
        del u, alpha_e, beta_e
        gc.collect()
        torch.cuda.empty_cache()
        if verbose:
            self._log_memory("After cleanup of u, alpha_e, beta_e")

        # Apply any stretching/clamping
        samples = self.stretch_and_clamp_z(samples)
        if verbose:
            self._log_memory("Stretched and clamped")

        if cache_masks:
            self.current_mask = samples
        return samples


    def complexity_loss(self, z):
        """
        Compute the negative log likelihood of the Gaussian copula plus penalties.
        Vectorized implementation that avoids explicit loops.

        Args:
            z: Mask variables (already sampled)

        Returns:
            Total complexity loss (scalar)
        """
        # Get base Beta complexity loss
        beta_loss = super().complexity_loss(z)

        # Setup shared computations
        with torch.no_grad():
            I_k = torch.eye(self.rank_k, device=self.Q.device)

        if not self.by_token_Q: # Q is (num_neurons, rank_k)
            # Compute Q-dependent terms (once)
            Q_t_Q = self.Q.T @ self.Q
            det_term = -torch.logdet(I_k + Q_t_Q)
            inv_matrix = torch.inverse(I_k + self.inv_epsilon * Q_t_Q)
            q_norms = torch.sum(self.Q ** 2, dim=1)
            diag_penalty = self.lambda_diag * torch.sum((q_norms - 1.0) ** 2)
            Q_sparsity = self.lambda_Q * torch.sum(torch.abs(self.Q))
            # Averaging over batch, tokens, or both. Math derivation would be slightly different, but this keeps the terms
            # on the same scale and is just a question of scalars here or folded into the lambda terms
        
            # Handle different input shapes
            if z.dim() == 1:  # (neurons,)
                # No batch dimension - simple case
                z_norm_term = -self.inv_epsilon * torch.sum(z ** 2)
                Q_t_z = self.Q.T @ z.to(self.Q.dtype)
                quad_term = self.inv_epsilon_sqrd * (Q_t_z @ inv_matrix @ Q_t_z)

            elif z.dim() == 2:
                # z is (batch, neurons) if seq_len is None
                # z is (seq_len, neurons) if seq_len is not None
                # Treating batch and seq the same not because it's mathematically exact but just to keep the scale of
                # the terms here the same regardless of whether or not we're doing per token models. As in, the likelihood
                # is separate over batches but the sum over tokens. Either way just take the mean. It just washes out
                # to a scalar on the lambda weights, so better to just learn one set of lambda weights.
                z_norm_term = -self.inv_epsilon * torch.sum(z.to(self.Q.dtype) ** 2) / z.shape[0]
                # Vectorized quadratic term
                Q_t_z_batch = z.to(self.Q.dtype) @ self.Q
                quad_terms = torch.sum(Q_t_z_batch @ inv_matrix * Q_t_z_batch, dim=1)
                quad_term = self.inv_epsilon_sqrd * torch.mean(quad_terms)

            else:  # z.dim() == 3, (batch, seq, neurons)
                # Both batch and sequence dimensions
                z_norm_term = -self.inv_epsilon * torch.sum(z.to(self.Q.dtype) ** 2, dim=-1).mean()
                # Compute Q^T @ z for all vectors using einsum
                # z: (batch, seq, neurons), Q: (neurons, rank_k) -> (batch, seq, rank_k)
                Q_t_z = z.to(self.Q.dtype) @ self.Q                
                # Compute quadratic term for all vectors: (Q^T @ z) @ inv_matrix @ (Q^T @ z)
                # First Q_t_z @ inv_matrix: (batch, seq, rank_k) @ (rank_k, rank_k) -> (batch, seq, rank_k)
                temp = einsum(Q_t_z, inv_matrix.to(self.Q.dtype), "b s k1, k1 k2 -> b s k2")
                # Then multiply with Q_t_z element-wise and sum over k dimension
                # This computes sum_k (temp_ik * Q_t_z_ik) for each (batch, seq) pair
                quad_terms = torch.sum(temp * Q_t_z, dim=-1)  # (batch, seq)
                # Average over both batch and sequence dimensions
                quad_term = self.inv_epsilon_sqrd * torch.mean(quad_terms)
            # Combine all terms for the global Q case
            copula_loss = det_term + self.epsilon_term + z_norm_term + quad_term + diag_penalty + Q_sparsity
            return beta_loss + copula_loss

        else: # Token-specific Q (seq_len, num_neurons, rank_k)
            q_norms = torch.sum(self.Q ** 2, dim=-1)
            diag_penalty = self.lambda_diag * torch.sum((q_norms - 1.0) ** 2, dim=-1).mean()
            Q_sparsity = self.lambda_Q * torch.sum(torch.abs(self.Q), dim=(-1, -2)).mean()
            # Mean over seq_len or (batch, seq_len)
            z_norm_term  = -self.inv_epsilon * torch.sum(z ** 2, dim=-1).mean()
            
            # 1. Calculate Q.T @ Q for all token positions at once
            Q_t_Q = einops.einsum(self.Q, self.Q, "s n k1, s n k2 -> s k1 k2")
            # 2. Create seq_len I_k's to match Q_t_Q shape (seq_len, rank_k, rank_k)
            I_k_expanded = torch.eye(Q_t_Q.shape[1], device=Q_t_Q.device).unsqueeze(0).expand(self.seq_len, -1, -1)
            # 3. Calculate the log determinant term for all positions
            det_term = -torch.logdet(I_k_expanded + Q_t_Q).mean()  # Shape: (seq_len,).mean()
            # 4. Calculate inverse matrices for all positions
            inv_matrix = torch.inverse(I_k_expanded + self.inv_epsilon * Q_t_Q)  # Shape: (seq_len, rank_k, rank_k)                

            if z.dim() == 2:  # (seq, neurons)
                # No batch dimension
                # 1. Compute Q^T @ z for all tokens at once
                Q_t_z = einsum(self.Q, z, "s n k, s n -> s k")
                # 2. Compute quadratic term for all tokens: (Q^T @ z) @ inv_matrix @ (Q^T @ z)
                left_quad = einsum(Q_t_z, inv_matrix, "s k1, s k1 k2 -> s k2")

            else:  # z.dim() == 3, (batch, seq, neurons)
                # 1. Compute Q^T @ z for all tokens at once
                Q_t_z = einsum(self.Q, z, "s n k, b s n -> b s k")
                # 2. Compute quadratic term for all tokens: (Q^T @ z) @ inv_matrix @ (Q^T @ z)
                left_quad = einsum(Q_t_z, inv_matrix, "b s k1, s k1 k2 -> b s k2")
            
            # Multiply with Q_t_z element-wise and sum over k dimension
            # This computes sum_k (left_quad[t] * Q_t_z[t]) for each token or token and batch and takes
            # the overall mean
            quad_term    = torch.sum(left_quad * Q_t_z, dim=-1).mean()
            copula_loss = det_term + self.epsilon_term + z_norm_term + quad_term + diag_penalty + Q_sparsity
            return beta_loss + copula_loss




class SAECircuitMasker(nn.Module):
    def __init__(self, 
                 saes,                  # List of SAE objects returned by:
                                        # [SAE.from_pretrained(release="gemma-scope-9b-pt-res",
                                        #                   sae_id=f"layer_{layers[i]}/width_16k/average_l0_{l0s[i]}", 
                                        #                   device=device)[0] for i in range(len(layers))]
                 model,                 # sae_lens model that we're actually running and hooking
                 seq_len,               # Sequence length
                 u_step=0.001,          # Step size for u pctile for Beta ICDF est
                 alpha_step=0.01,       # Step size for alpha param for Beta ICDF est
                 lambda_e=4.0,          # Beta distribution parameter
                 lambda_beta=0.1,       # Weight for Beta log-density
                 lambda_sim=0.1,        # Weight for similarity penalty
                 rank_k=500,            # Rank of covariance approximation
                 lambda_diag=0.1,       # Weight for diagonal penalty
                 lambda_Q=0.01,         # Weight for Q sparsity
                 by_token_Q=False,      # Use token-specific Q matrices?
                 stretch_left=1e-5,     # Minimum mask value
                 stretch_right=(1 - 1e-5), # Max mask value
                 per_token_mask=True,   # Whether to use token-specific masks
                 epsilon=1e-4,
                 device="cuda",         # Device for computation
                 batch_size=16,
                 binary_threshold=0.5,
                 mean_tokens=None,
                 sparsity_multiplier=1.0,  # Weight for complexity loss in total loss
                 lambda_e_idx_dict=None,
                 icdf_chunk=2**20,
                 debug_grad_hooks=False):
        super().__init__()
        
        self.saes = saes
        self.model_layer_to_sae_idx = {sae.cfg.hook_layer : i for i, sae in enumerate(saes)}
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.per_token_mask = per_token_mask
        self.d_sae = saes[0].cfg.d_sae
        self.u_step = u_step
        self.inv_u_step = 1 / self.u_step
        self.alpha_step = alpha_step
        self.inv_alpha_step = 1 / self.alpha_step
        u_steps_r = (1 / self.u_step) - 1
        self.u_steps   = int(np.round(u_steps_r))
        if u_steps_r != self.u_steps or self.u_steps < 2:
            raise ValueError(f"u_step ({self.u_step}) must divide evenly into 1.0")
        
        if lambda_e_idx_dict is None:
            self.lambda_e_idx_dict = {val : i for i, val in enumerate([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])}
        
        # Extract necessary info from SAEs
        self.hook_names = [sae.cfg.hook_name for sae in saes]
        self.hook_layers = [sae.cfg.hook_layer for sae in saes]
        nn = [sae.cfg.d_sae for sae in saes]
        # Check if all SAEs have the same number of neurons
        if not all(n == nn[0] for n in nn):
            raise ValueError("All SAEs must have the same number of neurons, but got different values: " + str(nn))

        # Create the GaussianCopulaMask that covers ALL SAE neurons across ALL layers
        self.num_neurons = len(saes) * nn[0]  # Total neurons = num_saes * d_sae_per_layer
        self.seq_len = seq_len if per_token_mask else None
        
        self.icdf_chunk = icdf_chunk        # or tune per-GPU
        # index buffer: need 4 × chunk int64 slots (u_lo, u_hi, a_lo, a_hi)
        self.register_buffer("_icdf_int64", torch.empty(4 * self.icdf_chunk, dtype=torch.int64, device=self.device))
        # bf16 scratch: need 4 × chunk slots (u_low, u_high, a_low, a_high)
        self.register_buffer("_icdf_f32", torch.empty(4 * self.icdf_chunk, dtype=torch.float32, device=self.device))

        self.gcm = GaussianCopulaMask(num_neurons=self.num_neurons,  # Changed from nn[0] to total_neurons
                                      lambda_e=lambda_e,
                                      lambda_beta=lambda_beta,
                                      lambda_sim=lambda_sim,
                                      rank_k=rank_k,
                                      epsilon=epsilon,
                                      lambda_diag=lambda_diag,
                                      lambda_Q=lambda_Q,
                                      by_token_Q=by_token_Q,
                                      stretch_left=stretch_left,
                                      stretch_right=stretch_right,
                                      seq_len=seq_len,
                                      batch_size=batch_size,
                                      device=device,
                                      binary_threshold=binary_threshold)
        # Tell the GaussianCopulaMask to use the fast lookup the *parent* provides
        self.gcm.fast_icdf = self.lookup_beta_icdf

        
        # Initialize mean ablation values for each SAE
        # if mean_tokens is not None:
        #     # Assuming mean_tokens of shape (n_batches, batch_size, seq_len)
        #     self.set_sae_means(mean_tokens)

        #self.create_beta_icdf_lookup_table()

        self.sparsity_multiplier = sparsity_multiplier
        self.debug_grad_hooks = debug_grad_hooks

        # ------------------------------------------------------------------
        # Optional: attach NaN/Inf-detecting gradient hooks to key parameters
        # ------------------------------------------------------------------
        if self.debug_grad_hooks:
            try:
                self.gcm.alpha_raw.register_hook(self._check_grad("alpha_raw"))
                self.gcm.beta_raw .register_hook(self._check_grad("beta_raw"))
                self.gcm.Q        .register_hook(self._check_grad("Q"))
                print("[DEBUG] Parameter gradient hooks attached (alpha_raw, beta_raw, Q)")
            except Exception as _e:
                print(f"[DEBUG] Failed to attach parameter hooks: {_e}")

        # flag fields for optional debug mode
        self._grad_debug_steps = 0  # disabled by default


    # ───────────────────────── DEBUG PUBLIC API ────────────────────────
    def enable_grad_debug(self, n_steps: int = 1, *, log_every: int = 1):
        """Enable gradient/parameter debugging for the next *n_steps*.

        Args:
            n_steps: number of training iterations to trace.
            log_every: print stats every *log_every* backward calls.
        """
        self._grad_debug_steps = n_steps
        _register_debug_hooks(self.gcm, log_every=log_every)
    # ───────────────────────────────────────────────────────────────────────

    def running_mean_tensor(self, old_mean, new_value, n):
        return old_mean + (new_value - old_mean) / n


    def set_sae_means(self, mean_tokens):
        # Assuming mean_tokens of shape (n_samples, seq_len)
        total_batches = mean_tokens.shape[0] // self.batch_size
        for sae in self.saes:
            sae.mean_ablation = torch.zeros(sae.cfg.d_sae).float().to(self.device)
            #print(f"dir(sae) = {dir(sae)}")
            #print(f"sae.mean_ablation has prior shape: {sae.mean_ablation.shape}")

        with tqdm(total=total_batches * self.batch_size, desc="Mean Accum Progress") as pbar:
            for i in range(total_batches):
                for j in range(self.batch_size):
                    with torch.no_grad():
                        _ = self.model.run_with_hooks(mean_tokens[i, j],
                                                 return_type="logits",
                                                 fwd_hooks=self.build_hooks_list(mean_tokens[i, j], use_mask=False,
                                                                                 cache_sae_activations=True))
                        for sae in self.saes:
                            # print(f"i = {i}, j = {j}")
                            # print(f"sae.mean_ablation.shape = {sae.mean_ablation.shape}")
                            # print(f"sae.feature_acts.shape = {sae.feature_acts.shape}")
                            sae.mean_ablation = self.running_mean_tensor(sae.mean_ablation, sae.feature_acts,
                                                                         i * self.batch_size + j + 1)
                            # print("Ran running_mean_tensor")
                            # print(f"sae.feature_acts.shape = {sae.feature_acts.shape}")
                            # print(f"sae.mean_ablation.shape = {sae.mean_ablation.shape}")
                        self.cleanup_cuda()
                    pbar.update(1)

                if i >= total_batches:
                    break
    
    def create_beta_icdf_lookup_table(self, device="cuda", dtype=torch.float32, verbose=False):
        """
        Create a lookup table for Beta ICDF values based on lambda parameter values.
        Uses fully vectorized operations for maximum efficiency.
        
        Args:
            device: Device to store the table on
        
        Returns:
            Nothing. Sets self.lookup_tables
        """
        # Create u grid (same for all lambda values) - always create on CPU first for numpy conversion
        u_grid = torch.linspace(self.u_step, 1 - self.u_step, self.u_steps, device='cpu').numpy()
        # linspace gives tiny floating point errors frequently
        u_grid = np.round(u_grid, int(np.ceil(-np.log10(self.u_step))))
        
        # Initialize lookup tables and alpha grids as lists
        self.lookup_tables = []
        
        # For each lambda value, compute a table
        for lambda_e in self.lambda_e_idx_dict.keys():
            # Create alpha grid for this lambda - always create on CPU first for numpy conversion
            alpha_steps_r = lambda_e / self.alpha_step - 1
            alpha_steps   = int(alpha_steps_r)
            if alpha_steps_r != alpha_steps or alpha_steps < 2:
                raise ValueError(f"alpha_step ({self.alpha_step}) must divide evenly into lambda_e ({lambda_e})")
            alpha_grid = torch.linspace(self.alpha_step, lambda_e - self.alpha_step, alpha_steps, device='cpu').numpy()
            # linspace gives tiny floating point errors frequently
            alpha_grid = np.round(alpha_grid, int(np.ceil(-np.log10(self.alpha_step))))
            # Compute corresponding beta values
            beta_grid = lambda_e - alpha_grid

            # We want to compute beta.ppf for each combination of (alpha, beta) and u
            # Reshape to enable broadcasting:
            #    alpha and beta shape: (alpha_steps,),
            #    u shape: (1 / u_step - 1,), already correct
            alpha_reshaped = alpha_grid.reshape(-1, 1)  # Shape: (alpha_steps, 1)
            beta_reshaped = beta_grid.reshape(-1, 1)    # Shape: (alpha_steps, 1)
            
            # Compute the ICDF values for all combinations at once. Then convert
            # to tensor and move to specified device
            # This will return a 2D array with shape (alpha_steps, self.u_step)
            table_cpu = torch.tensor(beta.ppf(u_grid, alpha_reshaped, beta_reshaped), dtype=dtype)
            if verbose and (torch.isnan(table_cpu).any() or torch.isinf(table_cpu).any()):
                print(f"[DEBUG] NaN or Inf inside lookup table for lambda_e = {lambda_e}")
            self.lookup_tables.append(table_cpu.to(device))

    

    def lookup_beta_icdf(self, u, alpha_e, *, chunk: int = 2**20, verbose=False):
        """
        Beta-ICDF via fp16/bf16 table look-up, chunked to keep VRAM flat.
        * Identical maths to the original version.
        * Only `u_weight`, `a_weight`, and the final `interp` carry gradients.
        * Peak extra memory  ≲  (chunk × 6 bytes)  (≈ 1–2 MB with default chunk).
        """
        u  = u.clamp_(self.u_step, 1.0 - self.u_step)          # (eps , 1−eps)
        table = self.lookup_tables[self.lambda_e_idx_dict[self.gcm.lambda_e]]

        # --- constants ----------------------------------------------------
        u_min, u_max   =  self.u_step,     1.0 - self.u_step
        a_min, a_max   =  self.alpha_step, self.gcm.lambda_e - self.alpha_step
        n_alpha, n_u   =  table.shape[-2], table.shape[-1]

        # --- flatten everything once -------------------------------------
        flat_u         =  u.reshape(-1)
        flat_out       =  torch.empty_like(flat_u, dtype=u.dtype)

        # broadcast alpha_e to (B·T·N,) lazily
        repeat_factor  =  flat_u.numel() // alpha_e.numel()
        flat_alpha_e   =  alpha_e.repeat_interleave(repeat_factor)

        # scratch buffers (views, no allocation)
        work_int64     =  self._icdf_int64
        work_f32       =  self._icdf_f32
        if verbose:
            self._log_memory("Initialized table and working variables")

        # -----------------------------------------------------------------
        for start in range(0, flat_u.numel(), self.icdf_chunk):
            end                =  min(start + self.icdf_chunk, flat_u.numel())
            u_chunk            =  flat_u[start:end]          # (C,)
            alpha_chunk        =  flat_alpha_e[start:end]    # (C,)
            C                  =  u_chunk.numel()            # chunk length

            # -------------------------------------------------- no-grad ----
            with torch.no_grad():
                if verbose and torch.isnan(u_chunk).any():
                    idx = torch.nonzero(torch.isnan(u_chunk))[:5].flatten().tolist()
                    print(f"[DEBUG] u_chunk has NaN at positions {idx}")

                if verbose and torch.isinf(u_chunk).any():
                    idx = torch.nonzero(torch.isinf(u_chunk))[:5].flatten().tolist()
                    print(f"[DEBUG] u_chunk has Inf at positions {idx}")
                # ----- integer indices (int64, identical to old code) -----
                u_idx_low   = torch.floor((u_chunk.float().clamp_(u_min, u_max) - u_min) * self.inv_u_step).long().clamp_(0, n_u - 1)
                u_idx_high  = torch.clamp_(u_idx_low + 1, max=n_u - 1)
                a_idx_low   = torch.floor((alpha_chunk.float().clamp_(a_min, a_max) - a_min) * self.inv_alpha_step).long().clamp_(0, n_alpha - 1)
                a_idx_high  = torch.clamp_(a_idx_low + 1, max=n_alpha - 1)

                # ----- store indices in reusable buffer -------------------
                idx_view = work_int64[: 4 * C].view(4, C)
                idx_view[0].copy_(u_idx_low)
                idx_view[1].copy_(u_idx_high)
                idx_view[2].copy_(a_idx_low)
                idx_view[3].copy_(a_idx_high)
                u_idx_low, u_idx_high, a_idx_low, a_idx_high = idx_view  # views

                # ----- gather values ---------------------------------
                val_00 = table[a_idx_low, u_idx_low]   # (C,)
                val_01 = table[a_idx_low, u_idx_high]
                val_10 = table[a_idx_high, u_idx_low]
                val_11 = table[a_idx_high, u_idx_high]

                # ----- grid points into f32 scratch ----------------------
                grid_view = work_f32[: 4 * C].view(4, C)
                u_low, u_high, a_low, a_high = grid_view     # views
                
                u_low.copy_(u_min + u_idx_low.to(torch.float32) * self.u_step)
                u_high.copy_(u_low + self.u_step)
                a_low.copy_(a_min + a_idx_low.to(torch.float32) * self.alpha_step)
                a_high.copy_(a_low + self.alpha_step)

            # -------------- weights (keep grad wrt alpha_e) ---------------
            # Stable: guard denominator, replace non-finite results *before* clamp
            # ─── DEBUG: detect zero-width cells away from table edges ────────────
            if verbose:
                bad_u = (u_high - u_low) <= 0
                if bad_u.any():
                    legal_u = (u_idx_low == u_idx_high) & (
                            (u_idx_low == 0) | (u_idx_low == n_u - 1))
                    unexpected = bad_u & (~legal_u)
                    if unexpected.any():
                        idx = torch.nonzero(unexpected)[:5].flatten().tolist()
                        print(f"[DEBUG] Unexpected zero-width u-cell at positions {idx}")

                bad_a = (a_high - a_low) <= 0
                if bad_a.any():
                    legal_a = (a_idx_low == a_idx_high) & (
                            (a_idx_low == 0) | (a_idx_low == n_alpha - 1))
                    unexpected = bad_a & (~legal_a)
                    if unexpected.any():
                        idx = torch.nonzero(unexpected)[:5].flatten().tolist()
                        print(f"[DEBUG] Unexpected zero-width alpha-cell at positions {idx}")
            # ─────────────────────────────────────────────────────────────────────
            den_u = (u_high - u_low).clamp_min(1e-6)
            den_a = (a_high - a_low).clamp_min(1e-6)

            u_weight = (u_chunk.to(torch.float32) - u_low)  / den_u
            a_weight = (alpha_chunk.to(torch.float32) - a_low) / den_a

            # Replace NaN / ±Inf so ClampBackward never sees them
            u_weight = torch.nan_to_num(u_weight, nan=0.5, posinf=1.0, neginf=0.0)
            a_weight = torch.nan_to_num(a_weight, nan=0.5, posinf=1.0, neginf=0.0)

            u_weight.clamp_(self.gcm.epsilon, 1.0 - self.gcm.epsilon)
            a_weight.clamp_(self.gcm.epsilon, 1.0 - self.gcm.epsilon)

            # -------------- bilinear interpolation (identical formula) ----
            interp_chunk = (
                val_00 * (1 - a_weight) * (1 - u_weight) +
                val_01 * (1 - a_weight) *        u_weight +
                val_10 *        a_weight  * (1 - u_weight) +
                val_11 *        a_weight  *        u_weight
            )

            flat_out[start:end] = interp_chunk.to(u.dtype)
            if verbose:
                self._log_memory(f"Processed chunk {start:,} to {end:,}")
            break

        return flat_out.reshape_as(u)


    def lookup_beta_icdf_hi_mem(self, u, alpha_e):
        """
        Fast Beta–ICDF via table look‑up.
        * Keeps gradients wrt `alpha_e`.
        * Never puts `u` or large table tensors in the autograd graph.
        * Uses fp16 tables to halve memory.
        """
        u  = u.clamp_(self.u_step, 1.0 - self.u_step)          # (eps , 1−eps)
        table = self.lookup_tables[self.lambda_e_idx_dict[self.gcm.lambda_e]]
        u_min, u_max      =  self.u_step, 1.0 - self.u_step
        a_min, a_max      =  self.alpha_step, self.gcm.lambda_e - self.alpha_step
        n_u, n_alpha      =  table.shape[-1], table.shape[-2]
        self._log_memory("Initialized table")

        # ---- everything below is "index math" – no gradients needed ----
        with torch.no_grad():
            # clamp & convert to integer indices
            u_idx_float     = (u.clamp(u_min, u_max) - u_min) / (u_max - u_min) * (n_u - 1)
            u_idx_low       = torch.clamp(torch.floor(u_idx_float).long(), 0, n_u - 1)
            u_idx_high      = torch.clamp(u_idx_low + 1, max=n_u - 1)
            self._log_memory("Created high/low indices")

            a_idx_float     = (alpha_e.clamp(a_min, a_max) - a_min) / (a_max - a_min) * (n_alpha - 1)
            a_idx_low       = torch.clamp(torch.floor(a_idx_float).long(), 0, n_alpha - 1)
            a_idx_high      = torch.clamp(a_idx_low + 1, max=n_alpha - 1)
            self._log_memory("Created high/low indices")

            # gather fp16 table values, then cast to fp32 for math
            val_00 = table[a_idx_low,  u_idx_low ].float()
            val_01 = table[a_idx_low,  u_idx_high].float()
            val_10 = table[a_idx_high, u_idx_low ].float()
            val_11 = table[a_idx_high, u_idx_high].float()
            self._log_memory("Created values for interpolation")

            # pre‑compute grid points (also fp32, no‑grad)
            u_low  = u_min + u_idx_low .float() * self.u_step
            u_high = u_min + u_idx_high.float() * self.u_step
            a_low  = a_min + a_idx_low .float() * self.alpha_step
            a_high = a_min + a_idx_high.float() * self.alpha_step
            self._log_memory("Created grid points")
        
        # ---- small tensors below keep grad via alpha_e only ----
        # interpolation weights (depend on alpha_e, so keep in graph)
        u_weight = (u      - u_low)  / (u_high - u_low).clamp_min(1e-6)
        a_weight = (alpha_e - a_low) / (a_high - a_low).clamp_min(1e-6)
        self._log_memory("Created interpolation weights")
        interp = (
            val_00 * (1 - a_weight) * (1 - u_weight) +
            val_01 * (1 - a_weight) * u_weight       +
            val_10 * a_weight       * (1 - u_weight) +
            val_11 * a_weight       * u_weight
        )
        self._log_memory("Created interpolation")

        return interp.to(u.dtype)            # match original dtype


    # def lookup_beta_icdf(self, u, alpha_e):
    #     """
    #     Ultra-fast lookup function optimized for when all inputs have the same lambda_e value.
    #     Memory-optimized version that detaches table lookups while preserving gradients through interpolation weights.
        
    #     Args:
    #         u: Tensor of probability values (any shape)
    #         alpha_e: Tensor of alpha parameters (same shape as u)
        
    #     Returns:
    #         Interpolated ICDF values (same shape as u)
    #     """

    #     self._log_memory("Beginning of lookup_beta_icdf")
    #     table = self.lookup_tables[self.lambda_e_idx_dict[self.gcm.lambda_e]]
        
    #     # Calculate grid parameters
    #     u_min = self.u_step
    #     u_max = 1 - self.u_step
        
    #     alpha_min = self.alpha_step
    #     alpha_max = self.gcm.lambda_e - self.alpha_step
    #     alpha_steps = int((alpha_max - alpha_min) / self.alpha_step) + 1
        
    #     # Clamp inputs to valid ranges
    #     u = torch.clamp(u, u_min, u_max)
    #     alpha_e = torch.clamp(alpha_e, alpha_min, alpha_max)
    #     self._log_memory("Clamped inputs")

    #     # Calculate indices in the grid - DETACH these since they're just for indexing
    #     with torch.no_grad():
    #         u_idx_low = torch.floor((u - u_min) / (u_max - u_min) * (self.u_steps - 1)).long()
    #         u_idx_high = torch.where(u_idx_low + 1 >= self.u_steps, self.u_steps - 1, u_idx_low + 1)

    #         alpha_idx_low = torch.floor((alpha_e - alpha_min) / (alpha_max - alpha_min) * (alpha_steps - 1)).long()
    #         alpha_idx_high = torch.where(alpha_idx_low + 1 >= alpha_steps, alpha_steps - 1, alpha_idx_low + 1)
    #         self._log_memory("Created high/low indices")

    #         # Calculate grid values at these indices - also detached since they're just for interpolation
    #         u_low_value = u_min + u_idx_low.float() * self.u_step
    #         u_high_value = u_min + u_idx_high.float() * self.u_step
    #         alpha_low_value = alpha_min + alpha_idx_low.float() * self.alpha_step
    #         alpha_high_value = alpha_min + alpha_idx_high.float() * self.alpha_step
    #         self._log_memory("Created high/low values")

    #         # Lookup table values - detached since table doesn't need gradients
    #         val_00 = table[alpha_idx_low, u_idx_low].detach()
    #         val_01 = table[alpha_idx_low, u_idx_high].detach()
    #         val_10 = table[alpha_idx_high, u_idx_low].detach()
    #         val_11 = table[alpha_idx_high, u_idx_high].detach()
    #         self._log_memory("Created values for interpolation")
        
    #     self._log_memory("Calculated indices and table values (detached)")

    #     # Calculate interpolation weights - KEEP gradients for these since they depend on u and alpha_e
    #     u_weight = torch.zeros_like(u)
    #     u_diff_mask = (u_idx_high != u_idx_low)
    #     u_weight[u_diff_mask] = (u[u_diff_mask] - u_low_value[u_diff_mask]) / (u_high_value[u_diff_mask] - u_low_value[u_diff_mask])
    #     self._log_memory("Calculated u_weight")

    #     alpha_weight = torch.zeros_like(alpha_e)
    #     alpha_diff_mask = (alpha_idx_high != alpha_idx_low)
    #     alpha_weight[alpha_diff_mask] = (alpha_e[alpha_diff_mask] - alpha_low_value[alpha_diff_mask]) / (alpha_high_value[alpha_diff_mask] - alpha_low_value[alpha_diff_mask])
        
    #     self._log_memory("Calculated interpolation weights (with gradients)")
        
    #     # Bilinear interpolation - gradients flow through the weights, not the table values
    #     interp_result = (
    #         val_00 * (1 - alpha_weight) * (1 - u_weight) +
    #         val_01 * (1 - alpha_weight) * u_weight +
    #         val_10 * alpha_weight * (1 - u_weight) +
    #         val_11 * alpha_weight * u_weight
    #     )
    #     self._log_memory("Calculated interp_result")
        
    #     del u_idx_low, u_idx_high, alpha_idx_low, alpha_idx_high, u_low_value, u_high_value, alpha_low_value, \
    #         alpha_high_value, u_weight, u_diff_mask, alpha_weight, alpha_diff_mask, val_00, val_01, val_10, val_11
    #     gc.collect()
    #     torch.cuda.empty_cache()
    #     self._log_memory("Cleared intermediate variables")

    #     return interp_result


    def _log_memory(self, stage):
        """Helper function to log memory usage at different stages"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            print(f"[{stage}] VRAM - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
            return allocated, reserved
        return 0, 0


    def cleanup_cuda(self):
        gc.collect()
        torch.cuda.empty_cache()
         

    def clear_cached_masks(self):
        """Clear any cached masks and force garbage collection"""
        if hasattr(self.gcm, 'current_mask'):
            del self.gcm.current_mask
        torch.cuda.empty_cache()
        gc.collect()

    def sample_joint_masks(self, cache_masks=True):
        """Sample mask values for the current training iteration."""
        # Clear any existing cached masks first
        if hasattr(self.gcm, 'current_mask'):
            del self.gcm.current_mask
        torch.cuda.empty_cache()
        gc.collect()
        
        return self.gcm.sample_mask(cache_masks=cache_masks)


    def get_masks(self, binarize_mask=False, layer=None):
        """
        Get the current masks for all SAE hooks. We'll call sample_joint_masks once per
        batch, then call this through the forward function in each hook function.
        Separated this way because we need to jointly sample all the masks at all layers.

        Args:
            binarize_mask: Binarize masks via comparison to binary_threshold
            layer: SAE hook_layer number (e.g., 3, 5, 7, 9) to get mask for that specific layer.
                   If None, all SAE masks are returned concatenated into a single long vector 
                   as the last dimension (this is how self.gcm.current_mask is stored)

        Returns:
            Current mask tensor
        """
        if not hasattr(self.gcm, 'current_mask'):
            raise RuntimeError("SAECircuitMasker.gcm.current_mask must be set before get_masks is called")
            
        if binarize_mask:
            mask = torch.where(self.gcm.current_mask > self.gcm.binary_threshold, 1.0, 0.0)
        else:
            mask = self.gcm.current_mask

        if layer is None:
            return mask
        else:
            # Find which SAE has this hook_layer and get its positional index
            layer_idx = self.model_layer_to_sae_idx[layer]
            if layer_idx is None:
                available_layers = [k for k in self.model_layer_to_sae_idx.keys()]
                raise ValueError(f"Layer {layer} not found in SAEs. Available layers: {available_layers}")
                
            start_idx = layer_idx * self.d_sae
            end_idx = (layer_idx + 1) * self.d_sae
            return mask[..., start_idx:end_idx]
    

    def build_hooks_list(self, sequence,
                         circuit_mask=None,
                         use_mask=False,
                         binarize_mask=False,
                         mean_mask=False,
                         ig_mask_threshold=None,
                         cache_sae_grads=False,
                         cache_masked_activations=False,
                         cache_sae_activations=False,
                         fake_activations=False):
        """Build a list of hooks for all SAEs.
        
        Args:
            sequence: Input token sequence
            cache_sae_activations: Whether to store SAE activations
            cache_sae_grads: Whether to store gradients
            circuit_mask: Optional external mask to apply
            use_mask: Whether to use the internal mask
            binarize_mask: Whether to use binary (thresholded) masks
            mean_mask: Whether to use mean ablation
            cache_masked_activations: Whether to store masked activations
            fake_activations: Optional tuple of (layer_idx, tensor) for IG calculations
            ig_mask_threshold: Optional threshold for IG mask
            
        Returns:
            List of (hook_name, hook_fn) tuples
        """
        hooks = []
        for i, sae in enumerate(self.saes):
            hooks.append(
                (sae.cfg.hook_name,
                 self.build_sae_hook_fn(i, sequence,
                          circuit_mask=circuit_mask,
                          use_mask=use_mask,
                          binarize_mask=binarize_mask,
                          mean_mask=mean_mask,
                          ig_mask_threshold=ig_mask_threshold,
                          cache_sae_grads=cache_sae_grads,
                          cache_masked_activations=cache_masked_activations,
                          cache_sae_activations=cache_sae_activations,
                          fake_activations=fake_activations)
                )
            )
        return hooks


    def build_sae_hook_fn(self,
                          # Core Inputs
                          sae_idx,
                          sequence, 
                          # Masking options
                          circuit_mask=None,
                          use_mask=False,
                          binarize_mask=False,
                          mean_mask=False,           # Controls mean ablation of the SAE
                          ig_mask_threshold=None,
                          # Caching behavior
                          cache_sae_grads=False,
                          cache_masked_activations=False,
                          cache_sae_activations=False,
                          # Ablation options
                          fake_activations=False):  # Controls whether to use fake activations
        """Build a hook function for a specific SAE in the circuit masker.
        
        Args:
            sae_idx: Index of the SAE in self.saes list
            sequence: Input token sequence
            circuit_mask: Optional mask to apply to the circuit
            use_mask: Whether to use masking
            binarize_mask: Whether to use binary (thresholded) masks
            mean_mask: Whether to use mean ablation for masking
            ig_mask_threshold: Threshold for integrated gradients mask
            cache_sae_grads: Whether to store SAE gradients
            cache_masked_activations: Whether to store masked activations
            cache_sae_activations: Whether to store SAE activations
            fake_activations: Optional tuple of (layer_idx, tensor) for IG calculations
        
        Returns:
            Hook function for the specified SAE
        """
        sae = self.saes[sae_idx]
        bos_token_id = self.model.tokenizer.bos_token_id
        
        # Create token mask
        token_mask = torch.ones_like(sequence, dtype=torch.bool)
        token_mask[sequence == bos_token_id] = False
        # print(f"1: token_mask.shape = {token_mask.shape}")

        def sae_hook(value, hook):
            # Get SAE activations (or use fake ones)
            #print(f"value.shape = {value.shape}")
            if fake_activations and sae.cfg.hook_layer == fake_activations[0]:
                feature_acts = fake_activations[1]
            else:
                feature_acts = sae.encode(value)
                #print(f"feature_acts.shape pre-token_mask = {feature_acts.shape}")
                feature_acts = feature_acts * token_mask.unsqueeze(-1)
            #print(f"feature_acts.shape = {feature_acts.shape}")

            # Cache original activations if needed
            if cache_sae_activations:
                sae.feature_acts = feature_acts.detach().clone()
            
            # Get mask for this SAE
            if use_mask:
                current_mask = self.get_masks(binarize_mask=binarize_mask, layer=sae.cfg.hook_layer)
            
            # Apply mask with or without mean ablation
            if use_mask and mean_mask:
                diff = feature_acts - sae.mean_ablation
                masked_acts = diff * current_mask + sae.mean_ablation
            elif use_mask:
                masked_acts = feature_acts * current_mask
            else:
                masked_acts = feature_acts            
            # Cache masked activations if needed
            if cache_masked_activations:
                sae.masked_feature_acts = masked_acts.detach().clone()
            
            # Decode and apply back to the residual stream
            out = sae.decode(masked_acts)
            
            # Only apply to non-masked tokens
            mask_expanded = token_mask.unsqueeze(-1).expand_as(value)
            value = torch.where(mask_expanded, out, value)
            return value
        
        return sae_hook
        
    
    def run_training(self, 
                     token_dataset, 
                     labels_dataset, 
                     corr_labels_dataset, 
                     hyperparams=None,  # Made optional - Dictionary of hyperparameters
                     task="training",
                     loss_function='logit_diff',
                     portion_of_data=0.5,
                     learning_rate=0.01,
                     verbose=False):
        """
        Run training to find a minimal circuit using the beta-copula mask.
        
        Args:
            token_dataset: Tensor of input tokens, shape (n_batches, batch_size, seq_len)
            labels_dataset: Tensor of correct labels, shape (n_batches, batch_size)
            corr_labels_dataset: Tensor of incorrect/corrupted labels, shape (n_batches, batch_size)
            hyperparams: Optional dictionary of hyperparameters to override defaults. Can include:
                - lambda_e: Beta distribution concentration parameter (default: use existing value)
                - lambda_beta: Weight for Beta log-density penalty (default: use existing value)
                - lambda_sim: Weight for similarity penalty (default: use existing value)
                - lambda_diag: Weight for diagonal penalty (default: use existing value)
                - lambda_Q: Weight for Q sparsity penalty (default: use existing value)
            task: String identifier for the training task (for logging)
            loss_function: Type of loss to use ('logit_diff' or 'ce')
            portion_of_data: Fraction of dataset to use for training (0.0 to 1.0)
            learning_rate: Learning rate for Adam optimizer
        
        Returns:
            Dictionary containing:
                - Final mask statistics and densities
                - Training hyperparameters used
                - Final evaluation metrics (CE loss, logit diff)
                - Active indices for each SAE layer
        """
        if verbose:
            self._log_memory("Training Start")
        
        # Update hyperparameters in the GaussianCopulaMask only if provided
        if hyperparams is not None:
            self.gcm.lambda_e = hyperparams.get('lambda_e', self.gcm.lambda_e)
            self.gcm.lambda_beta = hyperparams.get('lambda_beta', self.gcm.lambda_beta)
            self.gcm.lambda_sim = hyperparams.get('lambda_sim', self.gcm.lambda_sim)
            self.gcm.lambda_diag = hyperparams.get('lambda_diag', self.gcm.lambda_diag)
            self.gcm.lambda_Q = hyperparams.get('lambda_Q', self.gcm.lambda_Q)
            print(f"Updated hyperparameters: {hyperparams}")
        else:
            print("Using existing hyperparameters from SCM initialization")
        
        # Display current hyperparameters being used
        current_hyperparams = {
            'lambda_e': self.gcm.lambda_e,
            'lambda_beta': self.gcm.lambda_beta,
            'lambda_sim': self.gcm.lambda_sim,
            'lambda_diag': self.gcm.lambda_diag,
            'lambda_Q': self.gcm.lambda_Q
        }
        print(f"Running training with hyperparameters: {current_hyperparams}")
        
        # Configure wandb
        config = {
            "batch_size": self.batch_size,
            "learning_rate": learning_rate,
            "total_steps": token_dataset.shape[0] * portion_of_data,
            **current_hyperparams  # Include all current hyperparameters in the config
        }
        wandb.init(project="sae circuits beta-copula", config=config)
        if verbose:
            self._log_memory("After wandb init")
        
        # Setup optimizer
        optimized_params = list(self.gcm.parameters())
        optimizer = optim.Adam(optimized_params, lr=learning_rate)
        total_steps = int(config["total_steps"] * 1.1)  # Allow for slight overrun
        if verbose:
            self._log_memory("After optimizer setup")
        
        # Get available lambda_e values from lookup tables (sorted in descending order)
        available_lambda_e_values = sorted(self.lambda_e_idx_dict.keys(), reverse=True)
        if verbose:
            print(f"Available lambda_e values: {available_lambda_e_values}")
        
        # Always start from the maximum lambda_e value
        lambda_e_schedule = available_lambda_e_values  # Start from max (4.0) down to min (1.0)
        num_stages = len(lambda_e_schedule)
        if num_stages > total_steps:
            print(f"Warning: num_stages ({num_stages}) is less than total_steps ({total_steps}).")
            print(f"Adjusting total_steps to {num_stages}.")
            total_steps = num_stages
        steps_per_stage = total_steps // num_stages if num_stages > 0 else total_steps
        
        if verbose:
            print(f"Lambda_e annealing schedule: {lambda_e_schedule}")
            print(f"Steps per stage: {steps_per_stage}")
        
        # Training loop
        with tqdm(total=total_steps, desc="Training Progress") as pbar:
            for i, (x, y, z) in enumerate(zip(token_dataset, labels_dataset, corr_labels_dataset)):
                try:
                    if verbose:
                        self._log_memory(f"Step {i} - Start")
                    
                    # Zero gradients
                    optimizer.zero_grad()
                    if verbose:
                        self._log_memory(f"Step {i} - After zero_grad")
                    
                    # Calculate ratio trained (for annealing)
                    ratio_trained = i / total_steps
                    
                    # Staged lambda_e annealing through discrete values
                    if num_stages > 1:
                        current_stage = min(i // steps_per_stage, num_stages - 1)
                        current_lambda_e = lambda_e_schedule[current_stage]
                        self.gcm.lambda_e = current_lambda_e
                    else:
                        # Fallback to original continuous annealing if only one stage
                        if available_lambda_e_values[0] > available_lambda_e_values[-1]:
                            current_lambda_e = available_lambda_e_values[-1]
                        self.gcm.lambda_e = current_lambda_e
                    
                    # Sample masks for this batch
                    with torch.autograd.detect_anomaly():
                        masks = self.sample_joint_masks()
                    if verbose:
                        self._log_memory(f"Step {i} - After mask sampling")
                        print(f"Mask shape: {masks.shape if hasattr(masks, 'shape') else 'No shape attr'}")
                    
                    # Forward pass with current mask
                    with torch.autograd.detect_anomaly():
                        task_loss, complexity_loss, beta_loss, copula_loss = self._forward_pass(x, y, z, loss_function, verbose=verbose)
                    if verbose:
                        self._log_memory(f"Step {i} - After forward pass")
                    
                    # Total loss combines task + regularisation
                    total_loss = task_loss + self.sparsity_multiplier * complexity_loss
                    if verbose:
                        self._log_memory(f"Step {i} - After loss computation")
                    
                    # ---------------- BACKWARD (with optional anomaly trace) --------
                    debug_mode = self._grad_debug_steps > 0
                    ctx = torch.autograd.detect_anomaly() if debug_mode else contextlib.nullcontext()
                    with ctx:
                        total_loss.backward()

                    # decrement debug counter & expose step to hooks
                    if debug_mode:
                        self._grad_debug_steps -= 1
                    self.gcm._debug_step = i

                    if verbose:
                        self._log_memory(f"Step {i} - After backward")
                    
                    # ------------------ post-step parameter clamp -------------------
                    with torch.no_grad():
                        self.gcm.alpha_raw.clamp_(-10.0, 10.0)
                        self.gcm.beta_raw .clamp_(-10.0, 10.0)
                        self.gcm.Q        .clamp_(-3.0,  3.0)
 
                    # ----------------------------------------------------------------
                    # FREE UNUSED CUDA CACHES EACH STEP (prevents fragmentation & OOM)
                    # ----------------------------------------------------------------
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    # Memory cleanup every few steps
                    if i % 1 == 0:  # Clean up every 5 steps
                        del masks, task_loss, complexity_loss, total_loss
                        if beta_loss is not None:
                            del beta_loss
                        if copula_loss is not None:
                            del copula_loss
                        self.cleanup_cuda()
                        # torch.cuda.empty_cache()
                        self._log_memory(f"Step {i} - After cleanup")
                    
                    # Log stats (reduced frequency to save memory)
                    if i % 2 == 0:  # Log every 2 steps instead of every step
                        stats = {
                            "Step": i, 
                            "Progress": ratio_trained, 
                                "Task Loss": task_loss.item() if 'task_loss' in locals() else 0,
                                "Complexity Loss": complexity_loss.item() if 'complexity_loss' in locals() else 0,
                            "Current Lambda_e": self.gcm.lambda_e
                        }
                        wandb.log(stats)
                    
                        # Update progress bar (less frequently)
                    pbar.set_postfix({k: v for k, v in stats.items() if not isinstance(v, torch.Tensor)})
                    
                    pbar.update(1)
                    
                    # Check if we've reached the end
                    if i >= total_steps:
                        break
                        
                except KeyboardInterrupt:
                    print("Training interrupted by user.")
                    break
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"Out of memory at step {i}!")
                        self._log_memory(f"OOM at step {i}")
                        # Try to recover
                        self.cleanup_cuda()
                        torch.cuda.empty_cache()
                        if 'total_loss' in locals():
                            del total_loss
                        if 'masks' in locals():
                            del masks
                        self._log_memory(f"After OOM cleanup")
                        raise e
                    else:
                        raise e
        
        self._log_memory("After training loop")
        
        # Finish wandb logging
        wandb.finish()
        
        # Clean up
        optimizer.zero_grad()
        for sae in self.saes:
            for param in sae.parameters():
                if param.grad is not None:
                    param.grad = None
                
        self.cleanup_cuda()
        gc.collect()
        torch.cuda.empty_cache()
        self._log_memory("After final cleanup")
        
        # Evaluate final performance with logging
        print("Evaluating final performance...")
        with torch.no_grad():
            # Use original batch sizes for evaluation
            eval_batch = token_dataset[-1]
            eval_labels = labels_dataset[-1]
            eval_corr_labels = corr_labels_dataset[-1]
            
            ce_loss = self._eval_ce_loss(eval_batch, eval_labels)
            print(f"CE loss: {ce_loss.item()}")
            self._log_memory("After CE evaluation")
            self.cleanup_cuda()
            
            # Use original number of batches for logit diff evaluation
            logit_diff = self._eval_logit_diff(len(token_dataset), token_dataset, labels_dataset, corr_labels_dataset)
            print(f"Logit Diff: {logit_diff}")
            self._log_memory("After logit diff evaluation")
            self.cleanup_cuda()
        
        self._log_memory("After evaluation")
        
        # Create mask dictionary for saving
        mask_dict = {}
        total_density = 0
        
        # Get binary masks for each SAE
        with torch.no_grad():
            binary_masks = self.get_masks(binarize_mask=True)
            for idx, sae in enumerate(self.saes):
                if binary_masks.dim() > 1:
                    layer_mask = binary_masks[:, idx] if binary_masks.shape[1] > idx else binary_masks[:, 0]
                else:
                    layer_mask = binary_masks
            active_indices = torch.nonzero(layer_mask).flatten().tolist()
            mask_dict[sae.cfg.hook_name] = active_indices
            total_density += len(active_indices)
        
        mask_dict["total_density"] = total_density
        mask_dict['avg_density'] = total_density / len(self.saes)
        
        if self.per_token_mask:
            print(f"Total # latents in circuit: {total_density}")
        print(f"Average density: {mask_dict['avg_density']}")
        
        # Save hyperparameters and results
        mask_dict['hyperparams'] = current_hyperparams
        mask_dict['ce_loss'] = ce_loss.item()
        mask_dict['logit_diff'] = logit_diff
        mask_dict['faithfulness'] = logit_diff / self._get_baseline_logit_diff(token_dataset, labels_dataset, corr_labels_dataset)
        
        print("Training completed successfully!")
        self._log_memory("Training End")
        
        return mask_dict


    def _forward_pass(self, batch, clean_label_tokens, corr_label_tokens, loss_function='logit_diff', verbose=False):
        """
        Perform a forward pass through the model with the current mask.
        
        Args:
            batch: Input token batch [batch_size, seq_len]
            clean_label_tokens: Correct answer token IDs [batch_size]
            corr_label_tokens: Incorrect answer token IDs [batch_size]
            loss_function: Type of loss to use ('logit_diff' or 'ce')
        
        Returns:
            Tuple of (task_loss, complexity_loss, beta_loss, copula_loss)
            - task_loss: Primary task loss (logit difference or cross-entropy)
            - complexity_loss: Combined regularization loss from mask complexity
            - beta_loss: Individual beta distribution penalty (may be None)
            - copula_loss: Individual copula penalty (may be None)
        """
        
        # Log memory at start of forward pass
        if verbose:
            self._log_memory("Forward Pass Start")
        
        # Get model output with masked SAEs
        masked_logits = self.model.run_with_hooks(
            batch,
            return_type="logits",
            fwd_hooks=self.build_hooks_list(batch, use_mask=True, mean_mask=True)
        )
        if verbose:
            self._log_memory("After Masked Forward")
        
        # Get model output without masking (for reference)
        with torch.no_grad(): 
            model_logits = self.model.run_with_hooks(
                batch,
                return_type="logits",
                fwd_hooks=self.build_hooks_list(batch, use_mask=False, mean_mask=False)
            )
        if verbose:
            self._log_memory("After Reference Forward")
        
        # Calculate loss based on specified loss function
        if loss_function == 'ce':
            # Cross-entropy loss on the last token
            last_token_logits = masked_logits[:, -1, :]
            task_loss = F.cross_entropy(last_token_logits, clean_label_tokens)
        elif loss_function == 'logit_diff':
            # Logit difference between correct and incorrect answers
            fwd_logit_diff = self._logit_diff_fn(masked_logits, clean_label_tokens, corr_label_tokens)
            model_logit_diff = self._logit_diff_fn(model_logits, clean_label_tokens, corr_label_tokens)
            task_loss = torch.abs(model_logit_diff - fwd_logit_diff)
        else:
            raise ValueError(f"Unsupported loss function: {loss_function}")
        
        if verbose:
            self._log_memory("After Task Loss Computation")
        
        # Calculate complexity losses (beta and copula terms)
        complexity_loss = self.complexity_loss(self.gcm.current_mask)
        if verbose:
            self._log_memory("After Complexity Loss")
        
        # If the GaussianCopulaMask implementation exposes the individual loss components,
        # get them for monitoring purposes
        beta_loss = getattr(self.gcm, "beta_loss", None)
        copula_loss = getattr(self.gcm, "copula_loss", None)
        
        # Clean up intermediate tensors to save memory
        del model_logits, masked_logits
        if 'last_token_logits' in locals():
            del last_token_logits
        if 'fwd_logit_diff' in locals():
            del fwd_logit_diff
        if 'model_logit_diff' in locals():
            del model_logit_diff
        if verbose:
            self._log_memory("After Cleanup")
        self.cleanup_cuda()
        
        return task_loss, complexity_loss, beta_loss, copula_loss    

    def complexity_loss(self, z):
        """Compute complexity loss for the current mask"""
        return self.gcm.complexity_loss(z)
    
    def save_masks(self, save_dir):
        """Save the current masks to disk"""
        pass    

    def load_masks(self, load_dir):
        """Load masks from disk"""
        pass

    def _eval_ce_loss(self, batch, labels):
        """
        Evaluate cross-entropy loss on a batch.
        
        Args:
            batch: Input tokens
            labels: Correct labels
            
        Returns:
            Cross-entropy loss tensor
        """
        logits = self.model.run_with_hooks(
            batch,
            return_type="logits",
            fwd_hooks=self.build_hooks_list(batch, use_mask=True, mean_mask=True)
        )
        last_token_logits = logits[:, -1, :]
        return F.cross_entropy(last_token_logits, labels)

    def _eval_logit_diff(self, num_batches, token_dataset, labels_dataset, corr_labels_dataset):
        """
        Evaluate logit difference across multiple batches.
        
        Args:
            num_batches: Number of batches to evaluate
            token_dataset: Token dataset
            labels_dataset: Clean labels dataset  
            corr_labels_dataset: Corrupted labels dataset
            
        Returns:
            Mean logit difference across all evaluated batches
        """
        total_logit_diff = 0
        for i in range(min(num_batches, len(token_dataset))):
            logits = self.model.run_with_hooks(
                token_dataset[i],
                return_type="logits", 
                fwd_hooks=self.build_hooks_list(token_dataset[i], use_mask=True, mean_mask=True)
            )
            logit_diff = self._logit_diff_fn(logits, labels_dataset[i], corr_labels_dataset[i])
            total_logit_diff += logit_diff.item()
            
        return total_logit_diff / min(num_batches, len(token_dataset))

    def _get_baseline_logit_diff(self, token_dataset, labels_dataset, corr_labels_dataset):
        """
        Get baseline (unmasked) logit difference for faithfulness calculation.
        
        Args:
            token_dataset: Token dataset
            labels_dataset: Clean labels dataset
            corr_labels_dataset: Corrupted labels dataset
            
        Returns:
            Baseline logit difference without any masking
        """
        total_logit_diff = 0
        num_batches = len(token_dataset)
        
        for i in range(num_batches):
            logits = self.model.run_with_hooks(
                token_dataset[i],
                return_type="logits",
                fwd_hooks=self.build_hooks_list(token_dataset[i], use_mask=False, mean_mask=False)
            )
            logit_diff = self._logit_diff_fn(logits, labels_dataset[i], corr_labels_dataset[i])
            total_logit_diff += logit_diff.item()
            
        return total_logit_diff / num_batches

    def _logit_diff_fn(self, logits, clean_labels, corr_labels, token_wise=False):
        """
        Compute logit difference between clean and corrupted labels.
        
        Args:
            logits: Model output logits [batch_size, seq_len, vocab_size]
            clean_labels: Correct answer token IDs [batch_size]
            corr_labels: Incorrect answer token IDs [batch_size]
            token_wise: If True, return per-example differences; if False, return mean
        
        Returns:
            Logit difference (clean_logits - corr_logits)
            - If token_wise=False: scalar (mean across batch)
            - If token_wise=True: tensor [batch_size] (per-example)
        """
        # Get last token logits for final prediction
        last_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]
        
        # Get logits for correct and incorrect answers
        clean_logits = last_token_logits[torch.arange(last_token_logits.shape[0]), clean_labels]
        corr_logits = last_token_logits[torch.arange(last_token_logits.shape[0]), corr_labels]
        
        # Return mean logit difference or per-example differences
        return (clean_logits - corr_logits).mean() if not token_wise else (clean_logits - corr_logits)

    # ------------------- INTERNAL: gradient-sanity hook ---------------
    def _check_grad(self, name: str):
        """Return a hook that prints a warning if *grad* contains NaN/Inf."""
        def _hook(grad: torch.Tensor):
            if torch.isnan(grad).any() or torch.isinf(grad).any():
                bad_idx = torch.nonzero(~torch.isfinite(grad))
                first = bad_idx[0].item() if bad_idx.numel() else -1
                print(f"[GRAD-DEBUG] {name} has non-finite grad; first bad index {first}")
        return _hook
    # ------------------------------------------------------------------
