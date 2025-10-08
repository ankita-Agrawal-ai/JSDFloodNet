# utils/diffusion.py
import math
import torch
import torch.nn.functional as F
from torch import nn

# Simple linear beta schedule
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

class Diffusion:
    def __init__(self, timesteps=1000, device='cpu'):
        self.timesteps = timesteps
        self.device = device
        betas = linear_beta_schedule(timesteps).to(device)  # (T,)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1,0), value=1.0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        # Precompute useful terms
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        """
        Add noise to x_start at step t (vectorized).
        x_start: (B, C, H, W)
        t: tensor long (B,) with values in [0, T-1]
        noise: optional (B,C,H,W)
        returns x_t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        # gather sqrt_alphas_cumprod[t] etc
        sqrt_acp = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1).to(self.device)
        sqrt_1_acp = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1).to(self.device)
        return sqrt_acp * x_start + sqrt_1_acp * noise

    def predict_start_from_noise(self, x_t, t, noise):
        sqrt_acp = self.sqrt_alphas_cumprod[t].view(-1,1,1,1).to(self.device)
        sqrt_1_acp = self.sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1).to(self.device)
        return (x_t - sqrt_1_acp * noise) / (sqrt_acp + 1e-8)

    def p_loss(self, denoise_fn, x_start, t, noise=None, loss_type='l2'):
        """
        Standard DDPM training loss: predict noise.
        denoise_fn(x_t, t) -> predicted noise
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        pred = denoise_fn(x_t, t)  # expected to return noise prediction of same shape
        if loss_type == 'l1':
            loss = F.l1_loss(pred, noise)
        else:
            loss = F.mse_loss(pred, noise)
        return loss

    @torch.no_grad()
    def p_sample(self, denoise_fn, x_t, t):
        """
        Single denoising step using model that predicts noise.
        """
        betas_t = self.betas[t].to(self.device)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].to(self.device)
        alphas_cumprod_t = self.alphas_cumprod[t].to(self.device)
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[t]).to(self.device)

        # predict noise
        pred_noise = denoise_fn(x_t, t)
        # predict x0
        x0_pred = (x_t - sqrt_one_minus_alphas_cumprod_t.view(-1,1,1,1) * pred_noise) / torch.sqrt(alphas_cumprod_t).view(-1,1,1,1)
        # compute mean of posterior q(x_{t-1} | x_t, x0_pred)
        posterior_mean = (
            self.posterior_mean_coeff1(t).view(-1,1,1,1) * x0_pred +
            self.posterior_mean_coeff2(t).view(-1,1,1,1) * x_t
        )
        # sample if t > 0
        if (t == 0).all():
            return posterior_mean
        else:
            var = self.posterior_variance[t].view(-1,1,1,1).to(self.device)
            noise = torch.randn_like(x_t)
            return posterior_mean + torch.sqrt(var) * noise

    def posterior_mean_coeff1(self, t):
        # helper to compute coefficients (vectorized)
        # using formula in DDPM paper:
        # coef1 = betas_t * sqrt(alphas_cumprod_prev) / (1 - alphas_cumprod_t)
        betas_t = self.betas[t].to(self.device)
        acp = self.alphas_cumprod[t].to(self.device)
        acp_prev = self.alphas_cumprod_prev[t].to(self.device)
        coef1 = betas_t * torch.sqrt(acp_prev) / (1.0 - acp)
        return coef1

    def posterior_mean_coeff2(self, t):
        # coef2 = (1 - alphas_cumprod_prev) * sqrt(alpha_t) / (1 - alphas_cumprod_t)
        betas_t = self.betas[t].to(self.device)
        acp = self.alphas_cumprod[t].to(self.device)
        acp_prev = self.alphas_cumprod_prev[t].to(self.device)
        coef2 = (1.0 - acp_prev) * torch.sqrt(self.alphas[t]).to(self.device) / (1.0 - acp)
        return coef2

    @torch.no_grad()
    def sample(self, denoise_fn, shape):
        """
        Sample from noise to x0 by iterating t = T-1 .. 0
        shape: (B, C, H, W)
        denoise_fn receives (x_t, t) and returns predicted noise
        """
        b = shape[0]
        x_t = torch.randn(shape, device=self.device)
        for t_idx in reversed(range(self.timesteps)):
            t = torch.full((b,), t_idx, dtype=torch.long, device=self.device)
            x_t = self.p_sample(denoise_fn, x_t, t)
        return x_t
