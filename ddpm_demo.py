"""
Minimal Denoising Diffusion Probabilistic Model (DDPM)
Demonstrates the forward noising and reverse denoising processes
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

class SimpleDDPM:
    """
    Simplified DDPM for visualization purposes.
    In practice, you'd use a U-Net instead of this toy predictor.
    """
    
    def __init__(self, num_timesteps: int = 1000):
        self.T = num_timesteps
        
        # Linear beta schedule (as in the paper)
        self.betas = np.linspace(1e-4, 0.02, num_timesteps)
        
        # Precompute useful values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        
    def forward_diffusion(self, x0: np.ndarray, t: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward diffusion: add noise to x0 at timestep t
        
        x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * epsilon
        
        Args:
            x0: Original data (e.g., image)
            t: Timestep (0 to T-1)
            
        Returns:
            x_t: Noised data at timestep t
            epsilon: The noise that was added
        """
        # Sample noise
        epsilon = np.random.randn(*x0.shape)
        
        # Get noise schedule values for timestep t
        sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Apply forward diffusion formula
        x_t = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * epsilon
        
        return x_t, epsilon
    
    def reverse_diffusion_step(self, x_t: np.ndarray, t: int, epsilon_pred: np.ndarray) -> np.ndarray:
        """
        Single step of reverse diffusion (denoising)
        
        This is Algorithm 2 from the paper (simplified).
        In practice, epsilon_pred comes from a neural network.
        
        Args:
            x_t: Noisy data at timestep t
            t: Current timestep
            epsilon_pred: Predicted noise (from neural network)
            
        Returns:
            x_{t-1}: Less noisy data at timestep t-1
        """
        if t == 0:
            # Final step: just compute mean
            alpha_bar_t = self.alphas_cumprod[t]
            return (x_t - np.sqrt(1 - alpha_bar_t) * epsilon_pred) / np.sqrt(alpha_bar_t)
        
        # Get values for timestep t
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alphas_cumprod[t]
        beta_t = self.betas[t]
        
        # Compute mean of reverse distribution
        coef1 = 1.0 / np.sqrt(alpha_t)
        coef2 = beta_t / np.sqrt(1.0 - alpha_bar_t)
        mean = coef1 * (x_t - coef2 * epsilon_pred)
        
        # Add noise for stochasticity (except last step)
        sigma_t = np.sqrt(beta_t)
        z = np.random.randn(*x_t.shape)
        
        x_t_minus_1 = mean + sigma_t * z
        
        return x_t_minus_1


def visualize_forward_process(ddpm: SimpleDDPM, x0: np.ndarray, timesteps_to_show: list):
    """
    Visualize how an image gets progressively noisier
    """
    fig, axes = plt.subplots(1, len(timesteps_to_show), figsize=(15, 3))
    
    for idx, t in enumerate(timesteps_to_show):
        x_t, _ = ddpm.forward_diffusion(x0, t)
        
        # Clip for visualization
        x_t_vis = np.clip(x_t, 0, 1)
        
        axes[idx].imshow(x_t_vis, cmap='gray')
        axes[idx].set_title(f't = {t}')
        axes[idx].axis('off')
    
    plt.suptitle('Forward Process: Progressive Noising', fontsize=14)
    plt.tight_layout()
    return fig


def visualize_reverse_process(ddpm: SimpleDDPM, timesteps_to_show: list):
    """
    Visualize reverse process with a toy noise predictor.
    
    Note: This uses a VERY simplified noise predictor (just scaling).
    Real DDPM uses a trained U-Net.
    """
    # Start from pure noise
    x_t = np.random.randn(28, 28)
    
    fig, axes = plt.subplots(1, len(timesteps_to_show), figsize=(15, 3))
    
    sample_idx = 0
    for t in range(ddpm.T - 1, -1, -1):
        # Toy noise predictor (in reality, this is a trained U-Net)
        # For visualization, we just use a simple heuristic
        epsilon_pred = x_t * 0.1  # Simplified!
        
        # Take reverse diffusion step
        x_t = ddpm.reverse_diffusion_step(x_t, t, epsilon_pred)
        
        # Save frames at specific timesteps
        if t in timesteps_to_show:
            x_t_vis = np.clip(x_t, 0, 1)
            axes[sample_idx].imshow(x_t_vis, cmap='gray')
            axes[sample_idx].set_title(f't = {t}')
            axes[sample_idx].axis('off')
            sample_idx += 1
    
    plt.suptitle('Reverse Process: Progressive Denoising (Toy Example)', fontsize=14)
    plt.tight_layout()
    return fig


def plot_noise_schedule(ddpm: SimpleDDPM):
    """
    Visualize the noise schedule over time
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Beta schedule
    axes[0].plot(ddpm.betas)
    axes[0].set_xlabel('Timestep t')
    axes[0].set_ylabel('β_t')
    axes[0].set_title('Noise Schedule: β_t (amount of noise added per step)')
    axes[0].grid(True, alpha=0.3)
    
    # Cumulative alpha (signal retention)
    axes[1].plot(ddpm.alphas_cumprod)
    axes[1].set_xlabel('Timestep t')
    axes[1].set_ylabel('ᾱ_t')
    axes[1].set_title('Cumulative Signal: ᾱ_t (how much original signal remains)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# ============================================
# Example Usage
# ============================================

if __name__ == "__main__":
    # Initialize DDPM
    ddpm = SimpleDDPM(num_timesteps=1000)
    
    # Create a simple synthetic image (checkerboard pattern)
    x0 = np.zeros((28, 28))
    x0[::4, ::4] = 1.0  # Checkerboard
    x0[2::4, 2::4] = 1.0
    
    print("=" * 60)
    print("DDPM Demonstration")
    print("=" * 60)
    print(f"Number of timesteps: {ddpm.T}")
    print(f"Beta range: {ddpm.betas[0]:.6f} to {ddpm.betas[-1]:.6f}")
    print(f"Final alpha_bar (signal at t=T): {ddpm.alphas_cumprod[-1]:.6e}")
    print("=" * 60)
    
    # Visualize forward process
    print("\n1. Visualizing forward diffusion (adding noise)...")
    timesteps_forward = [0, 100, 250, 500, 750, 999]
    fig1 = visualize_forward_process(ddpm, x0, timesteps_forward)
    plt.savefig('forward_process.png', dpi=150, bbox_inches='tight')
    print("   Saved: forward_process.png")
    
    # Visualize noise schedule
    print("\n2. Visualizing noise schedule...")
    fig2 = plot_noise_schedule(ddpm)
    plt.savefig('noise_schedule.png', dpi=150, bbox_inches='tight')
    print("   Saved: noise_schedule.png")
    
    # Visualize reverse process (toy example)
    print("\n3. Visualizing reverse diffusion (toy example)...")
    print("   Note: This uses a simplified noise predictor.")
    print("   Real DDPM uses a trained U-Net.")
    timesteps_reverse = [999, 750, 500, 250, 100, 0]
    fig3 = visualize_reverse_process(ddpm, timesteps_reverse)
    plt.savefig('reverse_process.png', dpi=150, bbox_inches='tight')
    print("   Saved: reverse_process.png")
    
    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("• Forward process: Gradually adds Gaussian noise over T steps")
    print("• At t=999, the image is almost pure noise")
    print("• Reverse process: Neural network learns to predict and remove noise")
    print("• Training: Minimize ||ε - ε_θ(x_t, t)||²")
    print("• Sampling: Start from noise, denoise for T steps")
    print("=" * 60)
    
    plt.show()