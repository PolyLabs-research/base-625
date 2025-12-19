import numpy as np
import matplotlib.pyplot as plt
import time

# --- Constants ---
COMPLEX_TYPE = np.complex128

class MicrotubuleSimulator2D:
    """
    Solves the 2D Non-Linear Schrödinger Equation on a cylindrical surface
    representing a Microtubule (13 Protofilaments).
    """
    
    def __init__(self, n_protofilaments=13, length_nm=200, n_length_points=256, 
                 n_steps=3000, dt=0.05, alpha=0.5, gamma=2.0, pump=1.2, decay=0.1, seed=42):
        
        # --- Geometry ---
        self.Ny = n_protofilaments  # Y-axis: Circumference (13)
        self.Nx = n_length_points   # X-axis: Length
        self.L = length_nm
        
        self.dx = self.L / self.Nx
        self.dy = 1.0 # Normalized distance between protofilaments
        
        # --- Simulation Parameters ---
        self.dt = dt
        self.n_steps = n_steps
        
        # --- Physics Parameters ---
        self.alpha = alpha  # Diffusion/Tunneling rate
        self.gamma = gamma  # Non-linearity (Self-focusing strength)
        self.P = pump       # Metabolic Energy (ATP/GTP)
        self.delta = decay  # Decoherence/Loss
        self.H_BAR_THRESHOLD = 4.0 # Threshold for Penrose Objective Reduction
        
        # --- Fourier Space Setup ---
        # kx: Frequencies along the length
        self.kx = 2 * np.pi * np.fft.fftfreq(self.Nx, d=self.dx)
        # ky: Frequencies around the circumference
        self.ky = 2 * np.pi * np.fft.fftfreq(self.Ny, d=self.dy)
        
        # Meshgrid for 2D FFT operator
        self.KX, self.KY = np.meshgrid(self.kx, self.ky)
        
        # Linear Operator (Kinetic Energy / Diffusion)
        # Propagator: exp(-i * alpha * (kx^2 + ky^2) * dt)
        self.linear_operator = np.exp(-1j * self.alpha * (self.KX**2 + self.KY**2) * self.dt)
        
        # --- State Initialization ---
        self.rng = np.random.default_rng(seed)
        self.reset_wavefunction()
        
    def reset_wavefunction(self):
        """Initializes the state with low-level thermal noise."""
        noise_r = 0.1 * (self.rng.random((self.Ny, self.Nx)) - 0.5)
        noise_i = 0.1 * (self.rng.random((self.Ny, self.Nx)) - 0.5)
        self.psi = (noise_r + 1j * noise_i).astype(COMPLEX_TYPE)
        self.time_in_superposition = 0.0

    def _nonlinear_step(self):
        """
        Applies Non-linearity (Self-focusing) and Pump/Decay.
        P and delta affect Amplitude (Real exp).
        Gamma affects Phase (Imaginary exp).
        """
        density = np.abs(self.psi)**2
        nonlinear_phase = self.gamma * density
        
        # Split operator: Amplitude gain/loss * Phase rotation
        gain_factor = np.exp((self.P - self.delta) * self.dt)
        phase_factor = np.exp(1j * nonlinear_phase * self.dt)
        
        return gain_factor * phase_factor

    def run(self):
        """Runs the simulation and captures snapshots."""
        print(f"Starting 2D Microtubule Simulation ({self.Ny}x{self.Nx})...")
        start_time = time.time()
        
        snapshots = []
        capture_indices = [0, 200, 800, 1500, 2500] # Steps to save for visualization
        collapse_log = []

        for t in range(self.n_steps):
            
            # 1. Capture Snapshot if needed
            if t in capture_indices:
                snapshots.append((t, np.abs(self.psi)**2))
            
            # 2. Split-Step Evolution
            # Part A: Non-linear (Real Space)
            self.psi *= self._nonlinear_step()
            
            # Part B: Linear (Fourier Space)
            psi_k = np.fft.fft2(self.psi)
            psi_k *= self.linear_operator
            self.psi = np.fft.ifft2(psi_k)
            
            # 3. Objective Reduction (Penrose Logic)
            # In 2D, we measure how "peaked" the distribution is.
            # High Peak = High Gravitational Self-Energy (E_G)
            density = np.abs(self.psi)**2
            max_density = np.max(density)
            
            # If the wave is peaking (Soliton formation), E_G increases.
            # We use max_density as a proxy for mass concentration.
            if max_density > 1.0:
                time_threshold = self.H_BAR_THRESHOLD / max_density
                
                if self.time_in_superposition > time_threshold:
                    # COLLAPSE EVENT
                    print(f"--> BING! Objective Reduction at t={t}. Max Density={max_density:.2f}")
                    collapse_log.append(t)
                    
                    # Save the moment of collapse
                    snapshots.append((t, density.copy()))
                    
                    # Gamma Reset (Discharge to noise)
                    self.reset_wavefunction()
                else:
                    self.time_in_superposition += self.dt
            else:
                # Decay timer if wave is diffuse
                self.time_in_superposition = max(0, self.time_in_superposition - self.dt)

        print(f"Simulation complete. Total Collapses: {len(collapse_log)}")
        print(f"Duration: {time.time() - start_time:.2f}s")
        return snapshots, collapse_log

def plot_cylinder_unrolled(snapshots):
    """
    Visualizes the 2D cylinder surface as a series of unrolled heatmaps.
    """
    num_snaps = len(snapshots)
    fig, axes = plt.subplots(num_snaps, 1, figsize=(12, 2.5 * num_snaps))
    
    if num_snaps == 1: axes = [axes]
    
    for i, (t, state) in enumerate(snapshots):
        ax = axes[i]
        # Visualize the cylinder unrolled
        im = ax.imshow(state, aspect='auto', cmap='inferno', interpolation='bicubic')
        ax.set_title(f"Time Step: {t} (State of the Lattice)")
        ax.set_ylabel("Protofilaments (1-13)")
        ax.set_yticks([0, 6, 12])
        
        # Add a colorbar for the last plot to show scale
        if i == num_snaps - 1:
            plt.colorbar(im, ax=ax, orientation='horizontal', label='Quantum Probability Density')

    plt.xlabel("Position along Microtubule Length")
    plt.tight_layout()
    plt.savefig("microtubule_2d_evolution.png")
    print("Visualization saved to microtubule_2d_evolution.png")

if __name__ == "__main__":
    sim = MicrotubuleSimulator2D()
    history, collapses = sim.run()
    plot_cylinder_unrolled(history)