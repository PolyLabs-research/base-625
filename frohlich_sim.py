import numpy as np
import matplotlib.pyplot as plt
import time

# --- Constants ---
COMPLEX_TYPE = np.complex128

class NLSE_Simulator:
    """
    Solves the 1D Non-Linear Schrödinger Equation using the split-step Fourier method
    and implements the Diósi-Penrose Objective Reduction model.
    """
    
    def __init__(self, N_points=512, L=100.0, N_time_steps=2000, dt=0.1, 
                 alpha=0.2, gamma=1.0, pump=0.8, decay=0.08, seed=42):
        
        # --- Grid Parameters ---
        self.N_points = N_points
        self.L = L
        self.dx = L / N_points
        self.x = np.linspace(-L/2, L/2, N_points, endpoint=False)
        
        # --- Time Parameters ---
        self.N_time_steps = N_time_steps
        self.dt = dt
        
        # --- Physics Parameters ---
        self.alpha = alpha
        self.gamma = gamma
        self.P = pump
        self.delta = decay
        self.H_BAR = 2.0 # Increased to make it harder to trigger collapse
        
        # --- Fourier Space ---
        self.k = 2 * np.pi * np.fft.fftfreq(N_points, d=self.dx)
        
        # --- State Initialization ---
        self.rng = np.random.default_rng(seed)
        noise_real = 0.1 * (self.rng.random(self.N_points) - 0.5)
        noise_imag = 0.1 * (self.rng.random(self.N_points) - 0.5)
        self.psi_0 = (noise_real + 1j * noise_imag).astype(COMPLEX_TYPE)
        self.psi = self.psi_0.copy()
        
        # --- OR State ---
        self.time_in_superposition = 0.0
        
        # --- Data Storage ---
        self.history = np.zeros((self.N_time_steps, self.N_points))
        
        # --- Pre-calculate Operators ---
        self.linear_operator = np.exp(-1j * self.alpha * self.k**2 * self.dt)
        

    def _nonlinear_step(self):
        """Solves the non-linear part of the equation in real space."""
        nonlinear_term = self.gamma * np.abs(self.psi)**2
        growth_decay_factor = np.exp((self.P - self.delta) * self.dt)
        phase_factor = np.exp(1j * nonlinear_term * self.dt)
        return growth_decay_factor * phase_factor

    def _create_gaussian(self, center_x, amplitude, sigma):
        """Creates a Gaussian profile, ensuring it is a complex type."""
        return (amplitude * np.exp(-(self.x - center_x)**2 / (2 * sigma**2))).astype(COMPLEX_TYPE)

    def run(self):
        """Executes the main simulation loop with Objective Reduction."""
        print("Starting NLSE simulation with Objective Reduction (Gamma Reset)...")
        start_time = time.time()
        
        self.psi = self.psi_0.copy()
        self.time_in_superposition = 0.0

        for t in range(self.N_time_steps):
            self.history[t, :] = np.abs(self.psi)**2

            # --- Evolve the state by one time step ---
            self.psi *= self._nonlinear_step()
            
            psi_k = np.fft.fft(self.psi)
            psi_k *= self.linear_operator
            self.psi = np.fft.ifft(psi_k)
            
            # --- Objective Reduction Logic ---
            prob_density = np.abs(self.psi)**2
            spread = np.std(prob_density)

            if spread > 1e-6:
                threshold = self.H_BAR / spread
                
                if self.time_in_superposition > threshold:
                    print(f"BING! Collapse at t={t}. Discharging energy...")
                    
                    # Reset wavefunction to low-level thermal noise (Gamma Reset)
                    noise_real = 0.1 * (self.rng.random(self.N_points) - 0.5)
                    noise_imag = 0.1 * (self.rng.random(self.N_points) - 0.5)
                    self.psi = (noise_real + 1j * noise_imag).astype(COMPLEX_TYPE)
                    
                    self.time_in_superposition = 0.0
                else:
                    self.time_in_superposition += self.dt
            else:
                self.time_in_superposition = 0.0

        end_time = time.time()
        print(f"Simulation finished in {end_time - start_time:.2f} seconds.")
        return self.history

def plot_waterfall(data, L, N_time_steps, dt):
    """
    Generates a "waterfall" plot (space-time heatmap) of the probability density.
    """
    print("Generating waterfall plot...")
    plt.figure(figsize=(10, 8))
    
    plt.imshow(
        data,
        extent=[-L/2, L/2, N_time_steps*dt, 0],
        aspect='auto',
        cmap='hot'
    )
    
    plt.colorbar(label=r'Probability Density $|\psi|^2$')
    plt.title('Fröhlich Condensation with Objective Reduction')
    plt.xlabel('Position along Microtubule (x)')
    plt.ylabel('Time (t)')
    
    filename = "frohlich_condensation_waterfall.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved waterfall plot to {filename}")

if __name__ == "__main__":
    # --- Execute the Simulation ---
    sim = NLSE_Simulator()
    history_data = sim.run()
    
    # --- Visualize the Results ---
    plot_waterfall(history_data, sim.L, sim.N_time_steps, sim.dt)