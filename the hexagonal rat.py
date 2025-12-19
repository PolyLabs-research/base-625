import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

# --- Constants ---
COMPLEX_TYPE = np.complex128

class MicrotubuleSimulator2D:
    """
    A 2D Quantum Field simulation that acts as a 'Steering Engine'.
    """
    def __init__(self, n_protofilaments=13, length_points=64, dt=0.05, 
                 alpha=0.5, gamma=2.0, base_pump=1.2, decay=0.1):
        
        self.Ny = n_protofilaments
        self.Nx = length_points
        self.dt = dt
        self.alpha = alpha
        self.gamma = gamma
        self.delta = decay
        self.base_pump = base_pump
        self.H_BAR = 3.0 # Threshold constant
        
        # Fourier Operators
        self.kx = 2 * np.pi * np.fft.fftfreq(self.Nx, d=1.0)
        self.ky = 2 * np.pi * np.fft.fftfreq(self.Ny, d=1.0)
        self.KX, self.KY = np.meshgrid(self.kx, self.ky)
        self.linear_operator = np.exp(-1j * self.alpha * (self.KX**2 + self.KY**2) * self.dt)
        
        self.psi = np.zeros((self.Ny, self.Nx), dtype=COMPLEX_TYPE)
        self.reset_wavefunction()
        
        # State Tracking
        self.time_in_superposition = 0.0

    def reset_wavefunction(self):
        noise_r = 0.1 * (np.random.rand(self.Ny, self.Nx) - 0.5)
        noise_i = 0.1 * (np.random.rand(self.Ny, self.Nx) - 0.5)
        self.psi = (noise_r + 1j * noise_i).astype(COMPLEX_TYPE)
        self.time_in_superposition = 0.0

    def step(self, spatial_input, target_bias_map):
        """
        Evolves the quantum state.
        spatial_input: Energy from Grid Cells (Where am I?)
        target_bias_map: Energy from Smell (Where is the cheese?)
        """
        # Input Mixing: The Brain sees both Location and Goal
        P_total = self.base_pump + spatial_input + (target_bias_map * 0.5)
        
        # 1. Non-linear Step
        density = np.abs(self.psi)**2
        nonlinear_phase = self.gamma * density
        gain = np.exp((P_total - self.delta) * self.dt)
        phase = np.exp(1j * nonlinear_phase * self.dt)
        self.psi *= (gain * phase)
        
        # 2. Linear Step
        psi_k = np.fft.fft2(self.psi)
        psi_k *= self.linear_operator
        self.psi = np.fft.ifft2(psi_k)
        
        # 3. Check for Collapse (The Decision)
        # We return the "Center of Mass" of the collapse if it happens
        max_density = np.max(density)
        decision_vector = None
        
        if max_density > 1.0:
            threshold = self.H_BAR / max_density
            if self.time_in_superposition > threshold:
                # --- BING! (Objective Reduction) ---
                # Find where the collapse happened (y, x)
                y_center, x_center = np.unravel_index(np.argmax(density), density.shape)
                
                # Map Y-coordinate (Protofilament) to a Steering Angle
                # 0 to 13 -> 0 to 2*Pi
                angle = (y_center / self.Ny) * 2 * np.pi
                decision_vector = np.array([np.cos(angle), np.sin(angle)])
                
                # Gamma Reset (Discharge)
                self.reset_wavefunction()
            else:
                self.time_in_superposition += self.dt
        else:
            self.time_in_superposition *= 0.95 # Decay if signal is weak
            
        return np.abs(self.psi)**2, decision_vector

class QuantumRat:
    def __init__(self, arena_size=20.0):
        self.pos = np.array([arena_size/2, arena_size/2]) # Start center
        self.velocity = np.random.randn(2) * 0.2
        self.arena_size = arena_size
        self.target = np.array([arena_size * 0.8, arena_size * 0.8]) # Cheese location
        self.score = 0
        
        # Grid Cell Setup (3 plane waves)
        self.k_scale = 1.0
        self.wave_vecs = np.array([[np.cos(a), np.sin(a)] for a in [0, np.pi/3, 2*np.pi/3]]) * self.k_scale

    def get_grid_pattern(self):
        # Generate the Hexagonal Interference Pattern
        waves = np.sum([np.cos(np.dot(self.pos, k)) for k in self.wave_vecs])
        activation = (waves + 1.5) / 4.5 # Normalize 0-1
        return activation

    def get_target_bias(self, mt_shape):
        """
        Creates a 'Gradient Map' on the microtubule based on target direction.
        This represents the rat's 'sense of smell' biasing the quantum probability.
        """
        ny, nx = mt_shape
        to_target = self.target - self.pos
        target_angle = np.arctan2(to_target[1], to_target[0])
        if target_angle < 0: target_angle += 2*np.pi
        
        # Convert physical angle to Protofilament Index
        target_y = int((target_angle / (2*np.pi)) * ny)
        
        # Create a Gaussian strip along that protofilament
        bias_grid = np.zeros((ny, nx))
        Y, _ = np.ogrid[:ny, :nx]
        
        # Distance on cylinder (wrapping)
        dist = np.minimum(np.abs(Y - target_y), np.abs(Y - target_y - ny))
        dist = np.minimum(dist, np.abs(Y - target_y + ny))
        
        # A stripe of energy appearing on the "North" side of the tube if target is North
        bias_grid = np.exp(-dist**2 / 2.0) 
        return bias_grid

    def update(self, decision_vector):
        # 1. Physics Momentum
        self.pos += self.velocity
        
        # 2. Quantum Steering
        if decision_vector is not None:
            # The "Bing" applies a force in the decided direction
            steer_force = decision_vector * 0.5
            self.velocity += steer_force
            
            # Normalize speed (Rat can't run infinitely fast)
            speed = np.linalg.norm(self.velocity)
            if speed > 0.5:
                self.velocity = (self.velocity / speed) * 0.5
                
        # 3. Walls
        np.clip(self.pos, 0, self.arena_size, out=self.pos)
        
        # 4. Check Goal
        dist = np.linalg.norm(self.pos - self.target)
        found_goal = False
        if dist < 1.0:
            found_goal = True
            self.target = np.random.rand(2) * self.arena_size # Move cheese
            self.score += 1
            
        return found_goal

# --- Main Simulation Loop ---
def run_navigator():
    print("Initializing Quantum Navigator...")
    
    # Objects
    mt = MicrotubuleSimulator2D(n_protofilaments=13, length_points=64)
    rat = QuantumRat()
    
    # Visualization
    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(1, 2, figure=fig)
    
    # Plot 1: The Arena
    ax_arena = fig.add_subplot(gs[0, 0])
    ax_arena.set_title("Real World: Rat vs. Cheese")
    ax_arena.set_xlim(0, rat.arena_size)
    ax_arena.set_ylim(0, rat.arena_size)
    
    # Draw Grid Field Background
    x = np.linspace(0, rat.arena_size, 50)
    y = np.linspace(0, rat.arena_size, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(50):
        for j in range(50):
            # Simple grid pattern reconstruction for visuals
            pos = np.array([X[i,j], Y[i,j]])
            waves = np.sum([np.cos(np.dot(pos, k)) for k in rat.wave_vecs])
            Z[i,j] = waves
    ax_arena.contourf(X, Y, Z, cmap="Greys", alpha=0.3)
    
    rat_dot, = ax_arena.plot([], [], 'ro', markersize=12, label='Rat')
    target_dot, = ax_arena.plot([], [], 'y*', markersize=15, label='Cheese')
    path_line, = ax_arena.plot([], [], 'r-', alpha=0.5, lw=1)
    path_x, path_y = [], []
    
    # Plot 2: The Brain
    ax_brain = fig.add_subplot(gs[0, 1])
    ax_brain.set_title("Quantum Brain (Microtubule State)")
    ax_brain.set_xlabel("Length (x)")
    ax_brain.set_ylabel("Protofilament (Angle)")
    brain_img = ax_brain.imshow(np.zeros((mt.Ny, mt.Nx)), aspect='auto', cmap='inferno', vmin=0, vmax=8)
    
    print("Starting navigation loop...")

    def update(frame):
        nonlocal path_x, path_y
        
        # 1. Generate Brain Inputs
        grid_signal = rat.get_grid_pattern() # Scalar energy based on location
        target_map = rat.get_target_bias((mt.Ny, mt.Nx)) # Directional bias
        
        # Project Grid Signal as a spotlight on the tube (Self-Location)
        # We put the self-location at the 'center' of the tube for simplicity
        spotlight = np.zeros((mt.Ny, mt.Nx))
        spotlight[:, 32] = grid_signal * 2.0 
        
        # 2. Run Physics (5 sub-steps)
        final_density = None
        decision = None
        
        for _ in range(5):
            # Input is sum of Self-Location (Spotlight) and Goal (Target Map)
            total_input = spotlight + target_map
            density, vec = mt.step(total_input, target_map) # Step physics
            final_density = density
            if vec is not None:
                decision = vec # Capture the decision vector if a collapse happened
        
        # 3. Move Rat based on Physics Decision
        found = rat.update(decision)
        
        if found:
            print(f"Target Reached! Score: {rat.score}")
            path_x, path_y = [], [] # Reset trail
            
        # 4. Visuals
        rat_dot.set_data([rat.pos[0]], [rat.pos[1]])
        target_dot.set_data([rat.target[0]], [rat.target[1]])
        
        path_x.append(rat.pos[0])
        path_y.append(rat.pos[1])
        path_line.set_data(path_x, path_y)
        
        brain_img.set_data(final_density)
        
        return rat_dot, target_dot, brain_img, path_line

    ani = animation.FuncAnimation(fig, update, frames=300, interval=20, blit=True)
    ani.save('quantum_navigator.gif', writer='pillow', fps=30)
    print("Simulation saved to quantum_navigator.gif")

if __name__ == "__main__":
    run_navigator()