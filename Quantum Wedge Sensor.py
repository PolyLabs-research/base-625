import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches

# --- Physics Engine (With Inhibitory Input) ---
COMPLEX_TYPE = np.complex128

class MicrotubuleSimulator2D:
    def __init__(self, n_protofilaments=13, length_points=64, dt=0.05, 
                 alpha=0.5, base_gamma=2.0, base_pump=1.2, decay=0.1):
        self.Ny = n_protofilaments
        self.Nx = length_points
        self.dt = dt
        self.alpha = alpha
        self.gamma = base_gamma
        self.delta = decay
        self.base_pump = base_pump
        self.H_BAR = 3.0
        
        self.kx = 2 * np.pi * np.fft.fftfreq(self.Nx, d=1.0)
        self.ky = 2 * np.pi * np.fft.fftfreq(self.Ny, d=1.0)
        self.KX, self.KY = np.meshgrid(self.kx, self.ky)
        self.linear_operator = np.exp(-1j * self.alpha * (self.KX**2 + self.KY**2) * self.dt)
        
        self.psi = np.zeros((self.Ny, self.Nx), dtype=COMPLEX_TYPE)
        self.reset_wavefunction()
        self.time_in_superposition = 0.0

    def reset_wavefunction(self):
        noise_r = 0.1 * (np.random.rand(self.Ny, self.Nx) - 0.5)
        noise_i = 0.1 * (np.random.rand(self.Ny, self.Nx) - 0.5)
        self.psi = (noise_r + 1j * noise_i).astype(COMPLEX_TYPE)
        self.time_in_superposition = 0.0

    def step(self, spatial_input, target_bias, visual_inhibition):
        # Visual Inhibition SUBTRACTS from the pump
        # If there is a wall, P_total drops, preventing solitons at that angle
        P_total = self.base_pump + spatial_input + (target_bias * 0.5) - (visual_inhibition * 5.0)
        
        # Clamp P to avoid negative energy weirdness
        P_total = np.maximum(P_total, 0.0)

        density = np.abs(self.psi)**2
        nonlinear_phase = self.gamma * density
        gain = np.exp((P_total - self.delta) * self.dt)
        phase = np.exp(1j * nonlinear_phase * self.dt)
        self.psi *= (gain * phase)
        
        psi_k = np.fft.fft2(self.psi)
        psi_k *= self.linear_operator
        self.psi = np.fft.ifft2(psi_k)
        
        density = np.abs(self.psi)**2
        max_density = np.max(density)
        decision_vector = None
        
        if max_density > 1.0:
            threshold = self.H_BAR / max_density
            if self.time_in_superposition > threshold:
                y_center, x_center = np.unravel_index(np.argmax(density), density.shape)
                angle = (y_center / self.Ny) * 2 * np.pi
                decision_vector = np.array([np.cos(angle), np.sin(angle)])
                self.reset_wavefunction()
            else:
                self.time_in_superposition += self.dt
        else:
            self.time_in_superposition *= 0.95
            
        return density, decision_vector

# --- The Rat with Eyes ---
class RatWithEyes:
    def __init__(self, arena_size=20.0):
        self.pos = np.array([10.0, 2.0])
        self.velocity = np.array([0.0, 0.5]) # Moving Up
        self.speed = 0.2
        self.arena_size = arena_size
        self.target = np.array([10.0, 18.0])
        
        # Wedge Sensor Configuration
        self.num_rays = 13 # One ray per protofilament for direct mapping
        self.fov = np.pi / 2 # 90 degree field of view
        self.view_distance = 6.0
        self.rays = np.zeros((self.num_rays, 2)) # Store endpoints for drawing
        self.distances = np.ones(self.num_rays) * self.view_distance # Store readings
        
        self.walls = [(5, 10, 10, 1), (5, 10, 1, 6), (14, 10, 1, 6)] # U-Trap
        
        # Grid Cell Setup
        self.k_scale = 1.0
        self.wave_vecs = np.array([[np.cos(a), np.sin(a)] for a in [0, np.pi/3, 2*np.pi/3]])

    def cast_rays(self):
        """
        Casts 13 rays in a cone and measures distance to walls.
        """
        heading = np.arctan2(self.velocity[1], self.velocity[0])
        angles = np.linspace(heading - self.fov/2, heading + self.fov/2, self.num_rays)
        
        self.distances.fill(self.view_distance) # Reset
        
        for i, angle in enumerate(angles):
            ray_dir = np.array([np.cos(angle), np.sin(angle)])
            
            # Simple Raymarching
            for d in np.linspace(0, self.view_distance, 20):
                check_pos = self.pos + ray_dir * d
                
                # Wall Collision
                hit = False
                # Check Arena Walls
                if (check_pos[0] < 0 or check_pos[0] > self.arena_size or 
                    check_pos[1] < 0 or check_pos[1] > self.arena_size):
                    hit = True
                
                # Check Obstacles
                for (wx, wy, ww, wh) in self.walls:
                    if (check_pos[0] >= wx and check_pos[0] <= wx + ww and
                        check_pos[1] >= wy and check_pos[1] <= wy + wh):
                        hit = True
                
                if hit:
                    self.distances[i] = d
                    self.rays[i] = check_pos # End point
                    break
            
            # If no hit, ray goes to max view
            if self.distances[i] == self.view_distance:
                self.rays[i] = self.pos + ray_dir * self.view_distance

    def get_visual_map(self, mt_shape):
        """
        Maps the 13 ray distances to the 13 microtubule protofilaments.
        Close wall = High Inhibition.
        """
        ny, nx = mt_shape
        inhibition_grid = np.zeros((ny, nx))
        
        # Invert distance: Close = 1.0, Far = 0.0
        intensity = 1.0 - (self.distances / self.view_distance)
        intensity = np.clip(intensity, 0, 1) ** 2 # Square for sharper shadows
        
        # Map rays to protofilaments
        # Ray 0 (Left) -> Protofilament 0
        # Ray 12 (Right) -> Protofilament 12
        # We smear it across the length (X-axis) because visual info is global
        for i in range(ny):
            inhibition_grid[i, :] = intensity[i]
            
        return inhibition_grid

    def get_target_bias(self, mt_shape):
        # Same "Smell" logic as before
        ny, nx = mt_shape
        to_target = self.target - self.pos
        target_angle = np.arctan2(to_target[1], to_target[0])
        if target_angle < 0: target_angle += 2*np.pi
        target_y = int((target_angle / (2*np.pi)) * ny)
        bias_grid = np.zeros((ny, nx))
        Y, _ = np.ogrid[:ny, :nx]
        dist = np.minimum(np.abs(Y - target_y), np.abs(Y - target_y - ny))
        dist = np.minimum(dist, np.abs(Y - target_y + ny))
        return np.exp(-dist**2 / 2.0)

    def update(self, decision_vector):
        # 1. Sense
        self.cast_rays()
        
        # 2. Steer (Physics Output)
        if decision_vector is not None:
            # Visual steering is precise, low noise
            self.velocity += decision_vector * 0.2
            
            # Normalize speed
            speed = np.linalg.norm(self.velocity)
            if speed > 0.3: self.velocity = (self.velocity / speed) * 0.3

        # 3. Move
        new_pos = self.pos + self.velocity
        
        # Hard collision stop (if raycasting failed to prevent it)
        hit = False
        for (wx, wy, ww, wh) in self.walls:
            if (new_pos[0] >= wx and new_pos[0] <= wx + ww and
                new_pos[1] >= wy and new_pos[1] <= wy + wh):
                hit = True
        if (new_pos[0] < 0 or new_pos[0] > self.arena_size or
            new_pos[1] < 0 or new_pos[1] > self.arena_size):
            hit = True
            
        if hit:
            self.velocity *= -0.5
        else:
            self.pos = new_pos

        # Goal Check
        dist = np.linalg.norm(self.pos - self.target)
        return dist < 1.0

# --- Simulation Loop ---
def run_wedge_experiment():
    print("Initializing Quantum Wedge Sensor...")
    mt = MicrotubuleSimulator2D(n_protofilaments=13, length_points=64)
    rat = RatWithEyes()
    
    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(1, 2, figure=fig)
    
    # Arena Plot
    ax_arena = fig.add_subplot(gs[0, 0])
    ax_arena.set_title("Wedge Sight (Vision Cone)")
    ax_arena.set_xlim(0, rat.arena_size)
    ax_arena.set_ylim(0, rat.arena_size)
    
    for (wx, wy, ww, wh) in rat.walls:
        rect = patches.Rectangle((wx, wy), ww, wh, facecolor='black')
        ax_arena.add_patch(rect)
        
    rat_dot, = ax_arena.plot([], [], 'ro', zorder=10)
    target_dot, = ax_arena.plot([], [], 'y*', markersize=15)
    
    # Ray Visuals
    ray_lines = []
    for _ in range(rat.num_rays):
        line, = ax_arena.plot([], [], 'g-', alpha=0.5, lw=1)
        ray_lines.append(line)

    # Brain Plot
    ax_brain = fig.add_subplot(gs[0, 1])
    ax_brain.set_title("Quantum Brain (Inhibitory Shadows)")
    ax_brain.set_xlabel("Length")
    ax_brain.set_ylabel("Visual Angle (Protofilament)")
    brain_img = ax_brain.imshow(np.zeros((mt.Ny, mt.Nx)), aspect='auto', cmap='inferno', vmin=0, vmax=5)
    
    print("Running simulation...")

    def update(frame):
        # Inputs
        grid_signal = 0.5 # Keep grid signal constant to highlight vision
        target_map = rat.get_target_bias((mt.Ny, mt.Nx))
        
        # Get Visual Inhibition
        visual_shadow = rat.get_visual_map((mt.Ny, mt.Nx))
        
        spotlight = np.zeros((mt.Ny, mt.Nx))
        spotlight[:, 32] = grid_signal
        
        # Physics Step (With Visual Inhibition)
        decision = None
        final_density = None
        for _ in range(5):
            density, vec = mt.step(spotlight, target_map, visual_shadow)
            final_density = density
            if vec is not None: decision = vec
        
        # Rat Move
        found = rat.update(decision)
        
        # Visuals
        rat_dot.set_data([rat.pos[0]], [rat.pos[1]])
        target_dot.set_data([rat.target[0]], [rat.target[1]])
        
        # Update Rays
        for i, line in enumerate(ray_lines):
            line.set_data([rat.pos[0], rat.rays[i, 0]], [rat.pos[1], rat.rays[i, 1]])
            # Color ray red if hitting wall
            if rat.distances[i] < rat.view_distance:
                line.set_color('red')
            else:
                line.set_color('green')
        
        brain_img.set_data(final_density)
        
        if found:
            print("Goal Reached!")
            ani.event_source.stop()
            
        return [rat_dot, brain_img] + ray_lines

    ani = animation.FuncAnimation(fig, update, frames=600, interval=20, blit=True)
    ani.save('quantum_wedge_sensor.gif', writer='pillow', fps=30)
    print("Simulation saved to quantum_wedge_sensor.gif")

if __name__ == "__main__":
    run_wedge_experiment()