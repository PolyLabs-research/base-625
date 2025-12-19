import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches

# --- Physics Engine (Unchanged) ---
COMPLEX_TYPE = np.complex128

class MicrotubuleSimulator2D:
    def __init__(self, n_protofilaments=13, length_points=64, dt=0.05, 
                 alpha=0.5, gamma=2.0, base_pump=1.0, decay=0.1):
        self.Ny = n_protofilaments
        self.Nx = length_points
        self.dt = dt
        self.alpha = alpha
        self.gamma = gamma
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

    def step(self, spatial_input, target_bias_map, frustration_level):
        # Frustration increases the global pump (Temperature)
        # This creates chaos/noise, allowing "irrational" decisions
        P_total = self.base_pump + spatial_input + (target_bias_map * 0.5) + (frustration_level * 2.0)
        
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

# --- The Quantum Rat (With Persistence) ---
class QuantumRat:
    def __init__(self, arena_size=20.0):
        self.pos = np.array([10.0, 2.0]) # Start at bottom
        self.velocity = np.array([0.0, 0.0])
        self.arena_size = arena_size
        self.target = np.array([10.0, 18.0]) # Cheese at top
        
        # Frustration State
        self.frustration = 0.0
        self.stuck_counter = 0
        # Decreased decay further to allow longer "irrational" behavior
        self.frustration_decay = 0.001 # Was 0.002
        
        # Walls (x, y, width, height) - The U-Trap
        self.walls = [
            (5, 10, 10, 1),  # Bottom bar of U
            (5, 10, 1, 6),   # Left arm
            (14, 10, 1, 6)   # Right arm
        ]
        
        self.k_scale = 1.0
        self.wave_vecs = np.array([[np.cos(a), np.sin(a)] for a in [0, np.pi/3, 2*np.pi/3]]) * self.k_scale

    def get_grid_pattern(self):
        waves = np.sum([np.cos(np.dot(self.pos, k)) for k in self.wave_vecs])
        return (waves + 1.5) / 4.5

    def get_target_bias(self, mt_shape):
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

    def check_collision(self, new_pos):
        # Check Walls
        for (wx, wy, ww, wh) in self.walls:
            if (new_pos[0] >= wx and new_pos[0] <= wx + ww and
                new_pos[1] >= wy and new_pos[1] <= wy + wh):
                return True
        # Check Arena Borders
        if (new_pos[0] < 0 or new_pos[0] > self.arena_size or
            new_pos[1] < 0 or new_pos[1] > self.arena_size):
            return True
        return False

    def update(self, decision_vector):
        # 1. Apply Quantum Steering
        if decision_vector is not None:
            # If frustrated, steering is erratic (high gain)
            # Increased gain multiplier to encourage stronger turns
            steer_force = decision_vector * (0.5 + self.frustration * 2.0) 
            self.velocity += steer_force
            
            # Normalize speed
            speed = np.linalg.norm(self.velocity)
            max_speed = 0.5
            if speed > max_speed:
                self.velocity = (self.velocity / speed) * max_speed

        # 2. Attempt Move
        new_pos = self.pos + self.velocity
        
        if self.check_collision(new_pos):
            # HIT WALL!
            self.velocity *= -0.5 # Bounce/Stop
            self.stuck_counter += 1
            # Increased frustration gain to trigger faster escape response
            self.frustration = min(1.0, self.frustration + 0.1) # Was 0.05
        else:
            self.pos = new_pos
            # If moving well, reduce frustration slowly
            self.stuck_counter = max(0, self.stuck_counter - 1)
            # Slower decay allows persistence in "Exploration Mode"
            self.frustration = max(0.0, self.frustration - self.frustration_decay)

        # 3. Check Goal
        dist = np.linalg.norm(self.pos - self.target)
        return dist < 1.0, self.frustration

# --- Main Simulation ---
def run_maze():
    print("Initializing Quantum Maze Solver...")
    mt = MicrotubuleSimulator2D(n_protofilaments=13, length_points=64)
    rat = QuantumRat()
    
    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(1, 2, figure=fig)
    
    # Arena Plot
    ax_arena = fig.add_subplot(gs[0, 0])
    ax_arena.set_title("Deceptive Maze (U-Trap)")
    ax_arena.set_xlim(0, rat.arena_size)
    ax_arena.set_ylim(0, rat.arena_size)
    
    # Draw Walls
    for (wx, wy, ww, wh) in rat.walls:
        rect = patches.Rectangle((wx, wy), ww, wh, linewidth=1, edgecolor='black', facecolor='black')
        ax_arena.add_patch(rect)
    
    rat_dot, = ax_arena.plot([], [], 'ro', markersize=12, label='Rat')
    target_dot, = ax_arena.plot([], [], 'y*', markersize=15, label='Cheese')
    path_line, = ax_arena.plot([], [], 'r-', alpha=0.3, lw=1)
    path_x, path_y = [], []
    
    # Frustration Bar
    frust_bar = ax_arena.bar([1], [0], width=0.5, color='orange', align='center', label='Stress')
    ax_arena.text(1, 18, "Stress Level", ha='center')

    # Brain Plot
    ax_brain = fig.add_subplot(gs[0, 1])
    ax_brain.set_title("Quantum Brain Activity")
    ax_brain.set_xlabel("Length")
    ax_brain.set_ylabel("Angle (Steering)")
    brain_img = ax_brain.imshow(np.zeros((mt.Ny, mt.Nx)), aspect='auto', cmap='inferno', vmin=0, vmax=8)
    
    print("Starting simulation...")

    def update(frame):
        nonlocal path_x, path_y
        
        # Inputs
        grid_signal = rat.get_grid_pattern()
        target_map = rat.get_target_bias((mt.Ny, mt.Nx))
        spotlight = np.zeros((mt.Ny, mt.Nx))
        spotlight[:, 32] = grid_signal * 2.0
        
        # Physics Step (with Frustration Input)
        decision = None
        final_density = None
        for _ in range(5):
            total_input = spotlight + target_map
            density, vec = mt.step(total_input, target_map, rat.frustration)
            final_density = density
            if vec is not None: decision = vec
        
        # Rat Update
        success, frustration = rat.update(decision)
        
        # Visuals
        rat_dot.set_data([rat.pos[0]], [rat.pos[1]])
        target_dot.set_data([rat.target[0]], [rat.target[1]])
        path_x.append(rat.pos[0])
        path_y.append(rat.pos[1])
        path_line.set_data(path_x, path_y)
        brain_img.set_data(final_density)
        
        # Update Stress Bar (Hack to update bar height)
        frust_bar[0].set_height(frustration * 5) # Scale for visibility
        
        if success:
            print("Goal Reached!")
            ani.event_source.stop()
            
        return rat_dot, brain_img, path_line, frust_bar[0]

    ani = animation.FuncAnimation(fig, update, frames=1500, interval=20, blit=True)
    ani.save('quantum_maze_solver.gif', writer='pillow', fps=30)
    print("Simulation saved to quantum_maze_solver.gif")

if __name__ == "__main__":
    run_maze()