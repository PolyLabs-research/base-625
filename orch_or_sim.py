import numpy as np
import matplotlib.pyplot as plt
import time
import networkx as nx

# --- Constants ---

# Meso-Scale (Phase 4)
NUM_NEURONS = 100
GRID_SIZE = 10

# Physics
COMPLEX_TYPE = np.complex64
H_BAR = 0.4
LATTICE_ROWS = 13
LATTICE_COLS = 100
DT = 0.1
COUPLING_J = 0.15 # Lowered for hub-based topology
INPUT_GAIN = 10.0 # Increased to make signal louder
PHASE_DAMPENING = 0.99
BASELINE_ENERGY = (LATTICE_ROWS * LATTICE_COLS) / 2
GAP_JUNCTION_ENERGY = 1.0

# --- Batched Physics Engine ---
class BatchedMicrotubules:
    def __init__(self, num_neurons, coupling_j, seed=None):
        self.num_neurons = num_neurons
        self.rng = np.random.default_rng(seed)
        self.h_bar = H_BAR
        self.coupling_j = coupling_j
        self.reset()

    def reset(self):
        angles = self.rng.uniform(0, 2 * np.pi, (self.num_neurons, LATTICE_ROWS, LATTICE_COLS))
        self.lattices = (np.cos(angles) + 1j * np.sin(angles)).astype(COMPLEX_TYPE)
        self.coherence_times = np.zeros(self.num_neurons, dtype=np.float32)

    def _convolve_manual(self, lattice_batch):
        neighbor_influence = np.zeros_like(lattice_batch, dtype=COMPLEX_TYPE)
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0:
                    continue
                neighbor_influence += np.roll(lattice_batch, shift=(i, j), axis=(1, 2))
        return neighbor_influence

    def evolve(self, input_energies):
        # Intra-lattice evolution (within each neuron)
        neighbor_influence = self._convolve_manual(self.lattices)
        hamiltonian_psi = self.coupling_j * neighbor_influence + input_energies
        delta_psi = (-1j / self.h_bar) * hamiltonian_psi * DT
        delta_psi *= PHASE_DAMPENING
        self.lattices += delta_psi
        
        magnitudes = np.abs(self.lattices)
        mask = magnitudes > 1.0
        self.lattices[mask] /= magnitudes[mask]
        
        self.coherence_times += DT

    def calculate_Eg(self):
        phases = np.angle(self.lattices)
        return np.std(phases, axis=(1, 2))

    def check_collapse(self):
        E_gs = self.calculate_Eg()
        collapsed_mask = (E_gs * self.coherence_times) >= self.h_bar
        if not np.any(collapsed_mask):
            return collapsed_mask, None
        
        num_collapsed = np.sum(collapsed_mask)
        probs = np.abs(self.lattices[collapsed_mask])**2
        rand_vals = self.rng.random(probs.shape, dtype=np.float32)
        collapsed_states = (probs > rand_vals).astype(np.int8)
        output_levels = np.sum(collapsed_states, axis=(1, 2)) - BASELINE_ENERGY
        
        new_angles = self.rng.uniform(0, 2 * np.pi, (num_collapsed, LATTICE_ROWS, LATTICE_COLS))
        self.lattices[collapsed_mask] = (np.cos(new_angles) + 1j * np.sin(new_angles)).astype(COMPLEX_TYPE)
        self.coherence_times[collapsed_mask] = 0.0
        
        return collapsed_mask, output_levels

# --- Meso-Scale Layer ---
class OrchLayer:
    def __init__(self, num_neurons=100, coupling_j=0.15, input_gain=10.0, seed=42):
        self.num_neurons = num_neurons
        self.rng = np.random.default_rng(seed)
        self.input_gain = input_gain
        self.microtubules = BatchedMicrotubules(num_neurons, coupling_j, seed)
        self.synaptic_weights = self.rng.uniform(-0.5, 0.5, (num_neurons, LATTICE_ROWS, LATTICE_COLS)).astype(np.float32)
        self.refractory_timers = np.zeros(num_neurons, dtype=int)
        
        # Generate Scale-Free Graph for inter-neuron connectivity
        print("Generating Barabási–Albert scale-free graph...")
        self.graph = nx.barabasi_albert_graph(n=num_neurons, m=2, seed=seed)

    def forward(self, signal_vector, max_steps=100):
        fire_threshold = 50.0
        spikes = []
        feedback_energy = np.zeros_like(self.microtubules.lattices, dtype=COMPLEX_TYPE)
        signal_vector_b = signal_vector.reshape(self.num_neurons, 1, 1)

        for t in range(max_steps):
            active_mask = (self.refractory_timers == 0)
            active_mask_b = active_mask.reshape(self.num_neurons, 1, 1)
            self.refractory_timers[self.refractory_timers > 0] -= 1

            input_energy = signal_vector_b * self.synaptic_weights * self.input_gain
            total_energy = (input_energy + feedback_energy) * active_mask_b
            self.microtubules.evolve(total_energy)
            self.microtubules.coherence_times[~active_mask] = 0
            
            collapsed_mask, output_levels = self.microtubules.check_collapse()
            feedback_energy.fill(0)
            
            if np.any(collapsed_mask):
                collapsed_indices = np.where(collapsed_mask)[0]
                for i, neuron_idx in enumerate(collapsed_indices):
                    spikes.append((t, neuron_idx))
                    
                    if output_levels[i] > fire_threshold:
                        self.refractory_timers[neuron_idx] = 10
                        
                        # Excite neighbors based on the scale-free graph topology
                        for neighbor_idx in self.graph.neighbors(neuron_idx):
                            feedback_energy[neighbor_idx] += GAP_JUNCTION_ENERGY
        return spikes

# --- Experiment & Visualization ---
def generate_input_data(grid_size, noise_level):
    grid = np.zeros((grid_size, grid_size), dtype=int)
    for i in range(grid_size):
        for j in range(i + 1):
            grid[i, j] = 1
    noise_mask = np.random.rand(grid_size, grid_size) < noise_level
    grid[noise_mask] = 1 - grid[noise_mask]
    return grid.flatten()

def plot_raster(spikes, title):
    if not spikes:
        print(f"No spikes to plot for {title}.")
        return
    events_by_neuron = [[] for _ in range(NUM_NEURONS)]
    for t, neuron_id in spikes:
        events_by_neuron[neuron_id].append(t)
    
    plt.figure(figsize=(12, 6))
    plt.eventplot(events_by_neuron, lineoffsets=np.arange(NUM_NEURONS), linelengths=0.9, colors='black')
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Neuron ID")
    plt.yticks(np.arange(0, NUM_NEURONS, 10))
    plt.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5)
    
    filename = f"raster_plot_{title.replace(' ', '_').lower()}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved raster plot to {filename}")

def run_scale_free_simulation():
    """Main function to run the Barabási-Albert graph simulation."""
    print("--- Running Phase 4: Scale-Free Network Simulation ---")
    
    # Initialize the layer with the new topology and tuned parameters
    layer = OrchLayer(num_neurons=NUM_NEURONS, coupling_j=COUPLING_J, input_gain=INPUT_GAIN, seed=42)

    # 1. Experiment with hidden triangle signal
    print("\n[Experiment 1: Hidden Triangle]")
    noisy_triangle_input = generate_input_data(GRID_SIZE, noise_level=0.4)
    start_time = time.time()
    triangle_spikes = layer.forward(noisy_triangle_input)
    end_time = time.time()
    
    num_collapses_tri = len(triangle_spikes)
    print(f"Triangle Sim Duration: {end_time - start_time:.2f}s")
    print(f"Result: {num_collapses_tri} total collapses.")

    # 2. Experiment with pure random noise
    # Re-initialize layer for a fair comparison
    layer = OrchLayer(num_neurons=NUM_NEURONS, coupling_j=COUPLING_J, input_gain=INPUT_GAIN, seed=42)
    print("\n[Experiment 2: Pure Random Noise]")
    random_input = np.random.randint(0, 2, NUM_NEURONS)
    start_time = time.time()
    random_spikes = layer.forward(random_input)
    end_time = time.time()

    num_collapses_rand = len(random_spikes)
    print(f"Random Sim Duration: {end_time - start_time:.2f}s")
    print(f"Result: {num_collapses_rand} total collapses.")

    # 3. Hypothesis Check
    print("\n--- Hypothesis Check ---")
    if num_collapses_rand > 0:
        collapse_diff = ((num_collapses_tri - num_collapses_rand) / float(num_collapses_rand)) * 100
    else:
        collapse_diff = float('inf') if num_collapses_tri > 0 else 0.0
    
    print(f"Total Collapses: Triangle: {num_collapses_tri}, Random: {num_collapses_rand}")
    print(f"Collapse difference vs random: {collapse_diff:+.2f}%")
    
    # 4. Visualization
    plot_raster(triangle_spikes, title="Spike Raster with Triangle (Scale-Free)")
    plot_raster(random_spikes, title="Spike Raster with Random Noise (Scale-Free)")

if __name__ == "__main__":
    run_scale_free_simulation()
