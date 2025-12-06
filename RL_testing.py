import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

class LQRSystem:
    """Double integrator system with noise"""
    
    def __init__(self, A, B, Q, R, S, W, N=10):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.S = S
        self.W = W
        self.N = N
        
        self.state_dim = A.shape[0]
        # ensure B is 2D
        self.B = B.reshape(self.state_dim, -1)
        self.control_dim = self.B.shape[1]
        
    def step(self, x, u, noise=True):
        """Single step dynamics"""
        # ensure column shapes are consistent (1D arrays ok)
        x_next = self.A @ x + self.B @ np.atleast_1d(u)
        if noise:
            w = np.random.multivariate_normal(np.zeros(self.state_dim), self.W)
            x_next = x_next + w
        return x_next
    
    def rollout(self, x0, policy, N=None, noise=True):
        """Execute a trajectory"""
        if N is None:
            N = self.N
        
        states = [x0.copy()]
        controls = []
        
        x = x0.copy()
        for k in range(N):
            u = np.atleast_1d(policy(x, k))
            controls.append(u.copy())
            x = self.step(x, u, noise)
            states.append(x.copy())
        
        return np.array(states), np.array(controls)
    
    def compute_cost(self, states, controls):
        """Compute finite horizon cost"""
        cost = 0.0
        N = len(controls)
        
        for k in range(N):
            xk = states[k]
            uk = np.atleast_1d(controls[k])
            cost += float(xk @ self.Q @ xk) + float(uk @ self.R @ uk)
        
        # Terminal cost
        cost += float(states[N] @ self.S @ states[N])
        
        return cost


class OptimalLQR:
    """Optimal LQR solution via dynamic programming"""
    
    def __init__(self, system: LQRSystem):
        self.system = system
        self.gains = None
        
    def solve(self):
        """Solve finite horizon LQR via backward Riccati recursion"""
        A, B, Q, R, S = self.system.A, self.system.B, self.system.Q, self.system.R, self.system.S
        N = self.system.N
        
        # Initialize
        P = S.copy()
        gains = []
        
        # Backward recursion
        for k in range(N-1, -1, -1):
            inv_term = np.linalg.inv(R + B.T @ P @ B)
            K = inv_term @ (B.T @ P @ A)            # shape (m, n) where m = control_dim, n = state_dim
            gains.append(K)
            
            # Update P
            P = Q + A.T @ P @ A - A.T @ P @ B @ inv_term @ (B.T @ P @ A)
        
        # Reverse to get forward order
        self.gains = gains[::-1]
        return self.gains
    
    def get_policy(self):
        """Return policy function"""
        def policy(x, k):
            if self.gains is None or k >= len(self.gains):
                return np.zeros(self.system.control_dim)
            return -self.gains[k] @ x
        return policy


class ModelBasedRL:
    """Model-based RL: Learn system dynamics then solve LQR"""
    
    def __init__(self, system: LQRSystem):
        self.system = system
        self.A_learned = None
        self.B_learned = None
        self.gains = None
        
    def collect_data(self, n_samples, noise_std=1.0):
        """Collect random trajectory data (one-step transitions)"""
        states = []
        next_states = []
        controls = []
        
        for _ in range(n_samples):
            # Random initial state
            x = np.random.randn(self.system.state_dim) * 2.0
            # Random control (vector)
            u = np.random.randn(self.system.control_dim) * noise_std
            
            # Step
            x_next = self.system.step(x, u, noise=True)
            
            states.append(x.copy())
            controls.append(np.atleast_1d(u).copy())
            next_states.append(x_next.copy())
        
        return np.array(states), np.array(controls), np.array(next_states)
    
    def learn_model(self, n_samples, noise_std=1.0):
        """Learn A and B via least squares"""
        states, controls, next_states = self.collect_data(n_samples, noise_std)
        
        # Stack [x, u] as features
        # ensure shapes (n, state_dim) and (n, control_dim)
        states = states.reshape(n_samples, self.system.state_dim)
        controls = controls.reshape(n_samples, self.system.control_dim)
        X = np.hstack([states, controls])             # shape (n, state_dim + control_dim)
        Y = next_states                                # shape (n, state_dim)
        
        # Solve least squares: X @ Theta = Y  -> Theta shape (state_dim+control_dim, state_dim)
        theta = np.linalg.lstsq(X, Y, rcond=None)[0]
        
        # theta[:state_dim, :] is A.T, theta[state_dim:, :] is B.T (because row feature vector multiplies theta)
        n_x = self.system.state_dim
        self.A_learned = theta[:n_x, :].T              # (state_dim, state_dim)
        self.B_learned = theta[n_x:, :].T              # (state_dim, control_dim)
        
        return self.A_learned, self.B_learned
    
    def solve_with_learned_model(self):
        """Solve LQR with learned dynamics"""
        if self.A_learned is None or self.B_learned is None:
            raise ValueError("Must learn model first")
        
        # Create temporary system with learned dynamics
        learned_system = LQRSystem(
            self.A_learned, self.B_learned,
            self.system.Q, self.system.R, self.system.S,
            self.system.W, self.system.N
        )
        
        # Solve LQR
        optimal = OptimalLQR(learned_system)
        self.gains = optimal.solve()
        
        return self.gains
    
    def get_policy(self):
        """Return policy function"""
        def policy(x, k):
            if self.gains is None or k >= len(self.gains):
                return np.zeros(self.system.control_dim)
            return -self.gains[k] @ x
        return policy
    
    def get_model_error(self):
        """Compute model learning error"""
        if self.A_learned is None or self.B_learned is None:
            return np.inf, np.inf
        A_error = np.linalg.norm(self.A_learned - self.system.A, 'fro')
        B_error = np.linalg.norm(self.B_learned - self.system.B, 'fro')
        return A_error, B_error


class ModelFreeRL:
    """Model-free RL: Finite-difference policy gradient with safeguards"""
    def __init__(self, system: LQRSystem, learning_rate=1e-3, gamma=1.0, reg=1e-2):
        self.system = system
        self.lr = learning_rate
        self.gamma = gamma
        self.reg = reg   # L2 regularization on K
        
        # Initialize policy parameters (time-varying linear gains)
        # Start with small random values
        self.K = [np.random.randn(self.system.control_dim, self.system.state_dim) * 0.01
                  for _ in range(system.N)]
        
    def get_policy(self):
        """Return current policy"""
        def policy(x, k):
            if k >= len(self.K):
                return np.zeros(self.system.control_dim)
            return -self.K[k] @ x
        return policy
    
    def _stabilize_gain(self, Kk):
        """Project Kk so that spectral radius of (A - B Kk) < 1 (if possible)"""
        A = self.system.A
        B = self.system.B
        Acl = A - B @ Kk
        eigs = np.linalg.eigvals(Acl)
        radius = max(abs(eigs))
        if radius > 0.995:
            # scale Kk down
            factor = 0.995 / radius
            return Kk * factor
        return Kk
    
    def compute_policy_gradient(self, n_trajectories=30, epsilon=1e-2):
        """Estimate policy gradient via finite differences (baseline + perturbations).
           Lower-variance: smaller epsilon and more trajectories, plus L2 regularization.
        """
        baseline_cost = 0.0
        
        # Evaluate current policy baseline
        for _ in range(n_trajectories):
            x0 = np.random.randn(self.system.state_dim) * 2.0
            states, controls = self.system.rollout(x0, self.get_policy(), noise=True)
            baseline_cost += self.system.compute_cost(states, controls)
        baseline_cost /= n_trajectories
        
        # Compute gradient for each gain matrix
        gradients = []
        
        for t in range(self.system.N):
            grad = np.zeros_like(self.K[t])
            # finite differences per element
            for i in range(self.K[t].shape[0]):
                for j in range(self.K[t].shape[1]):
                    # Perturb parameter
                    self.K[t][i, j] += epsilon
                    cost_plus = 0.0
                    for _ in range(n_trajectories):
                        x0 = np.random.randn(self.system.state_dim) * 2.0
                        states, controls = self.system.rollout(x0, self.get_policy(), noise=True)
                        cost_plus += self.system.compute_cost(states, controls)
                    cost_plus /= n_trajectories
                    # Restore
                    self.K[t][i, j] -= epsilon
                    # Finite difference
                    grad[i, j] = (cost_plus - baseline_cost) / epsilon
            # Add L2 regularization gradient
            grad += 2.0 * self.reg * self.K[t]
            gradients.append(grad)
        
        return gradients, baseline_cost
    
    def train(self, n_iterations=100, n_trajectories=30, verbose=True):
        """Train via policy gradient with adaptive learning rate and stability projection"""
        costs = []
        best_cost = float('inf')
        best_K = [K.copy() for K in self.K]
        patience = 0
        max_patience = 20

        for iteration in range(n_iterations):
            gradients, cost = self.compute_policy_gradient(n_trajectories=n_trajectories)
            costs.append(cost)

            # Track best policy
            if cost < best_cost:
                best_cost = cost
                best_K = [K.copy() for K in self.K]
                patience = 0
            else:
                patience += 1

            # Adaptive learning rate with decay
            current_lr = self.lr / (1 + iteration / 100)

            # Update parameters with gradient clipping and stability projection
            for t in range(self.system.N):
                grad_norm = np.linalg.norm(gradients[t])
                if grad_norm > 1.0:
                    gradients[t] = gradients[t] * (1.0 / grad_norm)

                self.K[t] -= current_lr * gradients[t]

                # Stability projection: ensure A - B K has spectral radius < 1
                self.K[t] = self._stabilize_gain(self.K[t])

                # NaN guard
                if np.any(np.isnan(self.K[t])):
                    print(f"Warning: NaN detected in K[{t}] at iteration {iteration}, resetting...")
                    self.K[t] = best_K[t].copy()

            # Reset to best if stuck
            if patience >= max_patience:
                if verbose:
                    print(f"  No improvement for {max_patience} iterations, resetting to best...")
                self.K = [K.copy() for K in best_K]
                patience = 0

            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}, Cost: {cost:.4f}, Best: {best_cost:.4f}")

        # Restore best policy
        self.K = best_K
        return costs


def run_comparison(system, x0, n_samples_mb=500, n_iterations_mf=100):
    """Run all three methods and compare"""
    
    print("=" * 60)
    print("Running Optimal LQR...")
    print("=" * 60)
    optimal = OptimalLQR(system)
    optimal.solve()
    optimal_policy = optimal.get_policy()
    
    # Evaluate optimal (no noise)
    states_opt, controls_opt = system.rollout(x0, optimal_policy, noise=False)
    cost_opt = system.compute_cost(states_opt, controls_opt)
    print(f"Optimal cost (no noise): {cost_opt:.4f}")
    
    # Evaluate optimal (with noise) - average over multiple runs
    costs_opt_noisy = []
    for _ in range(50):
        states, controls = system.rollout(x0, optimal_policy, noise=True)
        costs_opt_noisy.append(system.compute_cost(states, controls))
    print(f"Optimal cost (with noise): {np.mean(costs_opt_noisy):.4f} ± {np.std(costs_opt_noisy):.4f}")
    
    print("\n" + "=" * 60)
    print(f"Running Model-Based RL (n_samples={n_samples_mb})...")
    print("=" * 60)
    mb_rl = ModelBasedRL(system)
    A_learned, B_learned = mb_rl.learn_model(n_samples_mb)
    
    print("Learned model:")
    print("A_learned =")
    print(A_learned)
    print("B_learned =")
    print(B_learned)
    
    A_error, B_error = mb_rl.get_model_error()
    print(f"\nModel errors: ||A - A_learned|| = {A_error:.6f}, ||B - B_learned|| = {B_error:.6f}")
    
    mb_rl.solve_with_learned_model()
    mb_policy = mb_rl.get_policy()
    
    # Evaluate model-based
    costs_mb = []
    for _ in range(50):
        states, controls = system.rollout(x0, mb_policy, noise=True)
        costs_mb.append(system.compute_cost(states, controls))
    print(f"Model-based cost: {np.mean(costs_mb):.4f} ± {np.std(costs_mb):.4f}")
    
    print("\n" + "=" * 60)
    print(f"Running Model-Free RL (n_iterations={n_iterations_mf})...")
    print("=" * 60)
    mf_rl = ModelFreeRL(system, learning_rate=1e-3, gamma=1.0, reg=1e-2)
    training_costs = mf_rl.train(n_iterations=n_iterations_mf, n_trajectories=20, verbose=True)
    mf_policy = mf_rl.get_policy()
    
    # Evaluate model-free
    costs_mf = []
    for _ in range(50):
        states, controls = system.rollout(x0, mf_policy, noise=True)
        costs_mf.append(system.compute_cost(states, controls))
    print(f"Model-free cost: {np.mean(costs_mf):.4f} ± {np.std(costs_mf):.4f}")
    
    # Generate trajectories for plotting
    states_opt, controls_opt = system.rollout(x0, optimal_policy, noise=False)
    states_opt_noisy, controls_opt_noisy = system.rollout(x0, optimal_policy, noise=True)
    states_mb, controls_mb = system.rollout(x0, mb_policy, noise=True)
    states_mf, controls_mf = system.rollout(x0, mf_policy, noise=True)
    
    return {
        'optimal': {'states': states_opt, 'controls': controls_opt, 'cost': cost_opt},
        'optimal_noisy': {'states': states_opt_noisy, 'controls': controls_opt_noisy, 
                          'cost': np.mean(costs_opt_noisy)},
        'model_based': {'states': states_mb, 'controls': controls_mb, 
                        'cost': np.mean(costs_mb), 'model_error': (A_error, B_error)},
        'model_free': {'states': states_mf, 'controls': controls_mf, 
                       'cost': np.mean(costs_mf), 'training_costs': training_costs}
    }


def plot_comparison(results, x0, save_path=None):
    """Plot trajectories and costs"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # State trajectories x1
    ax = axes[0, 0]
    time = np.arange(len(results['optimal']['states']))
    ax.plot(time, results['optimal']['states'][:, 0], 'k-', label='Optimal (no noise)', linewidth=2)
    ax.plot(time, results['optimal_noisy']['states'][:, 0], 'k--', label='Optimal (noisy)', alpha=0.7)
    ax.plot(time, results['model_based']['states'][:, 0], 'b-', label='Model-based', linewidth=2)
    ax.plot(time, results['model_free']['states'][:, 0], 'r-', label='Model-free', linewidth=2)
    ax.set_xlabel('Time step')
    ax.set_ylabel('State x₁')
    ax.set_title(f'State x₁ Trajectory (x₀ = [{x0[0]:.2f}, {x0[1]:.2f}])')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # State trajectories x2
    ax = axes[0, 1]
    ax.plot(time, results['optimal']['states'][:, 1], 'k-', label='Optimal (no noise)', linewidth=2)
    ax.plot(time, results['optimal_noisy']['states'][:, 1], 'k--', label='Optimal (noisy)', alpha=0.7)
    ax.plot(time, results['model_based']['states'][:, 1], 'b-', label='Model-based', linewidth=2)
    ax.plot(time, results['model_free']['states'][:, 1], 'r-', label='Model-free', linewidth=2)
    ax.set_xlabel('Time step')
    ax.set_ylabel('State x₂')
    ax.set_title('State x₂ Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Control trajectories
    ax = axes[1, 0]
    time_u = np.arange(len(results['optimal']['controls']))
    ax.plot(time_u, results['optimal']['controls'].squeeze(), 'k-', label='Optimal (no noise)', linewidth=2)
    ax.plot(time_u, results['optimal_noisy']['controls'].squeeze(), 'k--', label='Optimal (noisy)', alpha=0.7)
    ax.plot(time_u, results['model_based']['controls'].squeeze(), 'b-', label='Model-based', linewidth=2)
    ax.plot(time_u, results['model_free']['controls'].squeeze(), 'r-', label='Model-free', linewidth=2)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Control u')
    ax.set_title('Control Input Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Cost comparison
    ax = axes[1, 1]
    methods = ['Optimal\n(no noise)', 'Optimal\n(noisy)', 'Model-based', 'Model-free']
    costs = [
        results['optimal']['cost'],
        results['optimal_noisy']['cost'],
        results['model_based']['cost'],
        results['model_free']['cost']
    ]
    colors = ['black', 'gray', 'blue', 'red']
    bars = ax.bar(methods, costs, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Total Cost')
    ax.set_title('Cost Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add cost values on bars
    for bar, cost in zip(bars, costs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{cost:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def sample_efficiency_experiment(system, x0, sample_sizes, n_trials=10):
    """Compare sample efficiency of model-based vs model-free"""
    mb_costs = {n: [] for n in sample_sizes}
    mb_errors_A = {n: [] for n in sample_sizes}
    mb_errors_B = {n: [] for n in sample_sizes}
    
    print("\n" + "=" * 60)
    print("Running Sample Efficiency Experiment")
    print("=" * 60)
    
    for n_samples in sample_sizes:
        print(f"\nSample size: {n_samples}")
        
        for trial in range(n_trials):
            # Model-based
            mb_rl = ModelBasedRL(system)
            mb_rl.learn_model(n_samples)
            A_err, B_err = mb_rl.get_model_error()
            mb_rl.solve_with_learned_model()
            mb_policy = mb_rl.get_policy()
            
            # Evaluate
            states, controls = system.rollout(x0, mb_policy, noise=True)
            cost = system.compute_cost(states, controls)
            
            mb_costs[n_samples].append(cost)
            mb_errors_A[n_samples].append(A_err)
            mb_errors_B[n_samples].append(B_err)
        
        print(f"  Model-based: {np.mean(mb_costs[n_samples]):.4f} ± {np.std(mb_costs[n_samples]):.4f}")
        print(f"  Model error A: {np.mean(mb_errors_A[n_samples]):.6f}")
        print(f"  Model error B: {np.mean(mb_errors_B[n_samples]):.6f}")
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Cost vs samples
    ax = axes[0]
    means = [np.mean(mb_costs[n]) for n in sample_sizes]
    stds = [np.std(mb_costs[n]) for n in sample_sizes]
    ax.errorbar(sample_sizes, means, yerr=stds, marker='o', capsize=5, linewidth=2, label='Model-based')
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Cost')
    ax.set_title('Sample Efficiency: Cost vs Data')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Model error vs samples
    ax = axes[1]
    means_A = [np.mean(mb_errors_A[n]) for n in sample_sizes]
    means_B = [np.mean(mb_errors_B[n]) for n in sample_sizes]
    ax.plot(sample_sizes, means_A, 'o-', linewidth=2, label='||A - Â||')
    ax.plot(sample_sizes, means_B, 's-', linewidth=2, label='||B - B̂||')
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Frobenius Norm Error')
    ax.set_title('Model Learning Error vs Data')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    return mb_costs, mb_errors_A, mb_errors_B


# Main execution
if __name__ == "__main__":
    # System setup
    A = np.array([[1, 1], [0, 1]])
    B = np.array([[0], [1]])
    Q = np.array([[1, 0], [0, 0.1]])
    R = np.array([[1]])
    S = Q.copy()
    W = 0.01 * np.eye(2)
    N = 10
    
    system = LQRSystem(A, B, Q, R, S, W, N)
    
    # Initial condition
    x0 = np.array([2.0, 0.5])
    
    # Run comparison
    results = run_comparison(system, x0, n_samples_mb=500, n_iterations_mf=100)
    
    # Plot results
    plot_comparison(results, x0)
    
    # Sample efficiency experiment
    sample_sizes = [50, 100, 200, 500, 1000]
    sample_efficiency_experiment(system, x0, sample_sizes, n_trials=5)
    
    print("\n" + "=" * 60)
    print("Experiments completed!")
    print("=" * 60)