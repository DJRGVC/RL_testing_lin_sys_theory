import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_lyapunov

##############################
# Problem Setup
##############################

A_true = np.array([[1.0, 1.0],
                   [0.0, 1.0]])

B_true = np.array([[0.0],
                   [1.0]])

Q = np.array([[1.0, 0.0],
              [0.0, 0.1]])

R = np.array([[1.0]])

W = 0.01 * np.eye(2)      # process noise
N_horizon = 10            # finite horizon
Tdata = 300               # number of data points for model-based fitting


###########################################
# Utility: simulate system under policy K
###########################################

def rollout(A, B, Klist, x0, noise=True):
    x = x0.copy()
    xs = [x.copy()]
    us = []
    cost = 0.0
    
    for k in range(len(Klist)):
        K = Klist[k]
        u = -K @ x
        us.append(u)
        
        cost += x.T @ Q @ x + u.T @ R @ u
        
        if noise:
            w = np.random.multivariate_normal(mean=np.zeros(2), cov=W)
        else:
            w = np.zeros(2)

        x = A @ x + B @ u + w
        xs.append(x.copy())
    
    cost += xs[-1].T @ Q @ xs[-1]    # terminal cost
    return np.array(xs), np.array(us), cost


############################################################
# Optimal Finite Horizon LQR via Riccati recursion
############################################################

def finite_horizon_lqr(A, B, Q, R, N):
    P = Q.copy()
    Klist = []
    
    # Backward Riccati
    for _ in range(N):
        S = R + B.T @ P @ B
        K = np.linalg.solve(S, B.T @ P @ A)
        Klist.append(K)
        P = Q + A.T @ P @ (A - B @ K)
    
    Klist = Klist[::-1]  # reverse (K_0, ..., K_{N-1})
    return Klist

K_opt = finite_horizon_lqr(A_true, B_true, Q, R, N_horizon)



############################################################
# MODEL-BASED RL: Learn A,B from data (least squares)
############################################################

def collect_data(A, B, noiseW, T=300):
    X = []
    U = []
    Y = []
    
    x = np.random.randn(2)
    
    for _ in range(T):
        u = np.random.randn(1) * 0.4   # random input excitation
        
        X.append(x.copy())
        U.append(u.copy())
        
        w = np.random.multivariate_normal(mean=np.zeros(2), cov=noiseW)
        x_next = A @ x + B @ u + w
        
        Y.append(x_next.copy())
        x = x_next.copy()
    
    X = np.array(X)
    U = np.array(U)
    Y = np.array(Y)
    return X, U, Y

# regression for A,B
def learn_model(X, U, Y):
    Z = np.hstack([X, U])     # shape (T, 3)
    theta = np.linalg.lstsq(Z, Y, rcond=None)[0]   # theta shape: (3, 2) - maps 3→2
    # theta = [A.T; B.T] where rows are features
    Ahat = theta[:2, :].T     # first 2 rows, transposed -> (2, 2)
    Bhat = theta[2:, :].T     # last row(s), transposed -> (2, 1)
    return Ahat, Bhat


##############################
# MODEL-FREE RL (Stable Version)
##############################

def estimate_state_action_covariance(A, B, K, rollout_len=2000):
    """Estimate Σ_K from rollouts."""
    x = np.random.randn(2)
    S = np.zeros((2,2))
    
    for _ in range(rollout_len):
        u = -K @ x
        w = np.random.multivariate_normal([0,0], W)
        x_next = A @ x + B @ u + w
        S += np.outer(x, x)
        x = x_next
    return S / rollout_len

def estimate_costate_matrix(A, B, K):
    """Solve Lyapunov for P_K: P = Q + (A-BK)^T P (A-BK)"""
    M = A - B @ K
    # solve P = Q + M^T P M  -> discrete Lyapunov equation: M^T P M - P = -Q
    P = solve_discrete_lyapunov(M.T, Q)
    return P


def model_free_policy_gradient(A, B, K0, lr=0.001, iters=80):
    K = K0.copy()
    Klist = []
    
    for t in range(iters):
        Σ = estimate_state_action_covariance(A, B, K)
        P = estimate_costate_matrix(A, B, K)
        
        grad = 2 * (R @ K - B.T @ P @ A) @ Σ
        
        K -= lr * grad
        Klist.append(K.copy())
    
    return K, Klist


##############################################
# Run Policies
##############################################

# 1. Optimal (no noise)
x0 = np.array([2.0, 0.5])
xs_opt, us_opt, cost_opt = rollout(A_true, B_true, K_opt, x0, noise=False)

# 2. Model-Based RL
X,U,Y = collect_data(A_true, B_true, W, T=Tdata)
Ahat, Bhat = learn_model(X,U,Y)
K_mb = finite_horizon_lqr(Ahat, Bhat, Q, R, N_horizon)
xs_mb, us_mb, cost_mb = rollout(A_true, B_true, K_mb, x0)

# 3. Model-Free RL
# Initialize with a stabilizing gain (eigenvalues of A-BK should be < 1 in magnitude)
# For A=[[1,1],[0,1]], B=[[0],[1]], we have A-BK = [[1, 1], [0, 1]] - [[0],[1]]@[[k1,k2]]
# A-BK = [[1, 1-k2], [-k1, 1-k1]]
# We need eigenvalues with magnitude < 1 for stability
# Use the LQR optimal gain as initial guess
K_opt_steady = K_opt[0]  # Use first time-step optimal gain
K0 = K_opt_steady.copy()
print(f"Initial gain K0 = {K0}")
print(f"Closed-loop eigenvalues: {np.linalg.eigvals(A_true - B_true @ K0)}")
K_mf, Ktraj = model_free_policy_gradient(A_true, B_true, K0)
Klist_mf = [K_mf for _ in range(N_horizon)]
xs_mf, us_mf, cost_mf = rollout(A_true, B_true, Klist_mf, x0)


###############################################
# Plot trajectories
###############################################

t = np.arange(N_horizon+1)

plt.figure(figsize=(12,6))
plt.subplot(2,2,1)
plt.plot(t, xs_opt[:,0], 'k', label='Optimal')
plt.plot(t, xs_mb[:,0], 'b', label='Model-based')
plt.plot(t, xs_mf[:,0], 'r', label='Model-free')
plt.title("State x1")
plt.legend()

plt.subplot(2,2,2)
plt.plot(t, xs_opt[:,1], 'k')
plt.plot(t, xs_mb[:,1], 'b')
plt.plot(t, xs_mf[:,1], 'r')
plt.title("State x2")

plt.subplot(2,2,3)
plt.step(np.arange(N_horizon), us_opt[:,0], 'k')
plt.step(np.arange(N_horizon), us_mb[:,0], 'b')
plt.step(np.arange(N_horizon), us_mf[:,0], 'r')
plt.title("Control u")

plt.subplot(2,2,4)
plt.bar(["Optimal", "Model-based", "Model-free"], 
        [cost_opt, cost_mb, cost_mf], color=['black','blue','red'])
plt.title("Cost Comparison")

plt.tight_layout()
plt.show()
