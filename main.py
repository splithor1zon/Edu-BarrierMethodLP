#%% Load imports and all required methods
'''
This code depends on three primary packages: numpy, matplotlib, cvxpy
You can either install them directly, or what is recommended is to create a virtual python environment (venv) for this project.
To use interactive python mode in VS Code (#%%), you need to install Jupyter extension.
'''

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cvxpy
from cvxpy import Variable, Minimize, Problem


'''
A, b, c, x0 = minimization problem variables
alpha, beta = backtracking parameters (0 < alpha < 0.5) (0 < beta < 1) [cvxbook p.464]
maxiter = interrupt the calculation in a defined number of steps
epsilon = stopping criterion [cvxbook p.487]
lambdasq = square of Newton decrement [cvxbook p.486-487]
deltax =  Newton step [cvxbook p.484]
x* = primal optimal point
nu = dual optimal point
lambda_log = record of lambdasq/2 for each Newton step
'''
def newton_centering(A, b, c, x0, alpha = 0.01, beta = 0.5, maxiter = 1000, epsilon = 1e-6, fcheck = True):
    '''
    minimize        c^T @ x - sum(log(x))
    subject to      A @ x = b

    Newton's method centering function. Accepts only strictly feasible starting point x0.
    Returns primal and dual optimal points with log of lambda^2/2 for each step. 
    '''
    lambda_log = []
    x = x0

    if fcheck:
        # Feasibility check (disabled for Phase 2)
        if np.min(x0) <= 0 or np.linalg.norm(A @ x0 - b) >= epsilon:
            print("Provided x0 vector is not feasible")
            return np.array([]), np.array([]), lambda_log
    
    for i in range(maxiter):
        # Calculate gradient
        gradient = c - x ** (-1)
        x_diag_sq = np.diag(x.reshape(x.shape[0]) ** 2)

        '''
        If desirable, lambdasq can be calculated using hessian, but it is easier to calculate it using just gradient.
        If the x is nearly optimal, this approach can cause that lambdasq is negative.
        This would mean a failure to progress further. (complex sqrt)
        For larger epsilon i.e. 1e-6 this shouldn't be a problem.
        '''
        # hessian = np.diag(x.reshape(x.shape[0]) ** (-2))

        # Calculate Newton step using elimination method
        nu = np.linalg.lstsq(A @ x_diag_sq @ A.transpose(), (-A @ x_diag_sq @ gradient), rcond=None)[0]
        deltax = -x_diag_sq @ (A.transpose() @ nu + gradient)

        # Calculate Newton decrement squared.
        lambdasq = -gradient.transpose() @ deltax
        # Alternative calculation using hessian.
        # lambdasq = deltax.transpose() @ hessian @ deltax

        lambda_log.append(lambdasq / 2)

        # Stopping condition
        if lambdasq / 2 <= epsilon:
            return x, nu, lambda_log

        # Backtracking line search
        t = 1
        while np.min(x + t * deltax) <= 0:
            t *= beta
        while c.transpose() @ (t * deltax) - np.sum(np.log(x + t * deltax)) > -np.sum(np.log(x)) + alpha * t * (gradient.transpose() @ deltax):
            t *= beta

        x += t * deltax

    print("Solution within provided stopping condition was not found, maxiter condition reached.")
    return np.array([]), np.array([]), lambda_log

def lp_barrier_method(A, b, c, x0, t0 = 1, mu = 50, epsilon = 1e-3, fcheck = True):
    '''
    minimize        c^T @ x
    subject to      A @ x = b
                    x >= 0

    Barrier method is requiring strictly feasible x0.
    Uses centering provided by newton_centering function.
    Returns primal optimal point x*, list of newton step count for each iteration, and list of duality gaps for each iteration.
    '''
    t = t0
    x = x0
    n = x.shape[0]
    ns_count = []
    gaps = []
    gap = float(n) / t
    while True:
        # Centering step
        xs, nus, lambda_log = newton_centering(A, b, t * c, x, fcheck=fcheck)

        # Feasibility check
        if len(xs) == 0:
            return np.array([]), gap, ns_count, gaps

        # Update variables
        x = xs
        gap = float(n) / t

        ns_count.append(len(lambda_log))
        gaps.append(gap)

        # Stopping condition
        if gap < epsilon:
            return xs, gap, ns_count, gaps

        t *= mu

def lp_solve(A, b, c):
    '''
    minimize        c^T @ x
    subject to      A @ x = b
                    x >= 0

    Encapsulating Phase 1 and 2 function, accepts just three parameters.
    First, the function calculates strictly feasible starting point (if possible) using a Phase 1 method.
    Returns primal optimal point x*, primal optimal value p*, latest identified duality gap, and number of newton steps for each phase.
    '''
    m, n = A.shape
    step_num = [0, 0]

    # PHASE 1
    '''
    minimize        t
    subject to      A @ x = b
                    x >= (1 - t) @ 1
                    t >= 0

    First we initialize x0 with any one satisfying  A @ x0 = b.
    If min(x0) <= 0, then we already have strictly feasible point and we can skip the phase 1.
    (in this example we continue anyway)
    Then we initialize t0 = 2 - min(x0).
    Using z0 we transform this problem to a compatible one with the lp_barrier_method.
    '''
    x0 = np.linalg.lstsq(A, b, rcond=None)[0]
    t0 = 2 + max(0, -np.min(x0))
    A1 = np.hstack((A, -(A @ np.ones(n)).reshape(m, 1)))
    b1 = b - (A @ np.ones(n)).reshape(m, 1)
    z0 = x0.reshape(n, 1) + t0 * np.ones((n, 1)) - np.ones((n, 1))
    c1 = np.vstack((np.zeros((n, 1)), np.ones((1, 1))))
    x0 = np.vstack((z0, t0)).reshape(n + 1, 1)
    zs, gap, ns_count, gaps = lp_barrier_method(A1, b1, c1, x0)

    step_num[0] = np.sum(ns_count)
    if len(zs) == 0:
        print("PHASE 1: INFEASIBLE")
        return np.array([]), np.inf, np.inf, step_num
    
    x0 = zs[:n] - zs[n][0] * np.ones((n, 1)) + np.ones((n, 1))

    # PHASE 2
    xs, gap, ns_count, gaps = lp_barrier_method(A, b, c, x0, fcheck = False)
    step_num[1] = np.sum(ns_count)
    if len(xs) == 0:
        print("PHASE 2: INFEASIBLE")
        return np.array([]), np.inf, np.inf, step_num
    
    ps = c.reshape(len(c)) @ xs.reshape(len(c))
    return xs, ps, gap, step_num


#%% Testing code of lp_barrier_method function plots lambdasq/2 progression over iterations
# Matrix A size
m, n = 10, 200
# To ensure comparable results with students set the random seed to for example '1'.
np.random.seed(None)

# Generate feasible starting values
A = np.vstack((np.random.randn(m - 1, n), np.ones((1, n))))
x0 = np.random.rand(n, 1) + 0.1
b = A @ x0
c = np.random.randn(n, 1)

#x, nu, logg = newton_step(A, b, c, x0)
xs, gap, ns_count, gaps = lp_barrier_method(A, b, c, x0)
fig, ax = plt.subplots()
plt.yscale("log")
plt.xlabel("iteration")
plt.ylabel("lambda^2/2")
ax.step(np.cumsum(ns_count), gaps, where="post")
plt.show()

#%% Testing code with infeasible starting values. + Comparison with CVXPY.
# Matrix A size
m, n = 100, 500
# To ensure comparable results with students set the random seed to for example '1'.
np.random.seed(None)

# Initialize variables
A = np.vstack((np.random.randn(m - 1, n), np.ones((1, n))))
b = np.random.randn(m)
c = np.random.randn(n, 1)

# USE: Implemented barrier method
xs, ps, gap, step_num = lp_solve(A, b, c)
print("Barrier: ", ps)

# USE: CVXPY library - Use the same variables
c = np.squeeze(c) # Need to flatten the variable from (n,1) to (n) for cvxpy
x = Variable(n)
obj = Minimize(cvxpy.sum(cvxpy.multiply(c, x)))
prob = Problem(obj, [A @ x == b, x >= 0])
prob.solve()
print("CVXPY status: " , prob.status)
print("CVXPY value: " , prob.value)

#%% Testing code with feasible starting values. + Comparison with CVXPY.
# Matrix A size
m, n = 100, 500
# To ensure comparable results with students set the random seed to for example '1'.
np.random.seed(None)

# Initialize variables
A = np.vstack((np.random.randn(m - 1, n), np.ones((1, n))))
v = np.random.rand(n, 1) + 0.1
b = A @ v
c = np.random.randn(n, 1)

# USE: Implemented barrier method
xs, ps, gap, step_num = lp_solve(A, b, c)
print("Barrier value: ", ps)

# USE: CVXPY library - same variables
v = np.squeeze(v)
c = np.squeeze(c) # Need to flatten the variables from (n,1) to (n) for cvxpy
b = A @ v
x = Variable(n)

obj = Minimize(cvxpy.sum(cvxpy.multiply(c, x)))
prob = Problem(obj, [A @ x == b, x >= 0])
prob.solve()
print("CVXPY status: " , prob.status)
print("CVXPY value: " , prob.value)

#%%