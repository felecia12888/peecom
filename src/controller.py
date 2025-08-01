# src/controller.py

from typing import Dict, Optional
import cvxpy as cp
import numpy as np

class ModelPredictiveController:
    """
    Model Predictive Controller for PEECOM.
    Uses a simple linear dynamical model and CVXPY to optimize control inputs.
    """
    def __init__(self, system_model, config: Dict):
        self.model = system_model
        # Expecting config keys under `mpc` and `model` sections
        mpc_cfg = config['mpc']
        self.horizon = mpc_cfg['horizon']
        self.min_u, self.max_u = mpc_cfg['min_action'], mpc_cfg['max_action']
        # input_shape is not used by the CVXPY problem directly, 
        # but could guide reshaping if needed
        self.input_shape = tuple(config['model']['input_shape'])

    def optimize(
        self,
        state_history: np.ndarray,
        attention_label: Optional[float] = None
    ) -> float:
        """
        Solve an MPC problem over `self.horizon` timesteps.
        Returns the first control action in the optimal sequence.
        """
        # The CVXPY problem may occasionally fail; catch exceptions
        try:
            # Current state is the last row of the history
            x0 = state_history[-1].flatten()
            n = x0.shape[0]

            # Decision variables: controls U[0..H-1], states X[0..H]
            U = cp.Variable(self.horizon)
            X = cp.Variable((self.horizon + 1, n))

            # Cost: heavy penalty on last state component (e.g., accumulator),
            # moderate on other states, plus control effort & smoothness
            cost = (
                5.0 * cp.sum_squares(X[:, -1]) +
                2.0 * cp.sum_squares(X[:, :n-1]) +
                0.1 * cp.sum_squares(U) +
                0.5 * cp.sum_squares(cp.diff(U))
            )

            # Constraints
            constr = []
            constr += [X[0] == x0]
            constr += [U >= self.min_u, U <= self.max_u]
            constr += [cp.abs(cp.diff(U)) <= 0.2]

            # Simple linear dynamics: X[t+1] = X[t] + 0.1 * U[t]
            for t in range(self.horizon):
                constr.append(X[t+1] == X[t] + 0.1 * U[t])

            # Build and solve
            prob = cp.Problem(cp.Minimize(cost), constr)
            prob.solve(solver=cp.OSQP, warm_start=True)

            if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                # Return first control action
                return float(U.value[0])
            else:
                # fallback if solver did not converge
                return 0.0

        except Exception as e:
            print(f"[MPC] Optimization error: {e}")
            return 0.0
