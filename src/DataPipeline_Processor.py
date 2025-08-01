# src/controller.py

from typing import Dict, Tuple
import numpy as np
import cvxpy as cp
import pandas as pd

# Import your data processor from the same package (relative import)
from PEECOM_data import PEECOMDataProcessor

class DataPipelineProcessor:
    """
    Wrapper around the real PEECOM_data processor to produce
    feature matrix X and target DataFrame y in PEECOM format.
    """
    def __init__(self, config_path: str):
        self.processor = PEECOMDataProcessor(config_path)

    def process_all(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # PEECOMDataProcessor.process() returns X_train, X_test, y_train, y_test
        X_train, X_test, y_train, y_test = self.processor.process()

        # Concatenate to full datasets
        X = pd.concat([X_train, X_test], ignore_index=True)
        y = pd.concat([y_train, y_test], ignore_index=True)

        return X, y


class ModelPredictiveController:
    """
    MPC aligned to your `mpc:` YAML section, using CVXPY.
    """
    def __init__(self, system_model, config: Dict):
        self.model = system_model
        mpc_cfg = config['mpc']
        self.horizon = mpc_cfg['horizon']
        self.min_u, self.max_u = mpc_cfg['min_action'], mpc_cfg['max_action']
        self.input_shape = tuple(config['model']['input_shape'])

    def optimize(
        self,
        state_history: np.ndarray,
        attention_label: float = 0.0
    ) -> float:
        """
        Solve for a control sequence U[0..H-1], return U[0].
        """
        x0 = state_history[-1].flatten()
        n  = x0.size

        U = cp.Variable(self.horizon)
        X = cp.Variable((self.horizon + 1, n))

        # Cost: heavy penalty on last state, moderate on others, plus effort & smoothness
        cost = (
            5.0 * cp.sum_squares(X[:, -1]) +
            2.0 * cp.sum_squares(X[:, :n-1]) +
            0.1 * cp.sum_squares(U) +
            0.5 * cp.sum_squares(cp.diff(U))
        )

        cons = [
            X[0] == x0,
            U >= self.min_u,
            U <= self.max_u,
            cp.abs(cp.diff(U)) <= 0.2
        ]
        for t in range(self.horizon):
            cons.append(X[t+1] == X[t] + 0.1 * U[t])

        prob = cp.Problem(cp.Minimize(cost), cons)
        prob.solve(solver=cp.OSQP, warm_start=True)

        if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return float(U.value[0])
        return 0.0
