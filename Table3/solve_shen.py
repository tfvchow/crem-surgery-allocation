from gurobipy import GRB

import gurobipy as gp
import numpy as np


def solve_shen(alpha, time_limit, c_z, c_y, mu, cov, no_rooms=6, no_cases=32):
    mdl = gp.Model('Shen')
    y = mdl.addMVar((no_rooms, no_cases), vtype=GRB.BINARY)
    z = mdl.addMVar(no_rooms, vtype=GRB.BINARY)

    mdl.modelSense = GRB.MINIMIZE

    # Declaring the objective function
    # The operator "@" here has the same function as quicksum(...) or matrix multiplications
    # c_z @ z is equivalent to (c_z^T) z or z^T (c_z)
    # No transpose is needed here, the same applies on the quadratic form y^T Q y
    # We just need y @ Q @ y.
    obj = c_z @ z + sum(c_y[i, :] @ y[i, :] for i in range(no_rooms))
    mdl.setObjective(obj)

    # Constraints on assignments
    # No case should be assigned to a non-open OR.
    # Each case must be assigned to exactly 1 OR.
    mdl.addConstrs((
        y[:, j] <= z[:]
        for j in range(no_cases)
    ))

    mdl.addConstrs((
        y[:, j].sum() == 1
        for j in range(no_cases)
    ))

    # Owing to the limitation of "@" operator, we need to prepare something like yy^T in advance
    # Notice that the shape of mu_i is in form of (32,) but we need (32,1) so that they can be multiplied
    # together. Hence, the reshape() function is used.
    mu_col_vector = mu.reshape((mu.shape[0], 1))
    mu_mu_transpose = np.matmul(mu_col_vector, np.transpose(mu_col_vector))

    # Finally adding the clumsy quadratic constraints.
    # Again, the alpha must be put between thw two "@" operators.
    mdl.addConstrs((
        y[i, :] @ ((1 - alpha) / alpha * cov) @ y[i, :]
        <= time_limit[i] * time_limit[i] - (2 * time_limit[i] * mu) @ y[i, :] +
        y[i, :] @ mu_mu_transpose @ y[i, :]
        for i in range(no_rooms)
    ))

    # Ensuring that the solution returned is feasible in deterministic
    # settings. Otherwise, the results returned could violate this.
    mdl.addConstrs((
        time_limit[i] - mu @ y[i, :] >= 0 for i in range(no_rooms)
    ))

    # No output to be displayed
    # Time Limit for Shen's model is 120 minutes, which can be extended.
    mdl.setParam('OutputFlag', 0)
    mdl.setParam('TimeLimit', 7200)
    mdl.optimize()

    # Return the objective value and assignments only when an optimal solution is found
    # Otherwise, return a value for subsequent handling
    if mdl.status == GRB.OPTIMAL:
        return [mdl.objVal, y.x]
    else:
        print(f"Timeout. Gap: {round(mdl.MIPGap * 100, 2)}%")
        return [-999, np.zeros((no_rooms, no_cases))]
