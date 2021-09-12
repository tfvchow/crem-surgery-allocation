from gurobipy import GRB

import gurobipy as gp
import numpy as np


# noinspection PyPep8Naming
def solve_CREM(k_val, budget, time_limit, c_z, c_y, mu, cov, no_rooms=6, no_cases=32):
    mdl = gp.Model('CREM')
    y = mdl.addMVar((no_rooms, no_cases), vtype=GRB.BINARY)
    z = mdl.addMVar(no_rooms, vtype=GRB.BINARY)

    mdl.modelSense = GRB.MINIMIZE
    mdl.setObjective(1)

    mdl.addConstrs((
        y[:, j] <= z[:]
        for j in range(no_cases)
    ))

    mdl.addConstrs((
        y[:, j].sum() == 1
        for j in range(no_cases)
    ))

    mdl.addConstr(c_z @ z + sum(c_y[i, :] @ y[i, :] for i in range(no_rooms)) <= budget, name="")
    capital_m = 1 / 4 * cov

    mdl.addConstrs((
        - k_val * (time_limit[i] - mu @ y[i, :]) + y[i, :] @ capital_m @ y[i, :] <= 0
        for i in range(no_rooms)
    ))

    mdl.addConstrs((
        k_val - 0.5 * (-1 * (time_limit[i] - mu @ y[i, :]) + k_val) >= 0
        for i in range(no_rooms)
    ))

    mdl.setParam('OutputFlag', 0)
    mdl.setParam('NonConvex', 2)
    mdl.setParam('TimeLimit', 60)
    mdl.optimize()

    if mdl.status == GRB.OPTIMAL:
        return [mdl.status, y.x]
    else:
        return [mdl.status, np.zeros((no_rooms, no_cases))]
