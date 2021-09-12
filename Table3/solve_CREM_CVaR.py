from gurobipy import GRB

import gurobipy as gp
import numpy as np


# noinspection PyPep8Naming
def solve_CREM_CVaR(k_val, budget, time_limit, c_z, c_y, mu, cov, no_rooms=6, no_cases=32):
    M = 100000

    mdl = gp.Model('CREM_CVaR')
    y = mdl.addMVar((no_rooms, no_cases), vtype=GRB.BINARY)
    z = mdl.addMVar(no_rooms, vtype=GRB.BINARY)
    v = mdl.addMVar(no_rooms, vtype=GRB.CONTINUOUS, lb=-float('inf'))
    phi = mdl.addMVar((no_rooms, no_cases), vtype=GRB.CONTINUOUS)
    theta = mdl.addMVar((no_rooms, no_cases), vtype=GRB.CONTINUOUS)
    hat_phi = mdl.addMVar(no_rooms, vtype=GRB.CONTINUOUS)
    hat_theta = mdl.addMVar(no_rooms, vtype=GRB.CONTINUOUS)

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

    mdl.addConstr(c_z @ z + sum(c_y[i, :] @ y[i, :] for i in range(no_rooms)) <= budget, name="budget")

    mdl.addConstrs((
        k_val * (mu @ y[i, :] - time_limit[i] - v[i] - k_val) +
        y[i, :] @ (0.25 * cov) @ y[i, :] <= 0
        for i in range(no_rooms)
    ))

    mdl.addConstrs((
        v[i] * time_limit[i] -
        sum(mu[j] * (phi[i, j] - theta[i, j]) for j in range(no_cases)) +
        y[i, :] @ (0.25 * cov) @ y[i, :] <= 0
        for i in range(no_rooms)
    ))

    mdl.addConstr(
        v == hat_phi - hat_theta,
        name="v=hat_phi-hat_theta")

    mdl.addConstrs((
        phi[:, j] <= hat_phi
        for j in range(no_cases)
    ), name="phi_ij<=hat_phi_i")

    mdl.addConstrs((
        phi[i, :] <= M * y[i, :]
        for i in range(no_rooms)
    ), name="phi_ij<=M*y_ij")

    mdl.addConstrs((
        phi[:, j] >= hat_phi - M * (1 - y[:, j])
        for j in range(no_cases)
    ), name="phi_ij>=hat_phi_i-M(1-y_ij)")

    mdl.addConstrs((
        theta[:, j] <= hat_theta
        for j in range(no_cases)
    ), name="theta_ij<=hat_theta_i")

    mdl.addConstrs((
        theta[i, :] <= M * y[i, :]
        for i in range(no_rooms)
    ), name="theta_ij<=hat_theta_i")

    mdl.addConstrs((
        theta[:, j] >= hat_theta - M * (1 - y[:, j])
        for j in range(no_cases)
    ), name="theta_ij<=hat_theta_i-M(1-y_ij)")

    mdl.addConstrs((
        k_val + 0.5 * (v[i] + time_limit[i] - mu @ y[i, :]) >= 0
        for i in range(no_rooms)
    ))

    mdl.addConstrs((
        -v[i] + 0.5 * (v[i] + time_limit[i] - mu @ y[i, :]) >= 0
        for i in range(no_rooms)
    ))

    mdl.setParam('OutputFlag', 0)
    mdl.setParam('TimeLimit', 60)
    mdl.optimize()

    if mdl.status == GRB.OPTIMAL:
        return [mdl.status, y.x]
    else:
        return [mdl.status, np.zeros((no_rooms, no_cases))]
