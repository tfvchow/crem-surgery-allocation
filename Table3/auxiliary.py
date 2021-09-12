from gurobipy import GRB
from solve_CREM import solve_CREM
from solve_CREM_CVaR import solve_CREM_CVaR

import numpy as np


def bisection_on_k(budget, time_limit, c_z, c_y, mu, cov, mode=1,
                   no_rooms=6, no_cases=32):
    k_min = 1000.0
    solution_status = GRB.OPTIMAL

    # Keep shrinking k until the problem is feasible
    while solution_status == GRB.OPTIMAL:
        k_min = k_min / 4

        if k_min < 1e-2:
            break

        if mode == 1:
            solution_status, _ = solve_CREM(k_min, budget, time_limit, c_z, c_y, mu, cov,
                                            no_rooms=no_rooms, no_cases=no_cases)

        if mode == 2:
            solution_status, _ = solve_CREM_CVaR(k_min, budget, time_limit, c_z, c_y, mu, cov,
                                                 no_rooms=no_rooms, no_cases=no_cases)

    k_max = k_min * 4

    # Keep shrinking the search space until the two endpoints are close enough
    # Replace the upper bound by k if the problem (given k) is feasible
    # Otherwise, replace the lower bound by k
    if k_min > 1e-2:
        while k_max - k_min > 1e-2:
            k = (k_min + k_max) / 2
            if mode == 1:
                solution_status, _ = solve_CREM(k, budget, time_limit, c_z, c_y, mu, cov,
                                                no_rooms=no_rooms, no_cases=no_cases)

            if mode == 2:
                solution_status, _ = solve_CREM_CVaR(k, budget, time_limit, c_z, c_y, mu, cov,
                                                     no_rooms=no_rooms, no_cases=no_cases)

            if solution_status == GRB.OPTIMAL:
                k_max = k
            else:
                k_min = k

    if mode == 1:
        _, sol = solve_CREM(k_max, budget, time_limit, c_z, c_y, mu, cov,
                            no_rooms=no_rooms, no_cases=no_cases)

    if mode == 2:
        _, sol = solve_CREM_CVaR(k_max, budget, time_limit, c_z, c_y, mu, cov,
                                 no_rooms=no_rooms, no_cases=no_cases)

    return k_max, sol


def data_generation(seed, avg, lognormal=False,
                    diagonal=False, no_rooms=6, no_cases=32, no_sample=10000):
    np.random.seed(seed)

    time_limit = np.random.randint(420, 540 + 1, no_rooms)
    c_z = (time_limit * time_limit) / 3600 + (3 * time_limit) / 60
    c_y = np.random.choice([6, 18, 30, 42], (no_rooms, no_cases))

    mean = np.array([avg] * int(no_cases / 2) + [avg * 0.5] * int(no_cases / 2))
    sigma = np.array([avg] * int(no_cases / 4) + [avg * 0.3] * int(no_cases / 4) +
                     [avg * 0.5] * int(no_cases / 4) + [avg * 0.15] * int(no_cases / 4))

    if lognormal:
        log_mean = np.log(mean) - 0.5 * np.log(np.power(sigma / mean, 2) + 1)
        log_sigma = np.sqrt(np.log(np.power(sigma / mean, 2) + 1))
        data = np.random.lognormal(mean=log_mean, sigma=log_sigma, size=(no_sample, no_cases))
    else:
        data = np.random.normal(loc=mean, scale=sigma, size=(no_sample, no_cases))
        data = np.maximum(data, 0)

    mu = np.mean(data, axis=0)
    if diagonal:
        cov = np.diag(np.diag(np.cov(data, rowvar=False)))
    else:
        cov = np.cov(data, rowvar=False)

    return [time_limit, c_z, c_y, mu, cov, mean, sigma]


def outsample_test(out_sample, y, time_limit, no_rooms, no_samples):
    y = np.vectorize(lambda x: round(x))(y)

    room_usage = (out_sample * y).sum(axis=2)
    ot = np.maximum(room_usage - time_limit,0)

    no_open_room = np.sum(y.sum(axis=1) > 0)
    avg = np.mean(ot) * no_rooms / no_open_room
    std = np.std(ot) * np.sqrt(no_rooms / no_open_room)

    if np.sum(ot > 0) > 0:
        cavg = np.sum(ot) / np.sum(ot > 0)
        cstd = np.std(ot[ot > 0])
    else:
        cavg = 0.0
        cstd = 0.0

    prob1 = np.sum(ot > 0) / (no_samples * no_open_room) * 100
    prob2 = ((ot > 0).sum(axis=1) > 1).sum() / no_samples * 100
    var100 = np.max(ot)

    return [avg, std, cavg, cstd, prob1, prob2, var100]
