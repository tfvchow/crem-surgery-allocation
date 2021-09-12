from auxiliary import *
from solve_shen import solve_shen

import numpy as np
import pandas as pd

result = []
for _alpha in [0.05, 0.1, 0.3, 0.5, 0.7, 0.9]:
    _done = 0
    _seed = 0
    _lognormal = False

    while _done < 10:
        _seed = _seed + 1

        print(f"Trying Seed: {_seed}, alpha: {_alpha}")
        _avg = 50
        _no_samples = 100000
        _no_rooms = 3
        _no_cases = 16

        [_time_limit, _c_z, _c_y, _mu, _cov, _mean, _sigma] \
            = data_generation(_seed, _avg, diagonal=False, lognormal=_lognormal, no_rooms=_no_rooms, no_cases=_no_cases)

        [shen_obj, shen_y] = solve_shen(_alpha, _time_limit, _c_z, _c_y, _mu, _cov, no_rooms=_no_rooms,
                                        no_cases=_no_cases)

        # Among the solution methods, the chance-constrained approach takes longest to optimise.
        # Hence, we are imposing a time limit on the benchmark approach.
        # The time limit is 2 hours for the benchmark and 60 seconds for
        # each iteration in the bisection search for CREM and its variant.
        if shen_obj == -999:
            print(f"When Seed = {_seed}, the benchmark approach takes too long to compute.")
            continue

        [CREM_k, CREM_y] = bisection_on_k(shen_obj, _time_limit, _c_z, _c_y, _mu, _cov,
                                          mode=1, no_rooms=_no_rooms, no_cases=_no_cases)
        [CREM_CVaR_k, CREM_CVaR_y] = bisection_on_k(shen_obj, _time_limit, _c_z, _c_y, _mu, _cov,
                                                    mode=2, no_rooms=_no_rooms, no_cases=_no_cases)

        if _lognormal:
            log_mean = np.log(_mean) - 0.5 * np.log(np.power(_sigma / _mean, 2) + 1)
            log_std = np.sqrt(np.log(np.power(_sigma / _mean, 2) + 1))
            out_sample = np.random.lognormal(mean=log_mean, sigma=log_std, size=(_no_samples, _no_rooms, _no_cases))
        else:
            out_sample = np.random.normal(loc=_mu, scale=np.sqrt(np.diagonal(_cov)),
                                          size=(_no_samples, _no_rooms, _no_cases))
            out_sample = np.vectorize(lambda x: max(x, 0))(out_sample)

        shen_metrics = outsample_test(out_sample, shen_y, _time_limit, _no_rooms, _no_samples)
        CREM_metrics = outsample_test(out_sample, CREM_y, _time_limit, _no_rooms, _no_samples)
        CREM_CVaR_metrics = outsample_test(out_sample, CREM_CVaR_y, _time_limit, _no_rooms, _no_samples)

        result += [[_seed, _alpha, "Shen"] + shen_metrics]
        result += [[_seed, _alpha, "CREM"] + CREM_metrics]
        result += [[_seed, _alpha, "CREM_CVaR"] + CREM_CVaR_metrics]

        _done = _done + 1

result = pd.DataFrame(result,
                      columns=['seed', 'alpha', 'model', 'avg', 'std', 'cavg', 'cstd', 'prob1', 'prob2', 'var100'])

print(result.groupby(['alpha', 'model']).mean())
print("Done")
