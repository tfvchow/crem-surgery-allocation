from auxiliary import bisection_on_k
from auxiliary import outsample_test
from solve_shen import solve_shen

import os
import pandas as pd
import pickle

out = []

# To use this file, you need to download one/some/all of the pickles (files ending with pkl) on GitHub.
# Put them inside a folder and move all of the Python scripts into that folder as well.
# Then type python reproduce.py in Command Prompt or Terminal.
# Advanced users please ignore this instruction and run this script in your own way.

path = os.getcwd() + "/"
files = os.listdir(path)
filenames = [filename for filename in files
    if "pkl" in filename and not filename.startswith("._")]

out = []

for filename in filenames:
    parameters = filename.split("_")

    _avg = int(parameters[1])
    _alpha = float(parameters[2])
    _seed = int(parameters[3].split(".")[0])

    print(f"Running alpha = {_alpha}, seed = {_seed}")

    open_file = open(path + filename, "rb")
    _alpha, _time_limit, _c_z, _c_y, _mu, _cov, _no_rooms, _no_cases, _no_samples, out_sample = pickle.load(open_file)
    open_file.close()

    [shen_obj, shen_y] = solve_shen(_alpha, _time_limit, _c_z, _c_y, _mu, _cov,
                                    no_rooms=_no_rooms, no_cases=_no_cases)

    # If the benchmark approach takes too long to complete, you can:
    # (a) Ignore this instance by NOT commenting the if-statement below
    # (b) Increase the time limit in solve_shen.py (Line 59)
    if shen_obj == -999:
        continue

    [CREM_k, CREM_y] = bisection_on_k(shen_obj, _time_limit, _c_z, _c_y, _mu, _cov,
                                      mode=1, no_rooms=_no_rooms, no_cases=_no_cases)
    [CREM_CVaR_k, CREM_CVaR_y] = bisection_on_k(shen_obj, _time_limit, _c_z, _c_y, _mu, _cov,
                                                mode=2, no_rooms=_no_rooms, no_cases=_no_cases)

    shen_metrics = outsample_test(out_sample, shen_y, _time_limit, _no_rooms, _no_samples)
    CREM_metrics = outsample_test(out_sample, CREM_y, _time_limit, _no_rooms, _no_samples)
    CREM_CVaR_metrics = outsample_test(out_sample, CREM_CVaR_y, _time_limit, _no_rooms, _no_samples)

    out += [[_seed, _alpha, "Shen"] + shen_metrics]
    out += [[_seed, _alpha, "CREM"] + CREM_metrics]
    out += [[_seed, _alpha, "CREM_CVaR"] + CREM_CVaR_metrics]

result = pd.DataFrame(out,
                      columns=['seed', 'alpha', 'model', 'avg', 'std', 'cavg', 'cstd', 'prob1', 'prob2', 'var100'])

agg_result = result.groupby(['alpha', 'model']).mean()
agg_result.to_csv("agg_result.csv")

print(agg_result)
print("Done")
