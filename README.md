# crem-surgery-allocation

Data and results for the computational studies in the IJOC 2021 paper on coherent risk enveloping measure in surgery allocation problems

The folder “Table3” contains the codes and the data (see below for a remark) for the results in Table 3 under Section 5. Specifically,

- the files "solve_CREM_CVaR.py", "solve_CREM.py", and "solve_shen.py" contains the actual implementations of the three approaches in Python. GUROBI 9.1 is used throughout the test 
- the file "auxiliary.py" contains functions for generating data (for both constructing ambiguity sets and out-of-sample data) and the bisection search algorithm 
- the file "main.py" is the main function/routine for picking random seeds, generating data, grabbing optimal solutions, and evaluating the out-of-sample tests for the three approaches. 
- the file "reproduce.py" is for reproducing Tables 3 and 4 based on the samples we have used in our testing. 

Owing to large file sizes, instances containing the  will be uploaded at a later stage. Once uploaded, they will be available in "Release".

To obtain the results in Table 4, it suffices to replace the statement "_lognormal=False" by _lognormal=True" in Line 12 of the file "main.py".
