# pd_polLearn
Primal-Dual Policy Learning Simple Example
approxMPCTrain.m file trains the Primal and Dual policies. This must be run to generate and store all data.
approxMPCTest.m file then loads the trained policies and data and runs test on new samples.
If test statistics are deemed unsatisfactory, please go back to training code and retrain
# Requirements
Multiparametric toolbox with explicit MPC solver. YALMIP. Gurobi.
