%% Training Data Generation for MPC Policy  Learning
% Monimoy Bujarbaruah 
% 06/22/2019
% Run this code and save data before running test code

clc; clear all; close all; 
%% Generate parameters with known bounds, instead of loading Jongsung data files 
N_mpc = 3;                                                        % MPC horizon length fixed at 3
input_num           =    1;                                       % Dimension of inputs 
state_num           =    2;                                       % Dimension of states 
output_num          =   state_num;                                %  Full state feedback 
%% Arrays for storing NN training data 
sample_number   = 1e2;                                              % Uniform samples' count. Training data size
x_nn_train = nan(state_num, sample_number);                         % Training parameters. Current state
y_nn_train = nan(N_mpc*(input_num), sample_number);                 % Labels. Learning the whole sequence
options = sdpsettings('verbose',0, 'solver', 'gurobi');

%% Define the system matrices. Simple LTI System 
Ad = [1 0.5; 0 -1];
Bd = [1; 1];
%% Weight Matrices for MPC 
Q = eye(state_num); R = 5*eye(input_num); 
Qx_vec = kron(eye(N_mpc),Q); Rx_vec = kron(eye(N_mpc),R);           % Append along horizon 
%% State and input Constraints 
u_min = -5;
u_max = 5;
state_min =  -[10;10];                     % Min state 
state_max =  [10;10];                      % Max state  
%% Append constraints over horizon. QP with substitution  
G = [eye(input_num) ; -eye(input_num)];
g = [u_max; -u_min];
ng = length(g);
G_vec = kron(eye(N_mpc), G);
g_vec = repmat(g,N_mpc, 1);

F = [eye(state_num) ; -eye(state_num)];
f = [state_max ; -state_min];
nf = length(f);
F_vec = kron(eye(N_mpc), F);
f_vec = repmat(f,N_mpc, 1);    
y_nn_train_dual = nan(N_mpc*(nf+ng),sample_number);
opt_val = zeros(sample_number,1); 

%% Append system matrices over the horizon. QP with substitution  
[Ax_vec, Bx_vec] = appendMat(N_mpc, state_num, input_num, Ad, Bd);

%% Solve MPC problem and get training data 

ii = 1;                                                                  % starting loop 

while ii < sample_number

    state01 = state_min(1) + (state_max(1)-state_min(1))*rand(1);
    state02 = state_min(2) + (state_max(2)-state_min(2))*rand(1);
    
    param0 = [state01;state02];                                          % Initial state 
   
    %% Yalmip QP formulation 
    x_nn_train(:,ii) = param0;                                           % Getting features 
    yalmip clear; yalmip('clear');

    % State-Input stacked vectors  
    xvec_yp = sdpvar(N_mpc*state_num,1); 
    uvec_yp = sdpvar(N_mpc*input_num,1); 

    % Cost function    
    objective = xvec_yp'*Qx_vec*xvec_yp + uvec_yp'*Rx_vec*uvec_yp;

    % Constraints 
    constraints =  [xvec_yp== Ax_vec*param0 + Bx_vec*uvec_yp;
                    F_vec * xvec_yp <= f_vec;                                                    % state constraints
                    G_vec*uvec_yp <= g_vec];                                                     % input constraints
                                                

    exitflag = solvesdp(constraints, objective, options);
    
    if exitflag.problem ~= 0
        problem = exitflag.problem;
        yalmiperror(problem);
    else
        y_nn_train(:,ii) = double(uvec_yp);   % whole input sequence  
        
        opt_val(ii,1) = double(objective); 
        
        %     save dual variables
        l_vec_double = [dual(constraints(2)) ; dual(constraints(3))];
        m_vec_double = dual(constraints(1));
        
        y_nn_train_dual(:,ii) = l_vec_double;                  % extract ALL dual variables 

   
        diagn = exitflag.problem; 
        ii = ii+1 
    end
       
    
end

%% Now Primal Neural Net Train for MPC Policy Learning  
% https://www.mathworks.com/help/releases/R2016b/nnet/ref/network.html

maxTrials_fixedStructure = 3;                         % max number of fitting trials for given networkSize
maxTrials = 3;                                        % total number of trials of neuron increase            
net = cell(1,1);                                      % Just one NN defined here 

%% 
trialNum_AddNeur = 1; 

% store all runs
net_all = cell(maxTrials_fixedStructure,maxTrials);
net_perf_all = nan(maxTrials_fixedStructure,maxTrials);

% init network parameters 
tr_mse = inf;                                                          % training error; stopping error
networkSize = [0 0];
tmp = sqrt(sum(y_nn_train.^2,1));                                      % length of indiv training data
eps_mse = 1e-5*max(tmp);                                               % desired MSE error

    %%
    while (tr_mse > eps_mse) && (trialNum_AddNeur <= maxTrials)

        networkSize = networkSize + 2;                                 % if passed vector, then multiple layers
        disp(['network size: ' , num2str(networkSize)])

        trialNum_FixedStruc = 0;

        for ii = 1 : maxTrials_fixedStructure                        

            trialNum_FixedStruc = trialNum_FixedStruc+1;               % increase trial number
            disp(['trial number with fixed structure: ' num2str(trialNum_FixedStruc)])


              %% Define the Neural Network

            net = feedforwardnet(networkSize);                                            

            net.layers{1}.transferFcn = 'poslin';                                                   

          % Different sets are randomly created for training, validation and testing the network

            net.divideParam.trainRatio = 80/100;
            net.divideParam.valRatio =     0/100;
            net.divideParam.testRatio =    20/100;

            net.trainParam.epochs = 500; 
            % view(net)
            net = configure(net,x_nn_train,y_nn_train);                                    % configure network (#inputs/output, params, ...)
            % view(net)    
            [net  tr] = train(net,x_nn_train,y_nn_train,'useGPU','no');                    % train network

            % Once the network has been trained, we can obtain the Mean Squared Error
            % for the best epoch (time when the training has stopped in order to avoid
            % overfitting the network).
            tr_mse = tr.perf(tr.best_epoch + 1);                                                
            net_perf_all(trialNum_FixedStruc,trialNum_AddNeur) = tr_mse;
            net_all{trialNum_FixedStruc, trialNum_AddNeur} = net;

            disp(['training error: ', num2str(tr_mse) ])
            disp(['best training error so far: ', num2str(min(min(net_perf_all)))])
            if tr_mse <= eps_mse
                disp(['---> Success!   |   NetworkSize: ',  num2str(networkSize) , '   |   MSE: ', num2str(tr_mse) ])
                break
            end
        end

        trialNum_AddNeur = trialNum_AddNeur + 1; 
        disp(['Neuron Increase Index: ' num2str(trialNum_AddNeur)])

    end

%% Picking Best Trained Network (Lowest MSE) 
min_m = min(min(net_perf_all));
[idx1, idx2] = find(net_perf_all==min_m);
net = net_all{idx1,idx2};                                           % Best Network after training 

%% Now Dual Neural Net Train for MPC Policy Learning  

maxTrials_fixedStructureD = 3;                         % max number of fitting trials for given networkSize
maxTrialsD = 3;                                        % total number of trials of neuron increase         
net_dual = cell(1,1);                                  % Just one NN defined here 

%% 
trialNum_AddNeurD = 1; 

% store all runs
net_all_dual = cell(maxTrials_fixedStructureD,maxTrialsD);
net_perf_all_dual = nan(maxTrials_fixedStructureD,maxTrialsD);

% init network parameters 
tr_mse = inf;                                                            % training error; stopping error
networkSizeD = [0 0];
tmp = sqrt(sum(y_nn_train_dual.^2,1));                                   % length of indiv training data
eps_mse = 1e-5*max(tmp);                                                 % desired MSE error

    %%
    while (tr_mse > eps_mse) && (trialNum_AddNeurD <= maxTrialsD)

        networkSizeD = networkSizeD + 2;                                 % if passed vector, then multiple layers
        disp(['network size: ' , num2str(networkSizeD)])

        trialNum_FixedStrucD = 0;

        for ii = 1 : maxTrials_fixedStructureD                        

            trialNum_FixedStrucD = trialNum_FixedStrucD+1;               % increase trial number
            disp(['trial number with fixed structure: ' num2str(trialNum_FixedStrucD)])


              %% Define the Neural Network

            net_dual = feedforwardnet(networkSizeD);                                                 

            net_dual.layers{1}.transferFcn = 'poslin';                                                    

          % Different sets are randomly created for training, validation and testing the network

            net_dual.divideParam.trainRatio = 80/100;
            net_dual.divideParam.valRatio =     0/100;
            net_dual.divideParam.testRatio =    20/100;

            net_dual.trainParam.epochs = 500; 
            % view(net)
            net_dual = configure(net_dual,x_nn_train,y_nn_train_dual);                                    % configure network (#inputs/output, params, ...)
            % view(net)    
            [net_dual  tr] = train(net_dual,x_nn_train,y_nn_train_dual,'useGPU','no');                    % train network

            % Once the network has been trained, we can obtain the Mean Squared Error
            % for the best epoch (time when the training has stopped in order to avoid
            % overfitting the network).
            tr_mse = tr.perf(tr.best_epoch + 1);                                                
            net_perf_all_dual(trialNum_FixedStrucD,trialNum_AddNeurD) = tr_mse;
            net_all_dual{trialNum_FixedStrucD, trialNum_AddNeurD} = net_dual;

            disp(['training error: ', num2str(tr_mse) ])
            disp(['best training error so far: ', num2str(min(min(net_perf_all_dual)))])
            if tr_mse <= eps_mse
                disp(['---> Success!   |   NetworkSize: ',  num2str(networkSizeD) , '   |   MSE: ', num2str(tr_mse) ])
                break
            end
        end

        trialNum_AddNeurD = trialNum_AddNeurD + 1; 
        disp(['Neuron Increase Index: ' num2str(trialNum_AddNeurD)])

    end

%% Picking Best Trained Dual Network (Lowest MSE) 
min_mD = min(min(net_perf_all_dual));
[idx1, idx2] = find(net_perf_all_dual==min_mD);
net_dual = net_all_dual{idx1,idx2};                                           % Best Network after training 

%% Save Everything 
save('trainedNNData.mat')
