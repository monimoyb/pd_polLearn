%% Function Approximation Example: ME231B Spring 2019 
% Monimoy Bujarbaruah 
% This code generates all I-O data. Trains an NN approximation. 

clc; clear all; close all; 

%% Arrays for storing NN training data 
sample_number   = 1e3;                          
x_nn_train = nan(1, sample_number);       % Training values. 
y_nn_train = nan(1, sample_number);       % Labels. Evaluated function values. 

%% Get training data for a NN 

ii = 1;                                                                              % starting loop 

while ii < sample_number
    
    param0 = -10+20*rand(1);                                             % Training data. Uniform rand.                                
    
    x_nn_train(:,ii) = param0;                                               % Getting features 

    func_out = param0^2 - 4*param0 + 1;                             % Example function to be fitted. 
    
    y_nn_train(:,ii) = func_out;                                              % Training labels   
   
    ii = ii+1; 
 
end
 
%%%% Data Collection Ends Here %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Now Neural Net Train with all above data 
% https://www.mathworks.com/help/releases/R2016b/nnet/ref/network.html

% set training data
% second dimension must be same
maxTrials_fixedStructure = 1;                   % max number of fitting trials for given networkSize
maxTrials = 1;                                        % total number of trials of neuron increase                 (keep same for all networks) 
net = cell(1,1);                                       % Just one NN defined here 

%% 
trialNum_AddNeur = 1; 

% store all runs
net_all = cell(maxTrials_fixedStructure,maxTrials);
net_perf_all = nan(maxTrials_fixedStructure,maxTrials);

% init network parameters 
tr_mse = inf;                                                                    % training error; stopping error
networkSize = [0 0];
tmp = sqrt(sum(y_nn_train.^2,1));                                      % length of indiv training data
eps_mse = 1e-5*max(tmp);                                               % desired MSE error

    %%
    while (tr_mse > eps_mse) && (trialNum_AddNeur <= maxTrials)

        networkSize = networkSize + 2;                              % if passed vector, then multiple layers
        disp(['network size: ' , num2str(networkSize)])

        trialNum_FixedStruc = 0;

        for ii = 1 : maxTrials_fixedStructure                        

            trialNum_FixedStruc = trialNum_FixedStruc+1;                                                    % increase trial number
            disp(['trial number with fixed structure: ' num2str(trialNum_FixedStruc)])


              %% Define the Neural Network

            net = feedforwardnet(networkSize);                                                  % 2 good enough for u0

            net.layers{1}.transferFcn = 'poslin';                                                   % change to sigmoid (Rel-U Bad Now) 

          % Different sets are randomly created for training, validation and testing the network

            net.divideParam.trainRatio = 80/100;
            net.divideParam.valRatio =     0/100;
            net.divideParam.testRatio =    20/100;

            net.trainParam.epochs = 50; 
            % view(net)
            net = configure(net,x_nn_train,y_nn_train);                                       % configure network (#inputs/output, params, ...)
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


%% Testing the Trained NNs
test_valsNN = zeros(1,sample_number); 
test_valsTrue = zeros(1,sample_number); 

test_xs = -20 + 40*rand(1,sample_number);
for i = 1: sample_number
test_valsNN(1,i) = net(test_xs(1,i)); 
test_valsTrue(1,i) = test_xs(1,i)^2 - 4*test_xs(1,i) + 1;   
end

%% Figures showing Plots 
figure; 

plot(test_xs(1,1),test_valsTrue(1,1),'.--','color','b','linewidth',2); hold on;
plot(test_xs(1,1),test_valsNN(1,1),'*','color','k','linewidth',2); hold on;
legend({'True Value','NN Approximate'},'Fontsize',25); 

for i = 1:sample_number/10
    plot(test_xs(1,i),test_valsTrue(1,i),'.--','color','b','linewidth',2); hold on;
    plot(test_xs(1,i),test_valsNN(1,i),'*','color','k','linewidth',2); hold on;
end
grid on;  

xlabel('x'); ylabel('f(x)');
set(gca, 'fontsize',20,'fontweight','bold')

%% Show the NN used
view(net) 
