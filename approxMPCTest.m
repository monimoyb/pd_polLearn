%% PD Policy Learning: Testing of trained Neural Networks (Offline) 
%  Monimoy Bujarbaruah (monimoyb@berkeley.edu)
%  Xiaojing Zhang (xiaojing.zhang@berkeley.edu)
%  06/22/2019
%  Run code after running training code and saving .mat file 
%%
% The paper describing the theory can be found here:
% 	X. Zhang, M. Bujarbaruah and F. Borrelli; "Safe and Near Optimal Policy Learning for Model Predictive Control using Primal-Dual Neural Networks"; https://arxiv.org/abs/1906.08257]

%%
clear all; close all; clc; 

%% Loading trained neural nets and all required values 
load('trainedNNData.mat'); 
% This loads all required system matrices and parameters too

%% Test the quality of the trained neural networks  
num_testRuns = 1e2;                                   % Increase this if required
act_on_GapTest_all = nan(num_testRuns,1); 
rel_on_GapTest_all = nan(num_testRuns,1); 
act_pd_GapTest_all = nan(num_testRuns,1); 
rel_pd_GapTest_all = nan(num_testRuns,1); 

OptVal_all = nan(num_testRuns,1);
x_nn_test = nan(state_num, num_testRuns);    
y_nn_test = nan(N_mpc*input_num, num_testRuns); 

U_test = nan(N_mpc*(input_num),num_testRuns); 
options = sdpsettings('verbose',0, 'solver', 'gurobi','gurobi.BarConvTol',1e-8);
num_infeas = 0;                                      % number of infeasible random parameters

%% Main test loop 

ii = 1;

while ii <= num_testRuns

 state01 = state_min(1) + (state_max(1)-state_min(1))*rand(1);
 state02 = state_min(2) + (state_max(2)-state_min(2))*rand(1);
    
 param0 = [state01;state02];                                              % Initial state 
 
 x_nn_test(:,ii) = param0; 
   
%% Yalmip QP formulation 
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
    
    y_nn_test(:,ii) = double(uvec_yp); 
    
    if exitflag.problem == 0
       
        %% Evaluate primal network    
        U_test(:,ii) = net(param0);
        
        objective_onMPCTest = double(objective);               % cost of online problem 

        %% Compute the cost using NN solution %%%%%%%%%%%
        xvec_ypT= Ax_vec*param0 + Bx_vec*U_test(:,ii); 
        
        objective_Primal_test = xvec_ypT'*Qx_vec*xvec_ypT+ U_test(:,ii)'*Rx_vec*U_test(:,ii);
            
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       act_on_GapTest_all(ii,1) =  norm(objective_onMPCTest-objective_Primal_test);      
       rel_on_GapTest_all(ii,1) =  norm(objective_onMPCTest-objective_Primal_test)/norm(objective_onMPCTest);      
       OptVal_all(ii) = objective_onMPCTest;
        
       %% Call the dual network and compute cost 
       L_test = net_dual(param0);
       L_test = max(L_test,0);

       Q = 2*(Bx_vec'*Qx_vec*Bx_vec + Rx_vec);    
       c = (2*param0'*Ax_vec'*Qx_vec*Bx_vec)';     
       const = param0'*Ax_vec'*Qx_vec*Ax_vec*param0; 
        
        C_dual = [G_vec; F_vec*Bx_vec];
        d = [g_vec; f_vec - F_vec*Ax_vec*param0];
        Q_tmp = C_dual*(Q\(C_dual'));
        Q_tmp = 0.5*(Q_tmp+Q_tmp') + 0e-5*eye(N_mpc*(ng+nf));
        
       % Form dual cost 
       obj_Dual_test = -1/2 * L_test'*Q_tmp*L_test - (C_dual*(Q\c)+d)'*L_test - 1/2*c'*(Q\c) + const; 
       
       act_pd_GapTest_all(ii,1) =  norm(obj_Dual_test-objective_Primal_test);      
       rel_pd_GapTest_all(ii,1) =  norm(obj_Dual_test-objective_Primal_test)/norm(objective_onMPCTest);  

        % update counter
        ii = ii + 1
    else
        num_infeas = num_infeas + 1;
    end

end

%% Also solve the multiparametric problem here for comparison with Explicit MPC 

yalmip('clear'); yalmip clear
x_vecMP = sdpvar(N_mpc*state_num,1);            
u_vecMP = sdpvar(N_mpc*input_num,1);             
state0 = sdpvar(state_num,1);                        % parameter. Initial state 

objectiveMP = x_vecMP' * Qx_vec * x_vecMP + u_vecMP' * Rx_vec * u_vecMP;
    
constraintsMP = [state_min<= state0 <= state_max ...
                    x_vecMP == Ax_vec*state0 + Bx_vec*u_vecMP ...    % state dynamics
                    F_vec * x_vecMP <= f_vec ...
                    G_vec * u_vecMP <= g_vec];
            
[sol, diagnostics, aux, Valuefunction, Optimizer] = solvemp(constraintsMP,objectiveMP,[],state0,u_vecMP);


%% Suboptimality statistics 
disp(['online/offline gap rate below 0.1%: ' num2str(sum(rel_on_GapTest_all(:,1)<0.1/100)/num_testRuns*100) '%'])
disp(['online/offline gap rate below 0.5%: ' num2str(sum(rel_on_GapTest_all(:,1)<0.5/100)/num_testRuns*100) '%'])
disp(['online/offline gap rate below 1%: '   num2str(sum(rel_on_GapTest_all(:,1)<1/100)/num_testRuns*100) '%'])
disp(['online/offline gap rate below 5%: '   num2str(sum(rel_on_GapTest_all(:,1)<5/100)/num_testRuns*100) '%'])
disp(['online/offline gap rate below 10%: '   num2str(sum(rel_on_GapTest_all(:,1)<10/100)/num_testRuns*100) '%'])

%%

disp(['PD gap rate below 0.1%: ' num2str(sum(rel_pd_GapTest_all(:,1)<0.1/100)/num_testRuns*100) '%'])
disp(['PD gap rate below 0.5%: ' num2str(sum(rel_pd_GapTest_all(:,1)<0.5/100)/num_testRuns*100) '%'])
disp(['PD gap rate below 1%: '   num2str(sum(rel_pd_GapTest_all(:,1)<1/100)/num_testRuns*100) '%'])
disp(['PD gap rate below 5%: '   num2str(sum(rel_pd_GapTest_all(:,1)<5/100)/num_testRuns*100) '%'])
disp(['PD gap rate below 10%: '   num2str(sum(rel_pd_GapTest_all(:,1)<10/100)/num_testRuns*100) '%'])

%% Now plot the contours of the learned policy 
close all

for ii = 1 : N_mpc
    figure(100+ii-1);
    clear title xlabel ylabel zlabel
    title( ['Approximated Solution U_' , num2str(ii-1)] ); 
    hold on
    plot3(x_nn_test(1,:), x_nn_test(2,:),U_test(ii,:),'ob','MarkerSize',10,'MarkerFaceColor', 'k')
    plot3(x_nn_test(1,:), x_nn_test(2,:),y_nn_test(ii,:),'xk','MarkerSize',30,'MarkerFaceColor', 'b')
    legend('NN Approx', 'Optimal Policy')
    plot(Optimizer(ii))    % plot parametric solution
    xlabel('x_1')
    ylabel('x_2')
    zlabel(['input U_', num2str(ii-1)])
    grid on
    hold off
    set(gca, 'fontsize',20,'fontweight','bold'); 
end
