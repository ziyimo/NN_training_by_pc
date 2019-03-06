clear ; close all; clc;

% dfile = '0301_083410.mat'; % Iris
% load('fisheriris.mat');
% sin = meas';
% keySet = {'setosa', 'versicolor', 'virginica'};
% valueSet = [1, 2, 3];
% M = containers.Map(keySet,valueSet);
% y = cellfun(@(x) M(x), species);

dfile = 'training_runs/0305_085739.mat'; % MNIST5000
load('mnist_5000.mat');

load(dfile);
semilogy(J_history, 'LineWidth', 2);
ylabel('cost function') 
xlabel('training epoch') 

%plot(cost_history);
%ylim([0 inf]);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(y_pred == y')) * 100);
