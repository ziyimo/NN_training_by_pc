%% Initialization
clear ; close all; clc
timestamp = datestr(now, 'mmdd_HHMMSS');
% Setup the architecture of the PC network
% weights, biases, variables and errors are all stored in cell arrays.

params.act = 'tanh'; %activation function: 'tanh', 'sig', 'lin', 'reclin'
params.l_rate =  0.3; % learning rate
params.it_max = 300; % maximum iterations of inference
params.epochs = 1750; % number of epochs
params.d_rate = 0; % weight decay parameter
params.beta = 0.1; % euler integration constant

params.layer_sizes = [4, 5, 3, 3]; %units in each layer
% note: last two layers together constitute output layer, their size must
% equal!
params.n_layers = length(params.layer_sizes); % number of layers
var = ones(1, params.n_layers); % puts variance on all layers as 1
var(end)=1; % variance on last layer
params.var=var;

%% Data
% XOR problem
% sin = [0 0 1 1;
%         0 1 0 1];
% y = [1, 2, 2, 1]';
% sout = [1 0 0 1;
%          0 1 1 0];

% Iris dataset
load('fisheriris.mat');
sin = meas';
keySet = {'setosa', 'versicolor', 'virginica'};
valueSet = [1, 2, 3];
M = containers.Map(keySet,valueSet);
y = cellfun(@(x) M(x), species);
sout = double(bsxfun(@eq, 1:3, y))';

%% Training
[w_pc, b_pc] = rand_init(params); % get weights and biases parameters
log_f = fopen(strcat(timestamp, '.txt'), 'w');
cost_history = zeros(params.epochs, 1);

%learn
for epoch = 1:params.epochs
    [w_pc,b_pc] = learn_pc(sin,sout,w_pc,b_pc,params); %train pc
    [~, pc_out] = predict(sin, w_pc, b_pc, params); % examine cost
    J = -sum(sum(log(pc_out).*sout))/length(sin);
    cost_history(epoch) = J;
    fprintf(log_f, 'Epoch %d | Cost: %e\n', epoch, J);
end

[y_pred, O] = predict(sin, w_pc, b_pc, params);

fprintf(log_f, '\nTraining Set Accuracy: %f\n', mean(double(y_pred == y')) * 100);
fclose(log_f);
save(strcat(timestamp, '.mat'), 'w_pc', 'b_pc', 'cost_history', 'y_pred', 'O');

plot(1:params.epochs, cost_history);
