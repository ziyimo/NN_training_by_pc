% Initialization
clear ; close all; clc

% Setup the architecture of the PC network
% weights, biases, variables and errors are all stored in cell arrays.

params.act = 'tanh'; %activation function: 'tanh', 'sig', 'lin', 'reclin'
params.l_rate =  1; % learning rate
params.it_max = 100; % maximum iterations of inference
params.epochs = 200; % number of epochs
params.d_rate = 0; % weight decay parameter
params.beta = 0.1; % euler integration constant

params.layer_sizes = [4, 5, 3, 3]; %units in each layer
% note: last two layers together constitute output layer, their size must
% equal!
params.n_layers = length(params.layer_sizes); % number of layers
var = ones(1, params.n_layers); % puts variance on all layers as 1
var(end)=1; % variance on last layer
params.var=var;

% XOR problem
% sin = [0 0 1 1;
%        0 1 0 1];
% y = [1, 2, 2, 1]';
% sout = [1 0 0 1;
%         0 1 1 0];

% MNIST
%load('mnist_5000.mat');
%sin = X(1:500, :)';
%sout = y';
%sout = bsxfun(@eq, 1:params.layer_sizes(end), y(1:500))'; % Apply element-wise operation to two arrays with implicit expansion enabled

% Iris dataset
load('fisheriris.mat');
sin = meas';
keySet = {'setosa', 'versicolor', 'virginica'};
valueSet = [1, 2, 3];
M = containers.Map(keySet,valueSet);
y = cellfun(@(x) M(x), species);
sout = double(bsxfun(@eq, 1:3, y))';

[w_pc, b_pc] = rand_init(params); % get weights and biases parameters

%learn
for epoch = 1:params.epochs
    params.epoch_num = epoch;
    [w_pc,b_pc] = learn_pc(sin,sout,w_pc,b_pc,params); %train pc
    [~, pc_out] = predict(sin, w_pc, b_pc, params); % examine cost
    J = -sum(log(pc_out).*sout, 'all')/length(sin);
    fprintf('Epoch %d | Cost: %e\r', epoch, J);
end

[y_pred, O] = predict(sin, w_pc, b_pc, params);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(y_pred == y')) * 100);