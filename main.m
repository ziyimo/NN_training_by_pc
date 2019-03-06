function main(param_file_path, NN_arch_path, data_path)

%% Initialization
%clear ; close all; clc

% Setup the architecture of the PC network
% weights, biases, variables and errors are all stored in cell arrays.
params = tdfread(param_file_path);
% params.act_type; %activation function: 'tanh', 'sig', 'lin', 'reclin'
% params.l_rate % learning rate
% params.beta1 % "momentum" beta_1
% params.numint_its; % maximum iterations of inference
% params.epochs % number of epochs
% params.d_rate % weight decay parameter
% params.int_step % euler integration constant
% params.lr_decay % learning rate decay
% params.rate_decay_win % min number of epochs between rate decays

arch_f = fopen(NN_arch_path, 'r');
cell = textscan(arch_f,'%f');
params.layer_sizes = cell{1}'; %units in each layer
fclose(arch_f);
%params.layer_sizes = [2, 5, 2, 2]; %units in each layer
% note: last two layers together constitute output layer, their size must
% equal!
params.n_layers = length(params.layer_sizes); % number of layers
var = ones(1, params.n_layers); % puts variance on all layers as 1
var(end)=10; % variance on last layer
params.var=var;

params.timestamp = datestr(now, 'mmdd_HHMMSS');
params.log_f = fopen(strcat('training_runs/', params.timestamp, '.txt'), 'w');
%params.log_f = 1; % for std output
fprintf(params.log_f, evalc(['disp(params)']));

%% Data

% MNIST
load(data_path);
sin = X';
sout = bsxfun(@eq, 1:params.layer_sizes(end), y)'; % Apply element-wise operation to two arrays with implicit expansion enabled

%% Training
[w_pc, b_pc] = rand_init(params); % randomly initialize weights and biases parameters
[w_pc,b_pc, J_history] = batch_learn_pc(sin,sout,w_pc,b_pc,params); % batch is a lot quicker
[y_pred, O] = predict(sin, w_pc, b_pc, params);

fprintf(params.log_f, '\nTraining Set Accuracy: %f\n', mean(double(y_pred == y')) * 100);
fclose(params.log_f);
save(strcat('training_runs/', params.timestamp, '.mat'), 'w_pc', 'b_pc', 'J_history', 'y_pred', 'O', 'params');

%plot(1:params.epochs, J_history);

end
