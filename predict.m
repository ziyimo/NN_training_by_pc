function [Y, O_layer] = predict(X, w, b, params)
%function Y = predict(X, w, b, params)
%w,b - these are the weights and biases
%params - a structure containing parameters
%X - input data, in an array of size (input data dimension x number of data samples) 

type = params.act;
n_layers = params.n_layers;

% Data structures 
% A - Activation (post-non-linearity)
% Z - Incoming connection (pre-non-linearity)

A = X; % input layer activation, c

for i = 2:n_layers-2
    Z = w{i-1}*A+b{i-1}; % incoming to layer i, c
    A = act_func(Z, type); % layer i activation, c 
end

% use softmax for output layer
Z = w{n_layers-2}*A+b{n_layers-2}; % incoming to output layer, c
O_layer = softmax(Z); % output, each col is a sample (c)

[~, Y] = max(O_layer, [], 1);

end

