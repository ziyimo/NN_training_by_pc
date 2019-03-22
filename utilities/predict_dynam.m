function [Y, O_layer_ts, its] = predict_dynam(X, w, b, params, conv_it, conv_th, max_it, fb_strength)
%function Y = predict(X, w, b, params)
%prediction using the dynamical system
%w,b - these are the weights and biases
%params - a structure containing parameters
%X - ONE input data, in a col vector (input data dimension x 1) 
%conv_it - threshold for convergence test

type = params.act_type;
n_layers = params.n_layers;
beta = params.int_step;
e = cell(n_layers-1,1);
x = cell(n_layers-1,1);
x{1} = X;

O_layer_ts = [];
its = 0;
% initialization
for ii = 2:n_layers-1
    x{ii} = unifrnd(-1,1,params.layer_sizes(ii), 1) * sqrt(1/params.layer_sizes(ii));
    %x{ii} = zeros(params.layer_sizes(ii), 1);
end

%calculate initial errors
e{2} = (x{2} - (w{1}*x{1}+b{1}));
for ii=3:n_layers-1
    e{ii} = (x{ii} - w{ii-1} * act_func(x{ii-1}, type) - b{ii-1});
end

conv_cnt = 0;
Y = 0;
max_Y = zeros(params.layer_sizes(end), 1);
while conv_cnt < conv_it
    %update varaible nodes
    for ii=2:n_layers-2
        g = ( w{ii}' *  e{ii+1} ) .* act_func_grad(x{ii}, type);
        g = fb_strength*g; % weaken feedback link
        x{ii} = x{ii} + beta * ( - e{ii} + g );
    end
    % for the last hidden layer (aka output layer)
    x{n_layers-1} = x{n_layers-1} + beta * (-e{n_layers-1});
    %calculate errors
    e{2} = (x{2} - (w{1}*x{1}+b{1}));
    for ii=3:n_layers-1
        e{ii} = (x{ii} - w{ii-1} * act_func(x{ii-1}, type) - b{ii-1});
    end
    % use softmax for output layer
    Ol = softmax(x{n_layers-1});
    O_layer_ts = [O_layer_ts, Ol]; % output, c
    its = its + 1;
    if its == max_it
        break 
    end
    [M, I] = max(Ol);
    if I == Y && M > max_Y(I) - 0.0001 %I == Y && M > conv_th
        conv_cnt = conv_cnt + 1;
        max_Y(I) = M;
    else
        Y = I;
        max_Y(I) = M;
        conv_cnt = 0;
    end
end

%[~, Y] = max(O_layer, [], 1);

end

