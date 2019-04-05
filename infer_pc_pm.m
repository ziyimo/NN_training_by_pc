function [x,e,its] = infer_pc_pm(x,w,b,params)
%function [x,e,its] = infer_pc(x,w,b,params)
% w,b - these are the weights and biases (n_layer - 2)
% x - Variable nodes: x{1} is input layer. x{n_layer} is output layer (n_layer in total)

% e - Error nodes: x{1} empty. x{n_layer} is output layer
% params - a structure containing parameters
it_max = params.numint_its;
n_layers = params.n_layers;
type = params.act_type;
beta = params.int_step;
e = cell(n_layers,1);
var = params.var;

for i = 1:it_max
    x_t = x;
    % for input layer
    F_t = w{1} * x_t{1} + b{1} + (w{2}' *  (x_t{3} - w{2} * act_func(x_t{2}, type) - b{2})).* act_func_grad(x_t{2}, type);
    %x{2} = x_t{2} + beta * (-x_t{2} + F_t); % Fwd Euler scheme
    x{2} = x_t{2}/(1+beta) + beta/(1+beta)*F_t;
    % update varaible nodes
    for ii=3:n_layers-2
        F_t = w{ii-1} * act_func(x_t{ii-1}, type) + b{ii-1} + (w{ii}' *  (x_t{ii+1} - w{ii} * act_func(x_t{ii}, type) - b{ii})).* act_func_grad(x_t{ii}, type);
        %x{ii} = x_t{ii} + beta * (-x_t{ii} + F_t); % Fwd Euler scheme
        x{ii} = x_t{ii}/(1+beta) + beta/(1+beta)*F_t;
    end
    % for the last hidden layer (aka output layer), the activation function
    % is fixed to be softmax
    F_t = w{n_layers-2} * act_func(x_t{n_layers-2}, type) + b{n_layers-2} + (x_t{n_layers} - softmax(x_t{n_layers-1})).*(softmax(x_t{n_layers-1}).*(1 - softmax(x_t{n_layers-1})));
    %x{n_layers-1} = x_t{n_layers-1} + beta * (-x_t{n_layers-1} + F_t); % Fwd Euler scheme
    x{n_layers-1} = x_t{n_layers-1}/(1+beta) + beta/(1+beta)*F_t;
end
its=i;

%calculate FINAL errors
e{2} = (x{2} - (w{1}*x{1}+b{1}))/var(2);
for ii=3:n_layers-1
    e{ii} = (x{ii} - w{ii-1} * act_func(x{ii-1}, type) - b{ii-1})/var(ii) ;
end

e{n_layers} = (x{n_layers} - softmax(x{n_layers-1}))/var(ii);

end