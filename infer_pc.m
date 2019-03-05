function [x,e,its] = infer_pc(x,w,b,params)
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

%calculate initial errors
e{2} = (x{2} - (w{1}*x{1}+b{1}))/var(2);
for ii=3:n_layers-1
    e{ii} = (x{ii} - w{ii-1} * act_func(x{ii-1}, type) - b{ii-1})/var(ii) ;
end

e{n_layers} = (x{n_layers} - softmax(x{n_layers-1}))/var(ii);

for i = 1:it_max
    %update varaible nodes
    for ii=2:n_layers-2
        g = ( w{ii}' *  e{ii+1} ) .* act_func_grad(x{ii}, type);
        x{ii} = x{ii} + beta * ( - e{ii} + g );
    end
    % for the last hidden layer (aka output layer), the activation function
    % is fixed to be softmax
    g = e{n_layers}.*(softmax(x{n_layers-1}).*(1 - softmax(x{n_layers-1})));
    x{n_layers-1} = x{n_layers-1} + beta * (-e{n_layers-1} + g);
    %calculate errors
    e{2} = (x{2} - (w{1}*x{1}+b{1}))/var(2);
    for ii=3:n_layers-1
        e{ii} = (x{ii} - w{ii-1} * act_func(x{ii-1}, type) - b{ii-1})/var(ii) ;
    end
    e{n_layers} = (x{n_layers} - softmax(x{n_layers-1}))/var(ii);
end
its=i;
end