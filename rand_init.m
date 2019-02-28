function [w,b] = rand_init(params)
%w,b are cell arrays. Their i^th cell elements correspond the the i^th
%weights/biases of the network. For example the first cell element
%corresponds to the first weights/biases that convey information form input
%to first hidden layer of network.
%params is a structure of parameters

type = params.act;
n_layers = params.n_layers;
w = cell(n_layers-2,1);
b = cell(n_layers-2,1);
neurons=params.layer_sizes;

% w{n_layers-1} = I
% b{n_layers-1} = 0

for i = 1:n_layers-2
    norm_b = 0;
    switch type
        case 'lin'
            norm_w = sqrt(1/(neurons(i+1) + neurons(i))) ;
        case 'tanh'
            norm_w = sqrt(6/(neurons(i+1) + neurons(i))) ;
        case 'sig'
            norm_w = 4 * sqrt(6/(neurons(i+1) + neurons(i))) ;
        case 'reclin'
            norm_w = sqrt(1/(neurons(i+1) + neurons(i))) ;
            norm_b = 0.1;
    end    
    w{i} = unifrnd(-1,1,neurons(i+1),neurons(i)) * norm_w ;
    b{i} = zeros(neurons(i+1),1) + norm_b * ones(neurons(i+1),1) ;  
end