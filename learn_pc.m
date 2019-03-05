function [w, b] = learn_pc(X_in,Y_out,w,b,params)
%function [w,b] = learn_pc(in,out,w,b,params)
% w,b - these are the weights and biases (n_layer - 2)
% X_in - input data, in an array of size (input data dimension x number of data samples) 
% Y_out - output data, in an array of size (output data dimension x number of data samples)
% params - a structure containing parameters
n_layers = params.n_layers;
type = params.act;
l_rate = params.l_rate/length(X_in); % scale per sample learning rate by # of samples
d_rate = params.d_rate;
var = params.var;
v_out = var(end);
a = n_layers-2;

iterations = length(X_in);
for its = 1:iterations
    x = cell(n_layers,1);
    grad_b = cell(size(b));
    grad_w = cell(size(w));
    
    %organise date into cells arrays
    x{1} = X_in(:,its);
    x{n_layers} = Y_out(:,its);
    %make a prediciton 
    for ii = 2:n_layers-1
        x{ii} = w{ii-1} * ( act_func(x{ii-1}, type) ) +  b{ii-1};
    end
    %infer
    [x,e,~] = infer_pc(x,w,b,params);
    %calculate gradients
    for ii = 1:a
        grad_b{ii} = v_out * e{ii+1};
        grad_w{ii} = v_out * e{ii+1} * act_func( x{ii}, type )' - d_rate*w{ii};
    end
    %update weights
    for ii = 1:a        
        w{ii} = w{ii} + l_rate * grad_w{ii}   ;
        b{ii} = b{ii} + l_rate * grad_b{ii}   ;
    end
end

end