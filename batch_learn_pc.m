function [w, b, cost_history] = batch_learn_pc(X_in,Y_out,w,b,params, y)
%function [w,b, cost_history] = batch_learn_pc(in,out,w,b,params)
% w,b - these are the weights and biases (n_layer - 2)
% X_in - input data, in an array of size (input data dimension x number of data samples) 
% Y_out - output data, in an array of size (output data dimension x number of data samples)
% params - a structure containing parameters
% y - output data in 1D format, used to calculate training set accuracy

n_layers = params.n_layers;
type = params.act_type;
l_rate = params.l_rate/length(X_in);
beta1 = params.beta1;
d_rate = params.d_rate;
lr_decay = params.lr_decay;
var = params.var;
v_out = var(end);
a = n_layers-2;

x = cell(n_layers,1);
grad_b = cell(size(b));
V_db = cell(size(b));
grad_w = cell(size(w));
V_dw = cell(size(w));

for ii = 1:a
    V_dw{ii} = zeros(params.layer_sizes(ii+1), params.layer_sizes(ii));
    V_db{ii} = zeros(params.layer_sizes(ii+1), 1);
end

%organize data into cell arrays
x{1} = X_in;
x{n_layers} = Y_out;

cost_history = zeros(params.epochs, 1);

rdw = params.buf_win;
rde = 1;
acc_tmax = 0;
%learn
for epoch = 1:params.epochs
    %make a prediciton 
    for ii = 2:n_layers-1
        x{ii} = w{ii-1} * ( act_func(x{ii-1}, type) ) +  b{ii-1};
    end
    %infer
    [x,e,~] = infer_pc(x,w,b,params);
    for ii = 1:a
        %calculate gradients
        grad_b{ii} = v_out * sum(e{ii+1}, 2);
        grad_w{ii} = v_out * e{ii+1} * act_func( x{ii}, type )' - d_rate*w{ii};
        %calculate momentum
        V_db{ii} = (beta1*V_db{ii} + (1-beta1)*grad_b{ii})/(1-beta1^epoch);
        V_dw{ii} = (beta1*V_dw{ii} + (1-beta1)*grad_w{ii})/(1-beta1^epoch);
        %update weights   
        w{ii} = w{ii} + l_rate * V_dw{ii};
        b{ii} = b{ii} + l_rate * V_db{ii};
    end
    [y_pred, pc_out] = predict(X_in, w, b, params); % examine cost
    J = -sum(sum(log(pc_out).*Y_out))/length(X_in);
    cost_history(epoch) = J;
    acc_t = mean(double(y_pred == y')) * 100;
    fprintf(params.log_f, 'Epoch %d | Cost: %e | Training Set Accuracy: %f\n', epoch, J, acc_t);
    if acc_t > acc_tmax
        save(strcat('training_runs/', params.timestamp, '_best.mat'), 'w', 'b', 'J', 'acc_t', 'epoch', 'params');
        acc_tmax = acc_t;
    end
    if epoch > rde+rdw && cost_history(epoch) > cost_history(epoch-1)
        l_rate = l_rate*lr_decay;
        params.numint_its = params.numint_its*params.ADAPTIT_ri;
        params.int_step = params.int_step/params.ADAPTIT_rs;
        fprintf(params.log_f, 'New learning rate: %.5f\n', l_rate*length(X_in));
        fprintf(params.log_f, 'New inference its: %d\n', params.numint_its);
        fprintf(params.log_f, 'New inference steps: %d\n', params.int_step);
        rde = epoch;
    end
    if epoch > rde+params.ADAPTIT_w
        params.numint_its = params.numint_its*params.ADAPTIT_ri;
        params.int_step = params.int_step/params.ADAPTIT_rs;
        fprintf(params.log_f, 'New inference its: %d\n', params.numint_its);
        fprintf(params.log_f, 'New inference steps: %d\n', params.int_step);
        rde = epoch;
    end
end

end
