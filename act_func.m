function g = act_func(z, type)

switch type
    case 'lin'
        g = z;
    case 'tanh'
        g = tanh(z);
    case 'sig'
        g = 1.0 ./ (1.0 + exp(-z));
    case 'reclin'
        g = max(z,0);
    otherwise
        error('unsupported activation function type');
end
end