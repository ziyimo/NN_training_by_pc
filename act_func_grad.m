function g_p = act_func_grad(z, type)

switch type
    case 'lin'
        g_p = ones(size(z));
    case 'tanh'
        g_p = 1 - tanh(z).^2;
    case 'sig'
        g = act_func(z, 'sig');
        g_p = g .* (1 - g) ;
    case 'reclin'
        g_p = (sign(z) + 1)./2; % at 0, set arbitrarily to 0.5
    otherwise
        error('unsupported activation function type');
end

end