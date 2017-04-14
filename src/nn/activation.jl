function Threshold(input, threshold, value)
    output = map(x -> x > threshold ? x : value, input)
    return output
end

function Threshold!(input, threshold, value)
    input = map(x -> x > threshold ? x : value, input)
    return input
end

function ReLU(input)
    return max(0, input)
end

function ReLU!(input)
    input = max(0, input)
    return input
end

function Hardtanh(input, min_value, max_value)
    return map(x -> x > max_value ? 1 : x < min_value ? -1 : x, input)
end

function Hardtanh(input, min_value, max_value)
    input = map(x -> x > max_value ? 1 : x < min_value ? -1 : x, input)
    return input
end

function ReLU6(input)
    return min(max(0, x), 6)
end

function ReLU6!(input)
    input = min(max(0, x), 6)
    return input
end

Sigmoid(x) = 1 ./ (1 + exp(-x))

Tanh(x) = (exp(x) - exp(-x)) ./ (exp(x) + exp(-x))

ELU(x) = max(0, x) + min(0, alpha * (exp(x) - 1))

function hardshrink(input, lambda)
    return map(x -> x > lambda || x < lambda ? x : 0, input)
end

function hardshrink!(input, lambda)
    input = map(x -> x > lambda || x < lambda ? x : 0, input)
end

LeakyReLU(x, negative_slope) = max(0, x) + negative_slope(0, x)

LogSigmoid(x) = log(Sigmoid(x))

Softplus(x, beta) = 1 ./ beta * log(1 + exp(beta * x))

function Softshrink(input, lambda)
    output = map(input) do x
        if x > lambda
            return x - lambda
        elseif x < -lambda
            return x + lambda
        else
            return 0
        end
    end
    return output
end

function Softshrink!(input, lambda)
    input = map(input) do x
        if x > lambda
            return x - lambda
        elseif x < - lambda
            return x + lambda
        else 
            return 0
        end
    end
    return input
end

PReLU(x, a) = max(0, x) + a * min(0, x)

Softsign(x) = x ./ (1 + abs(x))

Tanhshrink(x) = x - Tanh(x)

function Softmin(input)
    shift(x) = max(input) - x
    return map(x -> exp(-x - shift(x)) ./ sum(exp(map(y -> y - shift(x), input))), input)
end

function Softmin!(input)
    shift(x) = max(input) - x
    input = map(x -> exp(-x - shift(x)) ./ sum(map(y -> exp(y - shift(x)), input)), input)   
end

function Softmax(input)
    return map(x -> exp(x - max(x)) ./ sum(map(y -> exp(y - max(x)), input)), input)
end

function Softmax!(input)
    input = map(x -> exp(x - max(x)) ./ sum(map(y -> exp(y - max(x)), input)), input) 
end

LogSoftmax(x) = log(Softmax(x))
