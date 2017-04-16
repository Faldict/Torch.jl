L1Loss(x, y) = mean(abs(x - y))

function NLLLoss(input::Array{Float64, 2}, weights::Array{Float64, 1}, size_average::Bool=true)
    if size_average
        return map((x, y) -> mean(-x * y), input, weights)
    else
        return map((x, y) -> sum(-x * y), input, weights)
    end
end

KLDivLoss(x, target) = mean(target .* (log(target) - x))

MSELoss(x, y) = mean((x - y) .* (x - y))

BCELoss(o, t, weights, size_average::Bool=true) = -mean(weights .* (t .* log(o) + (1 - t) * log(1 - o)))

HigheEmbeddingLoss(x, y, margin=1) = map((x, y) -> y == 1 ? x : max(0, margin - x), x, y)

SoftMarginLoss(x, y) = mean(log(1 + exp(- y .* x)))

CrossEntropyLoss(x, class, weights) = weights[class] * (-x[class] + log(sum(exp(x))))

CosineEmbeddingLoss(x1, x2, y, margin=0) = map((x1, x2, y) -> y == 1 ? 1 - cos(x1, x2) : max(0, cos(x1, x2) - margin), x1, x2, y)
