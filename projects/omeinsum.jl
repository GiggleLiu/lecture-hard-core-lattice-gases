using OMEinsum, Enzyme

function f(x::AbstractMatrix)
    return sum(x^2)
end

Enzyme.autodiff(Reverse, f, Duplicated(randn(4, 4), zeros(4, 4)))

function g(x)
    ein"ii->"(x)[]
end
Enzyme.autodiff(Reverse, g, Duplicated(randn(4, 4), zeros(4, 4)))