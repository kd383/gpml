% dAfun = derivative_OU(hyp, x)
% Return a function that applies the derivative of OU kernel with
% respect to hyperparameters
%
% Input:
%   hyp: [ell, sf, sigma]
%   x: data points
%   grid: omit last derivative if using grid
%
% Output:
%   dK: derivative of the OU kernel
function dK = derivative_OU(hyp, x, grid)
    if nargin < 3
        grid = 0;
    end
    if isstruct(hyp)
        ell = exp(hyp.cov(1));
        sf = exp(hyp.cov(2));
        sn2 = exp(2*hyp.lik);
    else
        ell = exp(hyp(1));
        sf = exp(hyp(2));
        sn2 = exp(2*hyp(3));
    end
    d = pdist2(x,x(1));
    c = sf^2*exp(-d/ell);
    if grid
        dK = @(X)[toeplitzmult(d.*c/ell,X), toeplitzmult(2*c,X)];
    else
        dK = @(X)[toeplitzmult(d.*c/ell,X), toeplitzmult(2*c,X), 2*sn2*X];
    end
end
