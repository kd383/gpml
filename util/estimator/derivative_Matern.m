% dAfun = derivative_Matern(d, hyp, x)
% Return a function that applies the derivative of Matern kernel with
% respect to hyper-parameters
%
% Input:
%   d: 1, 3, or 5
%   hyp: [ell, sf, sigma]
%   x: data points
%   grid: omit last derivative if using grid
%
% Output:
%   dK: derivative of the Matern kernel with d=1,3,5

function dK = derivative_Matern(d, hyp, x, grid)
if (d ~= 1 && d ~= 3 && d ~= 5)
    error('d has to be 1, 3, or 5... so... yeah... no.... aborting....')
end
if nargin < 4
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
dd = pdist2(x,x(1));

% Figure out the Toeplitz vectors
if d == 1
    c_ell = sf^2*dd.*exp(-dd/ell)/ell;
    c_sf  = 2*sf^2*exp(-dd/ell);
elseif d == 3
    c_ell = 3*sf^2*dd.^2.*exp(-sqrt(3)*dd/ell)/ell^2;
    c_sf  = 2*sf^2*(1+sqrt(3)*dd/ell).*exp(-sqrt(3)*dd/ell);
elseif d == 5
    c_ell = 5*sf^2*dd.^2.*(ell+sqrt(5)*dd).*exp(-sqrt(5)*dd/ell)/(3*ell^3);
    c_sf  = 2*sf^2*(1+sqrt(5)*dd/ell + 5*dd.^2/(3*ell^2)).*exp(-sqrt(5)*dd/ell);
end

% Create handle
if grid
    dK = @(X)[toeplitzmult(c_ell,X), toeplitzmult(c_sf,X)];
else
    dK = @(X)[toeplitzmult(c_ell,X), toeplitzmult(c_sf,X), 2*sn2*X];
end

end
