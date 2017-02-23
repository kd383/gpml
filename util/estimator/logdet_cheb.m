% [ldA, dldA] = logdet_cheb(Afun, n, nZ, N, dAfun, M)
%
% Apporximate the logdet and its derivative of matrix A using Chebyshev
% expansion
%
% Inputs:
%   Afun: A function to apply matrix (to multiple RHS)
%   n: Dimension of the space (if A is a function)
%   nZ: Number of probe vectors with which we want to compute moments
%   N: Number of moments to compute
%   dAfun: Array of functions to apply direction of change on A (one
%          for each parameter)
%   M: Number of parameters
%
% Output:
%   ldA: Logdet estimation
%   dldA: derivative of logdet
%
function [ldA,dldA] = logdet_cheb(Afun, n, nZ, N, sigma,dAfun,M)
    if nargin < 6, dAfun = []; M = 0;end
    dldA = [];
    dc = [];
    % Rescale the matrix so eigenvalues are in [-1,1]
    [Afuns,ab] = rescale_mfunc(Afun,n,sigma^2);
    % Compute the Chebyshev coefficients of spectral density and the
    % derivative
    if nargout<=1
        c = moments_cheb(Afuns,n,nZ,N);
    else
        dAfun = @(x) dAfun(x)/ab(1);
        [c,dc] = moments_cheb(Afuns,n,nZ,N,dAfun,M);
    end
    % Compute the Chebyshev coefficients of log function
    b = moments_log(N,ab(1)/ab(2));
    ldA = n*log(ab(2))+b'*c;
    if ~isempty(dc)
        dldA = b'*dc;
    end
end
