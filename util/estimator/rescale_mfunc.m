% [Hs, ab] = rescale_mfunc(Hfun, n, range)
% [Hs, ab] = rescale_mfunc(Hfun, n)
% [Hs, ab] = rescale_mfunc(H, range)
%
% Rescale the symmetric matrix H so the eigenvalue range maps to
% between -1 and 1.  If a range is not given, the function uses Lanczos
% to estimate the extremal eigenvalues, expanding by a relative fudge
% factor.
%
% Input:
%    H: The original matrix (or function)
%    n: The dimension of the space (iff H is a function)
%    range: A two-element vector representing an interval where eigs live
%    fudge: A scalar "fudge factor" to guard against range underestimates
%           (default is minimum of 0.01 or 1e-2*lowest eigenvalue)
%
% Output:
%    Hs: A function representing the scaled matrix
%    ab: Transformation parameters: Hs = (H-b)/a
%
function [Hs, ab] = rescale_mfunc(H, n, range)

  % Deal with matrix vs function
  if isa(H, 'function_handle')
    Hfun = H;
    if nargin < 2, error('Missing size argument'); end
    if nargin < 3, range = []; end
  else
    Hfun = @(x) H*x;
    if nargin < 2, range = [];
    elseif nargin <3, range = n; end
    n = size(H,1);
  end

  % Run Lanczos to estimate range (if needed)
  if length(range) < 2
    opts = [];
    opts.isreal = 1;
    opts.issym = 1;
    if length(range) < 1
        range(1) = eigs(Hfun,n,1,'sm',opts);
    end
    range(2) = eigs(Hfun,n,1,'lm',opts);
  end
  fudge = min(0.01, range(1)*1e-2);
  % Parameters for linear mapping
  ab = [(range(2)-range(1))/(2-fudge); (range(2)+range(1))/2];
  Hs = @(x) (Hfun(x)-ab(2)*x)/ab(1);
end
