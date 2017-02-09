% [Q, T] = moments_lanpro(A, n, Z, kmax)
%
% Grouped lanczos iterations with no reorthogonalization for
% multiple vectors at once
%
% Input:
%   A: Matrix-vector multiplication operator
%   n: Dimension of A
%   Z: Starting vector as columns
%   kmax: Maximum Lanczos iteration
%
% Output:
%   Q: 3d-matrix of returned basis for Krylov subpsace of each vector
%   T: 3d-matrix of tridiagonal by each vector
%
function [Q,T] = moments_lan(Afun, n, Z, kmax)
  if nargin < 4, kmax = 150; end
  % initialization
  nZ = size(Z, 2);
  Q = zeros(n, nZ, kmax);
  alpha = zeros(kmax, nZ);
  beta = zeros(kmax, nZ);

  k  = 0;
  qk = zeros(size(Z));
  n1 = sqrt(sum(Z.^2));
  r  = bsxfun(@rdivide,Z,n1);
  b  = ones(1,nZ);
  
  % Lanczos algorithm without reorthogonalization
  while k < kmax
      k = k+1;
      qkm1 = qk;
      qk = bsxfun(@rdivide,r,b);
      Q(:,:,k) = qk;
      Aqk = Afun(qk);
      alpha(k,:) = sum(qk.*Aqk);
      r = Aqk - bsxfun(@times,qk,alpha(k,:)) - bsxfun(@times,qkm1,b);
      b = sqrt(sum(r.^2));
      beta(k,:) = b;
  end

  T = zeros(kmax,kmax,nZ);
  for j = 1:nZ
      T(:,:,j) = diag(alpha(:,j)) + diag(beta(1:end-1,j),1) + diag(beta(1:end-1,j),-1);
  end
end
