% [ldB2] = logdet_lanczos(B,n,nZ,kmax,reorth)
%
% Compute an approximation of logdet and its derivative through Lanczos
% iteration
%
% Input:
%   B: I + W^(1/2)*K*W^(1/2)
%   n: Dimension of the space
%   nZ: Number of probe vectors with which we want to compute moments
%   kmax: Number of Lanczos steps
%   reorth: No reorthogonalization or partial reorthogonaliztion
%
% Output:
%   ldB2: log(det(B))/2 estimation
%
function ldB2 = logdet_lanczos(B,n,nZ,kmax,reorth)
    if nargin < 5,  reorth = 0; end
    if nargin < 4,  kmax = 150;  end
    if nargin < 3,  nZ = ceil(log(n)); end
        
    if length(nZ) > 1
        Z = nZ;
        nZ = size(Z,2);
    else
        Z = sign(randn(n,nZ));
    end
    
    if ~isa(B, 'function_handle')
        c = B(:,1);
        B = @(x)toeplitzmult(c,x);
    end
    
    ldB2 = zeros(nZ,1);
    
    if reorth
        [Q,T] = moments_lanpro(B, n, Z, kmax); % partial reorthogonalization
    else
        [Q,T] = moments_lan(B, n, Z, kmax); % no reorthogonalization
    end
    for k = 1:nZ
        [V,theta] = eig(T(:,:,k),'vector');
        wts = (V(1,:).').^2 * norm(Z(:,k))^2;
        ldB2(k) = sum(wts.*log(theta));
    end
    ldB2 = real(mean(ldB2))/2;
end
