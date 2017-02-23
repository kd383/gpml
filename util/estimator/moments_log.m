% c = moments_log(N, alpha)
%
% Compute the Chebyshev expansion of log(1+alpha*x) for degree N-1
% using Chebyshev nodes (Chebyshev-Gauss Quadrature)
%
% Input:
%   N: number of moments
%   alpha: scaling factor (alpha<1)
%
% Output:
%   c: Chebyshev moments from 0 to N-1
%
function c = moments_log(N,alpha)
    xk = cos(pi*((1:N)'-1/2)/N);
    fk = log(1+alpha*xk);
    c = zeros(N,1);
    T1 = ones(N,1);
    T2 = xk;
    c(1) = T1'*fk/N;
    c(2) = 2*T2'*fk/N;
    for k=3:N
        T3 = 2*xk.*T2-T1;
        c(k) = 2*T3'*fk/N;
        T1 = T2;
        T2 = T3;
    end
end
