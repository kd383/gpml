% [c,dc] = moments_cheb(Afun, n, nZ, N, dAfun, M)
% [c, cs, dc, dcs] = moments_cheb(Afun, n, nZ, N, dAfun, M)
%
% Compute a column vector (or vectors) of Chebyshev moments of
% the form c(k) = tr(T_k(A)) for k = 0 to N-1.  This routine
% does no scaling; the spectrum of A should already lie in [-1,1].
% The traces are computed via a stochastic estimator with nZ probes.
% The derivative is computed if the direction of change dAfun is  given
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
%    c: an column vector of N moment estimates
%    cs: standard deviation of the moment estimator (std/sqrt(nZ))
%    dc: derivative of c in direction of dA
%    dcs: standard deviation of the derivative estimator
%
function [c, cs, dc, dcs] = moments_cheb(Afun,n,nZ,N,dAfun, M)
    % Construct random probe
    if length(nZ) > 1
        Z = nZ;
        nZ = size(Z,2);
    else
        Z = sign(randn(n,nZ));
    end
    % minimum number of moments
    if N < 10
        N = 10;
    end
    % when derivative not wanted
    if nargin < 5
        cZ = zeros(N,nZ);
        T1 = Z;
        T2 = Afun(Z);
        cZ(1,:) = sum(Z.*T1);
        cZ(2,:) = sum(Z.*T2);
        for k=3:N
            T3 = 2*Afun(T2)-T1;
            cZ(k,:) = sum(Z.*T3);
            T1 = T2;
            T2 = T3;
        end
        c  = mean(cZ,2);
        if nargout > 1
            cs = std(cZ,0,2)/sqrt(nZ);
        end
    % when direction of change is given
    else
        cZ = zeros(N,nZ);
        dcZ = zeros(N,M*nZ);
        T1 = Z;
        T2 = Afun(Z);
        dT1 = zeros(n,M*nZ);
        dT2 = dAfun(Z);
        cZ(1,:) = sum(Z.*T1);
        cZ(2,:) = sum(Z.*T2);
        dcZ(2,:)= sum(repmat(Z,[1 M]).*dT2);
        for k=3:N
            T3 = 2*Afun(T2)-T1;
            dT3 = 2*Afun(dT2)-dT1+2*dAfun(T2);
            cZ(k,:) = sum(Z.*T3);
            dcZ(k,:) = sum(repmat(Z,[1 M]).*dT3);
            T1 = T2;
            T2 = T3;
            dT1 = dT2;
            dT2 = dT3;
        end
        % do not return std of estimation (ues this)
        if nargout == 2
            c = mean(cZ,2);
            for i=0:M-1
                cs(:,i+1) = mean(dcZ(:,i*nZ+1:(i+1)*nZ),2);
            end
        else
            c = mean(cZ,2);
            cs = std(cZ,0,2)/sqrt(nZ);
            dc = zeros(n,M);
            dcs = zeros(n,M);
            for i=0:M-1
                dc(:,i+1) = mean(dcZ(:,i*nZ+1:(i+1)*nZ),2);
                dcs(:,i+1) = std(dcZ(:,i*nZ+1:(i+1)*nZ),0,2)/sqrt(nZ);
            end
        end
    end
end
