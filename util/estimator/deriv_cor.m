function dB = deriv_cor(hyp,xg,Mx,deriv)
    ell = exp(hyp.cov(1));
    sf = exp(hyp.cov(2));
    sn2 = exp(2*hyp.lik);
    
    if deriv
        d2 = pdist2(xg,xg(1)).^2;
        c = sf^2*exp(-d2/ell^2/2);
        dKxg_ell = @(x) toeplitzmult(d2.*c/ell^2,x);
    else
        d2 = pdist2(xg,xg(1));
        c = sf^2*exp(-d2/ell);
        dKxg_ell = @(x) toeplitzmult(d2.*c/ell,x);
    end
    dKxg_sf = @(x) toeplitzmult(2*c,x);
    diag_cor_ell = sum(Mx'.*dKxg_ell(full(Mx')));
    diag_cor_sf = sum(Mx'.*dKxg_sf(full(Mx')));
    dB = @(x)[Mx*dKxg_ell(full(Mx'*x)) - bsxfun(@times,diag_cor_ell',x),...
                Mx*dKxg_sf(full(Mx'*x)) + bsxfun(@times,2*sf^2-diag_cor_sf',x)]/sn2;
end

% y = toeplitzmult(c, r, x)
% y = toeplitzmult(c, x)
%
% O(n*log(n)) method for computing the matrix-vector multiplication
% of a Toeplitz matrix
%
% Input:
%    c: First column of toeplitz matrix (wins diagonal conflict)
%    r: First row of toeplitz matrix (r(1)=c(1))
%    x: Vector to multiply
%
% Output:
%    y: Right-side of matrix-vector multiplication

function y = toeplitzmult(c,r,x)
    if nargin < 3
        x = r;
        r = c;
    end

    [n,m]=size(x);
    y = ifft(bsxfun(@times,fft([c;r(end:-1:2)]),fft([x;zeros(n-1,m)])));
    y = y(1:n,:);
end
