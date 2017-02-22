function dB = deriv_cor(hyp,xg,Mx,deriv)
    ell = exp(hyp.cov(1));
    sf = exp(hyp.cov(2));
    sn2 = exp(2*hyp.lik);
    
    if deriv == 1
        d2 = pdist2(xg,xg(1)).^2;
        c = sf^2*exp(-d2/ell^2/2);
        cc = d2.*c/ell^2; % DE
        dKxg_ell = @(x) toeplitzmult(d2.*c/ell^2,x);
    elseif deriv == 0 
        d2 = pdist2(xg,xg(1));
        c = sf^2*exp(-d2/ell);
        cc = d2.*c/ell; % DE
        dKxg_ell = @(x) toeplitzmult(d2.*c/ell,x);
    elseif deriv == 2
        d2 = pdist2(xg,xg(1));
        c = sf^2*(1+sqrt(3)*d2/ell).*exp(-sqrt(3)*d2/ell);
        cc = 3*sf^2*d2.^2.*exp(-sqrt(3)*d2/ell)/ell^2;
        dKxg_ell = @(x) toeplitzmult(cc,x);
    end
    dKxg_sf = @(x) toeplitzmult(2*c,x);
    %diag_cor_ell = sum(Mx'.*dKxg_ell(full(Mx'))); % DE
    diag_cor_ell = sum((toeplitz(cc,cc) * Mx') .* Mx', 1); % DE
    %diag_cor_sf = sum(Mx'.*dKxg_sf(full(Mx'))); %DE
    diag_cor_sf = sum((toeplitz(2*c,2*c) * Mx') .* Mx', 1); % DE
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
