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
