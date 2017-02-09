% sur = build_surrogate(cov, x, opt)
%
% Building surrogate surface for fast evaluation of logdet and its
% derivative with respect to hyperparameters
% 
% Input:
%   cov: Covariance function
%   x: Data points
%   opt: Options for building surrogate (including exact or approximate evaluation)
%
% Output:
%   sur: Surrogate surface
%
function sur = build_surrogate(cov, x, opt)
    if ~all(isfield(opt, {'bounds','npts','ntrials','param','method'}))
        error('Missing options for building surrogate.');
    else
        bounds = opt.bounds; npts = opt.npts; ntrials = opt.ntrials;
        fields = opt.param{1}; idx = [0 cumsum(opt.param{2})];
    end
    
    N = size(bounds,2);
    if isfield(opt,'nZ'), nZ = opt.nZ; else nZ = ceil(log(N)); end
    if length(nZ)>1, Z = nZ; else Z = sign(randn(n,nZ)); end
    if isfield(opt,'kmax'), kmax = opt.kmax; else kmax = 150;  end
    kernel = CubicKernel(N);
    tail = LinearTail(N);
    exp_des = best_slhd(npts, N, ntrials);
    exp_des = bsxfun(@times, exp_des, bounds(2,:)-bounds(1,:));
    exp_des = bsxfun(@plus, exp_des, bounds(1,:));
    fX = zeros(npts, 1);
    
    for i = 1:npts
        for j = 1:length(fields)
            hyp.(fields{j}) = exp_des(i,idx(j)+1:idx(j+1));
        end
        sn2 = exp(2*hyp.lik);
        K = apx(hyp, cov, x, []);
        if strcmp(method,'lanczos')
            fX(i) = logdet_lanczos(@(X)K.mvm(X)/sn2+X,size(x,1),Z,[],kmax,0);
        else
            fX(i) = K.fun(ones(N,1)/sn2);
        end
    end
    sur = SurrogateLogDet(x, fX, exp_des, kernel, tail);
end