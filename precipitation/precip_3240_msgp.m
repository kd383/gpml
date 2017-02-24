%Load and split data
%===================
%{
[Xhyp, yhyp] = load_precip3240('data/precipitation3240/processed-data-2010-jan.csv');

d = size(Xhyp, 2);
limits = zeros(d, 2);
limits(:,1) = min(Xhyp)';
limits(:,2) = max(Xhyp)';

ng = [100, 100, 50]; % lat, long, time
% Setup MSGP
% ==========
xg = {};
%hyp = struct();
%hyp.cov = zeros(2*d, 1);
cov  = {};
sf = .5 * std(yhyp);
for i=1:d
   spann = limits(i,2) - limits(i, 1);
   ell = spann / 20;
   %hyp.cov(2 * (i-1)+1:2*i) = log([ell; sf^(1/d)]);
   cov{i} = {@covSEiso};
   xg{i} = {linspace(limits(i,1)-0.01*spann,limits(i, 2) + 0.01*spann,ng(i))'};
end

mopt.cg_maxit =1500; 
mopt.cg_tol = 1e-4;                 % keep these values fixed
mopt.pred_var = 0;
mopt.ldB2_lan = 1;
mopt.ldB2_lan_kmax = 50;
mopt.ldB2_lan_hutch = sign(randn(size(Xhyp,1),5));
meanfunc = {@meanConst}; %hyp.mean=[mean(yhyp)];
lik = @likGauss; sn = .5 * std(yhyp);  %hyp.lik = log(sn);
covg = {@covGrid,cov,xg};


fprintf('Optimize hypers\n');
inf_method = @(varargin) infGrid(varargin{:},mopt);
tic;
hyp = minimize(hyp,@gp,-35, inf_method, meanfunc, covg, lik, Xhyp,yhyp);
time = toc;
exp(hyp.cov(1:2:end)) ./ (limits(:,2) - limits(:,1))
fprintf('Coefficients\n');
exp(hyp.cov(2:2:end))
exp(hyp.lik)
fprintf('Mean\n');
hyp.mean
%}

a = load('precip_result.mat', 'hyp');
hyp = a.hyp;

ntest = 100000;
[X, y, Xtest, ytest] = load_precip3240('data/precipitation3240/processed-data-2010.csv', ntest);
ntrain = size(X, 1);

d = size(X, 2);
limits = zeros(d, 2);
limits(:,1) = min(min(X), min(Xtest))';
limits(:,2) = max(max(X), max(Xtest))';

ng = [130, 130, 600]; % lat, long, time
% Setup MSGP
% ==========
xg = {};
cov  = {};
sf = .5 * std(y);
for i=1:d
    spann = limits(i,2) - limits(i, 1);
    cov{i} = {@covSEiso};
    xg{i} = {linspace(limits(i,1)-0.01*spann,limits(i, 2) + 0.01*spann,ng(i))'};
end
covg = {@covGrid,cov,xg};
meanfunc = {@meanConst};
lik = @likGauss;
mopt.cg_maxit =1500; 
mopt.cg_tol = 1e-4;                 % keep these values fixed
mopt.pred_var = 0;
mopt.ldB2_lan = 1;
mopt.ldB2_lan_kmax = 50;
mopt.ldB2_lan_hutch = sign(randn(size(X,1),5));

fprintf('Start inference\n');
tic
[post nlZ] = infGrid(hyp,meanfunc,covg,lik,X, y, mopt);
t_train = toc;
fprintf('Finished inference after %.1f seconds\n', t_train);

% Predict
% =======
tic
y_pred = post.predict(Xtest);
t_predict = toc;
mae = mean(abs(y_pred - ytest))
mae_mp = mean(abs(bsxfun(@minus, ytest, mean(ytest))));
smae = mae / mae_mp
mse = mean((y_pred - ytest).^2)
%}