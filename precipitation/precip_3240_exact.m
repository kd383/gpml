% Load and split data
% ===================
ntest = 100000;
nhyp = 3000;
ntrain = 10000; % DE, Feb 16
%ntrain = 25000;
[X, y, Xtest, ytest] = load_precip3240('data/precipitation3240/processed-data-2010.csv', ntest, ntrain);



d = size(X, 2);
limits = zeros(d, 2);
limits(:,1) = min(X)';
limits(:,2) = max(X)';

hyp = struct();
hyp.cov = zeros(d+1, 1);
cov = {@covSEard};
sf = .5 * std(y);
hyp.cov(end) = log(sf);
for i=1:d
    spann = limits(i,2) - limits(i, 1);
    ell = spann / 20;
    hyp.cov(i) = log(ell);
end

meanfunc = {@meanConst}; hyp.mean=[mean(y)];
lik = @likGauss; sn = .2 * std(y);  hyp.lik = log(sn);
%hyp.cov
%hyp.mean
%hyp.lik
fprintf('Optimize hypers\n');
hyp = minimize(hyp,@gp,-300, @infExact, meanfunc, cov, lik, X(1:nhyp, :),y(1:nhyp));
fprintf('Resulting hyperparameters\n');
exp(hyp.cov(1:end-1)) ./ (limits(:,2) - limits(:,1))
fprintf('Coefficients\n');
exp(hyp.cov(end))
exp(hyp.lik)
fprintf('Mean\n');
hyp.mean

fprintf('Start inference on %i samples\n', ntrain);
tic
[post, nlZ] = infExact(hyp, meanfunc, cov, lik, X, y);
t_train = toc;
fprintf('Finished inference after %.1f seconds\n', t_train);

% Predict
% =======
tic
y_pred = gp(hyp, @infExact, meanfunc, cov, lik, X, post, Xtest);
t_predict = toc;
mae = mean(abs(y_pred - ytest))
mae_mp = mean(abs(bsxfun(@minus, ytest, mean(ytest))));
smae = mae / mae_mp
mse = mean((y_pred - ytest).^2)

