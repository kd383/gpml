%{
clear all
close all
clc

addpath('../cov4.0/')
addpath('../surrogate/')
%}
N = 1000;
opt.bounds = log([5e-2,0.1,1e-2; 1,1.5,1]);
opt.ntrials = 1000;
opt.npts = 300;
%{
% X = linspace(0, 4, N)';
X = 4*rand(N,1) - 2;
xg = apxGrid('create', X, true,200);
cov = {@apxGrid, {@covMaterniso}, xg};

kernel = CubicKernel(3);
tail = LinearTail(3);
npts = 200;
exp_des = best_slhd(npts, 3, 1000);
exp_des = bsxfun(@times, exp_des, opt.bounds(2,:) - opt.bounds(1,:));
exp_des = bsxfun(@plus, exp_des, opt.bounds(1,:));

% Print the test details
fprintf('Finite difference test for the surrogate\n')
fprintf('- Using X = linspace(0, 4, %d)\n', N)
fprintf('- Using the Gaussian RBF kernel\n')
fprintf('- Using %d design points\n\n', npts)

fX = zeros(npts, 1);
Z = sign(randn(N,ceil(log(N))));
fprintf('Building surrogate... Get some coffee?\n')
for i = 1:npts
    hyp.cov = exp_des(i, 1:2); % [ell, sf]
    hyp.lik = exp_des(i,3); % sigma
    sn2 = exp(2*hyp.lik);
    K = apx(hyp, cov, X);
    fX(i) = logdet_lanczos(@(X)K.mvm(X) + sn2*X, N, Z, [], 150, 0);
end
fprintf('Finished building surrogate\n')
sur = SurrogateLogDet(X, fX, exp_des, kernel, tail);
%}
% Check the derivatives for some random points and compare to finite
% differences
Ntest = 10;
h = 1e-6;
Xtest = rand(Ntest, 3);
Xtest = bsxfun(@times, Xtest, opt.bounds(2,:) - opt.bounds(1,:));
Xtest = bsxfun(@plus, Xtest, opt.bounds(1,:));

for i=1:Ntest
    deriv = zeros(3,1);
    for j=0:3
        % j == 1 is derivative on ell
        ell1 = Xtest(i,1)   - h*(j==1);
        ell2 = Xtest(i,1)   + h*(j==1);
        % j == 2 is derivative on sf
        sf1 = Xtest(i,2)    - h*(j==2);
        sf2 = Xtest(i,2)    + h*(j==2);
        % j == 3 is derivative on sigma
        sigma1 = Xtest(i,3) - h*(j==3);
        sigma2 = Xtest(i,3) + h*(j==3);
        
        Kleft = covMaterniso(1,[ell1,sf1],X);
        Kleft = (Kleft+Kleft')/2/exp(2*sigma1)+eye(N);
        Cleft = cholcov(Kleft);
        
        Kright = covMaterniso(1,[ell2,sf2],X);
        Kright = (Kright+Kright')/2/exp(2*sigma2)+eye(N);
        Cright = cholcov(Kright);
        
        if j==0 % Compute the value of the logdet
            val = sum(log(diag(Cright)));
        else % Compute the derivative of the logdet
            deriv(j) = (sum(log(diag(Cright))) - sum(log(diag(Cleft))))/(2*h);
        end
    end
    
    [rbf_val,rbf_deriv] = sur.predict(Xtest(i,:));
    
    % Print the results
    fprintf('=== Test point #%d log([ell,sf,sigma])=(%.3f,%.3f,%.3f) ===\n', i, Xtest(i,:))
    fprintf('\tTrue value:       %.3f,    Surrogate value:       %.3f\n', val, rbf_val)
    fprintf('\tTrue ell-deriv:   %.3f,    Surrogate ell-deriv:   %.3f\n', deriv(1), rbf_deriv(1))
    fprintf('\tTrue sf-deriv:    %.3f,    Surrogate sf-deriv:    %.3f\n', deriv(2), rbf_deriv(2))
    fprintf('\tTrue sigma-deriv: %.3f,    Surrogate sigma-deriv: %.3f\n', deriv(3), rbf_deriv(3))
end

rmpath('../cov4.0/')
rmpath('../surrogate/')