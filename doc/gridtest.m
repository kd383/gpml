N = 5000;
X = sort(4*rand(N,1)-2,'ascend');
cov = {@(varargin)covMaterniso(1,varargin{:})};
lik = {@likGauss};
means = {@meanZero};
hyp = struct('mean', [], 'cov', log([0.2,1]), 'lik', log(0.1));
K = feval(cov{:},hyp.cov,X);
K = (K+K')/2+exp(2*hyp.lik)*eye(N);
Y = chol(K)'*randn(N,1);
opt.cg_maxit = 1200; opt.cg_tol = 1e-3;opt.replace_diag = 1;
first_term = zeros(2,10);
second_term = zeros(3,10);
K = apx(hyp,cov,X,opt);
[ldB2,solveKiW,dW,dldB2,L] = K.fun(ones(5000,1)/exp(hyp.lik*2));
[post,nlZ,dnlZ] = infGaussLik(hyp,means,cov,lik,X,Y,opt);
deriv_true = [nlZ,dnlZ.cov',dnlZ.lik];
deriv_grid = zeros(10,4);
deriv_fitc = zeros(10,4);
first_term_true = Y'*solveKiW(Y);
second_term_true = ldB2;
for npts = 1:10
    xg = apxGrid('create',X,true,500*npts);
    xf = linspace(-2,2,500*npts)';
    covg = {@apxGrid,cov,xg};
    covf = {@apxSparse,cov,50*xf};
    K2 = apx(hyp,covg,X,opt);
    [ldB2g,solveKiWg,dWg,dldB2g,Lg] = K2.fun(ones(5000,1)/exp(hyp.lik*2));
    [post,nlZ,dnlZ] = infGaussLik(hyp,means,covg,lik,X,Y,opt);
    deriv_grid(npts,:) = [nlZ,dnlZ.cov',dnlZ.lik];

    K3 = apx(hyp,covf,X,opt);
    [ldB2f,solveKiWf,dWf,dldB2f,Lf] = K3.fun(ones(5000,1)/exp(hyp.lik*2));
    [post,nlZ,dnlZ] = infGaussLik(hyp,means,covf,lik,X,Y,opt);
    deriv_fitc(npts,:) = [nlZ,dnlZ.cov',dnlZ.lik];
    first_term(:,npts)=[solveKiWg(Y),solveKiWf(Y)]'*Y;
    second_term(1:2,npts) = [ldB2g;ldB2f];
    optt = opt; optt.ldB2_lan = 1;
    K4 = apx(hyp,covg,X,opt);
    [ldB2g,solveKiWg,dWg,dldB2g,Lg] = K2.fun(ones(5000,1)/exp(hyp.lik*2));
    second_term(3,npts) = ldB2g;
    npts
end