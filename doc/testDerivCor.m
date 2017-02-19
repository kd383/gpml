clear
N = 1000;
X = sort(4*rand(N,1)-2,'ascend');
xg = apxGrid('create',X,true,500);
cov = {@covSEiso};
%cov = {@(varargin)covMaterniso(1,varargin{:})};
covg = {@apxGrid,cov,xg};
hyp = struct('mean', [], 'cov', log([0.3,1.2]), 'lik', log(0.1));
opt.cg_maxit = 700;opt.cg_tol = 1e-5;opt.replace_diag = 1;
K = apx(hyp,covg,X,opt);
B = eye(N)+K.mvm(eye(N))/exp(2*hyp.lik);
dB = deriv_cor(hyp,xg{:},K.Mx,1);
dB = dB(eye(N));

hyp1 = hyp;
hyp1.cov(1)=hyp1.cov(1)+1e-8;
K1 = apx(hyp1,covg,X,opt);
B1 = eye(N)+K1.mvm(eye(N))/exp(2*hyp1.lik);
dB_exact = (B1-B)*1e8;
fprintf('Finite difference relative error is %.6f.\n',norm(dB_exact-dB(:,1:N),'fro')/norm(dB_exact,'fro'));

hyp1 = hyp;
hyp1.cov(2)=hyp1.cov(2)+1e-8;
K1 = apx(hyp1,covg,X,opt);
B1 = eye(N)+K1.mvm(eye(N))/exp(2*hyp1.lik);
dB_exact = (B1-B)*1e8;
fprintf('Finite difference relative error is %.6f.\n',norm(dB_exact-dB(:,N+1:2*N),'fro')/norm(dB_exact,'fro'));

hyp1 = hyp;
hyp1.lik=hyp1.lik+1e-8;
K1 = apx(hyp1,covg,X,opt);
B1 = eye(N)+K1.mvm(eye(N))/exp(2*hyp1.lik);
dB_exact = (B1-B)*1e8;
fprintf('Finite difference relative error is %.6f.\n',norm(dB_exact+2/exp(2*hyp.lik)*K.mvm(eye(N)),'fro')/norm(dB_exact,'fro'));