function f = generate_data(X,opt)
    n = length(X);
    if strcmp(opt.type,'RBF') || strcmp(opt.type,'OU')
        K = apx(opt.hyp,opt.cov,X);
        K = K.mvm(eye(n));
        K = 0.5*(K + K') + exp(2*opt.hyp.lik)*eye(n);
        f = chol(K)'* randn(n,1);
    elseif strcmp(opt.type,'step')
        disc = X(randi(n,4,1));
        f1 = @(x)heaviside(x-disc(1));
        f2 = @(x)heaviside(x-disc(2));
        f3 = @(x)heaviside(x-disc(3));
        f4 = @(x)heaviside(x-disc(4));
        f = @(x) randn*f1(x)+randn*f2(x)+randn*f3(x)+randn*f4(x)+0.1*randn(length(x),1);
    elseif strcmp(opt.type,'expsin')
        f = @(x)sin(5*x).*exp(-x.^2/2) + 0.1*randn(length(x),1);
    end
end