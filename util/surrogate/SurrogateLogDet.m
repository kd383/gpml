classdef SurrogateLogDet
    %SurrogateLogDet Approximates the logdet with a surrogate model
    %
    % More information....
    %
    % David Eriksson, dme65@cornell.edu
    %

    properties(SetAccess = public)
        X % GP samples
        kernel % RBF kernel
        tail % RBF tail
        exp_des % Experimental design
        w % RBF weights
        lambda % Kernel weights
        c % Tail weights
        s % Surrogate model
        name = 'Surrogate'; % Name
    end
    methods
        function obj = SurrogateLogDet(X, fX, exp_des, kernel, tail)
            %SURROGATELOGDET Constructor
            %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %
            % Evaluate the logdet for the experimental design
            %
            % X       - The samples of the GP
            % fX      - Function values
            % exp_des - Points in the experimental design, each row is
            %           log10([ell, sigma])
            % kernel  - RBF kernel
            % tail    - Polynomial tail
            %
            % TODO:   - Add fifth argument for a fast log-det approximation
            %         - Add estimation of a2 as well
            %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            obj.X = X;
            obj.kernel = kernel;
            obj.tail = tail;
            obj.exp_des = exp_des;

            Phi = kernel.eval(exp_des, exp_des);
            P = tail.eval(exp_des);
            A = [Phi P; P' zeros(tail.dim_tail)];

            obj.w = A\[fX; zeros(tail.dim_tail, 1)];
            obj.lambda = obj.w(1:size(exp_des, 1));
            obj.c = obj.w(size(exp_des, 1) + 1:end);
            obj.s = @(y) obj.kernel.eval(y, obj.exp_des)*obj.lambda + ...
                obj.tail.eval(y)*obj.c;
        end

        function [val, g] = predict(obj, hyp)
            %PREDICT Predicts the value of the logdet for a given set of
            % hyper-parameters
            %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %
            % The input hyper-parameters are on a log10-scale
            % Returns the surrogate approximation and the approximated
            % gradient of the logdet
            %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            val = obj.s(hyp);

            if nargout > 1 % gradient required
                Jv = [obj.kernel.deval(hyp, obj.exp_des); ...
                    obj.tail.deval(hyp)];
                g = [obj.lambda; obj.c]' * Jv;
            end
        end
    end
end
