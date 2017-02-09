classdef TpsKernel
    %TPSKERNEL Thin-plate spline RBF kernel
    properties
        dim % Number of dimensions
        m % Order of conditional positive definiteness
    end
    methods
        function obj = TpsKernel(dim)
            obj.m = 2;
            obj.dim = dim;
        end

        function out = eval(~, Y, X)
            %EVAL Computes the radial kernel at Y based on centers X
            D = pdist2(Y, X);
            out = D.^2 .* log(D + 1e-14);
        end

        function Jac = deval(~, y, X)
            %DEVAL Computes the Jacobian at y for the kernel
            %
            % The radial derivative of the TPS kernel is r * (2log(r) + 1)
            % There derivative of ||y-x_i|| is (y-x_i)/r
            % Each row of the Jacobian is therefore (2log(r) + 1) * (y-x_i)
            %
            assert(numel(y) == length(y));
            y = y(:)';
            Jac = bsxfun(@minus, y, X);
            r = pdist2(y, X);
            Jac = bsxfun(@times, Jac, 2*log(r' + 1e-14) + 1);
        end
    end
end
