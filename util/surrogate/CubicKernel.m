classdef CubicKernel
    %CUBICKERNEL Cubic RBF kernel
    properties
        dim % Number of dimensions
        m % Order of conditional positive definiteness
    end
    methods
        function obj = CubicKernel(dim)
            %CUBICKERNEL Constructor
            obj.m = 2;
            obj.dim = dim;
        end

        function out = eval(~, Y, X)
            %EVAL Computes the radial kernel at Y based on centers X
            out = pdist2(Y, X).^3;
        end

        function Jac = deval(~, y, X)
            % DEVAL Computes the Jacobian at y for the kernel
            %
            % The radial derivative of the cubic kernel is 3r^2
            % There derivative of ||y-x_i|| is (y-x_i)/r
            % Each row of the Jacobian is therefore 3r(y-x_i)
            %
            assert(numel(y) == length(y));
            y = y(:)';
            Jac = bsxfun(@minus, y, X);
            r = pdist2(y, X);
            Jac = 3 * bsxfun(@times, Jac, r');
        end
    end
end
