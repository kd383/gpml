classdef LinearTail
    %LINEARTAIL Linear polynomial tail
    properties
        dim; % Number of dimensions
        degree = 1; % Polynomial degree
        dim_tail; % Dimensionality of the polynomial space
    end

    methods
        function obj = LinearTail(dim)
            %LINEARTAIL Constructor
            obj.dim = dim;
            obj.dim_tail = dim + 1;
        end
        function out = eval(~, Y)
            %EVAL Evaluate the polynomial for a set of points Y
            out = [ones(size(Y, 1), 1), Y];
        end
        function Jac = deval(obj, y)
            %DEVAL Computes the Jacobian at a point y
            assert(numel(y) == length(y));
            Jac = [zeros(1, obj.dim); eye(obj.dim)];
        end
    end
end
