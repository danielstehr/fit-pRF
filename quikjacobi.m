function jac = quikjacobi(x,y,varargin)
% jac = quikjacobi(x,y,varargin)
% Estimate jacobian matrix by finite differences

np = length(x);
nd = length(y);

jac = zeros(nd,np);
epsilon = sqrt(eps);

% iterate over params to estimate cols of Jacobian by finite differences
for i=1:np
    x_offset = x;
    x_offset(i) = x_offset(i) + epsilon;
    
    %get model predictions for offset parameter vector
    f_offset = feval(varargin{1},x_offset,varargin{2:end});
    jac(:,i) = (f_offset - y)/epsilon;
    
end

