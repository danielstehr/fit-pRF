function T = projmat(X)
% T = projmat(X)
% Construct a matrix, T, that when multiplied with a 
% dataset, y (samples x parameters), it will subtract out
% the fit of X (also samples x parameters). 
% This is implemented as:
% y-X*inv(X'*X)*X'*y = (I-X*inv(X'*X)*X')*y = T*y

% First, unit-length normalize each col of X
X = X./sqrt(dot(X,X,1));

% Compute T
T = eye(size(X,1)) - X*((X'*X)\X');

end

