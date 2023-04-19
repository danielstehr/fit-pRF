function m = polymat(n,maxdeg)
% m = polymat(n,maxdeg)
% Construct matrix of polynomial regressors
% <n> is the number of timepoints or samples in model
% <maxdeg> is the maximum degree of polynomial to include

deg = 0:maxdeg;
t = linspace(-1,1,n);
m = NaN(n,length(deg));

for d = 1:length(deg)
    m(:,d) = t.^deg(d);
end

end

