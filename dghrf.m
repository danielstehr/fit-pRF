function hrf = dghrf(TR,par)
% hrf = dghrf(TR,par)
% 
% Generate a canonical haemodynamic response function (hrf) 
% modeled as a linear combination of two Gamma functions.
% 
%   <TR>            Time resolution (in seconds). 
%                       default = 2 secs
%   <par(1)>        Time-to-peak of positive response (alpha1)
%                       default = 6
%   <par(2)>        Time-to-peak of negative response (alpha2)
%                       default = 16
%   <par(3)>        Dispersion of positive response (beta1)
%                       default = 1
%   <par(4)>        Dispersion of negative response (beta2)
%                       default = 1
%   <par(5)>        Ratio of negative to positive amplitude
%                       default = 6
%   <par(6)>        Onset of response
%                       default = 0
%   <par(7)>        Length of kernel (seconds)
%                       default = 32

if nargin < 2
    a1 = 6;
    a2 = 16;
    b1 = 1;
    b2 = 1;
    c = 6;
    t0 = 0;
    kernelLen = 32;
else
    a1 = par(1);
    a2 = par(2);
    b1 = par(3);
    b2 = par(4);
    c = par(5);
    t0 = par(6);
    kernelLen = par(7);
end


% Define the time axis
dt = TR/16;
t = 0:dt:kernelLen;
u = 0:length(t)-1;
u = u - (t0/dt);
u(u<0) = 0;     %positively rectify

% Define PDF of gamma distribution
gpdf = @(t,alpha,beta) (t.^(alpha-1).*beta^alpha.*exp(-beta*t))/gamma(alpha);

% Compute HRF as difference of two gammas
g1 = gpdf(u,a1/b1,dt/b1);
g2 = gpdf(u,a2/b2,dt/b2); g2(isnan(g2)) = 0;
hrf = g1 - g2/c;

hrf = hrf(1:16:length(u));
hrf = hrf/sum(hrf);

end

