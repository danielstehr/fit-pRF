function model = modelprf_seedts(x,dd,res,xx,yy,hrf,runidx)
% function model = modelPRF(x,dd,res,xx,yy,hrf,trainT,dataSD,pow)
% <x> contains input parameters x-coord, y-coord, sigma, gain, 
%   and exponent (last argument is optional)
% <dd> is the stimulus masks (in vector form) for each timepoint as
%   ntimepoints x index
% <res> is the resolution size of the masks

% Build 2d gaussian (step-by-step)
% r = (x(1) - ((res(1)+1)/2));
% c = (x(2) - ((res(2)+1)/2));
% gauss2d = exp(-((xx-c).^2 + (yy-r).^2)/2/x(3)^2);
% gauss2d = gauss2d / (2*pi*abs(x(3))^2);     %normalize area to one

% Build 2d gaussians (n_timepoints x n_gaussians)
gauss2d = exp(-((xx - (x(:,2) - ((res(2)+1)/2))').^2 + (yy - (x(:,1) - ((res(1)+1)/2))').^2)/2./(x(:,3)'.^2)) ./ (2*pi*abs(x(:,3)').^2);
gauss2d(end+1,:) = 0;

% compute stimulus-related time series, g*(s(t)*G)^n
%   result is (n_timepoints x n_vox)
stimts = (dd*gauss2d).^(x(:,5)'/100).*(x(:,4)'/100);
    
% convolve with hrf, separately by run
hrflen = length(hrf);
% runidx = dd(:,prod(res)+1);
model = zeros(size(stimts),class(stimts));
% for each run
for p=1:max(runidx)
  temp = conv2(stimts(runidx==p,:),hrf);
  model(runidx==p,:) = temp(1:end-hrflen+1,:);
end


end

