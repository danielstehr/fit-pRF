function model = modelprf(x,dd,res,xx,yy,hrf,trainT,dataSD,nOverride)
% model = modelprf(x,dd,res,xx,yy,hrf,trainT,dataSD,nOverride)
% Geneate the model time series for a set of pRF parameters and masks
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

% Build 2d gaussian (all in one step instead)
gaussflat = (1/(2*pi*x(3)^2))*(exp(-((xx-(x(2) - ((res(2)+1)/2))).^2 + (yy-(x(1) - ((res(1)+1)/2))).^2)/2/x(3)^2));
gaussflat(end+1) = 0;     % for book-keeping of run index stored in stimulus masks

% compute stimulus-related time series, g*(s(t)*G)^n
if ~exist('nOverride','var') || isempty(nOverride)
    stimts = (x(4)/100)*(dd*gaussflat).^(x(5)/100);
else
    stimts = (x(4)/100)*(dd*gaussflat).^(nOverride/100);
end
    
% convolve with hrf, separately by run
hrflen = length(hrf);
runidx = dd(:,prod(res)+1);
model = zeros(size(stimts),class(stimts));
for p=1:max(runidx)      % do once for each run
  temp = conv2(stimts(runidx==p,:),hrf);
  model(runidx==p,:) = temp(1:end-hrflen+1,:);
end

% Alternatively, we could convolce in Fourier domain (not really any faster)
% hrflen = length(hrf);
% runidx = dd(:,prod(res)+1);
% model = zeros(size(stimts),class(stimts));
% % loop over runs
% for p=1:max(runidx)
%     n = length(stimts(runidx==p,:)) + length(hrf) -1;
%     temp = ifft(fft(stimts(runidx==p,:),n).*fft(hrf,n));
%     model(runidx==p,:) = temp(1:end-hrflen+1,:);
% end

% regress out nuisance vars and normalize by sd of observed data
model = (trainT*model)/dataSD;

end

