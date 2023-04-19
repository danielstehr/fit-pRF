function simdata = simprf(stimulus,nvox,TR,options)
% simdata = simprf(stimulus,nvox,TR,options)
% Simulates a collection of pRFs and synthesizes their time series
% <stimulus> is a cell array containing the masks (apertures) presented on
%       each run of the experiment. Each mask should contain values in the 
%       range of [0,1] and there should be exactly one mask per TR. The
%       format inside each cell is row x col x time.
% <nvox> is the number of voxels to simululate
% <TR> is the repetition time 
% <options> is a struct (not required)
%        <mu> is a vector of mean values for [eccentricity, angle, size,
%             gain, exponent] respectively
%             (defaults, [10 0 20 10 0.5])
%        <sigma> is a vector of standard deviations for [eccentricity,
%             angle, size, gain, exponent] respectively
%             (defaults: [5 0 5 1 1])
%        <corEccSz> is the desired correlation between eccentricities and
%             pRF size
%             (default: 0.5)
%        <angleRange>, range of pRF angles to sample from (in degrees)
%             (default: [0 15])
%        <hrf> the hemodynamic response function to use in generating model
%             (default: two gamma hrf used in spm)
%        <rejectSamp> specifies whether rejection sampling should be
%             performed to ensure pRFs are not too far away from stimulus
%             (default: true)
%        <plotflag>, plots pRFs position and size
%             (default: true)
%        <usegpu> create the model time series on gpu for speed
%             (default: false)
%        <noiseSigma>, the standard deviation of noise to add
%
% <outputs>
%        <data>, the simulated pRF timeseries (ntimepoints x nvox) in a
%             cell array, one cell per run of data
%        <model>, the noise-free model timeseries 
%        <truepars>, the ground truth parameter values used to generate
%             the data. The values are [x, y, size, gain, exponent]
%             respectively
%
% Daniel Stehr
% Daniel.A.Stehr@dartmouth.edu
% Department of Psychological and Brain Sciences, Dartmouth College

% for reproducibility
rng('default');

%% fill in missing inputs and init params
if ~exist('options','var') || isempty(options)
    options = struct();
end
if ~isfield(options,'usegpu') || isempty(options.usegpu)
    options.usegpu = false;
end
if ~exist('nvox','var') || isempty(nvox)
    nvox = 100;
end
if ~exist('TR','var') || isempty(TR)
    error('must specify the TR!');
end
if ~isfield(options,'hrf') || isempty(options.hrf)
    % default, canonical double gamma HRF
    options.hrf = dghrf(0.1,[6.68 14.66 1.82 3.15 3.08 0.1 48.9]);  
    options.hrf = conv(options.hrf,ones(1,TR/0.1));     % predict response to a 1 TR stimulus
    options.hrf = options.hrf(1:(TR/0.1):end);          % downsample
    options.hrf = options.hrf/max(options.hrf);                 % scale peak to 1
end
if ~isfield(options,'mu') || isempty(options.mu)
    options.mu = [10 0 20 10 0.5];
end
if ~isfield(options,'sigma') || isempty(options.sigma)
    options.sigma = [5 0 5 1 1];
end
if ~isfield(options,'corrEccSz') || isempty(options.corrEccSz)
    options.corrEccSz = 0.5;
end
if ~isfield(options,'rejectSamp') || isempty(options.rejectSamp)
    options.rejectSamp = true;
end
if ~isfield(options,'plotflag') || isempty(options.plotflag)
    options.plotflag = true;
end
if ~isfield(options,'angRange') || isempty(options.angRange)
    options.angRange = [0 15];
end
if ~isfield(options,'noiseSigma') || isempty(options.noiseSigma)
    options.noiseSigma = 10;
end

stimres = size(stimulus{1}(:,:,1));
resmax = max(stimres);
pars = NaN(nvox,5);
nruns = length(stimulus);
totalvols = sum(cellfun(@(x) size(x,3),stimulus));

simdata = struct;
simdata.stimulus = stimulus;

%% Sample the parameter values for simulation
options.sigma = options.sigma.^2;
options.covEccSz = options.corrEccSz*sqrt(options.sigma(1)*options.sigma(3));   % desired covariance between pRF eccentricity and size

% sample eccentricity and size parameters (with correlation, if wanted)
%   from a multivariate normal distribution
muvec = [options.mu(1) options.mu(3)];
covmat = [options.sigma(1) options.covEccSz; options.covEccSz options.sigma(3)];
mvpars = mvnrnd(muvec,covmat,nvox);
% Reject samples with RF center outside stimulus (optional)
if options.rejectSamp
    margin = 0.1*resmax;
    toosmall = mvpars(:,1) < 0;
    toolarge = mvpars(:,1) > (floor(resmax/2)-margin);
    mvpars = mvpars(~toosmall & ~toolarge,:);
    while size(mvpars,1) < nvox
        voxneeded = nvox - size(mvpars,1);
        mvpars = [mvpars; mvnrnd(muvec,covmat,voxneeded)];
        toosmall = mvpars(:,1) < 0;
        toolarge = mvpars(:,1) > (floor(resmax/2)-margin);
        mvpars = mvpars(~toosmall & ~toolarge,:);
    end
end
pars(:,1) = mvpars(:,1);        % eccentricity
pars(:,3) = abs(mvpars(:,2));        % sigma (size)

% sample angle parameter from a uniform distribution over specified range
options.angRange = options.angRange .* (pi/180);
pars(:,2) = (options.angRange(2)-options.angRange(1)).*rand(nvox,1) + options.angRange(1);

% sample gain parameter from truncated normal distribution (to ensure gain
% doesn't fall too low and zero out the time series)
fa = normcdf(1,options.mu(4),options.sigma(4));
fb = normcdf(100,options.mu(4),options.sigma(4));
u = unifrnd(fa,fb,[nvox 1]);
pars(:,4) = icdf('Normal',u,options.mu(4),options.sigma(4));

% Fix exponent
pars(:,5) = 0.5;

%% Model prep
% [~,xx,yy] = makegaussian2d(resmax,resmax/2,resmax/2,5,5);
[xx, yy] = meshgrid(linspace(-stimres(2)/2,stimres(2)/2,stimres(2)), ...
    linspace(-stimres(1)/2,stimres(1)/2,stimres(1)));
xx = xx(:);
yy = yy(:);

% convert angle and eccentricity measures to Cartesian coordinates
pars(:,6) = (1+stimres(1))/2 - pars(:,1).*(sin(pars(:,2)));     % y coord (row index)
pars(:,7) = (1+stimres(2))/2 + pars(:,1).*(cos(pars(:,2)));     % x coord (col index)

% plot and check (optional)
if options.plotflag
    plot(pars(:,7),pars(:,6),'+');
    for i = 1:size(pars,1)
        drawellipse(pars(i,7),pars(i,6),0,2*(pars(i,3)/sqrt(0.5)),2*pars(i,3)/sqrt(0.5));
    end
    axis equal;
    xlim([0 resmax]);ylim([0 resmax]);
    xline(resmax/2);
    yline(resmax/2);
end

%% Simulate the data
stimulus = cellfun(@(x) reshape(x,stimres(1)*stimres(2),[])',stimulus,'UniformOutput',false);
% add dummy column to indicate run breaks
for run = 1:length(stimulus)
    stimulus{run} = [stimulus{run} run*ones(size(stimulus{run},1),1)];
end
stimulus = cat(1,stimulus{:});
if nnz(stimulus)/numel(stimulus) < 0.15
    stimulus = sparse(stimulus);        % make sparse for performance
end

% generate the (noise-free) model time series
runidx = stimulus(:,prod(stimres)+1);
if options.usegpu && (gpuDeviceCount > 0)
    stimulus = gpuArray(stimulus);      % send to GPU, if available
end
pars0 = pars(:,[6 7 3 4 5]);
pars0(:,4) = pars0(:,4)*100;
pars0(:,5) = pars0(:,5)*100;
modelts = modelprf_seedts(pars0,stimulus,stimres,repmat(xx,1,nvox),repmat(yy,1,nvox),options.hrf',runidx);

% collect data, if its on the GPU
if options.usegpu && (gpuDeviceCount > 0)
    modelts = gather(modelts);
end

% add noise to time series
% 1. Normally distributed noise
rng('default');     % Once again for reproducibility, just to be safe!
noise = normrnd(0,options.noiseSigma,[totalvols nvox]);

% 2. Low freq drift
lowdrift = linspace(0,1,size(modelts,1))';
lowdrift = repmat(lowdrift,1,nvox);
tmp = (.01-.5).*rand(nvox,1) + .8;
tmp = repmat(tmp',size(modelts,1),1);
lowdrift = lowdrift .* tmp;

% add noise components to models
DC = 500;
model_noise = modelts + noise + lowdrift + DC;
% close all;
% plot(model_noise(:,1));


%% Collect and tidy-up results
simdata.data = cell(1,nruns);
simdata.model = cell(1,nruns);

% store run data
for run = 1:nruns
   runidx = find(stimulus(:,end)==run) ;
   simdata.data{run} = model_noise(runidx,:);
   simdata.model{run} = modelts(runidx,:);
end

% store ground truth parameters
simdata.truepars = [pars(:,6:7) pars(:,3:5)];

end

