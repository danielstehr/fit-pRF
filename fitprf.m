function results = fitprf(stimulus,data,TR,options)
% results = fitprf(stimulus,data,TR,options)
% Estimate population receptive fields (pRFs) from retinotopic data
%
% <stimulus> is a cell array containing the masks (apertures) presented on
%       each run of the experiment. Each mask should contain values in the 
%       range of [0,1] and there should be exactly one mask per TR. The
%       format inside each cell is row x col x time.
% <data> is a cell array providing the data from each run, organized as
%       ntimepoints x nvoxels. The first row must include the vertex id
%       for each voxel and will be removed before fitting the models.
% <TR> is the repetition time (in seconds).
% <options> is a struct (not required)
%       <useCachedSupergrid> path to cached model time series for seed fit
%       <optimizeHRF>, pick best fitting HRF from a library (default,
%       false)
%       <avgData> if true, average all runs together (default, false)
%       <quikfit> exit early and only return best fitting seed parameters

% Daniel Stehr
% Daniel.A.Stehr@dartmouth.edu
% Department of Psychological and Brain Sciences, Dartmouth College

%% Initialize parameters and settings
fprintf(1,'*** fitPRF: execution started %s. ***\n\n',datestr(now));

% print system info
sysinfo = cpuinfo();
fprintf(1,'[%i cores, %s, %.2f GB Memory, %s]\n\n',sysinfo.TotalCores,sysinfo.CPUName,sysinfo.TotalMemory/1073741824,sysinfo.OSType);

% fill in missing inputs
if ~exist('options','var') || isempty(options)
    options = struct();
end
if ~isfield(options,'maxpolydeg') || isempty(options.maxpolydeg)
    options.maxpolydeg = [];
end
if ~isfield(options,'quikfit') || isempty(options.quikfit)
    options.quikfit = false;
end
if ~isfield(options,'parstofit') || isempty(options.parstofit)
    options.parstofit = logical([1 1 1 1 1]);
end
if ~isfield(options,'resample') || isempty(options.resample)
    options.resample = 'none';
end
if ~isfield(options,'useCachedSupergrid') || isempty(options.useCachedSupergrid)
    options.useCachedSupergrid = false;
end
if ~isfield(options,'avgData') || isempty(options.avgData)
    options.avgData = false;
end
if ~isfield(options,'optimizeHRF') || isempty(options.optimizeHRF)
    options.optimizeHRF = false;
end
if ~isfield(options,'modelmode') || isempty(options.modelmode)
    options.modelmode = 1;
end
if ~isfield(options,'include_vertxid') || isempty(options.include_vertxid)
    options.include_vertxid = true;
end

% init results
results = struct;

% extract vertex identifier, if provided
if options.include_vertxid
    results.vertxid = data{1}(1,:)';
    data = cellfun(@(x) x(2:end,:),data,'UniformOutput',false);
end

% average data (option)
if options.avgData
    data = cat(3,data{:});
    data = {mean(data,3)};
    stimulus = stimulus(1);
    fprintf(1,'++ Averaging all runs in time\n\n');
end

% Constants
nvol = cellfun(@(x) size(x,1),data);
nruns = length(data);
fprintf(1,'++ %i runs detected\n',nruns);
stimres = size(stimulus{1},1:2);
fprintf(1,'++ %i x %i stimulus resolution\n',stimres(1),stimres(2));

% remove potential bad voxels that have all zeros
bad = find(all(data{1,1}==0));
for r=1:nruns
    data{:,r}(:,bad) = [];
end
if options.include_vertxid
    results.vertxid(bad) = [];
end

nvox = size(data{1},2);
fprintf(1,'++ %i vox/vertices in dataset\n',nvox);

% compute each voxel's mean
meanvol = cellfun(@(x) mean(x),data,'UniformOutput',false);   % mean for each run
meanvol = mean(cat(1,meanvol{:}));       % mean across all runs
results.meanvol = meanvol';
clear meanvol

% set options for non-linear solver
nlopts = optimoptions('lsqcurvefit',...
    'Display','off',...
    'Algorithm','levenberg-marquardt',...
    'MaxIterations',500,...
    'StepTolerance',1e-6,...
    'UseParallel',false);   % UseParallel actually slows things down here

% set boundary constraints
lb_fit1 = [-stimres(2) -stimres(1) 0 0];
ub_fit1 = [2*stimres(2) 2*stimres(2) Inf 10000];

lb_fit2 = [-stimres(2) -stimres(1) 0 0 0.1];
ub_fit2 = [2*stimres(2) 2*stimres(2) Inf 10000 150];

%% Prepare stimuli
% reshape stimuli from 2d matrices to column vectors
stimulus = cellfun(@(x) reshape(x,stimres(1)*stimres(2),[])',stimulus,'UniformOutput',false);
% add dummy column to indicate run number
for run = 1:length(stimulus)
    stimulus{run} = [stimulus{run} run*ones(size(stimulus{run},1),1)];
end

%% Prepare noise regressors
% Compute degree of maximum polynomial regressor
if isempty(options.maxpolydeg)
    options.maxpolydeg = cellfun(@(x) round(size(x,1)*TR/60/2),data);
end
fprintf(1,'++ using the following maximum polynomial degrees: %s\n\n',mat2str(options.maxpolydeg));
polyregs = arrayfun(@(i) polymat(nvol(i),options.maxpolydeg(i)),1:nruns,'UniformOutput',false);
polyregs_allruns = blkdiag(polyregs{:});    % concatenate and stagger by run
polyregT = projmat(polyregs_allruns);       % form projection matrix

%% Prepare model and helper functions
% cache coordinates for gaussians
[xx, yy] = meshgrid(linspace(-stimres(2)/2,stimres(2)/2,stimres(2)), ...
    linspace(-stimres(1)/2,stimres(1)/2,stimres(1)));
xx = xx(:);
yy = yy(:);

% Prepare model(s) of HRF
if ~options.optimizeHRF
    hrf = dghrf(0.1,[6.68 14.66 1.82 3.15 3.08 0.1 48.9]);                   % double gamma hrf model
    hrf = conv(hrf,ones(1,TR/0.1));     % predict response to a 1 TR stimulus
    hrf = hrf(1:(TR/0.1):end);          % downsample
    hrf = hrf/max(hrf);                 % scale peak to 1
else
    hrf = getcanonicalhrflibrary(TR,TR);  % 20 HRFs x N time points
end
nhrf = size(hrf,1);

% for computing coefficient of determination (R^2)
calcR2 = @(y,x) 1-((sum((y-x).^2))/(sum((y-mean(y)).^2)));

%% Choose best seed for each vox
% a large grid search is performed separately for each HRF in consideration
gridresults = struct('r',NaN(nvox,1),'seedidx',NaN(nvox,1),'seeds',NaN(nvox,5));
gridresults = repmat(gridresults,nhrf,1);

% Define parameters for grid search
if isfield(options,'seedgrid')
    if ~isfield(options.seedgrid,'eccs')
        options.seedgrid.eccs = [0 0.00551 0.014 0.0269 0.0459 0.0731 0.112 0.166 0.242 0.348 0.498 0.707 1];
    end
    if ~isfield(options.seedgrid,'angs')
        linspace_circ = @(x1,x2,n) linspace(x1,x2-((x2-x1)/n),n);
        options.seedgrid.angs = linspace_circ(0,2*pi,16);
    end
    if ~isfield(options.seedgrid,'n')
        options.seedgrid.n = [0.5 0.25 0.1250] *100;
    end
else
    options.seedgrid.eccs = [0 0.00551 0.014 0.0269 0.0459 0.0731 0.112 0.166 0.242 0.348 0.498 0.707 1];
    linspace_circ = @(x1,x2,n) linspace(x1,x2-((x2-x1)/n),n);
    options.seedgrid.angs = linspace_circ(0,2*pi,16);
    options.seedgrid.n = [0.5 0.25 0.1250];
end

% pRF size grid: 1 px, 2 px, 4 px, ... , up to max resolution
maxn = floor(log2(max(stimres)));
options.seedgrid.sigma = 2.^(0:maxn);

% pull out seed params to make code easier to read
eccs = options.seedgrid.eccs;
angs = options.seedgrid.angs;
n = options.seedgrid.n;
sigma = options.seedgrid.sigma;

% construct full list of seeds (seeds x params)
%       params are [row col sigma gain exponent]
allseeds = NaN(length(eccs)*length(angs)*length(n)*length(sigma),5);
counter = 1;
for r = 1:length(eccs)
    for theta = 1:length(angs)
        if ~(eccs(r)==0 && angs(theta)>0)       % Note, for radius of 0 all angles give same result
            for s = 1:length(sigma)
                for pow = 1:length(n)
                    allseeds(counter,:) = [(1+stimres(1))/2 - sin(angs(theta)) * (eccs(r)*max(stimres)) ...
                        (1+stimres(2))/2 + cos(angs(theta)) * (eccs(r)*max(stimres)) ...
                        sigma(s)*sqrt(n(pow)) 1 n(pow)];
                    counter = counter + 1;
                end
            end
        end
    end
end
% Remove any NaNs (due to seeds that were skipped above)
allseeds = allseeds(~any(isnan(allseeds),2),:);
nseeds = size(allseeds,1);

% scale gain and exponent parameters,
%   because we want to have a similar range in all parameters during
%   optimization
allseeds(:,4) = allseeds(:,4) * 100;
allseeds(:,5) = allseeds(:,5) * 100;

if ~options.useCachedSupergrid
    % generate predicted time series for each seed (time x seeds x hrf)
    seedstim = cat(1,stimulus{:});
    runidx = seedstim(:,prod(stimres)+1);
    if nnz(seedstim)/numel(seedstim) < 0.15
        seedstim = sparse(seedstim);        % make sparse for performance
    end
    if gpuDeviceCount > 0
        seedstim = gpuArray(seedstim);      % send to GPU, if available
    end
    
    predts = cell(1,nhrf);
    for h = 1:nhrf
        predts{h} = NaN(sum(cellfun(@(x) size(x,1),stimulus)),size(allseeds,1));
        fprintf(1,'generating super-grid time-series . . . . . . . . . .');
        tic;
        % generate time series for each seed
        predts{h} = modelprf_seedts(allseeds,seedstim,stimres,repmat(xx,1,nseeds),repmat(yy,1,nseeds),hrf(h,:)',runidx);
        % collect data, if its on the GPU
        if gpuDeviceCount > 0
            predts{h} = gather(predts{h});
        end
        fprintf('done.'); toc; fprintf(1,'\n');
    end
    % clear masks;
else
    % Alternatively, if provided load pre-generated seed time series
    fprintf(1,'<strong>WARNING!!</strong> Using cached supergrid. Please, please, PLEASE make sure you have specified the right input.\n\n');
    load(options.useCachedSupergrid,'predts')
end

for h = 1:nhrf
    if options.optimizeHRF == false
        fprintf(1,'Finding best fitting seed for each vox . . . . . . .');
    else
        fprintf(1,'Finding best fitting seed for each vox, using hrf %i . . . . . . .',h);
    end
    % project out polynomial regressors
    predts{h} = polyregT*predts{h};
    datats = polyregT*catcell(1,data);
    
    % scale columns to have unit length
    predts_len = sqrt(dot(predts{h},predts{h},1));
    predts{h} = predts{h}./predts_len;
    
    datats_len = (sqrt(dot(datats,datats,1)));
    datats = datats./datats_len;
    
    % compute correlation and find maximum for each voxel
    chunksz = 10000;            % first, break data into smaller chunks
    chunks = diff([0:chunksz:(nvox-1),nvox]);
    datats = mat2cell(datats,size(datats,1),chunks);
    voxmodcor = cellfun(@(x) x'*predts{h},datats,'UniformOutput',false);
    [maxr,bestseedidx] = cellfun(@(x) max(x,[],2),voxmodcor,'UniformOutput',false);
    maxr = cat(1,maxr{:});
    bestseedidx = cat(1,bestseedidx{:});
    
    % estimate gain parameter (equivalent to best fitting beta weight)
    seedgain = maxr.*(datats_len'./predts_len(bestseedidx)');
    seedgain = seedgain * 100;          % scale to have similar range to other params
    
    % collect output
    gridresults(h).r = maxr;                            % maximum correlation found
    gridresults(h).seedidx = bestseedidx;               % index of best fitting seed
    gridresults(h).seeds = allseeds(bestseedidx,:);     % parameters of best fitting seed
    gridresults(h).seeds(:,4) = seedgain;               % estimated gain
    
    clear maxr bestseedidx
    fprintf(1,'done\n');
end

% find best fitting hrf for each voxel
[results.seedr,results.hrfpick] = max(cat(2,gridresults(:).r),[],2);
hrfpick = results.hrfpick;

% Exit early and return supergrid if no params were asked to be optimized
if options.quikfit
    return
end

%% Perform the fitting
% deal with resampling mode
switch options.resample
    case 'none'
        trainidx = 1:length(data);
        testidx = [];
        nresamp = size(trainidx,1);
    case 'LOO'
        trainidx = NaN(size(data,2),size(data,2)-1);
        testidx = NaN(size(data,2),1);
        for rr = 1:size(data,2)
            trainidx(rr,:) = setdiff(1:size(data,2),rr);
            testidx(rr,1) = rr;
        end
        nresamp = size(trainidx,1);
end

% loop over resampling cases
for resamp = 1:nresamp
    % resample data
    traindata = cat(1,data{trainidx(resamp,:)});
    if ~isempty(testidx)
        testdata = cat(1,data{testidx(resamp,:)});
    end
    
    trainstim = cat(1,stimulus{trainidx(resamp,:)});
    if ~isempty(testidx)
        teststim = cat(1,stimulus{testidx(resamp,:)});
    end
    
    trainT = projmat(blkdiag(polyregs{trainidx(resamp,:)}));
    if ~isempty(testidx)
        testT = projmat(blkdiag(polyregs{testidx(resamp,:)}));
    end
    
    % Convert to sparse matrix for efficiency
    if nnz(trainstim)/numel(trainstim) < .20
        trainstim = sparse(trainstim);
    end
    if nnz(trainT)/numel(trainT) < .20
        trainT = sparse(trainT);
    end
    
    % slice variables for parfor loop
    hrfpick_ts = hrf(hrfpick,:);
    returnseed = @(v) gridresults(hrfpick(v)).seeds(v,:);
    seedpick = arrayfun(returnseed,1:nvox,'UniformOutput',false);
    seedpick = cat(1,seedpick{:});
    
    % pre-allocate struct for storing results
    results0 = struct('ang',NaN,'ecc',NaN,'expt',NaN,'rfsize',NaN,'R2',NaN,'gain',NaN,'resnorm',NaN,'params',NaN(1,5),'seed',NaN(1,5));
    results0 = repmat(results0,nvox,1);
    
    clear predts gridresults
    
    % loop over voxels
    startloop = tic;
    parfor vox = 1:nvox
        
        % remove noise regressors from data
        traindataT = trainT*traindata(:,vox);
        
        % normalize by standard deviation
        dataSD = std(traindataT);
        if dataSD == 0
            dataSD = 1;
        end
        traindataT = traindataT / dataSD;
        
        % STEP 1: Optimize the first four parameters
        % select seed for this voxel
        initseed = seedpick(vox,:);
        x0 = seedpick(vox,:);
        nOverride = x0(5);
        x0(5) = [];
        
        % define model
        modelfunc = @(x,y) modelprf(x,trainstim,stimres,xx,yy,hrfpick_ts(vox,:)',trainT,dataSD,nOverride);
        
        % find minimum
        [pars0,resnorm,residual,exitflag,output] = ...
            lsqcurvefit(modelfunc,x0,[],traindataT,lb_fit1,ub_fit1,nlopts);
        
        % STEP 2: Optimize ALL parameters
        x0 = [pars0 nOverride];
        
        % define model
        modelfunc = @(x,y) modelprf(x,trainstim,stimres,xx,yy,hrfpick_ts(vox,:)',trainT,dataSD,[]);
        
        % find minimum
        [pars0,resnorm,residual,exitflag,output] = ...
            lsqcurvefit(modelfunc,x0,[],traindataT,lb_fit2,ub_fit2,nlopts);
        
        bestmodel = modelprf(pars0,trainstim,stimres,xx,yy,hrfpick_ts(vox,:)',trainT,dataSD,[]);
        bestR2 = 100*calcR2(traindataT,bestmodel);
        
        pars0(5) = pars0(5)/100;
        pars0(4) = pars0(4)/100;
        initseed(4) = initseed(4)/100;
        initseed(5) = initseed(5)/100;
        
        % store output
        results0(vox).ang = mod(atan2((1+stimres(1))/2 - pars0(1),pars0(2) - (1+stimres(2))/2),2*pi)*(180/pi);
        results0(vox).ecc = sqrt(((1+stimres(1))/2 - pars0(1))^2 + (pars0(2) - (1+stimres(2))/2)^2);
        results0(vox).expt = pars0(5);
        results0(vox).rfsize = pars0(3)/sqrt(posrect(pars0(5)));
        results0(vox).R2 = bestR2;
        results0(vox).gain = pars0(4);
        results0(vox).resnorm = resnorm;
        results0(vox).params = pars0;
        results0(vox).seed = initseed;
        % results.jacobirank(vox,:) = rank(quikJacobi(pars0,bestmodel,modelfunc));
        
        % report
        if bestR2 > 5
            fprintf(1,'<strong>Resample %i/%i\tvox %i/%i\tx=%.3f,\t y=%.3f,\t sigma=%.3f,\t gain=%.3f,\t n=%.3f,\t hrf=%i,\t R2=%.1f</strong>\n',resamp,nresamp,vox,nvox,pars0,hrfpick(vox),bestR2);
        else
            fprintf(1,'Resample %i/%i\tvox %i/%i\tx=%.3f,\t y=%.3f,\t sigma=%.3f,\t gain=%.3f,\t n=%.3f,\t hrf=%i,\t R2=%.1f\n',resamp,nresamp,vox,nvox,pars0,hrfpick(vox),bestR2);
        end
    end
    elapsedfit = toc(startloop);
    
end

%% Tidy up results
results.elapsedfit = elapsedfit;
results.ang = cat(1,results0(:).ang);
results.ecc = cat(1,results0(:).ecc);
results.expt = cat(1,results0(:).expt);
results.rfsize = cat(1,results0(:).rfsize);
results.R2 = cat(1,results0(:).R2);
results.gain = cat(1,results0(:).gain);
results.resnorms = cat(1,results0(:).resnorm);
results.params = cat(1,results0(:).params);
results.seed = cat(1,results0(:).seed);
results.hrfpick = hrfpick;

fprintf(1,'Execution finished successfully %s. ***\n',datestr(now));

end

