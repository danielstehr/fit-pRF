
%% Prep
clear;
close all;

% load stimulus masks
load('sample-stimulus.mat','stimulus');
stimulus = {stimulus stimulus stimulus stimulus stimulus};  % replicate for 5 runs
stimres = size(stimulus{1}(:,:,1));         % get dimensions of stimulus
resmax = max(stimres);

%% Simulate pRF time series and estimate the parameters
simdata = simprf(stimulus,20,1,struct('mu',[10 0 20 10 0.5],'plotflag',false));

% First, fit pRFs on simulated data without noise added
results_noisefree = fitprf(stimulus,simdata.model,1,struct('avgData',false,'include_vertxid',false));

% Plot ground truth pRFs
subplot(1,2,1);
plot(simdata.truepars(:,2),simdata.truepars(:,1),'b+');
for i = 1:size(simdata.truepars,1)
   drawprf(simdata.truepars(i,2),simdata.truepars(i,1),simdata.truepars(i,3),simdata.truepars(i,5),'b-'); 
end

% Plot estimated pRFs
hold on;
plot(results_noisefree.params(:,2),results_noisefree.params(:,1),'g+');
for i = 1:size(results_noisefree.params,1)
    drawprf(results_noisefree.params(i,2),results_noisefree.params(i,1),results_noisefree.params(i,3),results_noisefree.params(i,5),'g-');
end
axis equal;
xlim([0 resmax]);ylim([0 resmax]);
set(gca, 'YDir','reverse')
xline(resmax/2);
yline(resmax/2);
h = zeros(2,1);
hold on; h(1) = plot(NaN,NaN,'b-');
hold on; h(2) = plot(NaN,NaN,'g-');
legend(h,'ground truth','fitted pRF');
title('Simulation | Noise Free');


% Now, fit pRFs on simulated data WITH added noise
results_noiseAdded = fitprf(stimulus,simdata.data,1,struct('avgData',false,'include_vertxid',false));

% Plot ground truth pRFs
subplot(1,2,2);
plot(simdata.truepars(:,2),simdata.truepars(:,1),'b+');
for i = 1:size(simdata.truepars,1)
   drawprf(simdata.truepars(i,2),simdata.truepars(i,1),simdata.truepars(i,3),simdata.truepars(i,5),'b-'); 
end

% Plot estimated pRFs
hold on;
plot(results_noiseAdded.params(:,2),results_noiseAdded.params(:,1),'g+');
for i = 1:size(results_noiseAdded.params,1)
    drawprf(results_noiseAdded.params(i,2),results_noiseAdded.params(i,1),results_noiseAdded.params(i,3),results_noiseAdded.params(i,5),'g-');
end
axis equal;
xlim([0 resmax]);ylim([0 resmax]);
set(gca, 'YDir','reverse')
xline(resmax/2);
yline(resmax/2);
h = zeros(2,1);
hold on; h(1) = plot(NaN,NaN,'b-');
hold on; h(2) = plot(NaN,NaN,'g-');
legend(h,'ground truth','fitted pRF');
title('Simulation | Noise Added');


%% Load sample real fMRI retinotopy data and estimate pRF parameters
clear;
clc;

load('sample-data.mat','data');
% load stimulus masks
load('sample-stimulus.mat','stimulus');
stimulus = {stimulus stimulus stimulus stimulus stimulus};  % replicate for 5 runs
stimres = size(stimulus{1}(:,:,1));         % get dimensions of stimulus
resmax = max(stimres);

% perform the fitting
results_human = fitprf(stimulus,data,1,struct('avgData',true));


% threshold data based on R squared
keep = find(results_human.R2 >= 20);
par = results_human.params(keep,:);

% plot pRFs
figure;
plot(par(:,2),par(:,1),'b+');
for i = 1:size(par,1)
    drawprf(par(i,2),par(i,1),par(i,3),par(i,5),'r-');
end
axis equal;
xlim([0 resmax]);ylim([0 resmax]);
set(gca, 'YDir','reverse')
xline(resmax/2);
yline(resmax/2);
title('Human fMRI Data');
