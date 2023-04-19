# fit-pRF
Fit population receptive field (pRF) parameters from retinotopic fMRI data using nonlinear least-squares solver

This code depends on having installed the Matlab Optimization Toolbox.

Feaures include
1. Fit pRF parameters using the compressive spatial summation (CSS) model (fitprf.m).
2. Simulate a collection of pRFs and synthesize their time series (simprf.m). This can be useful for validating the fitting routine.
3. Optimize the HRF on a per-voxel basis.
4. Utilize GPU for generation of seed grid. 

To get started, see example.m
