# High resolution neural connectivity from incomplete tracing data using nonnegative spline regression

Authors: Kameron Decker Harris (kamdh@uw.edu), 
Stefan Mihalas (stefanm@alleninstitute.org),
Eric Shea-Brown (etsb@uw.edu).

Published: NIPS, 2016

The paper is available here as well as at [https://arxiv.org/abs/1605.08031].

### Code

The majority of code is split into separate repositories:

* [allen-voxel-network](https://github.com/kharris/allen-voxel-network) - utilities for setting up voxel matrices by pulling from allensdk
* [spatial-network-regression](https://github.com/kharris/spatial-network-regression) - solves (P1) using L-BFGS-B

Furthermore, we provide here the MATLAB code used to solve (P2), 
the low-rank version, using projected gradient descent:

* proj_grad_low_rank.m

### Supplemental videos of the inferred AIBS mouse brain connectivity.

Projections from a source voxel in VISp, depicted in blue, to the rest of the 
visual areas. The main discrepancy between the full and low rank solutions is 
confined to the medial-posterior area of VISp. There, the low rank solution 
undershoots the full rank solution in proximal projections, and overshoots it 
with distal projections.

* movie_full.mp4 - solution of (P1), lambda=10^5
* movie_low_rank.mp4 - solution of (P2), lambda=10^5, r=120
* movie_res.mp4 - residual (W_lowrank - W_full) plotted in the same way
* movie_full_retrograde.mp4 - solution of (P1), lambda=10^5, 
visualized retrograde with W^T