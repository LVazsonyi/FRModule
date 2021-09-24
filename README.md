Cosmosis modules to compute the f(R) power spectrum into the nonlinear regime.

This code requires gpflow

We have needed to slightly modified the code presented in: https://arxiv.org/abs/2010.00596 to be compataible with Cosmosis and MPI

The module load_fR_z.py indicates which z values to compute the ratio of the LCDM and f(R) power spectrum. The  module fR.py computes the f(R) nonlinear power spectrum, interpolating between these values.  

Some applications may also require the user to interpolate the power spectrum outside the current k-range using the Cosmosis extrapolate_power module or changing the k-range in halofit and camb. 
