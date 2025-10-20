# DyCOMS-II RF02 Moist Case

This directory contains an implementation of the DyCOMS-II RF02 nocturnal
marine stratocumulus case for ERF. The configuration follows the LES
intercomparison specifications and includes:

* 6.4 km × 6.4 km × 1.5 km domain with 50 m horizontal spacing
* 96 stretched vertical levels (minimum 5 m near the surface/inversion)
* Warm-rain Kessler microphysics with a cloud droplet concentration of 55 cm⁻³
* Large-scale subsidence defined by a constant divergence of 3.76×10⁻⁶ s⁻¹
* Constant geostrophic wind (u = 5 m s⁻¹, v = −5.5 m s⁻¹)
* Monin–Obukhov surface layer over a 292 K sea surface with 98% relative humidity
* Case-specific longwave radiation based on Beer's-law attenuation
* Rayleigh damping above 1250 m with a 100 s e-folding time

## Running the case

```bash
# CMake
cd MyBuild
cmake .. -DERF_ENABLE_MPI=ON -DERF_DIM=3
cmake --build . --target erf_dycoms_rf02 -j
mpirun -n 4 ./Exec/MoistRegTests/DyCOMS_RF02/erf_dycoms_rf02                                  \
       ./Exec/MoistRegTests/DyCOMS_RF02/inputs_dycoms_rf02

# GNUmake
cd Exec/MoistRegTests/DyCOMS_RF02
make -j4 DIM=3
mpirun -n 4 ./erf_dycoms_rf02 inputs_dycoms_rf02
```

The `inputs_dycoms_rf02_flux` configuration prescribes the observed fluxes from
van Zanten & Stevens (2005): sensible heat flux of 16 W m⁻², latent heat flux of
93 W m⁻², and a fixed friction velocity of 0.25 m s⁻¹ (giving `tstar = -0.052682`
and `qstar = -1.2298×10⁻⁴`). Use it exactly as above, substituting the desired
inputs file on the run line.

Small random potential-temperature perturbations (±0.1 K below 200 m by
default) are applied at initialization to seed boundary-layer turbulence. The
amplitude and depth can be overridden with `prob.dycoms.theta_pert_amp` and
`prob.dycoms.theta_pert_height`.

Adjust the MPI configuration and build options as needed for your machine.
