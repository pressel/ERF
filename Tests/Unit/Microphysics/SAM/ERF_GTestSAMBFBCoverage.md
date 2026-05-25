# SAM BFB Coverage Ledger

The SAM BFB tests drive the frozen current-behavior reference path and the
production SAM path through the same public flow:

- `Init`
- `Copy_State_to_Micro`
- `Advance`
- `Copy_Micro_to_State`

Every case still checks exact equality for:

- conserved `rho theta`, `qv`, `qc`, `qi`, `qr`, `qs`, `qg`
- `rain_accum`, `snow_accum`, `graup_accum`

The conserved-state comparison now includes the grown region for `rho theta`
and `qv` through `qg`. The `qmoist` accumulators remain valid-cell-only because
this harness does not initialize or fill their ghost cells.

## Cases

- `ShocNoPrecipNoIcePublicFlowExact`
- `NoIceRainMatrixPublicFlowExact`
- `FullSAMCloudMatrixPublicFlowExact`
- `FullSAMPrecipMatrixPublicFlowExact`
- `FullSAMSedimentationDetJColumnPublicFlowExact`

## Executable Branch Hits

- `cloud_noop`: `ShocNoPrecipNoIcePublicFlowExact`, `FullSAMSedimentationDetJColumnPublicFlowExact`
- `cloud_t_bgmin_or_less`: `FullSAMCloudMatrixPublicFlowExact`
- `cloud_t_between_bgmin_bgmax`: `FullSAMCloudMatrixPublicFlowExact`
- `cloud_t_bgmax_or_greater`: `FullSAMCloudMatrixPublicFlowExact`
- `cloud_qt_gt_qsat`: `FullSAMCloudMatrixPublicFlowExact`
- `cloud_qt_le_qsat`: `FullSAMCloudMatrixPublicFlowExact`
- `cloud_evap_all_then_recheck`: `FullSAMCloudMatrixPublicFlowExact`
- `cloud_newton`: `FullSAMCloudMatrixPublicFlowExact`
- `cloud_condensate_limiter`: `FullSAMCloudMatrixPublicFlowExact`
- `icefall_noop`: `ShocNoPrecipNoIcePublicFlowExact`, `NoIceRainMatrixPublicFlowExact`
- `icefall_bottom_face`, `icefall_interior_face`, `icefall_top_face`: `FullSAMSedimentationDetJColumnPublicFlowExact`
- `icefall_detj_present`: `FullSAMSedimentationDetJColumnPublicFlowExact`
- `precip_noop`: `ShocNoPrecipNoIcePublicFlowExact`
- `precip_cloud_only`: `FullSAMPrecipMatrixPublicFlowExact`
- `precip_only`, `precip_cloud_and_precip`: `NoIceRainMatrixPublicFlowExact`, `FullSAMPrecipMatrixPublicFlowExact`
- `precip_water_autoconversion_off`: `NoIceRainMatrixPublicFlowExact`, `FullSAMPrecipMatrixPublicFlowExact`
- `precip_water_autoconversion_on`: `FullSAMPrecipMatrixPublicFlowExact`
- `precip_ice_autoconversion_off`, `precip_ice_autoconversion_on`: `FullSAMPrecipMatrixPublicFlowExact`
- `precip_rain_accretion_off`: `FullSAMPrecipMatrixPublicFlowExact`
- `precip_rain_accretion_on`: `NoIceRainMatrixPublicFlowExact`, `FullSAMPrecipMatrixPublicFlowExact`
- `precip_snow_accretion_off`, `precip_snow_accretion_on`: `FullSAMPrecipMatrixPublicFlowExact`
- `precip_graupel_accretion_off`, `precip_graupel_accretion_on`: `FullSAMPrecipMatrixPublicFlowExact`
- `precip_sink_limited_cloud_water`, `precip_sink_limited_cloud_ice`: `FullSAMPrecipMatrixPublicFlowExact`
- `precip_evaporation_off`, `precip_evaporation_on`: `NoIceRainMatrixPublicFlowExact`, `FullSAMPrecipMatrixPublicFlowExact`
- `precipfall_noop`: `ShocNoPrecipNoIcePublicFlowExact`
- `precipfall_bottom_face`, `precipfall_interior_face`, `precipfall_top_face`: `NoIceRainMatrixPublicFlowExact`, `FullSAMSedimentationDetJColumnPublicFlowExact`
- `precipfall_detj_present`: `FullSAMSedimentationDetJColumnPublicFlowExact`
- `precipfall_zero_flux_below_threshold`: `NoIceRainMatrixPublicFlowExact`
- `precipfall_nonzero_rain_flux`: `NoIceRainMatrixPublicFlowExact`, `FullSAMSedimentationDetJColumnPublicFlowExact`
- `precipfall_nonzero_snow_flux`, `precipfall_nonzero_graupel_flux`: `FullSAMSedimentationDetJColumnPublicFlowExact`
- `precipfall_surface_rain_accum`: `NoIceRainMatrixPublicFlowExact`, `FullSAMSedimentationDetJColumnPublicFlowExact`
- `precipfall_surface_snow_accum`, `precipfall_surface_graupel_accum`: `FullSAMSedimentationDetJColumnPublicFlowExact`
- `precipfall_nonnegative_clipping`: `FullSAMSedimentationDetJColumnPublicFlowExact`

## Documented But Not Yet Asserted

- `precip_evaporation_species_limited`: the current frozen matrices do not reliably drive the final `std::min` species clamp after Cloud and Precip coupling, so this remains documented but not asserted as a nonzero hit.

## Configuration-Only Coverage

These items are controlled directly by the test inputs rather than by runtime
branch-hit counters:

- `moisture_type = SAM_NoPrecip_NoIce`: `ShocNoPrecipNoIcePublicFlowExact`
- `moisture_type = SAM_NoIce`: `NoIceRainMatrixPublicFlowExact`
- `moisture_type = SAM`: `FullSAMCloudMatrixPublicFlowExact`, `FullSAMPrecipMatrixPublicFlowExact`, `FullSAMSedimentationDetJColumnPublicFlowExact`
- `use_shoc = true`: `ShocNoPrecipNoIcePublicFlowExact`, `FullSAMSedimentationDetJColumnPublicFlowExact`
- `use_shoc = false`: `NoIceRainMatrixPublicFlowExact`, `FullSAMCloudMatrixPublicFlowExact`, `FullSAMPrecipMatrixPublicFlowExact`
- `detJ absent`: `ShocNoPrecipNoIcePublicFlowExact`, `NoIceRainMatrixPublicFlowExact`, `FullSAMCloudMatrixPublicFlowExact`, `FullSAMPrecipMatrixPublicFlowExact`
- `detJ present`: `FullSAMSedimentationDetJColumnPublicFlowExact`
- `real_width = 0`: `ShocNoPrecipNoIcePublicFlowExact`, `FullSAMCloudMatrixPublicFlowExact`, `FullSAMPrecipMatrixPublicFlowExact`, `FullSAMSedimentationDetJColumnPublicFlowExact`
- `real_width > 0`: `NoIceRainMatrixPublicFlowExact`
- `one-column vertical case`: `FullSAMSedimentationDetJColumnPublicFlowExact`
- `multi-cell horizontal case`: `ShocNoPrecipNoIcePublicFlowExact`, `NoIceRainMatrixPublicFlowExact`, `FullSAMCloudMatrixPublicFlowExact`, `FullSAMPrecipMatrixPublicFlowExact`