# SAM BFB Coverage Ledger

This ledger documents the exact-comparison GTest cases that drive the frozen
current-behavior SAM reference path and the production SAM path through the
same public flow:

- `Init`
- `Copy_State_to_Micro`
- `Advance`
- `Copy_Micro_to_State`

Each listed case checks exact equality for:

- conserved `rho theta`
- conserved `qv`, `qc`, `qi`, `qr`, `qs`, `qg`
- `rain_accum`
- `snow_accum`
- `graup_accum`

## Cases

- `ShocNoPrecipNoIcePublicFlowExact`
- `NoIceRainMatrixPublicFlowExact`
- `FullSAMCloudMatrixPublicFlowExact`
- `FullSAMPrecipMatrixPublicFlowExact`
- `FullSAMSedimentationDetJColumnPublicFlowExact`

## Branch And Option Checklist

- `SAM_NoPrecip_NoIce`: `ShocNoPrecipNoIcePublicFlowExact`
- `SAM_NoIce`: `NoIceRainMatrixPublicFlowExact`
- `SAM`: `FullSAMCloudMatrixPublicFlowExact`, `FullSAMPrecipMatrixPublicFlowExact`, `FullSAMSedimentationDetJColumnPublicFlowExact`
- `use_shoc = true`: `ShocNoPrecipNoIcePublicFlowExact`, `FullSAMSedimentationDetJColumnPublicFlowExact`
- `use_shoc = false`: `NoIceRainMatrixPublicFlowExact`, `FullSAMCloudMatrixPublicFlowExact`, `FullSAMPrecipMatrixPublicFlowExact`
- `Cloud no-op`: `ShocNoPrecipNoIcePublicFlowExact`, `FullSAMSedimentationDetJColumnPublicFlowExact`
- `Cloud active`: `NoIceRainMatrixPublicFlowExact`, `FullSAMCloudMatrixPublicFlowExact`, `FullSAMPrecipMatrixPublicFlowExact`
- `T <= tbgmin`: `FullSAMCloudMatrixPublicFlowExact`
- `tbgmin < T < tbgmax`: `FullSAMCloudMatrixPublicFlowExact`
- `T >= tbgmax`: `FullSAMCloudMatrixPublicFlowExact`
- `qt > qsat`: `FullSAMCloudMatrixPublicFlowExact`
- `qt <= qsat`: `FullSAMCloudMatrixPublicFlowExact`
- `evaporate-all-then-recheck path`: `FullSAMCloudMatrixPublicFlowExact`
- `Newton path`: `FullSAMCloudMatrixPublicFlowExact`
- `condensate limiter path`: `FullSAMCloudMatrixPublicFlowExact`
- `qn + qp == 0 no-op`: `ShocNoPrecipNoIcePublicFlowExact`
- `cloud only`: `NoIceRainMatrixPublicFlowExact`, `FullSAMPrecipMatrixPublicFlowExact`
- `precip only`: `NoIceRainMatrixPublicFlowExact`, `FullSAMPrecipMatrixPublicFlowExact`
- `cloud + precip`: `NoIceRainMatrixPublicFlowExact`, `FullSAMPrecipMatrixPublicFlowExact`
- `water autoconversion off`: `NoIceRainMatrixPublicFlowExact`, `FullSAMPrecipMatrixPublicFlowExact`
- `water autoconversion on`: `NoIceRainMatrixPublicFlowExact`, `FullSAMPrecipMatrixPublicFlowExact`
- `ice autoconversion off`: `FullSAMPrecipMatrixPublicFlowExact`
- `ice autoconversion on`: `FullSAMPrecipMatrixPublicFlowExact`
- `rain accretion off`: `NoIceRainMatrixPublicFlowExact`, `FullSAMPrecipMatrixPublicFlowExact`
- `rain accretion on`: `NoIceRainMatrixPublicFlowExact`, `FullSAMPrecipMatrixPublicFlowExact`
- `snow accretion off`: `FullSAMPrecipMatrixPublicFlowExact`
- `snow accretion on`: `FullSAMPrecipMatrixPublicFlowExact`
- `graupel accretion off`: `FullSAMPrecipMatrixPublicFlowExact`
- `graupel accretion on`: `FullSAMPrecipMatrixPublicFlowExact`
- `sink-limited cloud water`: `FullSAMPrecipMatrixPublicFlowExact`
- `sink-limited cloud ice`: `FullSAMPrecipMatrixPublicFlowExact`
- `evaporation off`: `NoIceRainMatrixPublicFlowExact`, `FullSAMPrecipMatrixPublicFlowExact`
- `evaporation on`: `NoIceRainMatrixPublicFlowExact`, `FullSAMPrecipMatrixPublicFlowExact`
- `evaporation species-limited`: `NoIceRainMatrixPublicFlowExact`, `FullSAMPrecipMatrixPublicFlowExact`
- `IceFall no-op`: `ShocNoPrecipNoIcePublicFlowExact`, `NoIceRainMatrixPublicFlowExact`
- `IceFall active`: `FullSAMSedimentationDetJColumnPublicFlowExact`
- `PrecipFall no-op`: `ShocNoPrecipNoIcePublicFlowExact`
- `PrecipFall active`: `NoIceRainMatrixPublicFlowExact`, `FullSAMSedimentationDetJColumnPublicFlowExact`
- `bottom face`: `NoIceRainMatrixPublicFlowExact`, `FullSAMSedimentationDetJColumnPublicFlowExact`
- `interior face`: `NoIceRainMatrixPublicFlowExact`, `FullSAMSedimentationDetJColumnPublicFlowExact`
- `top face`: `NoIceRainMatrixPublicFlowExact`, `FullSAMSedimentationDetJColumnPublicFlowExact`
- `zero flux below threshold`: `NoIceRainMatrixPublicFlowExact`, `FullSAMSedimentationDetJColumnPublicFlowExact`
- `nonzero rain flux`: `NoIceRainMatrixPublicFlowExact`, `FullSAMSedimentationDetJColumnPublicFlowExact`
- `nonzero snow flux`: `FullSAMSedimentationDetJColumnPublicFlowExact`
- `nonzero graupel flux`: `FullSAMSedimentationDetJColumnPublicFlowExact`
- `surface rain accumulation`: `NoIceRainMatrixPublicFlowExact`, `FullSAMSedimentationDetJColumnPublicFlowExact`
- `surface snow accumulation`: `FullSAMSedimentationDetJColumnPublicFlowExact`
- `surface graupel accumulation`: `FullSAMSedimentationDetJColumnPublicFlowExact`
- `nonnegative clipping`: `FullSAMSedimentationDetJColumnPublicFlowExact`
- `detJ absent`: `ShocNoPrecipNoIcePublicFlowExact`, `NoIceRainMatrixPublicFlowExact`, `FullSAMCloudMatrixPublicFlowExact`, `FullSAMPrecipMatrixPublicFlowExact`
- `detJ present`: `FullSAMSedimentationDetJColumnPublicFlowExact`
- `real_width = 0`: `ShocNoPrecipNoIcePublicFlowExact`, `FullSAMCloudMatrixPublicFlowExact`, `FullSAMPrecipMatrixPublicFlowExact`, `FullSAMSedimentationDetJColumnPublicFlowExact`
- `real_width > 0`: `NoIceRainMatrixPublicFlowExact`
- `one-column vertical case`: `FullSAMSedimentationDetJColumnPublicFlowExact`
- `multi-cell horizontal case`: `ShocNoPrecipNoIcePublicFlowExact`, `NoIceRainMatrixPublicFlowExact`, `FullSAMCloudMatrixPublicFlowExact`, `FullSAMPrecipMatrixPublicFlowExact`