#ifndef ERF_MORRISON_UTILS_H_
#define ERF_MORRISON_UTILS_H_

#include <algorithm>
#include <cmath>

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

struct MorrisonCellState {
    amrex::Real rho;
    amrex::Real temperature;
    amrex::Real pressure;
    amrex::Real qv;
    amrex::Real qc;
    amrex::Real qi;
    amrex::Real qr;
    amrex::Real qs;
    amrex::Real qg;
    amrex::Real nc;
    amrex::Real ni;
    amrex::Real nr;
    amrex::Real ns;
    amrex::Real ng;
};

struct MorrisonEffectiveRadii {
    amrex::Real effc;
    amrex::Real effi;
    amrex::Real effr;
    amrex::Real effs;
    amrex::Real effg;
};

struct MorrisonQSmallCleanupDiagnostics {
    int qc_zeroed;
    int qi_zeroed;
    int qr_zeroed;
    int qs_zeroed;
    int qg_zeroed;
};

struct MorrisonDistributionParameters {
    amrex::Real lambda;
    amrex::Real intercept;
    amrex::Real number;
    int active;
    int limited_to_min;
    int limited_to_max;
};

struct MorrisonAutoconversionRates {
    amrex::Real prc;
    amrex::Real nprc;
    amrex::Real nprc1;
    int active;
    int nprc_limited;
    int nprc1_limited;
};

struct MorrisonAccretionRates {
    amrex::Real pra;
    amrex::Real npra;
    int active;
};

struct MorrisonCloudWaterLimiterDiagnostics {
    amrex::Real ratio;
    int limited;
};

struct MorrisonSedimentationBudget {
    amrex::Real flux_divergence;
    amrex::Real mixing_ratio_tendency;
    amrex::Real density_content_delta;
};

struct MorrisonSurfacePrecipitationIncrement {
    amrex::Real precipitation;
    amrex::Real snow;
    amrex::Real snow_plus_ice;
    amrex::Real graupel;
};

static constexpr amrex::Real morrison_subsaturation_ratio_threshold = amrex::Real(0.9);
static constexpr amrex::Real morrison_subsaturation_cleanup_threshold = amrex::Real(1.0e-8);
static constexpr amrex::Real morrison_warm_small_ice_melt_threshold = amrex::Real(1.0e-6);
static constexpr amrex::Real morrison_autoconversion_cloud_water_threshold = amrex::Real(1.0e-6);
static constexpr amrex::Real morrison_cloud_rain_accretion_threshold = amrex::Real(1.0e-8);

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real morrison_total_water_full (const MorrisonCellState& state) noexcept
{
    return state.qv + state.qc + state.qi + state.qr + state.qs + state.qg;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real morrison_total_water_no_ice (const MorrisonCellState& state) noexcept
{
    return state.qv + state.qc + state.qr;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MorrisonQSmallCleanupDiagnostics morrison_apply_qsmall_mass_number_cleanup (
    MorrisonCellState& state,
    MorrisonEffectiveRadii& effective_radii,
    const amrex::Real qsmall) noexcept
{
    MorrisonQSmallCleanupDiagnostics diagnostics{0, 0, 0, 0, 0};

    // MUST MATCH: current Morrison qsmall mass-number cleanup branches.
    if (state.qc < qsmall) {
        state.qc = amrex::Real(0);
        state.nc = amrex::Real(0);
        effective_radii.effc = amrex::Real(0);
        diagnostics.qc_zeroed = 1;
    }
    if (state.qr < qsmall) {
        state.qr = amrex::Real(0);
        state.nr = amrex::Real(0);
        effective_radii.effr = amrex::Real(0);
        diagnostics.qr_zeroed = 1;
    }
    if (state.qi < qsmall) {
        state.qi = amrex::Real(0);
        state.ni = amrex::Real(0);
        effective_radii.effi = amrex::Real(0);
        diagnostics.qi_zeroed = 1;
    }
    if (state.qs < qsmall) {
        state.qs = amrex::Real(0);
        state.ns = amrex::Real(0);
        effective_radii.effs = amrex::Real(0);
        diagnostics.qs_zeroed = 1;
    }
    if (state.qg < qsmall) {
        state.qg = amrex::Real(0);
        state.ng = amrex::Real(0);
        effective_radii.effg = amrex::Real(0);
        diagnostics.qg_zeroed = 1;
    }

    return diagnostics;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void morrison_apply_subsaturation_small_hydrometeor_cleanup (
    MorrisonCellState& state,
    const amrex::Real qvqvs,
    const amrex::Real qvqvsi,
    const amrex::Real latent_vaporization,
    const amrex::Real latent_sublimation,
    const amrex::Real cpm) noexcept
{
    // MUST MATCH: current Morrison subsaturated tiny liquid cleanup thresholds.
    if (qvqvs < morrison_subsaturation_ratio_threshold) {
        if (state.qr < morrison_subsaturation_cleanup_threshold) {
            state.qv += state.qr;
            state.temperature -= state.qr * latent_vaporization / cpm;
            state.qr = amrex::Real(0);
        }
        if (state.qc < morrison_subsaturation_cleanup_threshold) {
            state.qv += state.qc;
            state.temperature -= state.qc * latent_vaporization / cpm;
            state.qc = amrex::Real(0);
        }
    }

    // MUST MATCH: current Morrison subsaturated tiny ice cleanup thresholds.
    if (qvqvsi < morrison_subsaturation_ratio_threshold) {
        if (state.qi < morrison_subsaturation_cleanup_threshold) {
            state.qv += state.qi;
            state.temperature -= state.qi * latent_sublimation / cpm;
            state.qi = amrex::Real(0);
        }
        if (state.qs < morrison_subsaturation_cleanup_threshold) {
            state.qv += state.qs;
            state.temperature -= state.qs * latent_sublimation / cpm;
            state.qs = amrex::Real(0);
        }
        if (state.qg < morrison_subsaturation_cleanup_threshold) {
            state.qv += state.qg;
            state.temperature -= state.qg * latent_sublimation / cpm;
            state.qg = amrex::Real(0);
        }
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void morrison_apply_warm_small_ice_melt_to_rain (
    MorrisonCellState& state,
    const amrex::Real latent_fusion,
    const amrex::Real cpm) noexcept
{
    // MUST MATCH: current Morrison warm-branch small snow/graupel melt threshold.
    if (state.qs < morrison_warm_small_ice_melt_threshold) {
        state.qr = state.qr + state.qs;
        state.nr = state.nr + state.ns;
        state.temperature = state.temperature - state.qs * latent_fusion / cpm;
        state.qs = amrex::Real(0);
        state.ns = amrex::Real(0);
    }
    if (state.qg < morrison_warm_small_ice_melt_threshold) {
        state.qr = state.qr + state.qg;
        state.nr = state.nr + state.ng;
        state.temperature = state.temperature - state.qg * latent_fusion / cpm;
        state.qg = amrex::Real(0);
        state.ng = amrex::Real(0);
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MorrisonAutoconversionRates morrison_compute_warm_rain_autoconversion (
    const amrex::Real qc,
    const amrex::Real nc,
    const amrex::Real rho,
    const amrex::Real dt,
    const amrex::Real cons29) noexcept
{
    MorrisonAutoconversionRates rates{amrex::Real(0), amrex::Real(0), amrex::Real(0), 0, 0, 0};

    if (qc >= morrison_autoconversion_cloud_water_threshold) {
        rates.active = 1;
        rates.prc = amrex::Real(1350.0) * std::pow(qc, amrex::Real(2.47)) *
            std::pow(nc / amrex::Real(1.0e6) * rho, -amrex::Real(1.79));
        rates.nprc1 = rates.prc / cons29;
        rates.nprc = rates.prc / (qc / nc);

        const amrex::Real nprc_limit = nc / dt;
        if (rates.nprc > nprc_limit) {
            rates.nprc_limited = 1;
        }
        rates.nprc = std::min(rates.nprc, nprc_limit);

        if (rates.nprc1 > rates.nprc) {
            rates.nprc1_limited = 1;
        }
        rates.nprc1 = std::min(rates.nprc1, rates.nprc);
    }

    return rates;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MorrisonAccretionRates morrison_compute_cloud_rain_accretion (
    const amrex::Real qc,
    const amrex::Real qr,
    const amrex::Real nc) noexcept
{
    MorrisonAccretionRates rates{amrex::Real(0), amrex::Real(0), 0};

    if (qr >= morrison_cloud_rain_accretion_threshold && qc >= morrison_cloud_rain_accretion_threshold) {
        rates.active = 1;
        rates.pra = amrex::Real(67.0) * std::pow(qc * qr, amrex::Real(1.15));
        rates.npra = rates.pra / (qc / nc);
    }

    return rates;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MorrisonCloudWaterLimiterDiagnostics morrison_apply_cloud_water_sink_limiter (
    const amrex::Real qc,
    const amrex::Real dt,
    const amrex::Real qsmall,
    amrex::Real& prc,
    amrex::Real& pra) noexcept
{
    MorrisonCloudWaterLimiterDiagnostics diagnostics{amrex::Real(1), 0};
    const amrex::Real requested_cloud_water = (prc + pra) * dt;

    if (requested_cloud_water > qc && qc >= qsmall) {
        diagnostics.limited = 1;
        diagnostics.ratio = qc / requested_cloud_water;
        prc = prc * diagnostics.ratio;
        pra = pra * diagnostics.ratio;
    }

    return diagnostics;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MorrisonSedimentationBudget morrison_sedimentation_budget (
    const amrex::Real fallout_from_above,
    const amrex::Real fallout_to_below,
    const amrex::Real dz,
    const amrex::Real rho,
    const amrex::Real dt,
    const int nstep) noexcept
{
    const amrex::Real flux_divergence = (fallout_from_above - fallout_to_below) / dz;
    const amrex::Real step_scale = amrex::Real(1) / static_cast<amrex::Real>(nstep);
    return MorrisonSedimentationBudget{flux_divergence,
                                       flux_divergence * step_scale / rho,
                                       flux_divergence * dt * step_scale};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MorrisonSurfacePrecipitationIncrement morrison_surface_precipitation_increment (
    const amrex::Real fallout_rain,
    const amrex::Real fallout_cloud_water,
    const amrex::Real fallout_snow,
    const amrex::Real fallout_cloud_ice,
    const amrex::Real fallout_graupel,
    const amrex::Real dt,
    const int nstep) noexcept
{
    const amrex::Real step_dt = dt / static_cast<amrex::Real>(nstep);
    return MorrisonSurfacePrecipitationIncrement{
        (fallout_rain + fallout_cloud_water + fallout_snow + fallout_cloud_ice + fallout_graupel) * step_dt,
        (fallout_snow + fallout_cloud_ice + fallout_graupel) * step_dt,
        (fallout_cloud_ice + fallout_snow) * step_dt,
        fallout_graupel * step_dt};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MorrisonDistributionParameters morrison_exponential_distribution_parameters (
    const amrex::Real mass,
    const amrex::Real number,
    const amrex::Real coefficient,
    const amrex::Real lambda_min,
    const amrex::Real lambda_max) noexcept
{
    MorrisonDistributionParameters params{amrex::Real(0), amrex::Real(0), number, 0, 0, 0};

    params.active = 1;
    // This helper is intentionally limited to Morrison rain/snow/graupel exponential
    // distributions, where mass = coefficient * number / lambda^3.
    params.lambda = std::pow(coefficient * number / mass, amrex::Real(1) / amrex::Real(3));
    params.intercept = number * params.lambda;

    // MUST MATCH: current Morrison slope clamp and number readjustment policy.
    if (params.lambda < lambda_min) {
        params.lambda = lambda_min;
        params.intercept = std::pow(params.lambda, amrex::Real(4.0)) * mass / coefficient;
        params.number = params.intercept / params.lambda;
        params.limited_to_min = 1;
    } else if (params.lambda > lambda_max) {
        params.lambda = lambda_max;
        params.intercept = std::pow(params.lambda, amrex::Real(4.0)) * mass / coefficient;
        params.number = params.intercept / params.lambda;
        params.limited_to_max = 1;
    }

    return params;
}

#endif