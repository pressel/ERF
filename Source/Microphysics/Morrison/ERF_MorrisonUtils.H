#ifndef ERF_MORRISON_UTILS_H_
#define ERF_MORRISON_UTILS_H_

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

struct MorrisonConfig {
    amrex::Real dt;
    amrex::Real qsmall;
    bool no_ice;
    bool use_graupel;
    bool use_number_concentrations;
    bool enable_sedimentation;
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

static constexpr amrex::Real morrison_subsaturation_ratio_threshold = amrex::Real(0.9);
static constexpr amrex::Real morrison_subsaturation_cleanup_threshold = amrex::Real(1.0e-8);
static constexpr amrex::Real morrison_warm_small_ice_melt_threshold = amrex::Real(1.0e-6);

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
MorrisonDistributionParameters morrison_exponential_distribution_parameters (
    const amrex::Real mass,
    const amrex::Real number,
    const amrex::Real coefficient,
    const amrex::Real lambda_min,
    const amrex::Real lambda_max,
    const amrex::Real shape_exponent) noexcept
{
    MorrisonDistributionParameters params{amrex::Real(0), amrex::Real(0), number, 0, 0, 0};

    params.active = 1;
    params.lambda = std::pow(coefficient * number / mass, amrex::Real(1) / shape_exponent);
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