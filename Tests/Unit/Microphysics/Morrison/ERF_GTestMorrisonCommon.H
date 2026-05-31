#ifndef ERF_GTEST_MORRISON_COMMON_H_
#define ERF_GTEST_MORRISON_COMMON_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>

#include <AMReX_Gpu.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_Math.H>

#include <ERF_MorrisonUtils.H>

namespace morrison_test {

// Morrison unit-test contract, first testability slice
// ----------------------------------------------------
// These tests cover scalar helpers extracted from local Morrison source blocks
// that were already device-reachable in Morrison::Advance. They protect
// threshold equality, qsmall mass-number cleanup, warm tiny-ice melt semantics,
// latent-heating signs, exponential PSD slope bounds, and host/device parity.
// Broader source-term, sedimentation, public-path, and MPI contracts are
// documented as known gaps in the Morrison README.

constexpr amrex::Real kQSmall = amrex::Real(1.0e-14);
constexpr amrex::Real kPi = amrex::Real(3.1415926535897932384626434);
constexpr amrex::Real kRhoWater = amrex::Real(997.0);
constexpr amrex::Real kRhoSnow = amrex::Real(100.0);
constexpr amrex::Real kRhoGraupel = amrex::Real(400.0);
constexpr amrex::Real kRainCoefficient = kPi * kRhoWater;
constexpr amrex::Real kSnowCoefficient = kPi * kRhoSnow;
constexpr amrex::Real kGraupelCoefficient = kPi * kRhoGraupel;
constexpr amrex::Real kLamMinRain = amrex::Real(1.0) / amrex::Real(2800.0e-6);
constexpr amrex::Real kLamMaxRain = amrex::Real(1.0) / amrex::Real(20.0e-6);
constexpr amrex::Real kLamMinSnow = amrex::Real(1.0) / amrex::Real(2000.0e-6);
constexpr amrex::Real kLamMaxSnow = amrex::Real(1.0) / amrex::Real(10.0e-6);
constexpr amrex::Real kLamMinGraupel = amrex::Real(1.0) / amrex::Real(2000.0e-6);
constexpr amrex::Real kLamMaxGraupel = amrex::Real(1.0) / amrex::Real(20.0e-6);
constexpr amrex::Real kLatentVaporization = amrex::Real(2.5e6);
constexpr amrex::Real kLatentSublimation = amrex::Real(2.834e6);
constexpr amrex::Real kLatentFusion = kLatentSublimation - kLatentVaporization;
constexpr amrex::Real kCpm = amrex::Real(1007.0);

inline amrex::Real formula_abs_tol (const amrex::Real expected)
{
#ifdef AMREX_USE_FLOAT
    return amrex::Real(96.0) * std::numeric_limits<amrex::Real>::epsilon() *
           std::max(amrex::Real(1.0), std::abs(expected));
#else
    return amrex::Real(48.0) * std::numeric_limits<amrex::Real>::epsilon() *
           std::max(amrex::Real(1.0), std::abs(expected));
#endif
}

inline amrex::Real backend_math_abs_tol (const amrex::Real expected)
{
#ifdef AMREX_USE_FLOAT
    return amrex::Real(4.0e3) * std::numeric_limits<amrex::Real>::epsilon() *
           std::max(amrex::Real(1.0), std::abs(expected));
#else
    return amrex::Real(2.0e2) * std::numeric_limits<amrex::Real>::epsilon() *
           std::max(amrex::Real(1.0), std::abs(expected));
#endif
}

inline amrex::Real exact_zero_tol ()
{
#ifdef AMREX_USE_FLOAT
    return amrex::Real(96.0) * std::numeric_limits<amrex::Real>::epsilon();
#else
    return amrex::Real(48.0) * std::numeric_limits<amrex::Real>::epsilon();
#endif
}

inline amrex::Real property_tol (const int num_terms,
                                 const amrex::Real scale)
{
#ifdef AMREX_USE_FLOAT
    const amrex::Real factor = amrex::Real(512.0);
#else
    const amrex::Real factor = amrex::Real(128.0);
#endif
    return factor * std::numeric_limits<amrex::Real>::epsilon() *
           std::max(amrex::Real(1.0), std::abs(scale)) * std::max(1, num_terms);
}

inline amrex::Real latent_proxy_tol (const int num_terms,
                                     const amrex::Real temperature_scale,
                                     const amrex::Real cpm)
{
    // Provenance: latent tests reconstruct cpm * dT from the difference of two
    // O(temperature_scale) temperatures after an O(1e-5 K) update.
    return property_tol(num_terms, temperature_scale * cpm);
}

inline void sync ()
{
#ifdef AMREX_USE_GPU
    amrex::Gpu::streamSynchronize();
#endif
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MorrisonCellState make_state (const amrex::Real qv,
                              const amrex::Real qc,
                              const amrex::Real qi,
                              const amrex::Real qr,
                              const amrex::Real qs,
                              const amrex::Real qg,
                              const amrex::Real nc,
                              const amrex::Real ni,
                              const amrex::Real nr,
                              const amrex::Real ns,
                              const amrex::Real ng) noexcept
{
    return MorrisonCellState{amrex::Real(1.0), amrex::Real(280.0), amrex::Real(90000.0),
                             qv, qc, qi, qr, qs, qg, nc, ni, nr, ns, ng};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MorrisonEffectiveRadii make_effective_radii () noexcept
{
    return MorrisonEffectiveRadii{amrex::Real(11.0), amrex::Real(12.0), amrex::Real(13.0),
                                  amrex::Real(14.0), amrex::Real(15.0)};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real number_for_lambda (const amrex::Real mass,
                               const amrex::Real coefficient,
                               const amrex::Real lambda) noexcept
{
    return lambda * lambda * lambda * mass / coefficient;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real expected_adjusted_number (const amrex::Real mass,
                                      const amrex::Real coefficient,
                                      const amrex::Real lambda) noexcept
{
    return lambda * lambda * lambda * mass / coefficient;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real expected_intercept (const amrex::Real mass,
                                const amrex::Real coefficient,
                                const amrex::Real lambda) noexcept
{
    return lambda * lambda * lambda * lambda * mass / coefficient;
}

inline std::string state_label (const MorrisonCellState& state)
{
    return "qc=" + std::to_string(static_cast<double>(state.qc)) +
           " qi=" + std::to_string(static_cast<double>(state.qi)) +
           " qr=" + std::to_string(static_cast<double>(state.qr)) +
           " qs=" + std::to_string(static_cast<double>(state.qs)) +
           " qg=" + std::to_string(static_cast<double>(state.qg));
}

} // namespace morrison_test

#endif