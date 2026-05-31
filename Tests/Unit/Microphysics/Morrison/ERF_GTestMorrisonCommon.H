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

// MUST MATCH Source/Microphysics/Morrison/ERF_AdvanceMorrison.cpp: m_qsmall initialization.
constexpr amrex::Real kQSmall = amrex::Real(1.0e-14);
// MUST MATCH Source/Microphysics/Morrison/ERF_AdvanceMorrison.cpp: m_pi initialization.
constexpr amrex::Real kPi = amrex::Real(3.1415926535897932384626434);
// MUST MATCH Source/Microphysics/Morrison/ERF_AdvanceMorrison.cpp: m_rhow initialization.
constexpr amrex::Real kRhoWater = amrex::Real(997.0);
// MUST MATCH Source/Microphysics/Morrison/ERF_AdvanceMorrison.cpp: m_rhosn initialization.
constexpr amrex::Real kRhoSnow = amrex::Real(100.0);
// MUST MATCH Source/Microphysics/Morrison/ERF_AdvanceMorrison.cpp: m_rhog graupel initialization when m_ihail == 0.
constexpr amrex::Real kRhoGraupel = amrex::Real(400.0);
constexpr amrex::Real kRainCoefficient = kPi * kRhoWater;
constexpr amrex::Real kSnowCoefficient = kPi * kRhoSnow;
constexpr amrex::Real kGraupelCoefficient = kPi * kRhoGraupel;
// MUST MATCH Source/Microphysics/Morrison/ERF_AdvanceMorrison.cpp: m_lamminr and m_lammaxr initialization.
constexpr amrex::Real kLamMinRain = amrex::Real(1.0) / amrex::Real(2800.0e-6);
constexpr amrex::Real kLamMaxRain = amrex::Real(1.0) / amrex::Real(20.0e-6);
// MUST MATCH Source/Microphysics/Morrison/ERF_AdvanceMorrison.cpp: m_lammins and m_lammaxs initialization.
constexpr amrex::Real kLamMinSnow = amrex::Real(1.0) / amrex::Real(2000.0e-6);
constexpr amrex::Real kLamMaxSnow = amrex::Real(1.0) / amrex::Real(10.0e-6);
// MUST MATCH Source/Microphysics/Morrison/ERF_AdvanceMorrison.cpp: m_lamming and m_lammaxg initialization.
constexpr amrex::Real kLamMinGraupel = amrex::Real(1.0) / amrex::Real(2000.0e-6);
constexpr amrex::Real kLamMaxGraupel = amrex::Real(1.0) / amrex::Real(20.0e-6);
// Mirrors Morrison C++ path latent heat magnitudes used by helper-level latent residual tests.
constexpr amrex::Real kLatentVaporization = amrex::Real(2.5e6);
constexpr amrex::Real kLatentSublimation = amrex::Real(2.834e6);
constexpr amrex::Real kLatentFusion = kLatentSublimation - kLatentVaporization;
// Mirrors Morrison helper-level moist heat capacity scale for isolated residual tests.
constexpr amrex::Real kCpm = amrex::Real(1007.0);
// MUST MATCH Source/Microphysics/Morrison/ERF_AdvanceMorrison.cpp: m_cons29 initialization.
constexpr amrex::Real kCons29 = amrex::Real(4.0) / amrex::Real(3.0) * kPi * kRhoWater *
                                amrex::Real(25.0e-6) * amrex::Real(25.0e-6) * amrex::Real(25.0e-6);

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
                                     const amrex::Real latent_energy_scale,
                                     const amrex::Real reconstructed_temperature_energy_scale)
{
    return property_tol(num_terms, std::max(latent_energy_scale, reconstructed_temperature_energy_scale));
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