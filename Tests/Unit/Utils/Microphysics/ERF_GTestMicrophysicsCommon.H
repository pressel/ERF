#ifndef ERF_GTEST_MICROPHYSICS_COMMON_H_
#define ERF_GTEST_MICROPHYSICS_COMMON_H_

#include <algorithm>
#include <cmath>
#include <string>

#include <AMReX_Math.H>

#include <ERF_Constants.H>
#include <ERF_HSEUtils.H>
#include <ERF_MicrophysicsUtils.H>

namespace microphysics_test {

#ifdef AMREX_USE_FLOAT
constexpr amrex::Real kValueRelTol = amrex::Real(1.0e-4);
constexpr amrex::Real kDerivativeRelTol = amrex::Real(2.0e-2);
constexpr amrex::Real kEmpiricalAgreementRelTol = amrex::Real(1.0e-2);
constexpr amrex::Real kSwitchRelTol = amrex::Real(5.0e-2);
constexpr amrex::Real kFiniteDiffStep = amrex::Real(5.0e-2);
constexpr amrex::Real kKernelErrorLimit = amrex::Real(20.0);
#else
constexpr amrex::Real kValueRelTol = amrex::Real(1.0e-12);
constexpr amrex::Real kDerivativeRelTol = amrex::Real(2.0e-5);
constexpr amrex::Real kEmpiricalAgreementRelTol = amrex::Real(2.0e-3);
constexpr amrex::Real kSwitchRelTol = amrex::Real(5.0e-2);
constexpr amrex::Real kFiniteDiffStep = amrex::Real(1.0e-3);
constexpr amrex::Real kKernelErrorLimit = amrex::Real(10.0);
#endif

constexpr amrex::Real kSqrtPi = amrex::Real(1.7724538509055160273);
constexpr amrex::Real kEmpiricalTriplePointMbar = amrex::Real(6.11245202992297);
constexpr amrex::Real kEmpiricalTriplePointPa = amrex::Real(611.245202992297);
constexpr amrex::Real kWaterLowerSwitchK = amrex::Real(203.16);
constexpr amrex::Real kWaterUpperSwitchK = amrex::Real(343.16);

inline amrex::Real scaled_tol (const amrex::Real actual,
                               const amrex::Real expected,
                               const amrex::Real factor)
{
    return factor * std::max({amrex::Real(1.0), std::abs(actual), std::abs(expected)});
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real portable_scaled_tol (const amrex::Real actual,
                                 const amrex::Real expected,
                                 const amrex::Real factor)
{
    return factor * amrex::max(amrex::Real(1.0),
                                amrex::max(amrex::Math::abs(actual), amrex::Math::abs(expected)));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real normalized_error (const amrex::Real actual,
                              const amrex::Real expected,
                              const amrex::Real factor)
{
    return amrex::Math::abs(actual - expected) / portable_scaled_tol(actual, expected, factor);
}

template <typename Function>
inline amrex::Real central_difference (Function&& func,
                                       const amrex::Real x,
                                       const amrex::Real step = kFiniteDiffStep)
{
    return (func(x + step) - func(x - step)) / (amrex::Real(2.0) * step);
}

inline amrex::Real cc_derivative_formula (const amrex::Real temperature)
{
    constexpr amrex::Real svp2 = amrex::Real(17.67);
    constexpr amrex::Real svp3 = amrex::Real(29.65);
    constexpr amrex::Real svpt0 = amrex::Real(273.15);
    const amrex::Real esatw = erf_esatw_cc(temperature);

    return esatw * svp2 * (svpt0 - svp3) /
           ((temperature - svp3) * (temperature - svp3));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool uses_normal_qsat_branch (const amrex::Real esat, const amrex::Real pressure)
{
    return (pressure - esat) >= esat;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real capped_branch_pressure (const amrex::Real esat)
{
    return amrex::Real(1.5) * esat;
}

inline std::string tp_label (const char* label,
                             const amrex::Real temperature,
                             const amrex::Real pressure)
{
    return std::string(label) + " T=" + std::to_string(temperature) +
           " p=" + std::to_string(pressure);
}

} // namespace microphysics_test

#endif