#ifndef ERF_GTEST_SAM_COMMON_H_
#define ERF_GTEST_SAM_COMMON_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>

#include <AMReX_FArrayBox.H>

#include <ERF_Constants.H>
#include <ERF_EOS.H>
#include <ERF_IndexDefines.H>
#include <ERF_MicrophysicsUtils.H>
#include <ERF_SAMUtils.H>

namespace sam_test {

constexpr amrex::Real kRdOcp = R_d / Cp_d;
constexpr amrex::Real kFacCond = lcond / Cp_d;
constexpr amrex::Real kFacFus = lfus / Cp_d;

// MUST MATCH the SAM helper integer used in ERF_CloudSAM.cpp and ERF_Precip.cpp
// for the cloud-ice-enabled mode.
constexpr int kSAMWithIceMode = 1;

// MUST MATCH the SAM helper integer used in ERF_CloudSAM.cpp and ERF_Precip.cpp
// for the no-ice mode.
constexpr int kSAMNoIceMode = 2;

// MUST MATCH qp_threshold in ERF_Constants.H.
// This duplication is intentional because the tests sample both sides of the
// production precipitation-flux activation branch.
constexpr amrex::Real kQpThresholdForBranchSampling = qp_threshold;

// MUST MATCH the phase-fraction activation thresholds in sam_accretion_rates.
constexpr amrex::Real kPhaseActivationLow = amrex::Real(0.001);
constexpr amrex::Real kPhaseActivationHigh = amrex::Real(0.999);

// Provenance: exact-zero branches only accumulate machine roundoff.
inline amrex::Real exact_zero_tol ()
{
#ifdef AMREX_USE_FLOAT
    return amrex::Real(64.0) * std::numeric_limits<amrex::Real>::epsilon();
#else
    return amrex::Real(32.0) * std::numeric_limits<amrex::Real>::epsilon();
#endif
}

// Provenance: a handful of multiplies/divides on smooth algebraic identities.
inline amrex::Real roundoff_tol (const amrex::Real expected)
{
#ifdef AMREX_USE_FLOAT
    const amrex::Real factor = amrex::Real(128.0);
#else
    const amrex::Real factor = amrex::Real(32.0);
#endif
    return factor * std::numeric_limits<amrex::Real>::epsilon() *
           std::max(amrex::Real(1.0), std::abs(expected));
}

// Provenance: pow/sqrt helpers vary slightly across host math libraries.
inline amrex::Real pow_sqrt_tol (const amrex::Real expected)
{
#ifdef AMREX_USE_FLOAT
    const amrex::Real factor = amrex::Real(4096.0);
#else
    const amrex::Real factor = amrex::Real(256.0);
#endif
    return factor * std::numeric_limits<amrex::Real>::epsilon() *
           std::max(amrex::Real(1.0), std::abs(expected));
}

// Provenance: reduced sums over several flux/source terms in a single cell.
inline amrex::Real flux_conservation_tol (const amrex::Real scale)
{
#ifdef AMREX_USE_FLOAT
    const amrex::Real factor = amrex::Real(256.0);
#else
    const amrex::Real factor = amrex::Real(64.0);
#endif
    return factor * std::numeric_limits<amrex::Real>::epsilon() *
           std::max(amrex::Real(1.0), std::abs(scale));
}

// Provenance: MPI/global reduction order plus accumulation over many cellwise
// precip and accumulation contributions in one domain-wide budget.
inline amrex::Real mpi_reduction_tol (const amrex::Real scale,
                                      const int num_terms)
{
#ifdef AMREX_USE_FLOAT
    const amrex::Real factor = amrex::Real(512.0);
#else
    const amrex::Real factor = amrex::Real(128.0);
#endif
    return factor * std::numeric_limits<amrex::Real>::epsilon() *
           std::max(amrex::Real(1.0), std::abs(scale)) * std::max(1, num_terms);
}

// Provenance: Newton helper temperatures are in kelvin and the scalar formulas
// are smooth away from branch boundaries.
inline amrex::Real newton_temperature_tol (const amrex::Real expected)
{
    return roundoff_tol(expected);
}

inline amrex::Real normalized_error (const amrex::Real actual,
                                     const amrex::Real expected)
{
    return std::abs(actual - expected) /
           std::max(amrex::Real(1.0), std::abs(expected));
}

template <typename F>
inline amrex::Real central_difference (F&& func,
                                       const amrex::Real x,
                                       const amrex::Real h)
{
    return (func(x + h) - func(x - h)) / (amrex::Real(2.0) * h);
}

inline amrex::Real derivative_step (const amrex::Real scale)
{
#ifdef AMREX_USE_FLOAT
    return amrex::Real(2.0e-3) * std::max(amrex::Real(1.0), std::abs(scale));
#else
    return amrex::Real(1.0e-6) * std::max(amrex::Real(1.0), std::abs(scale));
#endif
}

inline amrex::Box single_cell_box ()
{
    return amrex::Box(amrex::IntVect(0, 0, 0), amrex::IntVect(0, 0, 0));
}

inline amrex::Array4<amrex::Real> single_cell_state_array (amrex::FArrayBox& fab)
{
    return fab.array();
}

inline std::array<amrex::Real, 5> threshold_samples (const amrex::Real lower,
                                                      const amrex::Real upper)
{
    return {
        lower - amrex::Real(1.0e-3),
        lower,
        amrex::Real(0.5) * (lower + upper),
        upper,
        upper + amrex::Real(1.0e-3)};
}

inline std::string branch_label (const char* helper_name,
                                 const amrex::Real sample)
{
    return std::string(helper_name) + " sample=" +
           std::to_string(static_cast<double>(sample));
}

} // namespace sam_test

#endif