#include <algorithm>

#include <AMReX_ParallelDescriptor.H>

#include <gtest/gtest.h>

#include "ERF_GTestMorrisonCommon.H"

using namespace morrison_test;

namespace {

amrex::Real normalized_error (const amrex::Real actual,
                              const amrex::Real expected,
                              const amrex::Real tol)
{
    return std::abs(actual - expected) / std::max(tol, std::numeric_limits<amrex::Real>::min());
}

} // namespace

// Motivation: This is the first MPI-aware Morrison helper contract. It splits a
// deterministic PSD branch sweep across ranks and reduces the maximum error.
// This protects the extracted scalar contract from hidden rank-local behavior;
// public Morrison decomposition and sedimentation-budget tests still need a
// later, broader production-path seam.
TEST(MorrisonParallel, DistributedExponentialDistributionSweepMatchesScalarReference)
{
    struct DistributedCase {
        amrex::Real coefficient;
        amrex::Real lambda_min;
        amrex::Real lambda_max;
        amrex::Real raw_lambda;
        amrex::Real expected_lambda;
    };

    const amrex::Real mass = amrex::Real(2.0e-4);
    const std::array<DistributedCase, 9> cases = {{
        {kRainCoefficient, kLamMinRain, kLamMaxRain, kLamMinRain * amrex::Real(0.5), kLamMinRain},
        {kRainCoefficient, kLamMinRain, kLamMaxRain, amrex::Real(0.5) * (kLamMinRain + kLamMaxRain), amrex::Real(0.5) * (kLamMinRain + kLamMaxRain)},
        {kRainCoefficient, kLamMinRain, kLamMaxRain, kLamMaxRain * amrex::Real(2.0), kLamMaxRain},
        {kSnowCoefficient, kLamMinSnow, kLamMaxSnow, kLamMinSnow * amrex::Real(0.5), kLamMinSnow},
        {kSnowCoefficient, kLamMinSnow, kLamMaxSnow, amrex::Real(0.5) * (kLamMinSnow + kLamMaxSnow), amrex::Real(0.5) * (kLamMinSnow + kLamMaxSnow)},
        {kSnowCoefficient, kLamMinSnow, kLamMaxSnow, kLamMaxSnow * amrex::Real(2.0), kLamMaxSnow},
        {kGraupelCoefficient, kLamMinGraupel, kLamMaxGraupel, kLamMinGraupel * amrex::Real(0.5), kLamMinGraupel},
        {kGraupelCoefficient, kLamMinGraupel, kLamMaxGraupel, amrex::Real(0.5) * (kLamMinGraupel + kLamMaxGraupel), amrex::Real(0.5) * (kLamMinGraupel + kLamMaxGraupel)},
        {kGraupelCoefficient, kLamMinGraupel, kLamMaxGraupel, kLamMaxGraupel * amrex::Real(2.0), kLamMaxGraupel}
    }};

    const int rank = amrex::ParallelDescriptor::MyProc();
    const int nprocs = amrex::ParallelDescriptor::NProcs();
    long local_checked = 0;
    amrex::Real local_max_normalized_error = amrex::Real(0.0);

    for (int idx = rank; idx < static_cast<int>(cases.size()); idx += nprocs) {
        const DistributedCase& test_case = cases[idx];
        const amrex::Real number = number_for_lambda(mass, test_case.coefficient, test_case.raw_lambda);
        const MorrisonDistributionParameters actual = morrison_exponential_distribution_parameters(
            mass, number, test_case.coefficient, test_case.lambda_min, test_case.lambda_max, amrex::Real(3.0));
        const amrex::Real expected_number = expected_adjusted_number(mass, test_case.coefficient,
                                                                     test_case.expected_lambda);
        const amrex::Real expected_n0 = expected_intercept(mass, test_case.coefficient,
                                                           test_case.expected_lambda);
        local_max_normalized_error = std::max(local_max_normalized_error,
            normalized_error(actual.lambda, test_case.expected_lambda, backend_math_abs_tol(test_case.expected_lambda)));
        local_max_normalized_error = std::max(local_max_normalized_error,
            normalized_error(actual.number, expected_number, backend_math_abs_tol(expected_number)));
        local_max_normalized_error = std::max(local_max_normalized_error,
            normalized_error(actual.intercept, expected_n0, backend_math_abs_tol(expected_n0)));
        ++local_checked;
    }

    long global_checked = local_checked;
    amrex::Real global_max_normalized_error = local_max_normalized_error;
    amrex::ParallelDescriptor::ReduceLongSum(global_checked);
    amrex::ParallelDescriptor::ReduceRealMax(global_max_normalized_error);

    EXPECT_EQ(global_checked, static_cast<long>(cases.size()))
        << "rank=" << rank << " nprocs=" << nprocs << " local_checked=" << local_checked;
    EXPECT_LE(global_max_normalized_error, amrex::Real(1.0))
        << "rank=" << rank << " nprocs=" << nprocs
        << " local_max_normalized_error=" << local_max_normalized_error
        << " global_max_normalized_error=" << global_max_normalized_error;
}