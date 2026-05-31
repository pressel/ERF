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

// Motivation: This helper-distribution test keeps the MPI harness honest for
// Morrison helper checks by splitting deterministic PSD cases across ranks and
// reducing the maximum error. It does not claim public-path decomposition
// invariance; separate public-path MPI tests must cover that contract.
TEST(MorrisonParallel, HelperDistributedExponentialDistributionSweepMatchesScalarReference)
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
            mass, number, test_case.coefficient, test_case.lambda_min, test_case.lambda_max);
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

// Motivation: Source-rate and sedimentation contracts must reduce cleanly when
// deterministic helper cases are partitioned over MPI ranks. This remains a
// helper-level MPI contract, not a public Morrison decomposition test.
TEST(MorrisonParallel, HelperDistributedSourceAndSedimentationBudgetsReduceConsistently)
{
    struct SourceCase {
        amrex::Real qc;
        amrex::Real qr;
        amrex::Real nc;
        amrex::Real rho;
        amrex::Real dt;
    };

    const std::array<SourceCase, 5> source_cases = {{{amrex::Real(0.5e-6), amrex::Real(2.0e-4), amrex::Real(8.0e7), amrex::Real(1.1), amrex::Real(1.0)},
                                                     {amrex::Real(1.2e-6), amrex::Real(0.5e-8), amrex::Real(8.0e7), amrex::Real(1.1), amrex::Real(1.0)},
                                                     {amrex::Real(1.0e-4), amrex::Real(2.0e-4), amrex::Real(8.0e7), amrex::Real(1.0), amrex::Real(2.0)},
                                                     {amrex::Real(2.0e-3), amrex::Real(8.0e-4), amrex::Real(1.0e6), amrex::Real(1.1), amrex::Real(1000.0)},
                                                     {amrex::Real(3.0e-4), amrex::Real(6.0e-4), amrex::Real(5.0e7), amrex::Real(0.9), amrex::Real(10.0)}}};
    const std::array<amrex::Real, 4> fallout_to_below = {{amrex::Real(2.0e-5), amrex::Real(4.0e-5),
                                                          amrex::Real(5.5e-5), amrex::Real(7.0e-5)}};
    const amrex::Real dz = amrex::Real(90.0);
    const amrex::Real rho = amrex::Real(1.05);
    const amrex::Real dt = amrex::Real(3.0);
    const int nstep = 3;

    const int rank = amrex::ParallelDescriptor::MyProc();
    const int nprocs = amrex::ParallelDescriptor::NProcs();
    long local_checked = 0;
    amrex::Real local_max_normalized_error = amrex::Real(0.0);
    amrex::Real local_column_integrated_delta = amrex::Real(0.0);

    for (int idx = rank; idx < static_cast<int>(source_cases.size()); idx += nprocs) {
        const SourceCase& test_case = source_cases[idx];
        const MorrisonAutoconversionRates autoconversion = morrison_compute_warm_rain_autoconversion(
            test_case.qc, test_case.nc, test_case.rho, test_case.dt, kCons29);
        const MorrisonAccretionRates accretion = morrison_compute_cloud_rain_accretion(
            test_case.qc, test_case.qr, test_case.nc);
        amrex::Real prc = autoconversion.prc;
        amrex::Real pra = accretion.pra;
        morrison_apply_cloud_water_sink_limiter(test_case.qc, test_case.dt, kQSmall, prc, pra);
        const amrex::Real liquid_residual = (-prc - pra) + (prc + pra);
        local_max_normalized_error = std::max(local_max_normalized_error,
            normalized_error(liquid_residual, amrex::Real(0.0), exact_zero_tol()));
        EXPECT_LE((prc + pra) * test_case.dt, test_case.qc + property_tol(2, test_case.qc));
        ++local_checked;
    }

    for (int idx = rank; idx < static_cast<int>(fallout_to_below.size()); idx += nprocs) {
        const amrex::Real fallout_from_above = (idx + 1 < static_cast<int>(fallout_to_below.size()))
            ? fallout_to_below[static_cast<std::size_t>(idx + 1)] : amrex::Real(0);
        const MorrisonSedimentationBudget budget = morrison_sedimentation_budget(
            fallout_from_above, fallout_to_below[static_cast<std::size_t>(idx)], dz, rho, dt, nstep);
        local_column_integrated_delta += budget.density_content_delta * dz;
        ++local_checked;
    }

    const MorrisonSurfacePrecipitationIncrement bottom_increment = morrison_surface_precipitation_increment(
        fallout_to_below[0], amrex::Real(0), amrex::Real(0), amrex::Real(0), amrex::Real(0), dt, nstep);

    long global_checked = local_checked;
    amrex::Real global_max_normalized_error = local_max_normalized_error;
    amrex::Real global_column_integrated_delta = local_column_integrated_delta;
    amrex::ParallelDescriptor::ReduceLongSum(global_checked);
    amrex::ParallelDescriptor::ReduceRealMax(global_max_normalized_error);
    amrex::ParallelDescriptor::ReduceRealSum(global_column_integrated_delta);

    const amrex::Real expected_column_loss = -bottom_increment.precipitation;
    EXPECT_EQ(global_checked, static_cast<long>(source_cases.size() + fallout_to_below.size()))
        << "rank=" << rank << " nprocs=" << nprocs << " local_checked=" << local_checked;
    EXPECT_LE(global_max_normalized_error, amrex::Real(1.0))
        << "rank=" << rank << " nprocs=" << nprocs;
    EXPECT_NEAR(global_column_integrated_delta, expected_column_loss,
                property_tol(static_cast<int>(fallout_to_below.size()), expected_column_loss))
        << "rank=" << rank << " nprocs=" << nprocs;
}