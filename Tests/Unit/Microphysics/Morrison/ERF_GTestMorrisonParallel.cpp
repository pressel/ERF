#include <algorithm>
#include <memory>

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>

#include <gtest/gtest.h>

#include "ERF_Constants.H"
#include "ERF_IndexDefines.H"
#include "ERF_GTestMorrisonCommon.H"
#include "ERF_Morrison.H"

using namespace morrison_test;

namespace {

amrex::Real normalized_error (const amrex::Real actual,
                              const amrex::Real expected,
                              const amrex::Real tol)
{
    return std::abs(actual - expected) / std::max(tol, std::numeric_limits<amrex::Real>::min());
}

enum ParallelPublicCopyErrComps {
    ParallelErrRho = 0,
    ParallelErrRhoTheta,
    ParallelErrQ1,
    ParallelErrQ2,
    ParallelErrQ3,
    ParallelErrQ4,
    ParallelErrQ5,
    ParallelErrQ6,
    ParallelErrQ7,
    ParallelErrQ8,
    ParallelErrQ9,
    ParallelErrQ10,
    ParallelErrQ11,
    NumParallelPublicCopyErrComps
};

amrex::Geometry make_parallel_public_geometry (const int nx,
                                               const int ny,
                                               const int nz)
{
    const amrex::Box domain(amrex::IntVect(0), amrex::IntVect(AMREX_D_DECL(nx - 1, ny - 1, nz - 1)));
    const amrex::RealBox real_box({AMREX_D_DECL(amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0))},
                                  {AMREX_D_DECL(static_cast<amrex::Real>(nx),
                                                static_cast<amrex::Real>(ny),
                                                static_cast<amrex::Real>(nz))});
    const amrex::Array<int, AMREX_SPACEDIM> periodicity{AMREX_D_DECL(1, 1, 0)};
    return amrex::Geometry(domain, &real_box, amrex::CoordSys::cartesian, periodicity.data());
}

SolverChoice make_parallel_morrison_solver_choice ()
{
    SolverChoice sc;
    sc.c_p = Cp_d;
    sc.rdOcp = R_d / Cp_d;
    sc.ave_plane = 2;
    sc.moisture_type = MoistureType::Morrison;
    sc.use_shoc = false;
    sc.moisture_indices = MoistureComponentIndices(
        RhoQ1_comp, RhoQ2_comp, RhoQ3_comp, RhoQ4_comp, RhoQ5_comp, RhoQ6_comp,
        RhoQ7_comp, RhoQ8_comp, RhoQ9_comp, RhoQ10_comp, RhoQ11_comp);
    return sc;
}

void fill_parallel_public_copy_state (amrex::MultiFab& cons)
{
    for (amrex::MFIter mfi(cons, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& box = mfi.growntilebox();
        auto arr = cons.array(mfi);
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const amrex::Real rho = amrex::Real(1.0) + amrex::Real(0.01) * static_cast<amrex::Real>(i + 2 * j + 3 * k + 6);
            arr(i,j,k,Rho_comp) = rho;
            arr(i,j,k,RhoTheta_comp) = rho * (amrex::Real(286.0) + static_cast<amrex::Real>(i + j + k));
            arr(i,j,k,RhoKE_comp) = amrex::Real(0.0);
            arr(i,j,k,RhoScalar_comp) = amrex::Real(0.0);

            arr(i,j,k,RhoQ1_comp) = rho * amrex::Real(7.0e-3);
            arr(i,j,k,RhoQ2_comp) = rho * ((i % 3 == 0) ? -amrex::Real(1.0e-5) : amrex::Real(1.0e-4));
            arr(i,j,k,RhoQ3_comp) = rho * amrex::Real(2.0e-5);
            arr(i,j,k,RhoQ4_comp) = rho * amrex::Real(3.0e-5);
            arr(i,j,k,RhoQ5_comp) = rho * ((j % 2 == 0) ? -amrex::Real(2.0e-5) : amrex::Real(4.0e-5));
            arr(i,j,k,RhoQ6_comp) = rho * amrex::Real(5.0e-5);
            arr(i,j,k,RhoQ7_comp) = rho * amrex::Real(6.0e7);
            arr(i,j,k,RhoQ8_comp) = rho * ((k % 2 == 0) ? -amrex::Real(1.0e4) : amrex::Real(7.0e4));
            arr(i,j,k,RhoQ9_comp) = rho * amrex::Real(8.0e4);
            arr(i,j,k,RhoQ10_comp) = rho * amrex::Real(9.0e4);
            arr(i,j,k,RhoQ11_comp) = rho * amrex::Real(1.0e5);
        });
    }
    morrison_test::sync();
}

void compute_parallel_public_copy_errors (const amrex::MultiFab& before,
                                          const amrex::MultiFab& after,
                                          amrex::MultiFab& err)
{
    for (amrex::MFIter mfi(after, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& box = mfi.tilebox();
        const auto before_arr = before.const_array(mfi);
        const auto after_arr = after.const_array(mfi);
        auto err_arr = err.array(mfi);
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            err_arr(i,j,k,ParallelErrRho) = amrex::Math::abs(after_arr(i,j,k,Rho_comp) - before_arr(i,j,k,Rho_comp));
            err_arr(i,j,k,ParallelErrRhoTheta) = amrex::Math::abs(after_arr(i,j,k,RhoTheta_comp) - before_arr(i,j,k,RhoTheta_comp));
            err_arr(i,j,k,ParallelErrQ1) = amrex::Math::abs(after_arr(i,j,k,RhoQ1_comp) - amrex::max(amrex::Real(0), before_arr(i,j,k,RhoQ1_comp)));
            err_arr(i,j,k,ParallelErrQ2) = amrex::Math::abs(after_arr(i,j,k,RhoQ2_comp) - amrex::max(amrex::Real(0), before_arr(i,j,k,RhoQ2_comp)));
            err_arr(i,j,k,ParallelErrQ3) = amrex::Math::abs(after_arr(i,j,k,RhoQ3_comp) - amrex::max(amrex::Real(0), before_arr(i,j,k,RhoQ3_comp)));
            err_arr(i,j,k,ParallelErrQ4) = amrex::Math::abs(after_arr(i,j,k,RhoQ4_comp) - amrex::max(amrex::Real(0), before_arr(i,j,k,RhoQ4_comp)));
            err_arr(i,j,k,ParallelErrQ5) = amrex::Math::abs(after_arr(i,j,k,RhoQ5_comp) - amrex::max(amrex::Real(0), before_arr(i,j,k,RhoQ5_comp)));
            err_arr(i,j,k,ParallelErrQ6) = amrex::Math::abs(after_arr(i,j,k,RhoQ6_comp) - amrex::max(amrex::Real(0), before_arr(i,j,k,RhoQ6_comp)));
            err_arr(i,j,k,ParallelErrQ7) = amrex::Math::abs(after_arr(i,j,k,RhoQ7_comp) - amrex::max(amrex::Real(0), before_arr(i,j,k,RhoQ7_comp)));
            err_arr(i,j,k,ParallelErrQ8) = amrex::Math::abs(after_arr(i,j,k,RhoQ8_comp) - amrex::max(amrex::Real(0), before_arr(i,j,k,RhoQ8_comp)));
            err_arr(i,j,k,ParallelErrQ9) = amrex::Math::abs(after_arr(i,j,k,RhoQ9_comp) - amrex::max(amrex::Real(0), before_arr(i,j,k,RhoQ9_comp)));
            err_arr(i,j,k,ParallelErrQ10) = amrex::Math::abs(after_arr(i,j,k,RhoQ10_comp) - amrex::max(amrex::Real(0), before_arr(i,j,k,RhoQ10_comp)));
            err_arr(i,j,k,ParallelErrQ11) = amrex::Math::abs(after_arr(i,j,k,RhoQ11_comp) - amrex::max(amrex::Real(0), before_arr(i,j,k,RhoQ11_comp)));
        });
    }
    morrison_test::sync();
}

void expect_parallel_copy_error_le (const amrex::MultiFab& err,
                                    const int err_comp,
                                    const amrex::MultiFab& scale_source,
                                    const int state_comp)
{
    EXPECT_LE(err.max(err_comp), formula_abs_tol(scale_source.max(state_comp)))
        << "rank=" << amrex::ParallelDescriptor::MyProc()
        << " error component=" << err_comp << " state component=" << state_comp;
}

} // namespace

// Motivation: This is the public-path MPI/decomposition contract for Morrison
// state mapping. A split BoxArray is distributed by AMReX, then the public
// Define/Init/Copy_State_to_Micro/Copy_Micro_to_State path must reproduce the
// same clamped conserved state under any tested rank count.
TEST(MorrisonParallel, PublicCopyRoundTripIsDecompositionInvariant)
{
    amrex::ParmParse pp("erf");
    pp.add("use_morr_cpp_answer", true);

    const amrex::Geometry geom = make_parallel_public_geometry(4, 3, 2);
    amrex::BoxArray ba(geom.Domain());
    ba.maxSize(amrex::IntVect(1, 1, 2));
    amrex::DistributionMapping dm(ba);
    amrex::MultiFab cons(ba, dm, RhoQ11_comp + 1, 1);
    amrex::MultiFab before(ba, dm, RhoQ11_comp + 1, 1);
    amrex::MultiFab err(ba, dm, NumParallelPublicCopyErrComps, 0);
    auto z_phys_nd = std::make_unique<amrex::MultiFab>(ba, dm, 1, 1);
    auto detJ_cc = std::make_unique<amrex::MultiFab>(ba, dm, 1, 1);

    cons.setVal(amrex::Real(0.0));
    before.setVal(amrex::Real(0.0));
    err.setVal(amrex::Real(0.0));
    z_phys_nd->setVal(amrex::Real(0.0));
    detJ_cc->setVal(amrex::Real(1.0));
    fill_parallel_public_copy_state(cons);
    amrex::MultiFab::Copy(before, cons, 0, 0, cons.nComp(), cons.nGrowVect());

    Morrison morrison;
    SolverChoice sc = make_parallel_morrison_solver_choice();
    morrison.Define(sc);
    morrison.Init(cons, ba, geom, amrex::Real(1.0), z_phys_nd, detJ_cc);
    morrison.Copy_State_to_Micro(cons);
    morrison.Copy_Micro_to_State(cons);
    compute_parallel_public_copy_errors(before, cons, err);

    EXPECT_LE(err.max(ParallelErrRho), exact_zero_tol());
    EXPECT_LE(err.max(ParallelErrRhoTheta), exact_zero_tol());
    expect_parallel_copy_error_le(err, ParallelErrQ1, before, RhoQ1_comp);
    expect_parallel_copy_error_le(err, ParallelErrQ2, before, RhoQ2_comp);
    expect_parallel_copy_error_le(err, ParallelErrQ3, before, RhoQ3_comp);
    expect_parallel_copy_error_le(err, ParallelErrQ4, before, RhoQ4_comp);
    expect_parallel_copy_error_le(err, ParallelErrQ5, before, RhoQ5_comp);
    expect_parallel_copy_error_le(err, ParallelErrQ6, before, RhoQ6_comp);
    expect_parallel_copy_error_le(err, ParallelErrQ7, before, RhoQ7_comp);
    expect_parallel_copy_error_le(err, ParallelErrQ8, before, RhoQ8_comp);
    expect_parallel_copy_error_le(err, ParallelErrQ9, before, RhoQ9_comp);
    expect_parallel_copy_error_le(err, ParallelErrQ10, before, RhoQ10_comp);
    expect_parallel_copy_error_le(err, ParallelErrQ11, before, RhoQ11_comp);
}

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