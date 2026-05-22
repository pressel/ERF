#include <gtest/gtest.h>

#include "GTestKesslerCommon.H"

using namespace kessler_bfb_test;

namespace {

void expect_no_mismatch (const std::optional<BitMismatch>& mismatch,
                         const std::string& label)
{
    ASSERT_FALSE(mismatch.has_value())
        << label
        << " first mismatch at fab=" << mismatch->fab_index
        << " cell=(" << mismatch->i << "," << mismatch->j << "," << mismatch->k << ")"
        << " comp=" << mismatch->comp
        << " expected_bits=" << format_bits(mismatch->expected_bits)
        << " actual_bits=" << format_bits(mismatch->actual_bits)
        << " expected_value=" << mismatch->expected_value
        << " actual_value=" << mismatch->actual_value;
}

void expect_micro_state_equal (const Kessler& expected,
                               const Kessler& actual)
{
    for (int mic_var = MicVar_Kess::rho; mic_var <= MicVar_Kess::rain_accum; ++mic_var) {
        SCOPED_TRACE(micro_var_name(mic_var));
        expect_no_mismatch(find_first_multifab_mismatch(KesslerBFBTestAccessor::MicroVar(expected, mic_var),
                                                        KesslerBFBTestAccessor::MicroVar(actual, mic_var)),
                           micro_var_name(mic_var));
    }
}

void expect_conserved_state_equal (const amrex::MultiFab& expected,
                                   const amrex::MultiFab& actual)
{
    expect_no_mismatch(find_first_multifab_mismatch(expected, actual), "conserved_state");
}

void run_legacy_and_refactored_and_compare (const FixtureOptions& options)
{
    Harness legacy(options);
    Harness refactored(options);

    KesslerBFBTestAccessor::AdvanceLegacy(legacy.model, legacy.solver_choice);
    KesslerBFBTestAccessor::AdvanceRefactored(refactored.model, refactored.solver_choice);

    expect_micro_state_equal(legacy.model, refactored.model);

    legacy.model.Copy_Micro_to_State(legacy.cons);
    refactored.model.Copy_Micro_to_State(refactored.cons);

    expect_conserved_state_equal(legacy.cons, refactored.cons);
}

} // namespace

// Branch inventory for this BFB harness:
// - RainDetJCropDoCond covers MoistureType::Kessler, do_cond=true, detJ present,
//   condensation, cloud evaporation, rain evaporation, autoconversion/accretion,
//   qcc <= qcw0, qcc > qcw0, nonnegative clipping of qv/qc/qp, k==k_lo,
//   k==k_hi+1, interior face averaging, sedimentation substeps, fz threshold to
//   zero, dq_sed threshold to zero, bottom rain accumulation, and lateral
//   real-width crop behavior.
// - RainNoDetJDoCondFalse covers MoistureType::Kessler, do_cond=false, detJ
//   absent, and keeps the same sedimentation surface while condensation/cloud
//   evaporation remain disabled.
// - RainQcCapAndSurfacePrecip forces the dq_cloud_to_rain min cap against qc
//   and checks that surface rain accumulation increases.
// - RainQpCap forces the dq_rain_to_vapor min cap against qp. This dedicated
//   fixture evaporates the available rain before sedimentation, so the surface
//   precipitation assertion stays on the qc-cap fixture instead.
// - NoRainDoCond covers MoistureType::Kessler_NoRain with condensation and cloud
//   evaporation branches.
// - NoRainShocNoOp covers MoistureType::Kessler_NoRain with do_cond=false early
//   return.
// - Not covered because they are not reachable under the current positive-qsat
//   warm-rain contract: qsat <= 0, the condensation std::min cap against qv,
//   and qp_avg negative clipping after qp is pre-clipped nonnegative.

TEST(KesslerAdvanceBFB, RainDetJCropDoCondLegacyMatchesRefactored)
{
    // This is a BFB refactor guard. It preserves current behavior only;
    // it does not assert physical correctness.
    const FixtureOptions options{
        MoistureType::Kessler,
        FixtureFlavor::BranchQuilt,
        false,
        true,
        true,
        1,
        4,
        4,
        4,
        amrex::Real(3.0),
        amrex::Real(0.5)};

    run_legacy_and_refactored_and_compare(options);
}

TEST(KesslerAdvanceBFB, RainNoDetJDoCondFalseLegacyMatchesRefactored)
{
    // This is a BFB refactor guard. It preserves current behavior only;
    // it does not assert physical correctness.
    const FixtureOptions options{
        MoistureType::Kessler,
        FixtureFlavor::BranchQuilt,
        true,
        false,
        false,
        0,
        4,
        4,
        4,
        amrex::Real(3.0),
        amrex::Real(0.5)};

    run_legacy_and_refactored_and_compare(options);
}

TEST(KesslerAdvanceBFB, NoRainDoCondLegacyMatchesRefactored)
{
    // This is a BFB refactor guard. It preserves current behavior only;
    // it does not assert physical correctness.
    const FixtureOptions options{
        MoistureType::Kessler_NoRain,
        FixtureFlavor::BranchQuilt,
        false,
        false,
        false,
        0,
        4,
        4,
        3,
        amrex::Real(1.5),
        amrex::Real(0.5)};

    run_legacy_and_refactored_and_compare(options);
}

TEST(KesslerAdvanceBFB, NoRainShocNoOpLegacyMatchesRefactored)
{
    // This is a BFB refactor guard. It preserves current behavior only;
    // it does not assert physical correctness.
    const FixtureOptions options{
        MoistureType::Kessler_NoRain,
        FixtureFlavor::BranchQuilt,
        true,
        false,
        false,
        0,
        4,
        4,
        3,
        amrex::Real(1.5),
        amrex::Real(0.5)};

    run_legacy_and_refactored_and_compare(options);
}

TEST(KesslerAdvanceBFB, RainQcCapAndSurfacePrecipLegacyMatchesRefactored)
{
    // This is a BFB refactor guard. It preserves current behavior only;
    // it does not assert physical correctness.
    const FixtureOptions options{
        MoistureType::Kessler,
        FixtureFlavor::ForceQcCap,
        false,
        false,
        false,
        0,
        4,
        4,
        4,
        amrex::Real(3.0),
        amrex::Real(0.5)};

    Harness legacy(options);
    Harness refactored(options);

    EXPECT_FALSE(condensation_qv_cap_is_reachable());
    EXPECT_TRUE(has_cloud_to_rain_qc_cap(legacy.model, options.dt));

    const amrex::Real rain_before = max_bottom_rain_accum(legacy.model);

    KesslerBFBTestAccessor::AdvanceLegacy(legacy.model, legacy.solver_choice);
    KesslerBFBTestAccessor::AdvanceRefactored(refactored.model, refactored.solver_choice);

    expect_micro_state_equal(legacy.model, refactored.model);

    const amrex::Real rain_after = max_bottom_rain_accum(legacy.model);
    EXPECT_GT(rain_after, rain_before);

    legacy.model.Copy_Micro_to_State(legacy.cons);
    refactored.model.Copy_Micro_to_State(refactored.cons);

    expect_conserved_state_equal(legacy.cons, refactored.cons);
}

TEST(KesslerAdvanceBFB, RainQpCapLegacyMatchesRefactored)
{
    // This is a BFB refactor guard. It preserves current behavior only;
    // it does not assert physical correctness.
    const FixtureOptions options{
        MoistureType::Kessler,
        FixtureFlavor::ForceQpCap,
        false,
        false,
        false,
        0,
        4,
        4,
        4,
        amrex::Real(3.0),
        amrex::Real(0.5)};

    Harness legacy(options);
    Harness refactored(options);

    EXPECT_FALSE(condensation_qv_cap_is_reachable());
    EXPECT_TRUE(has_rain_to_vapor_qp_cap(legacy.model, options.dt));

    KesslerBFBTestAccessor::AdvanceLegacy(legacy.model, legacy.solver_choice);
    KesslerBFBTestAccessor::AdvanceRefactored(refactored.model, refactored.solver_choice);

    expect_micro_state_equal(legacy.model, refactored.model);

    legacy.model.Copy_Micro_to_State(legacy.cons);
    refactored.model.Copy_Micro_to_State(refactored.cons);

    expect_conserved_state_equal(legacy.cons, refactored.cons);
}
