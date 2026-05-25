#include <memory>
#include <string>

#include <gtest/gtest.h>

#include "ERF_GTestSAMCommon.H"

using namespace sam_test;

namespace {

void expect_exact_state_match (const amrex::MultiFab& reference,
                               const amrex::MultiFab& production,
                               const std::string& case_name)
{
    const amrex::IntVect ngrow = reference.nGrowVect();

    EXPECT_EQ(component_mismatch_count(reference, production, RhoTheta_comp, ngrow), 0)
        << case_name << ": rho theta differs";
    EXPECT_EQ(component_mismatch_count(reference, production, RhoQ1_comp, ngrow), 0)
        << case_name << ": rho qv differs";
    EXPECT_EQ(component_mismatch_count(reference, production, RhoQ2_comp, ngrow), 0)
        << case_name << ": rho qc differs";
    EXPECT_EQ(component_mismatch_count(reference, production, RhoQ3_comp, ngrow), 0)
        << case_name << ": rho qi differs";
    EXPECT_EQ(component_mismatch_count(reference, production, RhoQ4_comp, ngrow), 0)
        << case_name << ": rho qr differs";
    EXPECT_EQ(component_mismatch_count(reference, production, RhoQ5_comp, ngrow), 0)
        << case_name << ": rho qs differs";
    EXPECT_EQ(component_mismatch_count(reference, production, RhoQ6_comp, ngrow), 0)
        << case_name << ": rho qg differs";
}

void expect_branch_hit (const amrex::Long count,
                        const char* branch_name)
{
    EXPECT_GT(count, 0L) << branch_name << " was not exercised";
}

template <typename FillStateFn>
SAMBFBBranchHits run_sam_bfb_case (const std::string& case_name,
                                   const amrex::Geometry& geom,
                                   SolverChoice sc,
                                   const int real_width,
                                   const bool use_detj,
                                   const bool compressed_detj,
                                   const FillStateFn& fill_state)
{
    amrex::MultiFab cons_reference;
    amrex::MultiFab cons_production;
    initialize_case(geom, fill_state, 1, cons_reference, cons_production);

    std::unique_ptr<amrex::MultiFab> z_phys_ref;
    std::unique_ptr<amrex::MultiFab> z_phys_prod;
    std::unique_ptr<amrex::MultiFab> detj_ref;
    std::unique_ptr<amrex::MultiFab> detj_prod;
    initialize_optional_detj(cons_reference, use_detj, compressed_detj, detj_ref);
    initialize_optional_detj(cons_reference, use_detj, compressed_detj, detj_prod);

    SAMCurrentBehaviorReference reference;
    SAM production;

    run_public_flow(reference, sc, geom, cons_reference, z_phys_ref, detj_ref, real_width);
    run_public_flow(production, sc, geom, cons_production, z_phys_prod, detj_prod, real_width);

    expect_exact_state_match(cons_reference, cons_production, case_name);

    // Qmoist accumulators are only initialized on the valid region in this harness.
    EXPECT_EQ(scalar_mismatch_count(*reference.Qmoist_Ptr(0), *production.Qmoist_Ptr(0)), 0)
        << case_name << ": rain accumulation differs";
    EXPECT_EQ(scalar_mismatch_count(*reference.Qmoist_Ptr(1), *production.Qmoist_Ptr(1)), 0)
        << case_name << ": snow accumulation differs";
    EXPECT_EQ(scalar_mismatch_count(*reference.Qmoist_Ptr(2), *production.Qmoist_Ptr(2)), 0)
        << case_name << ": graupel accumulation differs";

    return reference.BranchHits();
}

} // namespace

TEST(SAMBFB, ShocNoPrecipNoIcePublicFlowExact)
{
    const amrex::Geometry geom = make_geometry(3, 2, 2);
    const SolverChoice sc = make_solver_choice(MoistureType::SAM_NoPrecip_NoIce, true);
    const SAMBFBBranchHits hits = run_sam_bfb_case("ShocNoPrecipNoIcePublicFlowExact", geom, sc, 0, false, false,
                                                   fill_shoc_no_precip_no_ice_state_portable);

    expect_branch_hit(hits.cloud_noop, "cloud_noop");
    expect_branch_hit(hits.icefall_noop, "icefall_noop");
    expect_branch_hit(hits.precip_noop, "precip_noop");
    expect_branch_hit(hits.precipfall_noop, "precipfall_noop");
}

TEST(SAMBFB, NoIceRainMatrixPublicFlowExact)
{
    const amrex::Geometry geom = make_geometry(4, 3, 4);
    const SolverChoice sc = make_solver_choice(MoistureType::SAM_NoIce, false);
    const SAMBFBBranchHits hits = run_sam_bfb_case("NoIceRainMatrixPublicFlowExact", geom, sc, 1, false, false,
                                                   fill_no_ice_rain_matrix_state_portable);

    expect_branch_hit(hits.icefall_noop, "icefall_noop");
    expect_branch_hit(hits.precip_only, "precip_only");
    expect_branch_hit(hits.precip_cloud_and_precip, "precip_cloud_and_precip");
    expect_branch_hit(hits.precip_water_autoconversion_off, "precip_water_autoconversion_off");
    expect_branch_hit(hits.precip_rain_accretion_on, "precip_rain_accretion_on");
    expect_branch_hit(hits.precip_evaporation_off, "precip_evaporation_off");
    expect_branch_hit(hits.precip_evaporation_on, "precip_evaporation_on");
    expect_branch_hit(hits.precipfall_bottom_face, "precipfall_bottom_face");
    expect_branch_hit(hits.precipfall_interior_face, "precipfall_interior_face");
    expect_branch_hit(hits.precipfall_top_face, "precipfall_top_face");
    expect_branch_hit(hits.precipfall_zero_flux_below_threshold, "precipfall_zero_flux_below_threshold");
    expect_branch_hit(hits.precipfall_nonzero_rain_flux, "precipfall_nonzero_rain_flux");
    expect_branch_hit(hits.precipfall_surface_rain_accum, "precipfall_surface_rain_accum");
}

TEST(SAMBFB, FullSAMCloudMatrixPublicFlowExact)
{
    const amrex::Geometry geom = make_geometry(4, 2, 2);
    const SolverChoice sc = make_solver_choice(MoistureType::SAM, false);
    const SAMBFBBranchHits hits = run_sam_bfb_case("FullSAMCloudMatrixPublicFlowExact", geom, sc, 0, false, false,
                                                   fill_full_sam_cloud_matrix_state_portable);

    expect_branch_hit(hits.cloud_t_bgmin_or_less, "cloud_t_bgmin_or_less");
    expect_branch_hit(hits.cloud_t_between_bgmin_bgmax, "cloud_t_between_bgmin_bgmax");
    expect_branch_hit(hits.cloud_t_bgmax_or_greater, "cloud_t_bgmax_or_greater");
    expect_branch_hit(hits.cloud_qt_gt_qsat, "cloud_qt_gt_qsat");
    expect_branch_hit(hits.cloud_qt_le_qsat, "cloud_qt_le_qsat");
    expect_branch_hit(hits.cloud_evap_all_then_recheck, "cloud_evap_all_then_recheck");
    expect_branch_hit(hits.cloud_newton, "cloud_newton");
    expect_branch_hit(hits.cloud_condensate_limiter, "cloud_condensate_limiter");
}

TEST(SAMBFB, FullSAMPrecipMatrixPublicFlowExact)
{
    const amrex::Geometry geom = make_geometry(4, 3, 3);
    const SolverChoice sc = make_solver_choice(MoistureType::SAM, false);
    const SAMBFBBranchHits hits = run_sam_bfb_case("FullSAMPrecipMatrixPublicFlowExact", geom, sc, 0, false, false,
                                                   fill_full_sam_precip_matrix_state_portable);

    expect_branch_hit(hits.precip_cloud_only, "precip_cloud_only");
    expect_branch_hit(hits.precip_only, "precip_only");
    expect_branch_hit(hits.precip_cloud_and_precip, "precip_cloud_and_precip");
    expect_branch_hit(hits.precip_water_autoconversion_off, "precip_water_autoconversion_off");
    expect_branch_hit(hits.precip_water_autoconversion_on, "precip_water_autoconversion_on");
    expect_branch_hit(hits.precip_ice_autoconversion_off, "precip_ice_autoconversion_off");
    expect_branch_hit(hits.precip_ice_autoconversion_on, "precip_ice_autoconversion_on");
    expect_branch_hit(hits.precip_rain_accretion_off, "precip_rain_accretion_off");
    expect_branch_hit(hits.precip_rain_accretion_on, "precip_rain_accretion_on");
    expect_branch_hit(hits.precip_snow_accretion_off, "precip_snow_accretion_off");
    expect_branch_hit(hits.precip_snow_accretion_on, "precip_snow_accretion_on");
    expect_branch_hit(hits.precip_graupel_accretion_off, "precip_graupel_accretion_off");
    expect_branch_hit(hits.precip_graupel_accretion_on, "precip_graupel_accretion_on");
    expect_branch_hit(hits.precip_sink_limited_cloud_water, "precip_sink_limited_cloud_water");
    expect_branch_hit(hits.precip_sink_limited_cloud_ice, "precip_sink_limited_cloud_ice");
    expect_branch_hit(hits.precip_evaporation_off, "precip_evaporation_off");
    expect_branch_hit(hits.precip_evaporation_on, "precip_evaporation_on");
}

TEST(SAMBFB, FullSAMSedimentationDetJColumnPublicFlowExact)
{
    const amrex::Geometry geom = make_geometry(1, 1, 5);
    const SolverChoice sc = make_solver_choice(MoistureType::SAM, true);
    const SAMBFBBranchHits hits = run_sam_bfb_case("FullSAMSedimentationDetJColumnPublicFlowExact", geom, sc, 0, true, true,
                                                   fill_full_sam_sedimentation_column_state_portable);

    expect_branch_hit(hits.cloud_noop, "cloud_noop");
    expect_branch_hit(hits.icefall_bottom_face, "icefall_bottom_face");
    expect_branch_hit(hits.icefall_interior_face, "icefall_interior_face");
    expect_branch_hit(hits.icefall_top_face, "icefall_top_face");
    expect_branch_hit(hits.icefall_detj_present, "icefall_detj_present");
    expect_branch_hit(hits.precipfall_bottom_face, "precipfall_bottom_face");
    expect_branch_hit(hits.precipfall_interior_face, "precipfall_interior_face");
    expect_branch_hit(hits.precipfall_top_face, "precipfall_top_face");
    expect_branch_hit(hits.precipfall_detj_present, "precipfall_detj_present");
    expect_branch_hit(hits.precipfall_nonzero_rain_flux, "precipfall_nonzero_rain_flux");
    expect_branch_hit(hits.precipfall_nonzero_snow_flux, "precipfall_nonzero_snow_flux");
    expect_branch_hit(hits.precipfall_nonzero_graupel_flux, "precipfall_nonzero_graupel_flux");
    expect_branch_hit(hits.precipfall_surface_rain_accum, "precipfall_surface_rain_accum");
    expect_branch_hit(hits.precipfall_surface_snow_accum, "precipfall_surface_snow_accum");
    expect_branch_hit(hits.precipfall_surface_graupel_accum, "precipfall_surface_graupel_accum");
    expect_branch_hit(hits.precipfall_nonnegative_clipping, "precipfall_nonnegative_clipping");
}