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
    EXPECT_EQ(component_mismatch_count(reference, production, RhoTheta_comp), 0)
        << case_name << ": rho theta differs";
    EXPECT_EQ(component_mismatch_count(reference, production, RhoQ1_comp), 0)
        << case_name << ": rho qv differs";
    EXPECT_EQ(component_mismatch_count(reference, production, RhoQ2_comp), 0)
        << case_name << ": rho qc differs";
    EXPECT_EQ(component_mismatch_count(reference, production, RhoQ3_comp), 0)
        << case_name << ": rho qi differs";
    EXPECT_EQ(component_mismatch_count(reference, production, RhoQ4_comp), 0)
        << case_name << ": rho qr differs";
    EXPECT_EQ(component_mismatch_count(reference, production, RhoQ5_comp), 0)
        << case_name << ": rho qs differs";
    EXPECT_EQ(component_mismatch_count(reference, production, RhoQ6_comp), 0)
        << case_name << ": rho qg differs";
}

template <typename FillStateFn>
void run_sam_bfb_case (const std::string& case_name,
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

    EXPECT_EQ(scalar_mismatch_count(*reference.Qmoist_Ptr(0), *production.Qmoist_Ptr(0)), 0)
        << case_name << ": rain accumulation differs";
    EXPECT_EQ(scalar_mismatch_count(*reference.Qmoist_Ptr(1), *production.Qmoist_Ptr(1)), 0)
        << case_name << ": snow accumulation differs";
    EXPECT_EQ(scalar_mismatch_count(*reference.Qmoist_Ptr(2), *production.Qmoist_Ptr(2)), 0)
        << case_name << ": graupel accumulation differs";
}

} // namespace

TEST(SAMBFB, ShocNoPrecipNoIcePublicFlowExact)
{
    // Covers:
    //   moisture_type = SAM_NoPrecip_NoIce
    //   use_shoc = true, Cloud no-op
    //   IceFall no-op
    //   Precip no-op
    //   PrecipFall no-op
    //   qn + qp == 0 no-op on some cells
    //   detJ absent
    //   real_width = 0
    //   multi-cell horizontal case
    const amrex::Geometry geom = make_geometry(3, 2, 2);
    const SolverChoice sc = make_solver_choice(MoistureType::SAM_NoPrecip_NoIce, true);
    run_sam_bfb_case("ShocNoPrecipNoIcePublicFlowExact", geom, sc, 0, false, false,
                     fill_shoc_no_precip_no_ice_state_portable);
}

TEST(SAMBFB, NoIceRainMatrixPublicFlowExact)
{
    // Covers:
    //   moisture_type = SAM_NoIce
    //   use_shoc = false, Cloud active
    //   cloud only
    //   precip only
    //   cloud + precip
    //   water autoconversion off / on
    //   rain accretion off / on
    //   evaporation off / on
    //   evaporation species-limited to rain
    //   IceFall no-op
    //   PrecipFall active
    //   bottom face / interior face / top face
    //   zero flux below threshold
    //   nonzero rain flux
    //   surface rain accumulation
    //   real_width > 0
    //   detJ absent
    //   multi-cell horizontal case
    const amrex::Geometry geom = make_geometry(4, 3, 4);
    const SolverChoice sc = make_solver_choice(MoistureType::SAM_NoIce, false);
    run_sam_bfb_case("NoIceRainMatrixPublicFlowExact", geom, sc, 1, false, false,
                     fill_no_ice_rain_matrix_state_portable);
}

TEST(SAMBFB, FullSAMCloudMatrixPublicFlowExact)
{
    // Covers:
    //   moisture_type = SAM
    //   use_shoc = false, Cloud active
    //   T <= tbgmin
    //   tbgmin < T < tbgmax
    //   T >= tbgmax
    //   qt > qsat
    //   qt <= qsat
    //   evaporate-all-then-recheck path intent
    //   Newton path intent
    //   condensate limiter path intent
    //   detJ absent
    //   real_width = 0
    const amrex::Geometry geom = make_geometry(4, 2, 2);
    const SolverChoice sc = make_solver_choice(MoistureType::SAM, false);
    run_sam_bfb_case("FullSAMCloudMatrixPublicFlowExact", geom, sc, 0, false, false,
                     fill_full_sam_cloud_matrix_state_portable);
}

TEST(SAMBFB, FullSAMPrecipMatrixPublicFlowExact)
{
    // Covers:
    //   moisture_type = SAM
    //   use_shoc = false, Cloud active
    //   cloud only
    //   precip only
    //   cloud + precip
    //   water autoconversion off / on
    //   ice autoconversion off / on
    //   rain accretion off / on
    //   snow accretion off / on
    //   graupel accretion off / on
    //   sink-limited cloud water intent
    //   sink-limited cloud ice intent
    //   evaporation off / on
    //   evaporation species-limited
    //   detJ absent
    //   real_width = 0
    const amrex::Geometry geom = make_geometry(4, 3, 3);
    const SolverChoice sc = make_solver_choice(MoistureType::SAM, false);
    run_sam_bfb_case("FullSAMPrecipMatrixPublicFlowExact", geom, sc, 0, false, false,
                     fill_full_sam_precip_matrix_state_portable);
}

TEST(SAMBFB, FullSAMSedimentationDetJColumnPublicFlowExact)
{
    // Covers:
    //   moisture_type = SAM
    //   use_shoc = true, Cloud no-op
    //   IceFall active
    //   PrecipFall active
    //   bottom face / interior face / top face
    //   zero flux below threshold
    //   nonzero rain flux
    //   nonzero snow flux
    //   nonzero graupel flux
    //   surface rain accumulation
    //   surface snow accumulation
    //   surface graupel accumulation
    //   nonnegative clipping intent
    //   detJ present
    //   real_width = 0
    //   one-column vertical case
    const amrex::Geometry geom = make_geometry(1, 1, 5);
    const SolverChoice sc = make_solver_choice(MoistureType::SAM, true);
    run_sam_bfb_case("FullSAMSedimentationDetJColumnPublicFlowExact", geom, sc, 0, true, true,
                     fill_full_sam_sedimentation_column_state_portable);
}