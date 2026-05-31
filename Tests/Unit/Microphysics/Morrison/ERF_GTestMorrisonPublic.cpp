#include <memory>
#include <cmath>

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>

#include <gtest/gtest.h>

#include "ERF_Constants.H"
#include "ERF_IndexDefines.H"
#include "ERF_Morrison.H"
#include "ERF_GTestMorrisonCommon.H"

using namespace morrison_test;

namespace {

enum PublicCopyErrComps {
    ErrRho = 0,
    ErrRhoTheta,
    ErrQ1,
    ErrQ2,
    ErrQ3,
    ErrQ4,
    ErrQ5,
    ErrQ6,
    ErrQ7,
    ErrQ8,
    ErrQ9,
    ErrQ10,
    ErrQ11,
    NumPublicCopyErrComps
};

struct MorrisonPublicColumnBudget {
    amrex::Real column_water_mass;
    amrex::Real surface_precipitation_mass;
};

amrex::Geometry make_public_geometry (const int nx,
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

SolverChoice make_morrison_solver_choice (const MoistureType moisture_type,
                                          const bool use_shoc)
{
    SolverChoice sc;
    sc.c_p = Cp_d;
    sc.rdOcp = R_d / Cp_d;
    sc.ave_plane = 2;
    sc.moisture_type = moisture_type;
    sc.use_shoc = use_shoc;
    sc.moisture_indices = MoistureComponentIndices(
        RhoQ1_comp, RhoQ2_comp, RhoQ3_comp, RhoQ4_comp, RhoQ5_comp, RhoQ6_comp,
        RhoQ7_comp, RhoQ8_comp, RhoQ9_comp, RhoQ10_comp, RhoQ11_comp);
    return sc;
}

void fill_morrison_public_state (amrex::MultiFab& cons)
{
    for (amrex::MFIter mfi(cons, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& box = mfi.tilebox();
        auto arr = cons.array(mfi);
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const amrex::Real rho = amrex::Real(1.0) + amrex::Real(0.01) * static_cast<amrex::Real>(i + 2 * j + 3 * k);
            arr(i,j,k,Rho_comp) = rho;
            arr(i,j,k,RhoTheta_comp) = rho * (amrex::Real(285.0) + static_cast<amrex::Real>(i + j + k));
            arr(i,j,k,RhoKE_comp) = amrex::Real(0.0);
            arr(i,j,k,RhoScalar_comp) = amrex::Real(0.0);

            arr(i,j,k,RhoQ1_comp) = rho * amrex::Real(8.0e-3);
            arr(i,j,k,RhoQ2_comp) = rho * ((i == 0) ? -amrex::Real(1.0e-5) : amrex::Real(1.0e-4));
            arr(i,j,k,RhoQ3_comp) = rho * amrex::Real(2.0e-5);
            arr(i,j,k,RhoQ4_comp) = rho * amrex::Real(3.0e-5);
            arr(i,j,k,RhoQ5_comp) = rho * ((j == 0) ? -amrex::Real(2.0e-5) : amrex::Real(4.0e-5));
            arr(i,j,k,RhoQ6_comp) = rho * amrex::Real(5.0e-5);
            arr(i,j,k,RhoQ7_comp) = rho * amrex::Real(6.0e7);
            arr(i,j,k,RhoQ8_comp) = rho * ((k == 0) ? -amrex::Real(1.0e4) : amrex::Real(7.0e4));
            arr(i,j,k,RhoQ9_comp) = rho * amrex::Real(8.0e4);
            arr(i,j,k,RhoQ10_comp) = rho * amrex::Real(9.0e4);
            arr(i,j,k,RhoQ11_comp) = rho * amrex::Real(1.0e5);
        });
    }
    morrison_test::sync();
}

void fill_morrison_warm_rain_advance_state (amrex::MultiFab& cons,
                                            const amrex::Real cloud_water)
{
    for (amrex::MFIter mfi(cons, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& box = mfi.growntilebox();
        auto arr = cons.array(mfi);
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const amrex::Real rho = amrex::Real(1.05) + amrex::Real(0.005) * static_cast<amrex::Real>(k + 4);
            const amrex::Real theta = amrex::Real(302.0) - amrex::Real(0.25) * static_cast<amrex::Real>(k);
            const amrex::Real rain = amrex::Real(1.2e-4) + amrex::Real(2.0e-5) * static_cast<amrex::Real>(k + 4);
            arr(i,j,k,Rho_comp) = rho;
            arr(i,j,k,RhoTheta_comp) = rho * theta;
            arr(i,j,k,RhoKE_comp) = amrex::Real(0.0);
            arr(i,j,k,RhoScalar_comp) = amrex::Real(0.0);

            arr(i,j,k,RhoQ1_comp) = rho * amrex::Real(2.6e-2);
            arr(i,j,k,RhoQ2_comp) = rho * cloud_water;
            arr(i,j,k,RhoQ3_comp) = amrex::Real(0.0);
            arr(i,j,k,RhoQ4_comp) = rho * rain;
            arr(i,j,k,RhoQ5_comp) = amrex::Real(0.0);
            arr(i,j,k,RhoQ6_comp) = amrex::Real(0.0);
            arr(i,j,k,RhoQ7_comp) = rho * amrex::Real(1.0e8);
            arr(i,j,k,RhoQ8_comp) = amrex::Real(0.0);
            arr(i,j,k,RhoQ9_comp) = rho * amrex::Real(8.0e6);
            arr(i,j,k,RhoQ10_comp) = amrex::Real(0.0);
            arr(i,j,k,RhoQ11_comp) = amrex::Real(0.0);
        });
    }
    morrison_test::sync();
}

void compute_public_copy_errors (const amrex::MultiFab& before,
                                 const amrex::MultiFab& after,
                                 amrex::MultiFab& err)
{
    for (amrex::MFIter mfi(after, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& box = mfi.tilebox();
        const auto before_arr = before.const_array(mfi);
        const auto after_arr = after.const_array(mfi);
        auto err_arr = err.array(mfi);
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            err_arr(i,j,k,ErrRho) = amrex::Math::abs(after_arr(i,j,k,Rho_comp) - before_arr(i,j,k,Rho_comp));
            err_arr(i,j,k,ErrRhoTheta) = amrex::Math::abs(after_arr(i,j,k,RhoTheta_comp) - before_arr(i,j,k,RhoTheta_comp));

            err_arr(i,j,k,ErrQ1) = amrex::Math::abs(after_arr(i,j,k,RhoQ1_comp) - amrex::max(amrex::Real(0), before_arr(i,j,k,RhoQ1_comp)));
            err_arr(i,j,k,ErrQ2) = amrex::Math::abs(after_arr(i,j,k,RhoQ2_comp) - amrex::max(amrex::Real(0), before_arr(i,j,k,RhoQ2_comp)));
            err_arr(i,j,k,ErrQ3) = amrex::Math::abs(after_arr(i,j,k,RhoQ3_comp) - amrex::max(amrex::Real(0), before_arr(i,j,k,RhoQ3_comp)));
            err_arr(i,j,k,ErrQ4) = amrex::Math::abs(after_arr(i,j,k,RhoQ4_comp) - amrex::max(amrex::Real(0), before_arr(i,j,k,RhoQ4_comp)));
            err_arr(i,j,k,ErrQ5) = amrex::Math::abs(after_arr(i,j,k,RhoQ5_comp) - amrex::max(amrex::Real(0), before_arr(i,j,k,RhoQ5_comp)));
            err_arr(i,j,k,ErrQ6) = amrex::Math::abs(after_arr(i,j,k,RhoQ6_comp) - amrex::max(amrex::Real(0), before_arr(i,j,k,RhoQ6_comp)));
            err_arr(i,j,k,ErrQ7) = amrex::Math::abs(after_arr(i,j,k,RhoQ7_comp) - amrex::max(amrex::Real(0), before_arr(i,j,k,RhoQ7_comp)));
            err_arr(i,j,k,ErrQ8) = amrex::Math::abs(after_arr(i,j,k,RhoQ8_comp) - amrex::max(amrex::Real(0), before_arr(i,j,k,RhoQ8_comp)));
            err_arr(i,j,k,ErrQ9) = amrex::Math::abs(after_arr(i,j,k,RhoQ9_comp) - amrex::max(amrex::Real(0), before_arr(i,j,k,RhoQ9_comp)));
            err_arr(i,j,k,ErrQ10) = amrex::Math::abs(after_arr(i,j,k,RhoQ10_comp) - amrex::max(amrex::Real(0), before_arr(i,j,k,RhoQ10_comp)));
            err_arr(i,j,k,ErrQ11) = amrex::Math::abs(after_arr(i,j,k,RhoQ11_comp) - amrex::max(amrex::Real(0), before_arr(i,j,k,RhoQ11_comp)));
        });
    }
    morrison_test::sync();
}

void expect_copy_error_le (const amrex::MultiFab& err,
                           const int err_comp,
                           const amrex::MultiFab& scale_source,
                           const int state_comp)
{
    EXPECT_LE(err.max(err_comp), formula_abs_tol(scale_source.max(state_comp)))
        << "error component=" << err_comp << " state component=" << state_comp;
}

void run_morrison_public_flow (Morrison& morrison,
                               SolverChoice& sc,
                               const amrex::Geometry& geom,
                               amrex::MultiFab& cons,
                               std::unique_ptr<amrex::MultiFab>& z_phys_nd,
                               std::unique_ptr<amrex::MultiFab>& detJ_cc,
                               const amrex::Real dt)
{
    morrison.Define(sc);
    morrison.Set_dzmin(geom.CellSize(2));
    morrison.Init(cons, cons.boxArray(), geom, dt, z_phys_nd, detJ_cc);
    morrison.Copy_State_to_Micro(cons);
    morrison.Advance(dt, sc);
    morrison.Copy_Micro_to_State(cons);
    morrison_test::sync();
}

MorrisonPublicColumnBudget compute_morrison_public_column_budget (const amrex::Geometry& geom,
                                                                  const amrex::MultiFab& cons,
                                                                  const amrex::MultiFab& rain_accum)
{
    const amrex::Real cell_volume = geom.CellSize(0) * geom.CellSize(1) * geom.CellSize(2);
    const amrex::Real cell_area = geom.CellSize(0) * geom.CellSize(1);
    const amrex::Real column_water = cell_volume * (cons.sum(RhoQ1_comp) + cons.sum(RhoQ2_comp) +
                                                    cons.sum(RhoQ3_comp) + cons.sum(RhoQ4_comp) +
                                                    cons.sum(RhoQ5_comp) + cons.sum(RhoQ6_comp));
    return MorrisonPublicColumnBudget{column_water, cell_area * rain_accum.sum(0)};
}

void expect_finite_nonnegative_component (const amrex::MultiFab& cons,
                                          const int comp)
{
    const amrex::Real min_value = cons.min(comp);
    const amrex::Real max_value = cons.max(comp);
    EXPECT_TRUE(std::isfinite(static_cast<double>(min_value))) << "component=" << comp;
    EXPECT_TRUE(std::isfinite(static_cast<double>(max_value))) << "component=" << comp;
    EXPECT_GE(min_value, -formula_abs_tol(max_value)) << "component=" << comp;
}

void expect_morrison_public_state_finite_and_nonnegative (const amrex::MultiFab& cons)
{
    expect_finite_nonnegative_component(cons, Rho_comp);
    expect_finite_nonnegative_component(cons, RhoTheta_comp);
    expect_finite_nonnegative_component(cons, RhoQ1_comp);
    expect_finite_nonnegative_component(cons, RhoQ2_comp);
    expect_finite_nonnegative_component(cons, RhoQ3_comp);
    expect_finite_nonnegative_component(cons, RhoQ4_comp);
    expect_finite_nonnegative_component(cons, RhoQ5_comp);
    expect_finite_nonnegative_component(cons, RhoQ6_comp);
    expect_finite_nonnegative_component(cons, RhoQ7_comp);
    expect_finite_nonnegative_component(cons, RhoQ8_comp);
    expect_finite_nonnegative_component(cons, RhoQ9_comp);
    expect_finite_nonnegative_component(cons, RhoQ10_comp);
    expect_finite_nonnegative_component(cons, RhoQ11_comp);
}

amrex::Real public_surface_budget_tol (const amrex::Real surface_precipitation_mass)
{
    return std::max(property_tol(128, surface_precipitation_mass),
                    amrex::Real(5.0e-2) * std::abs(surface_precipitation_mass));
}

} // namespace

// Motivation: This public-path test exercises Morrison's MultiFab state mapping
// without running the full source/sedimentation kernel. Copy-in clamps negative
// moist species and number concentrations to zero; copy-out writes those mapped
// values back to conserved ERF state while preserving rho and rho-theta.
TEST(MorrisonPublic, CopyStateRoundTripPreservesMappedStateAndNonnegativity)
{
    amrex::ParmParse pp("erf");
    pp.add("use_morr_cpp_answer", true);

    const amrex::Geometry geom = make_public_geometry(3, 2, 2);
    amrex::BoxArray ba(geom.Domain());
    amrex::DistributionMapping dm(ba);
    amrex::MultiFab cons(ba, dm, RhoQ11_comp + 1, 1);
    amrex::MultiFab before(ba, dm, RhoQ11_comp + 1, 1);
    amrex::MultiFab err(ba, dm, NumPublicCopyErrComps, 0);
    auto z_phys_nd = std::make_unique<amrex::MultiFab>(ba, dm, 1, 1);
    auto detJ_cc = std::make_unique<amrex::MultiFab>(ba, dm, 1, 1);

    cons.setVal(amrex::Real(0.0));
    before.setVal(amrex::Real(0.0));
    err.setVal(amrex::Real(0.0));
    z_phys_nd->setVal(amrex::Real(0.0));
    detJ_cc->setVal(amrex::Real(1.0));
    fill_morrison_public_state(cons);
    amrex::MultiFab::Copy(before, cons, 0, 0, cons.nComp(), cons.nGrowVect());

    SolverChoice sc;
    sc.c_p = Cp_d;
    sc.rdOcp = R_d / Cp_d;
    sc.use_shoc = false;

    Morrison morrison;
    morrison.Define(sc);
    morrison.Init(cons, ba, geom, amrex::Real(1.0), z_phys_nd, detJ_cc);
    morrison.Copy_State_to_Micro(cons);
    morrison.Copy_Micro_to_State(cons);
    compute_public_copy_errors(before, cons, err);

    EXPECT_LE(err.max(ErrRho), exact_zero_tol());
    EXPECT_LE(err.max(ErrRhoTheta), exact_zero_tol());
    expect_copy_error_le(err, ErrQ1, before, RhoQ1_comp);
    expect_copy_error_le(err, ErrQ2, before, RhoQ2_comp);
    expect_copy_error_le(err, ErrQ3, before, RhoQ3_comp);
    expect_copy_error_le(err, ErrQ4, before, RhoQ4_comp);
    expect_copy_error_le(err, ErrQ5, before, RhoQ5_comp);
    expect_copy_error_le(err, ErrQ6, before, RhoQ6_comp);
    expect_copy_error_le(err, ErrQ7, before, RhoQ7_comp);
    expect_copy_error_le(err, ErrQ8, before, RhoQ8_comp);
    expect_copy_error_le(err, ErrQ9, before, RhoQ9_comp);
    expect_copy_error_le(err, ErrQ10, before, RhoQ10_comp);
    expect_copy_error_le(err, ErrQ11, before, RhoQ11_comp);
}

// Motivation: The public copy test proves mapping only. This test calls the
// actual Morrison C++ `Advance` path in a warm no-ice column, then checks the
// production copy-out state after warm-rain conversion and sedimentation.
TEST(MorrisonPublic, WarmNoIceAdvanceProducesFiniteStateAndSurfacePrecipitation)
{
    amrex::ParmParse pp("erf");
    pp.add("use_morr_cpp_answer", true);

    const amrex::Real dt = amrex::Real(0.25);
    const amrex::Geometry geom = make_public_geometry(1, 1, 4);
    amrex::BoxArray ba(geom.Domain());
    amrex::DistributionMapping dm(ba);
    amrex::MultiFab cons(ba, dm, RhoQ11_comp + 1, 3);
    amrex::MultiFab before(ba, dm, RhoQ11_comp + 1, 3);
    std::unique_ptr<amrex::MultiFab> z_phys_nd;
    std::unique_ptr<amrex::MultiFab> detJ_cc;

    cons.setVal(amrex::Real(0.0));
    before.setVal(amrex::Real(0.0));
    fill_morrison_warm_rain_advance_state(cons, amrex::Real(4.0e-4));
    amrex::MultiFab::Copy(before, cons, 0, 0, cons.nComp(), cons.nGrowVect());

    Morrison morrison;
    SolverChoice sc = make_morrison_solver_choice(MoistureType::Morrison_NoIce, true);
    run_morrison_public_flow(morrison, sc, geom, cons, z_phys_nd, detJ_cc, dt);

    const MorrisonPublicColumnBudget after_budget = compute_morrison_public_column_budget(geom, cons, *morrison.Qmoist_Ptr(0));

    expect_morrison_public_state_finite_and_nonnegative(cons);
    EXPECT_GT(after_budget.surface_precipitation_mass, amrex::Real(0.0));
    EXPECT_LT(cons.sum(RhoQ2_comp), before.sum(RhoQ2_comp));
}

// Motivation: In a warm rain-only column with no cloud source terms, the public
// C++ sedimentation path should remove column rain and report the corresponding
// surface precipitation accumulator within current production-path tolerance.
TEST(MorrisonPublic, RainOnlyAdvanceTracksColumnWaterLossWithSurfacePrecipitation)
{
    amrex::ParmParse pp("erf");
    pp.add("use_morr_cpp_answer", true);

    const amrex::Real dt = amrex::Real(0.25);
    const amrex::Geometry geom = make_public_geometry(1, 1, 4);
    amrex::BoxArray ba(geom.Domain());
    amrex::DistributionMapping dm(ba);
    amrex::MultiFab cons(ba, dm, RhoQ11_comp + 1, 3);
    amrex::MultiFab before(ba, dm, RhoQ11_comp + 1, 3);
    amrex::MultiFab zero_accum(ba, dm, 1, 0);
    std::unique_ptr<amrex::MultiFab> z_phys_nd;
    std::unique_ptr<amrex::MultiFab> detJ_cc;

    cons.setVal(amrex::Real(0.0));
    before.setVal(amrex::Real(0.0));
    zero_accum.setVal(amrex::Real(0.0));
    fill_morrison_warm_rain_advance_state(cons, amrex::Real(0.0));
    amrex::MultiFab::Copy(before, cons, 0, 0, cons.nComp(), cons.nGrowVect());

    Morrison morrison;
    SolverChoice sc = make_morrison_solver_choice(MoistureType::Morrison_NoIce, true);
    run_morrison_public_flow(morrison, sc, geom, cons, z_phys_nd, detJ_cc, dt);

    const MorrisonPublicColumnBudget before_budget = compute_morrison_public_column_budget(geom, before, zero_accum);
    const MorrisonPublicColumnBudget after_budget = compute_morrison_public_column_budget(geom, cons, *morrison.Qmoist_Ptr(0));
        const amrex::Real column_water_loss = before_budget.column_water_mass - after_budget.column_water_mass;

    expect_morrison_public_state_finite_and_nonnegative(cons);
    EXPECT_GT(after_budget.surface_precipitation_mass, amrex::Real(0.0));
    EXPECT_LT(cons.sum(RhoQ4_comp), before.sum(RhoQ4_comp));
        EXPECT_GT(column_water_loss, amrex::Real(0.0));
        EXPECT_NEAR(column_water_loss, after_budget.surface_precipitation_mass,
            public_surface_budget_tol(after_budget.surface_precipitation_mass));
}