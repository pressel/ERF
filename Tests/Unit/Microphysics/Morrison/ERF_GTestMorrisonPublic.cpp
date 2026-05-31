#include <memory>

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

amrex::Geometry make_public_geometry (const int nx,
                                      const int ny,
                                      const int nz)
{
    const amrex::Box domain(amrex::IntVect(0), amrex::IntVect(AMREX_D_DECL(nx - 1, ny - 1, nz - 1)));
    const amrex::RealBox real_box({AMREX_D_DECL(amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0))},
                                  {AMREX_D_DECL(static_cast<amrex::Real>(nx),
                                                static_cast<amrex::Real>(ny),
                                                static_cast<amrex::Real>(nz))});
    const amrex::Array<int, AMREX_SPACEDIM> periodicity{AMREX_D_DECL(1, 1, 1)};
    return amrex::Geometry(domain, &real_box, amrex::CoordSys::cartesian, periodicity.data());
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
    for (int comp = ErrQ1; comp <= ErrQ11; ++comp) {
        EXPECT_LE(err.max(comp), formula_abs_tol(amrex::Real(1.0e8))) << "component=" << comp;
    }
}