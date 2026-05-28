#include <array>
#include <string>

#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Gpu.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_RealBox.H>

#include <gtest/gtest.h>

#include <ERF_SAM.H>

#include "ERF_GTestSAMCommon.H"

using namespace sam_test;

namespace {

SolverChoice make_sam_solver_choice (const MoistureType moisture_type = MoistureType::SAM)
{
    SolverChoice sc{};
    sc.c_p = Cp_d;
    sc.rdOcp = kRdOcp;
    sc.ave_plane = 2;
    sc.moisture_type = moisture_type;
    sc.use_shoc = false;
    return sc;
}

void fill_uniform_conserved_state (amrex::MultiFab& cons,
                                   const amrex::Real rho,
                                   const amrex::Real theta,
                                   const amrex::Real qv,
                                   const amrex::Real qcl,
                                   const amrex::Real qci,
                                   const amrex::Real qpr,
                                   const amrex::Real qps,
                                   const amrex::Real qpg)
{
    cons.setVal(amrex::Real(0.0));

    for (amrex::MFIter mfi(cons, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto arr = cons.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            arr(i,j,k,Rho_comp) = rho;
            arr(i,j,k,RhoTheta_comp) = rho * theta;
            arr(i,j,k,RhoKE_comp) = amrex::Real(0.0);
            arr(i,j,k,RhoScalar_comp) = amrex::Real(0.0);
            arr(i,j,k,RhoQ1_comp) = rho * qv;
            arr(i,j,k,RhoQ2_comp) = rho * qcl;
            arr(i,j,k,RhoQ3_comp) = rho * qci;
            arr(i,j,k,RhoQ4_comp) = rho * qpr;
            arr(i,j,k,RhoQ5_comp) = rho * qps;
            arr(i,j,k,RhoQ6_comp) = rho * qpg;
        });
    }

    amrex::Gpu::streamSynchronize();
}

void expect_coefficient_row_near (const SAMCoefficientRow& actual,
                                  const SAMCoefficientRow& expected,
                                  const int k,
                                  const int rank)
{
    const std::string prefix = "rank=" + std::to_string(rank) + " k=" + std::to_string(k) + " ";
    EXPECT_NEAR(actual.accrrc, expected.accrrc, pow_sqrt_tol(expected.accrrc)) << prefix << "accrrc";
    EXPECT_NEAR(actual.accrsi, expected.accrsi, pow_sqrt_tol(expected.accrsi)) << prefix << "accrsi";
    EXPECT_NEAR(actual.accrsc, expected.accrsc, pow_sqrt_tol(expected.accrsc)) << prefix << "accrsc";
    EXPECT_NEAR(actual.coefice, expected.coefice, pow_sqrt_tol(expected.coefice)) << prefix << "coefice";
    EXPECT_NEAR(actual.evaps1, expected.evaps1, pow_sqrt_tol(expected.evaps1)) << prefix << "evaps1";
    EXPECT_NEAR(actual.evaps2, expected.evaps2, pow_sqrt_tol(expected.evaps2)) << prefix << "evaps2";
    EXPECT_NEAR(actual.accrgi, expected.accrgi, pow_sqrt_tol(expected.accrgi)) << prefix << "accrgi";
    EXPECT_NEAR(actual.accrgc, expected.accrgc, pow_sqrt_tol(expected.accrgc)) << prefix << "accrgc";
    EXPECT_NEAR(actual.evapg1, expected.evapg1, pow_sqrt_tol(expected.evapg1)) << prefix << "evapg1";
    EXPECT_NEAR(actual.evapg2, expected.evapg2, pow_sqrt_tol(expected.evapg2)) << prefix << "evapg2";
    EXPECT_NEAR(actual.evapr1, expected.evapr1, pow_sqrt_tol(expected.evapr1)) << prefix << "evapr1";
    EXPECT_NEAR(actual.evapr2, expected.evapr2, pow_sqrt_tol(expected.evapr2)) << prefix << "evapr2";
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real patterned_rho (const int i,
                           const int j,
                           const int k) noexcept
{
    return amrex::Real(1.0) + amrex::Real(0.01) * i + amrex::Real(0.02) * j + amrex::Real(0.03) * k;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real patterned_theta (const int i,
                             const int j,
                             const int k) noexcept
{
    return amrex::Real(300.0) + i + amrex::Real(2.0) * j + amrex::Real(3.0) * k;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real patterned_qv (const int i,
                          const int j,
                          const int k) noexcept
{
    return amrex::Real(1.0e-2) + amrex::Real(1.0e-4) * (i + 2 * j + 3 * k);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real patterned_qcl (const int i,
                           const int j,
                           const int k) noexcept
{
    return amrex::Real(2.0e-4) + amrex::Real(1.0e-5) * (2 * i + j + k);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real patterned_qci (const int i,
                           const int j,
                           const int k) noexcept
{
    return amrex::Real(1.0e-4) + amrex::Real(1.0e-5) * (i + 3 * j + 2 * k);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real patterned_qpr (const int i,
                           const int j,
                           const int) noexcept
{
    return amrex::Real(3.0e-5) + amrex::Real(1.0e-6) * (4 * i + j);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real patterned_qps (const int i,
                           const int,
                           const int k) noexcept
{
    return amrex::Real(4.0e-5) + amrex::Real(1.0e-6) * (i + 5 * k);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real patterned_qpg (const int,
                           const int j,
                           const int k) noexcept
{
    return amrex::Real(5.0e-5) + amrex::Real(1.0e-6) * (2 * j + 3 * k);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real patterned_cons_value (const int comp,
                                  const int i,
                                  const int j,
                                  const int k) noexcept
{
    const amrex::Real rho = patterned_rho(i, j, k);
    const amrex::Real theta = patterned_theta(i, j, k);

    switch (comp) {
    case Rho_comp: return rho;
    case RhoTheta_comp: return rho * theta;
    case RhoKE_comp: return amrex::Real(0.0);
    case RhoScalar_comp: return amrex::Real(0.0);
    case RhoQ1_comp: return rho * patterned_qv(i, j, k);
    case RhoQ2_comp: return rho * patterned_qcl(i, j, k);
    case RhoQ3_comp: return rho * patterned_qci(i, j, k);
    case RhoQ4_comp: return rho * patterned_qpr(i, j, k);
    case RhoQ5_comp: return rho * patterned_qps(i, j, k);
    case RhoQ6_comp: return rho * patterned_qpg(i, j, k);
    default: return amrex::Real(0.0);
    }
}

void fill_patterned_conserved_state (amrex::MultiFab& cons)
{
    cons.setVal(amrex::Real(-999.0));

    for (amrex::MFIter mfi(cons, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto arr = cons.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            for (int comp = Rho_comp; comp <= RhoQ6_comp; ++comp) {
                arr(i,j,k,comp) = patterned_cons_value(comp, i, j, k);
            }
        });
    }

    amrex::Gpu::streamSynchronize();
}

void poison_ghost_cells (amrex::MultiFab& cons,
                         const amrex::Real sentinel)
{
    for (amrex::MFIter mfi(cons, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& fab_box = mfi.fabbox();
        const amrex::Box& valid_box = mfi.validbox();
        auto arr = cons.array(mfi);
        amrex::ParallelFor(fab_box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            if (!valid_box.contains(amrex::IntVect(AMREX_D_DECL(i, j, k)))) {
                for (int comp = Rho_comp; comp <= RhoQ6_comp; ++comp) {
                    arr(i,j,k,comp) = sentinel;
                }
            }
        });
    }

    amrex::Gpu::streamSynchronize();
}

amrex::Real mixed_phase_tabs ()
{
    const amrex::Real lower = std::max(tprmin, tgrmin) + amrex::Real(1.0e-3);
    const amrex::Real upper = std::min(tprmax, tgrmax) - amrex::Real(1.0e-3);
    EXPECT_LT(lower, upper);
    return amrex::Real(0.5) * (lower + upper);
}

amrex::Real mixed_qsat_for_state (const MoistureType moisture_type,
                                  const amrex::Real tabs,
                                  const amrex::Real pres_mbar)
{
    amrex::Real qsatw;
    amrex::Real qsati;
    erf_qsatw(tabs, pres_mbar, qsatw);
    erf_qsati(tabs, pres_mbar, qsati);
    const int sam_mode = sam_is_no_ice(moisture_type) ? kSAMNoIceMode : kSAMWithIceMode;
    const amrex::Real omn = sam_cloud_liquid_fraction(sam_mode, tabs, a_bg, tbgmin * a_bg);
    return sam_mixed_qsat(omn, qsatw, qsati);
}

SAMCellState make_precip_budget_state (const int i,
                                       const int j,
                                       const int k,
                                       const MoistureType moisture_type,
                                       const amrex::Real pres_mbar)
{
    const int pattern = (i + 2 * j + 3 * k) % 4;

    amrex::Real tabs = amrex::Real(285.0);
    amrex::Real vapor_fraction = amrex::Real(0.75);
    amrex::Real qcl = amrex::Real(0.0);
    amrex::Real qci = amrex::Real(0.0);
    amrex::Real qpr = amrex::Real(0.0);
    amrex::Real qps = amrex::Real(0.0);
    amrex::Real qpg = amrex::Real(0.0);

    switch (pattern) {
    case 0:
        tabs = amrex::Real(285.0);
        vapor_fraction = amrex::Real(0.75);
        qcl = qcw0 + amrex::Real(5.0e-4);
        qpr = amrex::Real(3.0e-4);
        break;
    case 1:
        tabs = mixed_phase_tabs();
        vapor_fraction = amrex::Real(0.6);
        qcl = qcw0 + amrex::Real(8.0e-4);
        qci = qci0 + amrex::Real(7.0e-4);
        qpr = amrex::Real(4.0e-4);
        qps = amrex::Real(5.0e-4);
        qpg = amrex::Real(6.0e-4);
        break;
    case 2:
        tabs = amrex::Real(260.0);
        vapor_fraction = amrex::Real(0.7);
        qci = qci0 + amrex::Real(4.0e-4);
        qps = amrex::Real(4.0e-4);
        break;
    default:
        tabs = amrex::Real(245.0);
        vapor_fraction = amrex::Real(0.7);
        qci = qci0 + amrex::Real(4.0e-4);
        qpg = amrex::Real(4.0e-4);
        break;
    }

    if (sam_is_no_ice(moisture_type)) {
        qci = amrex::Real(0.0);
        qps = amrex::Real(0.0);
        qpg = amrex::Real(0.0);
        tabs = std::max(tbgmax + amrex::Real(1.0), tabs);
    }

    const amrex::Real qv = vapor_fraction * mixed_qsat_for_state(moisture_type, tabs, pres_mbar);

    SAMCellState state{};
    state.tabs = tabs;
    state.pres_mbar = pres_mbar;
    state.qv = qv;
    state.qcl = qcl;
    state.qci = qci;
    state.qpr = qpr;
    state.qps = qps;
    state.qpg = qpg;
    state.qn = qcl + qci;
    state.qt = qv + state.qn;
    state.qp = qpr + qps + qpg;
    state.theta = sam_theta_from_stored_mbar_converted_to_pa(tabs, pres_mbar, kRdOcp);
    state.rho = getRhogivenTandPress(tabs, sam_mbar_to_pa(pres_mbar), qv);
    return state;
}

void fill_precip_budget_conserved_state (amrex::MultiFab& cons,
                                         const MoistureType moisture_type,
                                         const amrex::Real pres_mbar)
{
    cons.setVal(amrex::Real(0.0));

    for (amrex::MFIter mfi(cons, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto arr = cons.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const SAMCellState state = make_precip_budget_state(i, j, k, moisture_type, pres_mbar);
            arr(i,j,k,Rho_comp) = state.rho;
            arr(i,j,k,RhoKE_comp) = amrex::Real(0.0);
            arr(i,j,k,RhoScalar_comp) = amrex::Real(0.0);
            sam_primitive_to_cons(state, arr, i, j, k);
        });
    }

    amrex::Gpu::streamSynchronize();
}

void fill_nonuniform_detj (amrex::MultiFab& detJ_cc)
{
    for (amrex::MFIter mfi(detJ_cc, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto arr = detJ_cc.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            arr(i,j,k) = amrex::Real(0.8) + amrex::Real(0.05) * i + amrex::Real(0.03) * j + amrex::Real(0.07) * k;
        });
    }

    amrex::Gpu::streamSynchronize();
}

struct PrecipFallCellState {
    amrex::Real rho;
    amrex::Real theta;
    amrex::Real tabs;
    amrex::Real qv;
    amrex::Real qpr;
    amrex::Real qps;
    amrex::Real qpg;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
PrecipFallCellState precipfall_cell_state (const int i,
                                           const int j,
                                           const amrex::Real pressure_pa) noexcept
{
    const int phase_index = (i + 2 * j) % 3;
    const amrex::Real tabs = (phase_index == 0) ? amrex::Real(281.0)
                            : (phase_index == 1) ? amrex::Real(263.0)
                                                 : amrex::Real(240.0);
    const amrex::Real qv = amrex::Real(9.0e-3) + amrex::Real(2.0e-4) * phase_index;
    const amrex::Real theta = getThgivenTandP(tabs, pressure_pa, kRdOcp);
    const amrex::Real rho = getRhogivenTandPress(tabs, pressure_pa, qv);

    return {
        rho,
        theta,
        tabs,
        qv,
        amrex::Real(2.0e-5) + amrex::Real(5.0e-7) * i + amrex::Real(3.0e-7) * j,
        amrex::Real(1.5e-5) + amrex::Real(2.0e-7) * i + amrex::Real(4.0e-7) * j,
        amrex::Real(1.0e-5) + amrex::Real(3.0e-7) * i + amrex::Real(2.0e-7) * j};
}

std::array<amrex::Real, 3> precipfall_terminal_velocities ()
{
    const amrex::Real gamr3 = erf_gammafff(amrex::Real(4.0) + b_rain);
    const amrex::Real gams3 = erf_gammafff(amrex::Real(4.0) + b_snow);
    const amrex::Real gamg3 = erf_gammafff(amrex::Real(4.0) + b_grau);

    return {
        (a_rain * gamr3 / amrex::Real(6.0)) * std::pow((PI * rhor * nzeror), -crain),
        (a_snow * gams3 / amrex::Real(6.0)) * std::pow((PI * rhos * nzeros), -csnow),
        (a_grau * gamg3 / amrex::Real(6.0)) * std::pow((PI * rhog * nzerog), -cgrau)};
}

void fill_precipfall_conserved_state (amrex::MultiFab& cons,
                                      const amrex::Real pressure_pa)
{
    cons.setVal(amrex::Real(0.0));

    for (amrex::MFIter mfi(cons, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto arr = cons.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const PrecipFallCellState state = precipfall_cell_state(i, j, pressure_pa);

            arr(i,j,k,Rho_comp) = state.rho;
            arr(i,j,k,RhoTheta_comp) = state.rho * state.theta;
            arr(i,j,k,RhoKE_comp) = amrex::Real(0.0);
            arr(i,j,k,RhoScalar_comp) = amrex::Real(0.0);
            arr(i,j,k,RhoQ1_comp) = state.rho * state.qv;
            arr(i,j,k,RhoQ2_comp) = amrex::Real(0.0);
            arr(i,j,k,RhoQ3_comp) = amrex::Real(0.0);
            arr(i,j,k,RhoQ4_comp) = state.rho * state.qpr;
            arr(i,j,k,RhoQ5_comp) = state.rho * state.qps;
            arr(i,j,k,RhoQ6_comp) = state.rho * state.qpg;
        });
    }

    amrex::Gpu::streamSynchronize();
}

int wrap_index (const int idx,
                const int lo,
                const int hi)
{
    const int len = hi - lo + 1;
    if (idx < lo) {
        return idx + len;
    }
    if (idx > hi) {
        return idx - len;
    }
    return idx;
}

} // namespace

// Motivation:
// Compute_Coefficients forms plane-averaged SAM coefficient rows. For a
// horizontally uniform state, those rows should be independent of rank count
// and box decomposition; CTest runs this same test under 1, 2, and 4 ranks
// when configured.
TEST(SAMParallel, ComputeCoefficientsRankInvariant)
{
    constexpr int nx = 8;
    constexpr int ny = 4;
    constexpr int nz = 4;

    const amrex::Real rho = amrex::Real(1.05);
    const amrex::Real theta = amrex::Real(296.0);
    const amrex::Real qv = amrex::Real(1.1e-2);
    const amrex::Real qcl = amrex::Real(4.0e-4);
    const amrex::Real qci = amrex::Real(2.0e-4);

    amrex::Box domain(amrex::IntVect(0, 0, 0), amrex::IntVect(nx - 1, ny - 1, nz - 1));
    amrex::RealBox real_box({AMREX_D_DECL(0.0, 0.0, 0.0)},
                            {AMREX_D_DECL(1.0, 1.0, 1.0)});
    amrex::Array<int, AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(1, 1, 1)};
    amrex::Geometry geom(domain, &real_box, 0, is_periodic.data());

    amrex::BoxArray ba(domain);
    ba.maxSize(amrex::IntVect(4, 2, nz));
    amrex::DistributionMapping dm(ba);

    amrex::MultiFab cons(ba, dm, RhoQ6_comp + 1, 0);
    fill_uniform_conserved_state(cons, rho, theta, qv, qcl, qci,
                                 amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0));

    std::unique_ptr<amrex::MultiFab> z_phys_nd;
    std::unique_ptr<amrex::MultiFab> detJ_cc;

    SolverChoice sc = make_sam_solver_choice();
    SAM sam;
    sam.Define(sc);
    sam.Set_dzmin(amrex::Real(100.0));
    sam.Set_RealWidth(0);
    sam.Init(cons, ba, geom, amrex::Real(1.0), z_phys_nd, detJ_cc);
    sam.Copy_State_to_Micro(cons);
    sam.Compute_Coefficients();

    const amrex::Real tabs = amrex::min(getTgivenRandRTh(rho, rho * theta, qv), amrex::Real(273.16));
    const amrex::Real gamr1 = erf_gammafff(three + b_rain);
    const amrex::Real gamr2 = erf_gammafff((amrex::Real(5.0) + b_rain) / two);
    const amrex::Real gams1 = erf_gammafff(three + b_snow);
    const amrex::Real gams2 = erf_gammafff((amrex::Real(5.0) + b_snow) / two);
    const amrex::Real gamg1 = erf_gammafff(three + b_grau);
    const amrex::Real gamg2 = erf_gammafff((amrex::Real(5.0) + b_grau) / two);
    const SAMCoefficientRow expected = sam_compute_coefficient_row(
        rho, tabs, gamr1, gamr2, gams1, gams2, gamg1, gamg2);

    const int rank = amrex::ParallelDescriptor::MyProc();
    for (int k = domain.smallEnd(2); k <= domain.bigEnd(2); ++k) {
        const SAMCoefficientRow actual = sam.CoefficientRowAt(k);
        expect_coefficient_row_near(actual, expected, k, rank);
    }
}

// Motivation:
// Copy_Micro_to_State ends with FillBoundary over the SAM geometry periodicity.
// After a state-to-micro roundtrip, periodic ghost cells should match the
// wrapped valid neighbors regardless of rank count or box ownership.
TEST(SAMParallel, CopyMicroToStateFillBoundaryParallel)
{
    constexpr int nx = 8;
    constexpr int ny = 4;
    constexpr int nz = 4;
    constexpr int ng = 1;
    const amrex::Real kSentinel = amrex::Real(-999.0);

    amrex::Box domain(amrex::IntVect(0, 0, 0), amrex::IntVect(nx - 1, ny - 1, nz - 1));
    amrex::RealBox real_box({AMREX_D_DECL(0.0, 0.0, 0.0)},
                            {AMREX_D_DECL(1.0, 1.0, 1.0)});
    amrex::Array<int, AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(1, 1, 1)};
    amrex::Geometry geom(domain, &real_box, 0, is_periodic.data());

    amrex::BoxArray ba(domain);
    ba.maxSize(amrex::IntVect(4, 2, nz));
    amrex::DistributionMapping dm(ba);

    amrex::MultiFab cons(ba, dm, RhoQ6_comp + 1, ng);
    fill_patterned_conserved_state(cons);

    std::unique_ptr<amrex::MultiFab> z_phys_nd;
    std::unique_ptr<amrex::MultiFab> detJ_cc;

    SolverChoice sc = make_sam_solver_choice();
    SAM sam;
    sam.Define(sc);
    sam.Set_dzmin(amrex::Real(100.0));
    sam.Set_RealWidth(0);
    sam.Init(cons, ba, geom, amrex::Real(1.0), z_phys_nd, detJ_cc);
    sam.Copy_State_to_Micro(cons);

    poison_ghost_cells(cons, kSentinel);
    sam.Copy_Micro_to_State(cons);
    amrex::Gpu::streamSynchronize();

    const int rank = amrex::ParallelDescriptor::MyProc();
    int local_checked = 0;

    for (amrex::MFIter mfi(cons); mfi.isValid(); ++mfi) {
        const amrex::Box& fab_box = mfi.fabbox();
        const amrex::Array4<const amrex::Real> arr = cons.const_array(mfi);

        for (int k = fab_box.smallEnd(2); k <= fab_box.bigEnd(2); ++k) {
            for (int j = fab_box.smallEnd(1); j <= fab_box.bigEnd(1); ++j) {
                for (int i = fab_box.smallEnd(0); i <= fab_box.bigEnd(0); ++i) {
                    const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
                    if (domain.contains(iv)) {
                        continue;
                    }

                    ++local_checked;
                    const int wrapped_i = wrap_index(i, domain.smallEnd(0), domain.bigEnd(0));
                    const int wrapped_j = wrap_index(j, domain.smallEnd(1), domain.bigEnd(1));
                    const int wrapped_k = wrap_index(k, domain.smallEnd(2), domain.bigEnd(2));

                    for (int comp = Rho_comp; comp <= RhoQ6_comp; ++comp) {
                        const amrex::Real expected = patterned_cons_value(comp, wrapped_i, wrapped_j, wrapped_k);
                        EXPECT_NEAR(arr(i,j,k,comp), expected, roundoff_tol(expected))
                            << "rank=" << rank
                            << " ghost_iv=" << iv
                            << " wrapped_iv=(" << wrapped_i << "," << wrapped_j << "," << wrapped_k << ")"
                            << " comp=" << comp;
                    }
                }
            }
        }
    }

    amrex::ParallelDescriptor::ReduceIntSum(local_checked);
    EXPECT_GT(local_checked, 0);
}

// Motivation:
// PrecipFall should preserve the global precip mass budget once bottom surface
// accumulation is converted back to mass. CTest runs this same test under 1,
// 2, and 4 ranks, so passing all entries provides the intended rank-count
// invariance check for the public PrecipFall path.
TEST(SAMParallel, PrecipFallGlobalMassAndSurfaceAccumulation)
{
    constexpr int nx = 8;
    constexpr int ny = 4;
    constexpr int nz = 1;
    const amrex::Real pressure_pa = amrex::Real(9.0e4);
    const amrex::Real dt = amrex::Real(0.1);
    const amrex::Real rho0 = amrex::Real(1.29);

    amrex::Box domain(amrex::IntVect(0, 0, 0), amrex::IntVect(nx - 1, ny - 1, nz - 1));
    amrex::RealBox real_box({AMREX_D_DECL(0.0, 0.0, 0.0)},
                            {AMREX_D_DECL(1.0, 1.0, 1.0)});
    amrex::Array<int, AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(1, 1, 0)};
    amrex::Geometry geom(domain, &real_box, 0, is_periodic.data());

    amrex::BoxArray ba(domain);
    ba.maxSize(amrex::IntVect(4, 2, nz));
    amrex::DistributionMapping dm(ba);

    amrex::MultiFab cons(ba, dm, RhoQ6_comp + 1, 0);
    fill_precipfall_conserved_state(cons, pressure_pa);

    const amrex::Real dx = geom.CellSize(0);
    const amrex::Real dy = geom.CellSize(1);
    const amrex::Real dz = geom.CellSize(2);
    const amrex::Real cell_volume = dx * dy * dz;
    const amrex::Real cell_area = dx * dy;
    const auto terminal_velocities = precipfall_terminal_velocities();

    amrex::Real expected_rain_accum_mass = amrex::Real(0.0);
    amrex::Real expected_snow_accum_mass = amrex::Real(0.0);
    amrex::Real expected_graup_accum_mass = amrex::Real(0.0);
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const PrecipFallCellState state = precipfall_cell_state(i, j, pressure_pa);
            const amrex::Real qp_total = state.qpr + state.qps + state.qpg;
            SAMPrecipFaceState face_state{};
            face_state.rho_avg = state.rho;
            face_state.tabs_avg = state.tabs;
            face_state.qp_avg = qp_total;
            face_state.omp = sam_precip_rain_fraction(kSAMWithIceMode, state.tabs);
            face_state.omg = sam_graupel_fraction(kSAMWithIceMode, state.tabs);
            face_state.qrr = face_state.omp * face_state.qp_avg;
            face_state.qss = (one - face_state.omp) * (one - face_state.omg) * face_state.qp_avg;
            face_state.qgg = (one - face_state.omp) * face_state.omg * face_state.qp_avg;

            const SAMSurfaceAccumulation expected_accum = sam_surface_accumulation(
                face_state, rho0,
                terminal_velocities[0], terminal_velocities[1], terminal_velocities[2], dt);

            expected_rain_accum_mass += expected_accum.rain * cell_area * rhor / amrex::Real(1000.0);
            expected_snow_accum_mass += expected_accum.snow * cell_area * rhos / amrex::Real(1000.0);
            expected_graup_accum_mass += expected_accum.graupel * cell_area * rhog / amrex::Real(1000.0);
        }
    }
    const amrex::Real expected_surface_accum_mass =
        expected_rain_accum_mass + expected_snow_accum_mass + expected_graup_accum_mass;

    const amrex::Real initial_precip_mass =
        (cons.sum(RhoQ4_comp) + cons.sum(RhoQ5_comp) + cons.sum(RhoQ6_comp)) * cell_volume;

    std::unique_ptr<amrex::MultiFab> z_phys_nd;
    std::unique_ptr<amrex::MultiFab> detJ_cc;

    SolverChoice sc = make_sam_solver_choice();
    SAM sam;
    sam.Define(sc);
    sam.Set_dzmin(dz);
    sam.Set_RealWidth(0);
    sam.Init(cons, ba, geom, dt, z_phys_nd, detJ_cc);
    sam.Copy_State_to_Micro(cons);
    sam.PrecipFall(sc);
    sam.Copy_Micro_to_State(cons);

    const amrex::Real final_precip_mass =
        (cons.sum(RhoQ4_comp) + cons.sum(RhoQ5_comp) + cons.sum(RhoQ6_comp)) * cell_volume;

    const amrex::Real rain_accum_mass = sam.Qmoist_Ptr(0)->sum(0) * cell_area * rhor / amrex::Real(1000.0);
    const amrex::Real snow_accum_mass = sam.Qmoist_Ptr(1)->sum(0) * cell_area * rhos / amrex::Real(1000.0);
    const amrex::Real graup_accum_mass = sam.Qmoist_Ptr(2)->sum(0) * cell_area * rhog / amrex::Real(1000.0);
    const amrex::Real surface_accum_mass = rain_accum_mass + snow_accum_mass + graup_accum_mass;

    const int num_terms = nx * ny;
    const int rank = amrex::ParallelDescriptor::MyProc();

    EXPECT_NEAR(final_precip_mass,
                initial_precip_mass,
                mpi_reduction_tol(initial_precip_mass, num_terms))
        << "rank=" << rank
        << " initial_precip_mass=" << initial_precip_mass
        << " final_precip_mass=" << final_precip_mass;
    EXPECT_NEAR(rain_accum_mass,
                expected_rain_accum_mass,
                mpi_reduction_tol(expected_rain_accum_mass, num_terms))
        << "rank=" << rank
        << " rain_accum_mass=" << rain_accum_mass
        << " expected_rain_accum_mass=" << expected_rain_accum_mass;
    EXPECT_NEAR(snow_accum_mass,
                expected_snow_accum_mass,
                mpi_reduction_tol(expected_snow_accum_mass, num_terms))
        << "rank=" << rank
        << " snow_accum_mass=" << snow_accum_mass
        << " expected_snow_accum_mass=" << expected_snow_accum_mass;
    EXPECT_NEAR(graup_accum_mass,
                expected_graup_accum_mass,
                mpi_reduction_tol(expected_graup_accum_mass, num_terms))
        << "rank=" << rank
        << " graup_accum_mass=" << graup_accum_mass
        << " expected_graup_accum_mass=" << expected_graup_accum_mass;
    EXPECT_NEAR(surface_accum_mass,
                expected_surface_accum_mass,
                mpi_reduction_tol(expected_surface_accum_mass, num_terms))
        << "rank=" << rank
        << " surface_accum_mass=" << surface_accum_mass
        << " expected_surface_accum_mass=" << expected_surface_accum_mass
        << " rain_accum_mass=" << rain_accum_mass
        << " snow_accum_mass=" << snow_accum_mass
        << " graup_accum_mass=" << graup_accum_mass;
}

// Motivation:
// The public SAM Precip path is cell-local except for the precomputed
// coefficient rows and MPI reductions used by the test budget checks. Running
// this same contract under 1, 2, and 4 ranks should preserve the global total
// water budget and leave the surface accumulation reservoirs untouched,
// independent of box ownership and reduction order.
TEST(SAMParallel, PrecipGlobalWaterBudgetDecompositionInvariant)
{
    constexpr int nx = 8;
    constexpr int ny = 4;
    constexpr int nz = 4;
    const amrex::Real pres_mbar = amrex::Real(900.0);
    const amrex::Real dt = amrex::Real(0.1);

    amrex::Box domain(amrex::IntVect(0, 0, 0), amrex::IntVect(nx - 1, ny - 1, nz - 1));
    amrex::RealBox real_box({AMREX_D_DECL(0.0, 0.0, 0.0)},
                            {AMREX_D_DECL(1.0, 1.0, 1.0)});
    amrex::Array<int, AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(1, 1, 1)};
    amrex::Geometry geom(domain, &real_box, 0, is_periodic.data());

    amrex::BoxArray ba(domain);
    ba.maxSize(amrex::IntVect(4, 2, 2));
    amrex::DistributionMapping dm(ba);

    amrex::MultiFab cons(ba, dm, RhoQ6_comp + 1, 0);
    fill_precip_budget_conserved_state(cons, MoistureType::SAM, pres_mbar);

    std::unique_ptr<amrex::MultiFab> z_phys_nd;
    std::unique_ptr<amrex::MultiFab> detJ_cc;

    SolverChoice sc = make_sam_solver_choice(MoistureType::SAM);
    SAM sam;
    sam.Define(sc);
    sam.Set_dzmin(geom.CellSize(2));
    sam.Set_RealWidth(0);
    sam.Init(cons, ba, geom, dt, z_phys_nd, detJ_cc);

    const SAMBudget initial_budget = compute_sam_budget(geom, cons, sam);

    sam.Copy_State_to_Micro(cons);
    sam.Compute_Coefficients();
    sam.Precip(sc);
    sam.Copy_Micro_to_State(cons);

    const SAMBudget final_budget = compute_sam_budget(geom, cons, sam);
    const int num_terms = nx * ny * nz;
    const int rank = amrex::ParallelDescriptor::MyProc();

    EXPECT_NEAR(final_budget.total_water_mass,
                initial_budget.total_water_mass,
                mpi_reduction_tol(initial_budget.total_water_mass, num_terms))
        << "rank=" << rank
        << " initial_total_water_mass=" << initial_budget.total_water_mass
        << " final_total_water_mass=" << final_budget.total_water_mass;
    EXPECT_NEAR(final_budget.rain_accum_mass, amrex::Real(0.0), exact_zero_tol())
        << "rank=" << rank
        << " rain_accum_mass=" << final_budget.rain_accum_mass;
    EXPECT_NEAR(final_budget.snow_accum_mass, amrex::Real(0.0), exact_zero_tol())
        << "rank=" << rank
        << " snow_accum_mass=" << final_budget.snow_accum_mass;
    EXPECT_NEAR(final_budget.graupel_accum_mass, amrex::Real(0.0), exact_zero_tol())
        << "rank=" << rank
        << " graupel_accum_mass=" << final_budget.graupel_accum_mass;
}

// Motivation:
// The same public Precip water-budget contract should remain decomposition-
// invariant when the budget is checked in terrain-like coordinates with a
// nonuniform detJ.
TEST(SAMParallel, PrecipDetJWeightedWaterBudgetDecompositionInvariant)
{
    constexpr int nx = 8;
    constexpr int ny = 4;
    constexpr int nz = 4;
    const amrex::Real pres_mbar = amrex::Real(900.0);
    const amrex::Real dt = amrex::Real(1.0);

    amrex::Box domain(amrex::IntVect(0, 0, 0), amrex::IntVect(nx - 1, ny - 1, nz - 1));
    amrex::RealBox real_box({AMREX_D_DECL(0.0, 0.0, 0.0)},
                            {AMREX_D_DECL(1.0, 1.0, 1.0)});
    amrex::Array<int, AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(1, 1, 1)};
    amrex::Geometry geom(domain, &real_box, 0, is_periodic.data());

    amrex::BoxArray ba(domain);
    ba.maxSize(amrex::IntVect(4, 2, 2));
    amrex::DistributionMapping dm(ba);

    amrex::MultiFab cons(ba, dm, RhoQ6_comp + 1, 0);
    fill_precip_budget_conserved_state(cons, MoistureType::SAM, pres_mbar);

    std::unique_ptr<amrex::MultiFab> z_phys_nd;
    std::unique_ptr<amrex::MultiFab> detJ_cc = std::make_unique<amrex::MultiFab>(ba, dm, 1, 0);
    fill_nonuniform_detj(*detJ_cc);

    SolverChoice sc = make_sam_solver_choice(MoistureType::SAM);
    SAM sam;
    sam.Define(sc);
    sam.Set_dzmin(geom.CellSize(2));
    sam.Set_RealWidth(0);
    sam.Init(cons, ba, geom, dt, z_phys_nd, detJ_cc);

    const SAMBudget initial_budget = compute_sam_budget(geom, cons, sam, detJ_cc.get());

    sam.Copy_State_to_Micro(cons);
    sam.Compute_Coefficients();
    sam.Precip(sc);
    sam.Copy_Micro_to_State(cons);

    const SAMBudget final_budget = compute_sam_budget(geom, cons, sam, detJ_cc.get());
    const int num_terms = nx * ny * nz;
    const int rank = amrex::ParallelDescriptor::MyProc();

    EXPECT_NEAR(final_budget.total_water_mass,
                initial_budget.total_water_mass,
                mpi_reduction_tol(initial_budget.total_water_mass, num_terms))
        << "rank=" << rank
        << " initial_total_water_mass=" << initial_budget.total_water_mass
        << " final_total_water_mass=" << final_budget.total_water_mass;
    EXPECT_NEAR(final_budget.rain_accum_mass, amrex::Real(0.0), exact_zero_tol())
        << "rank=" << rank
        << " rain_accum_mass=" << final_budget.rain_accum_mass;
    EXPECT_NEAR(final_budget.snow_accum_mass, amrex::Real(0.0), exact_zero_tol())
        << "rank=" << rank
        << " snow_accum_mass=" << final_budget.snow_accum_mass;
    EXPECT_NEAR(final_budget.graupel_accum_mass, amrex::Real(0.0), exact_zero_tol())
        << "rank=" << rank
        << " graupel_accum_mass=" << final_budget.graupel_accum_mass;
}