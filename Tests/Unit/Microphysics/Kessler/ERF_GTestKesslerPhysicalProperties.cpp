#include <memory>

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_MultiFab.H>

#include <gtest/gtest.h>

#include "ERF_GTestKesslerCommon.H"

using namespace kessler_test;

namespace {

void initialize_kessler (Kessler& kessler,
                         SolverChoice& sc,
                         const amrex::Geometry& geom,
                         amrex::MultiFab& cons,
                         std::unique_ptr<amrex::MultiFab>& z_phys_nd,
                         std::unique_ptr<amrex::MultiFab>& detJ_cc)
{
    kessler.Define(sc);
    kessler.Set_RealWidth(0);
    kessler.Set_dzmin(geom.CellSize(2));
    kessler.Init(cons, cons.boxArray(), geom, kDefaultDt, z_phys_nd, detJ_cc);
}

struct SedimentationColumnState {
    std::array<PrimitiveState, 2> states;
    amrex::Real rain_accum;
};

struct SedimentationColumnRegressionReference {
    SedimentationColumnState velocity_reference;
    SedimentationColumnState flux_like_reference;
    int velocity_substeps;
    int flux_like_substeps;
};

inline std::array<PrimitiveState, 2> make_velocity_cfl_sensitive_column_states ()
{
    amrex::Real qsat_bottom;
    amrex::Real qsat_top;
    erf_qsatw(amrex::Real(289.0), amrex::Real(900.0), qsat_bottom);
    erf_qsatw(amrex::Real(288.0), amrex::Real(900.0), qsat_top);

    return {
        make_primitive_state(amrex::Real(289.0), amrex::Real(900.0),
                             qsat_bottom + amrex::Real(2.0e-4), amrex::Real(0.0), amrex::Real(1.0e-3)),
        make_primitive_state(amrex::Real(288.0), amrex::Real(900.0),
                             qsat_top + amrex::Real(2.0e-4), amrex::Real(0.0), amrex::Real(0.0))
    };
}

inline void fill_velocity_cfl_sensitive_column_state_portable (
    amrex::MultiFab& cons,
    const std::array<PrimitiveState, 2>& states)
{
    for (amrex::MFIter mfi(cons, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto arr = cons.array(mfi);
        run_and_sync([=] {
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                set_conserved_cell(arr, i, j, k, (k == 0) ? states[0] : states[1]);
            });
        });
    }
}

inline void fill_reference_rain_accum_portable (amrex::MultiFab& rain_accum,
                                                const amrex::Real bottom_rain_accum)
{
    for (amrex::MFIter mfi(rain_accum, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto arr = rain_accum.array(mfi);
        run_and_sync([=] {
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                arr(i,j,k) = (k == 0) ? bottom_rain_accum : amrex::Real(0.0);
            });
        });
    }
}

inline KesslerFaceState reference_column_face_state (const std::array<PrimitiveState, 2>& states,
                                                     const int face_k)
{
    KesslerFaceState face_state{amrex::Real(0.0), amrex::Real(0.0)};
    if (face_k == 0) {
        face_state.rho = states[0].rho;
        face_state.qp = states[0].qp;
    } else if (face_k == 2) {
        face_state.rho = states[1].rho;
        face_state.qp = states[1].qp;
    } else {
        face_state.rho = myhalf * (states[0].rho + states[1].rho);
        face_state.qp = myhalf * (states[0].qp + states[1].qp);
    }

    face_state.qp = std::max(amrex::Real(0.0), face_state.qp);
    return face_state;
}

// Provenance: pow roundoff in terminal velocity, arithmetic over a small fixed
// column, accumulation over n substeps, and float-vs-double scaling.
inline amrex::Real sedimentation_cfl_reference_tol (const int n_substeps,
                                                    const amrex::Real scale)
{
    return std::max(backend_math_abs_tol(scale),
                    property_accumulation_tol(2 * std::max(1, n_substeps), scale));
}

inline SedimentationColumnState advance_reference_sedimentation_column (
    const std::array<PrimitiveState, 2>& initial_states,
    const amrex::Real dt,
    const amrex::Real dz,
    const int n_substeps)
{
    SedimentationColumnState result{initial_states, amrex::Real(0.0)};
    const amrex::Real substep_dt = dt / static_cast<amrex::Real>(n_substeps);
    const amrex::Real coef = substep_dt / dz;

    for (int nsub = 0; nsub < n_substeps; ++nsub) {
        std::array<amrex::Real, 3> face_fluxes{};
        for (int face_k = 0; face_k < 3; ++face_k) {
            const KesslerFaceState face_state = reference_column_face_state(result.states, face_k);
            const amrex::Real terminal_velocity = reference_terminal_velocity(face_state.rho, face_state.qp);
            face_fluxes[face_k] = reference_precip_flux(face_state.rho, terminal_velocity, face_state.qp);
        }

        result.rain_accum += face_fluxes[0] * substep_dt / amrex::Real(1000.0) * amrex::Real(1000.0);
        for (int k = 0; k < 2; ++k) {
            amrex::Real dq_sed = reference_sedimentation_tendency(
                face_fluxes[k + 1], face_fluxes[k], result.states[k].rho, amrex::Real(1.0), coef);
            if (kessler_is_small_sedimentation_value(dq_sed)) {
                dq_sed = amrex::Real(0.0);
            }
            result.states[k].qp += dq_sed;
            result.states[k].qp = std::max(amrex::Real(0.0), result.states[k].qp);
        }
    }

    return result;
}

inline SedimentationColumnRegressionReference make_sedimentation_cfl_regression_reference (
    const std::array<PrimitiveState, 2>& initial_states,
    const amrex::Real dt,
    const amrex::Real dz)
{
    amrex::Real max_terminal_velocity = amrex::Real(0.0);
    amrex::Real max_flux_like_value = amrex::Real(0.0);
    for (int face_k = 0; face_k < 3; ++face_k) {
        const KesslerFaceState face_state = reference_column_face_state(initial_states, face_k);
        const amrex::Real terminal_velocity = reference_terminal_velocity(face_state.rho, face_state.qp);
        const amrex::Real flux_like_value = reference_precip_flux(face_state.rho, terminal_velocity, face_state.qp);
        max_terminal_velocity = std::max(max_terminal_velocity, terminal_velocity);
        max_flux_like_value = std::max(max_flux_like_value, flux_like_value);
    }

    const int velocity_substeps = reference_velocity_cfl_substeps(max_terminal_velocity, dt, dz);
    const int flux_like_substeps = reference_substeps_from_reduced_value(max_flux_like_value, dt, dz);

    return {
        advance_reference_sedimentation_column(initial_states, dt, dz, velocity_substeps),
        advance_reference_sedimentation_column(initial_states, dt, dz, flux_like_substeps),
        velocity_substeps,
        flux_like_substeps
    };
}

} // namespace

// Motivation: Local warm-rain source terms exchange water among qv, qc, and
// qp only. This closure-independent property must hold across representative
// condensation, cloud-evaporation, rain-evaporation, autoconversion, and mixed
// cases.
TEST(KesslerPhysicalProperties, PhysicalProperties_LocalSourcesConserveTotalWater)
{
    struct LocalSourceCase {
        amrex::Real qv;
        amrex::Real qc;
        amrex::Real qp;
        amrex::Real rho;
        amrex::Real pressure_mbar;
        amrex::Real qsat;
        amrex::Real dtqsat;
        amrex::Real dt;
        bool do_cond;
    };

    const std::array<LocalSourceCase, 6> cases = {{
        {amrex::Real(1.2e-2), amrex::Real(4.0e-4), amrex::Real(0.0), amrex::Real(1.1), amrex::Real(900.0), amrex::Real(1.0e-2), amrex::Real(0.02), amrex::Real(1.0), true},
        {amrex::Real(8.5e-3), amrex::Real(2.0e-3), amrex::Real(0.0), amrex::Real(1.0), amrex::Real(900.0), amrex::Real(1.0e-2), amrex::Real(0.02), amrex::Real(1.0), true},
        {amrex::Real(9.0e-3), amrex::Real(5.0e-4), amrex::Real(1.0e-3), amrex::Real(1.1), amrex::Real(900.0), amrex::Real(1.0e-2), amrex::Real(0.02), amrex::Real(10.0), true},
        {amrex::Real(1.0e-2), qcw0 + amrex::Real(5.0e-4), amrex::Real(0.0), amrex::Real(1.0), amrex::Real(900.0), amrex::Real(1.0e-2), amrex::Real(0.02), amrex::Real(1.0), true},
        {amrex::Real(1.0e-2), amrex::Real(8.0e-4), amrex::Real(1.5e-3), amrex::Real(1.0), amrex::Real(900.0), amrex::Real(1.0e-2), amrex::Real(0.02), amrex::Real(1.0), true},
        {amrex::Real(9.3e-3), amrex::Real(1.5e-3), amrex::Real(1.2e-3), amrex::Real(1.1), amrex::Real(900.0), amrex::Real(1.0e-2), amrex::Real(0.02), amrex::Real(5.0), true}
    }};

    for (const LocalSourceCase& test_case : cases) {
        SCOPED_TRACE("qv=" + std::to_string(static_cast<double>(test_case.qv)) +
                     " qc=" + std::to_string(static_cast<double>(test_case.qc)) +
                     " qp=" + std::to_string(static_cast<double>(test_case.qp)));
        const KesslerSourceTerms source_terms = kessler_warm_rain_sources(
            test_case.qv, test_case.qc, test_case.qp, test_case.rho, test_case.pressure_mbar,
            test_case.qsat, test_case.dtqsat, test_case.dt, test_case.do_cond);
        amrex::Real qv = test_case.qv;
        amrex::Real qc = test_case.qc;
        amrex::Real qp = test_case.qp;
        const amrex::Real total_before = qv + qc + qp;
        apply_local_sources(qv, qc, qp, source_terms);
        EXPECT_NEAR(qv + qc + qp, total_before, formula_abs_tol(total_before));
    }
}

// Motivation: The local Kessler caps should keep qv, qc, and qp nonnegative
// when the inputs are valid nonnegative mass fractions. This checks a physical
// sanity property independently of the exact closure coefficients.
TEST(KesslerPhysicalProperties, PhysicalProperties_LocalSourcesDoNotCreateNegativeSpeciesWhenCapsSuffice)
{
    const std::array<KernelCase, 4> cases = {make_kernel_cases()[0], make_kernel_cases()[1],
                                             make_kernel_cases()[2], make_kernel_cases()[3]};

    for (const KernelCase& test_case : cases) {
        SCOPED_TRACE(case_label(test_case));
        const KesslerSourceTerms source_terms = kessler_warm_rain_sources(
            test_case.qv, test_case.qc, test_case.qp, test_case.rho, test_case.pressure_mbar,
            test_case.qsat, test_case.dtqsat, test_case.dt, test_case.do_cond);
        amrex::Real qv = test_case.qv;
        amrex::Real qc = test_case.qc;
        amrex::Real qp = test_case.qp;
        apply_local_sources(qv, qc, qp, source_terms);
        EXPECT_GE(qv, -exact_zero_or_near_zero_tol());
        EXPECT_GE(qc, -exact_zero_or_near_zero_tol());
        EXPECT_GE(qp, -exact_zero_or_near_zero_tol());
    }
}

// Motivation: Availability caps are independent physical constraints. This
// protects the cloud-evaporation, cloud-to-rain, and rain-evaporation caps with
// representative inputs that exercise each branch.
TEST(KesslerPhysicalProperties, PhysicalProperties_WarmRainSourcesRespectAvailabilityCaps)
{
    const KesslerSourceTerms cloud_evap = kessler_warm_rain_sources(
        amrex::Real(8.0e-3), amrex::Real(1.0e-4), amrex::Real(0.0), amrex::Real(1.0),
        amrex::Real(900.0), amrex::Real(1.0e-2), amrex::Real(0.01), amrex::Real(1.0), true);
    const KesslerSourceTerms cloud_to_rain = kessler_warm_rain_sources(
        amrex::Real(1.0e-2), amrex::Real(2.0e-4), amrex::Real(1.0), amrex::Real(1.0),
        amrex::Real(900.0), amrex::Real(1.0e-2), amrex::Real(0.01), amrex::Real(10.0), true);
    const KesslerSourceTerms rain_evap = kessler_warm_rain_sources(
        amrex::Real(1.0e-3), amrex::Real(2.0e-4), amrex::Real(1.0e-4), amrex::Real(1.2),
        amrex::Real(900.0), amrex::Real(1.0e-2), amrex::Real(0.02), amrex::Real(200.0), true);

    EXPECT_LE(cloud_evap.dq_cloud_to_vapor, amrex::Real(1.0e-4) + formula_abs_tol(amrex::Real(1.0e-4)));
    EXPECT_LE(cloud_to_rain.dq_cloud_to_rain, amrex::Real(2.0e-4) + formula_abs_tol(amrex::Real(2.0e-4)));
    EXPECT_LE(rain_evap.dq_rain_to_vapor, amrex::Real(1.0e-4) + formula_abs_tol(amrex::Real(1.0e-4)));
}

// Motivation: For valid rho > 0 and qp >= 0, terminal velocity and precip flux
// should be nonnegative. This is closure-independent physical sanity coverage.
TEST(KesslerPhysicalProperties, PhysicalProperties_TerminalVelocityAndFluxAreNonnegativeForValidInputs)
{
    const std::array<amrex::Real, 3> rho_values = {amrex::Real(0.9), amrex::Real(1.16), amrex::Real(1.4)};
    const std::array<amrex::Real, 4> qp_values = {
        amrex::Real(0.0), amrex::Real(1.0e-6), amrex::Real(1.0e-3), amrex::Real(1.0e-2)};

    for (const amrex::Real rho : rho_values) {
        for (const amrex::Real qp : qp_values) {
            SCOPED_TRACE("rho=" + std::to_string(static_cast<double>(rho)) +
                         " qp=" + std::to_string(static_cast<double>(qp)));
            const amrex::Real velocity = kessler_terminal_velocity(rho, qp);
            const amrex::Real flux = kessler_precip_flux(rho, velocity, qp);
            EXPECT_GE(velocity, -exact_zero_or_near_zero_tol());
            EXPECT_GE(flux, -exact_zero_or_near_zero_tol());
        }
    }
}

// Motivation: Sedimentation tendency should follow the sign of the flux
// divergence with the current Kessler convention, and vanish for zero
// divergence. This is a physical-property sign test, not a formula copy test.
TEST(KesslerPhysicalProperties, PhysicalProperties_SedimentationTendencyHasFluxDivergenceSign)
{
    const amrex::Real positive_tendency = kessler_sedimentation_tendency(
        amrex::Real(5.0e-3), amrex::Real(1.0e-3), amrex::Real(1.0), amrex::Real(1.0), amrex::Real(0.5));
    const amrex::Real zero_tendency = kessler_sedimentation_tendency(
        amrex::Real(2.0e-3), amrex::Real(2.0e-3), amrex::Real(1.0), amrex::Real(1.0), amrex::Real(0.5));
    const amrex::Real negative_tendency = kessler_sedimentation_tendency(
        amrex::Real(1.0e-3), amrex::Real(5.0e-3), amrex::Real(1.0), amrex::Real(1.0), amrex::Real(0.5));

    EXPECT_GT(positive_tendency, amrex::Real(0.0));
    EXPECT_NEAR(zero_tendency, amrex::Real(0.0), exact_zero_or_near_zero_tol());
    EXPECT_LT(negative_tendency, amrex::Real(0.0));
}

// Motivation: Kessler_NoRain should not change qp or rain accumulation through
// the public Init -> Copy_State_to_Micro -> Advance -> Copy_Micro_to_State
// path, even if the conserved state carries rain water.
TEST(KesslerPhysicalProperties, PhysicalProperties_KesslerNoRainDoesNotAdvanceRainWaterOrAccumulation)
{
    const amrex::Geometry geom = make_geometry(1, 1, 4);
    amrex::BoxArray ba(geom.Domain());
    amrex::DistributionMapping dm(ba);
    amrex::MultiFab cons(ba, dm, RhoQ3_comp + 1, 1);
    amrex::MultiFab cons_before(ba, dm, RhoQ3_comp + 1, 1);
    amrex::MultiFab err(ba, dm, NumRainStateErrComps, 0);
    cons.setVal(amrex::Real(0.0));
    cons_before.setVal(amrex::Real(0.0));
    err.setVal(amrex::Real(0.0));
    fill_conserved_column_state_portable(cons);
    copy_multifab(cons, cons_before);

    std::unique_ptr<amrex::MultiFab> z_phys_nd;
    std::unique_ptr<amrex::MultiFab> detJ_cc;
    Kessler kessler;
    SolverChoice sc = make_solver_choice(MoistureType::Kessler_NoRain, false);

    initialize_kessler(kessler, sc, geom, cons, z_phys_nd, detJ_cc);
    run_and_sync([&] { kessler.Copy_State_to_Micro(cons); });
    amrex::MultiFab rain_before(ba, dm, 1, 1);
    copy_multifab(*kessler.Qmoist_Ptr(0), rain_before);
    run_and_sync([&] {
        kessler.Advance(kDefaultDt, sc);
        kessler.Copy_Micro_to_State(cons);
    });

    compute_rain_state_difference(cons, cons_before, *kessler.Qmoist_Ptr(0), rain_before, err);

    EXPECT_LE(err.max(RainStateErrQp), formula_abs_tol(cons_before.max(RhoQ3_comp)));
    EXPECT_LE(err.max(RainStateErrRainAccum), exact_zero_or_near_zero_tol());
}

// Motivation: The scalar substep helper test proves the terminal-velocity CFL
// formula, but it does not prove `AdvanceKessler` passes terminal velocity into
// that helper. This public-flow regression uses a column where velocity-based
// and flux-like substep counts differ, then compares the production result to
// an independent velocity-subcycled reference. It intentionally avoids the
// shared-face threshold branch and does not assert the full column water budget.
TEST(KesslerPhysicalProperties, KesslerPublicFlow_SedimentationSubstepsUseVelocityCFL)
{
    const amrex::Geometry geom = make_geometry(1, 1, 2);
    amrex::BoxArray ba(geom.Domain());
    amrex::DistributionMapping dm(ba);
    amrex::MultiFab cons(ba, dm, RhoQ3_comp + 1, 1);
    amrex::MultiFab velocity_cons(ba, dm, RhoQ3_comp + 1, 1);
    amrex::MultiFab flux_like_cons(ba, dm, RhoQ3_comp + 1, 1);
    amrex::MultiFab velocity_rain(ba, dm, 1, 1);
    amrex::MultiFab flux_like_rain(ba, dm, 1, 1);
    amrex::MultiFab velocity_err(ba, dm, NumRainStateErrComps, 0);
    amrex::MultiFab flux_like_err(ba, dm, NumRainStateErrComps, 0);
    cons.setVal(amrex::Real(0.0));
    velocity_cons.setVal(amrex::Real(0.0));
    flux_like_cons.setVal(amrex::Real(0.0));
    velocity_rain.setVal(amrex::Real(0.0));
    flux_like_rain.setVal(amrex::Real(0.0));
    velocity_err.setVal(amrex::Real(0.0));
    flux_like_err.setVal(amrex::Real(0.0));

    const std::array<PrimitiveState, 2> initial_states = make_velocity_cfl_sensitive_column_states();
    fill_velocity_cfl_sensitive_column_state_portable(cons, initial_states);
    const SedimentationColumnRegressionReference reference =
        make_sedimentation_cfl_regression_reference(initial_states, kDefaultDt, geom.CellSize(2));

    EXPECT_NE(reference.velocity_substeps, reference.flux_like_substeps);

    fill_velocity_cfl_sensitive_column_state_portable(velocity_cons, reference.velocity_reference.states);
    fill_velocity_cfl_sensitive_column_state_portable(flux_like_cons, reference.flux_like_reference.states);
    fill_reference_rain_accum_portable(velocity_rain, reference.velocity_reference.rain_accum);
    fill_reference_rain_accum_portable(flux_like_rain, reference.flux_like_reference.rain_accum);

    std::unique_ptr<amrex::MultiFab> z_phys_nd;
    std::unique_ptr<amrex::MultiFab> detJ_cc;
    Kessler kessler;
    SolverChoice sc = make_solver_choice(MoistureType::Kessler, true);
    run_kessler_public_flow(kessler, sc, geom, cons, z_phys_nd, detJ_cc);

    compute_rain_state_difference(cons, velocity_cons, *kessler.Qmoist_Ptr(0), velocity_rain, velocity_err);
    compute_rain_state_difference(cons, flux_like_cons, *kessler.Qmoist_Ptr(0), flux_like_rain, flux_like_err);

    const amrex::Real qp_scale = std::max(reference.velocity_reference.states[0].rho * reference.velocity_reference.states[0].qp,
                                          reference.velocity_reference.states[1].rho * reference.velocity_reference.states[1].qp);
    const amrex::Real qp_tol = sedimentation_cfl_reference_tol(reference.velocity_substeps, qp_scale);
    const amrex::Real rain_tol = sedimentation_cfl_reference_tol(reference.velocity_substeps,
                                                                 reference.velocity_reference.rain_accum);

    EXPECT_LE(velocity_err.max(RainStateErrQp), qp_tol);
    EXPECT_LE(velocity_err.max(RainStateErrRainAccum), rain_tol);
    EXPECT_GT(std::max(flux_like_err.max(RainStateErrQp), flux_like_err.max(RainStateErrRainAccum)),
              std::max(qp_tol, rain_tol));
}

// Motivation: This is a characterization test, not a correctness test. It
// pins current behavior so that any future change must also address
// `DISABLED_KesslerPublicFlow_LocalSourcesConserveTotalWaterAndLatentEnergyProxy`.
TEST(KesslerPhysicalProperties, Characterization_KesslerPublicFlow_LocalSourcesLatentEnergyProxyCurrentBehavior)
{
    const amrex::Geometry geom = make_geometry(4, 3, 2);
    amrex::BoxArray ba(geom.Domain());
    amrex::DistributionMapping dm(ba);
    amrex::MultiFab cons(ba, dm, RhoQ3_comp + 1, 1);
    cons.setVal(amrex::Real(0.0));
    fill_local_source_only_state_portable(cons);
    amrex::MultiFab cons_before(ba, dm, RhoQ3_comp + 1, 1);
    copy_multifab(cons, cons_before);

    std::unique_ptr<amrex::MultiFab> z_phys_nd;
    std::unique_ptr<amrex::MultiFab> detJ_cc;
    Kessler kessler;
    SolverChoice sc = make_solver_choice(MoistureType::Kessler, false);
    run_kessler_public_flow(kessler, sc, geom, cons, z_phys_nd, detJ_cc);

    const ColumnBudget before = compute_column_budget(geom, cons_before, *kessler.Qmoist_Ptr(0));
    const ColumnBudget after = compute_column_budget(geom, cons, *kessler.Qmoist_Ptr(0));
    const int num_terms = geom.Domain().numPts();

    EXPECT_NEAR(after.rho_sum, before.rho_sum, property_accumulation_tol(num_terms, before.rho_sum));
    EXPECT_NEAR(after.total_water_sum + after.rain_accum_sum,
                before.total_water_sum,
                property_accumulation_tol(num_terms, before.total_water_sum));
    EXPECT_GT(std::abs(after.latent_proxy_sum - before.latent_proxy_sum),
              property_accumulation_tol(num_terms, before.latent_proxy_sum));
}

// Motivation: Expected behavior:
//   In the no-sedimentation public-flow case, rho, rho*(qv+qc+qp), and the
//   latent-energy proxy rho*[T + (lcond/cp) qv] should be conserved.
// Observed current behavior:
//   Current development conserves rho and total water here, but the latent-
//   energy proxy exposes the saturation/theta latent-factor inconsistency.
// Minimal reproducer:
//   A 4x3x2 deterministic fixture with qp == 0 in every cell.
// Why disabled:
//   The current production latent factors are intentionally preserved.
// Enable trigger:
//   Production should adopt a consistent latent factor across saturation and
//   theta updates.
TEST(KesslerPhysicalProperties, DISABLED_KesslerPublicFlow_LocalSourcesConserveTotalWaterAndLatentEnergyProxy)
{
    const amrex::Geometry geom = make_geometry(4, 3, 2);
    amrex::BoxArray ba(geom.Domain());
    amrex::DistributionMapping dm(ba);
    amrex::MultiFab cons(ba, dm, RhoQ3_comp + 1, 1);
    cons.setVal(amrex::Real(0.0));
    fill_local_source_only_state_portable(cons);
    amrex::MultiFab cons_before(ba, dm, RhoQ3_comp + 1, 1);
    copy_multifab(cons, cons_before);

    std::unique_ptr<amrex::MultiFab> z_phys_nd;
    std::unique_ptr<amrex::MultiFab> detJ_cc;
    Kessler kessler;
    SolverChoice sc = make_solver_choice(MoistureType::Kessler, false);
    run_kessler_public_flow(kessler, sc, geom, cons, z_phys_nd, detJ_cc);

    const ColumnBudget before = compute_column_budget(geom, cons_before, *kessler.Qmoist_Ptr(0));
    const ColumnBudget after = compute_column_budget(geom, cons, *kessler.Qmoist_Ptr(0));
    const int num_terms = geom.Domain().numPts();

    EXPECT_NEAR(after.rho_sum, before.rho_sum, property_accumulation_tol(num_terms, before.rho_sum));
    EXPECT_NEAR(after.total_water_sum + after.rain_accum_sum,
                before.total_water_sum,
                property_accumulation_tol(num_terms, before.total_water_sum));
    EXPECT_NEAR(after.latent_proxy_sum,
                before.latent_proxy_sum,
                property_accumulation_tol(num_terms, before.latent_proxy_sum));
}

// Motivation: This is a characterization test, not a correctness test. It
// pins current behavior so that any future change must also address
// `DISABLED_KesslerPublicFlow_ColumnWaterBudgetIncludesSurfaceRain`.
TEST(KesslerPhysicalProperties, Characterization_ColumnWaterBudgetCurrentBehavior)
{
    const amrex::Geometry geom = make_geometry(1, 1, 4);
    amrex::BoxArray ba(geom.Domain());
    amrex::DistributionMapping dm(ba);
    amrex::MultiFab cons(ba, dm, RhoQ3_comp + 1, 1);
    cons.setVal(amrex::Real(0.0));
    fill_conserved_column_state_portable(cons);
    amrex::MultiFab cons_before(ba, dm, RhoQ3_comp + 1, 1);
    copy_multifab(cons, cons_before);

    std::unique_ptr<amrex::MultiFab> z_phys_nd;
    std::unique_ptr<amrex::MultiFab> detJ_cc;
    Kessler kessler;
    SolverChoice sc = make_solver_choice(MoistureType::Kessler, false);
    run_kessler_public_flow(kessler, sc, geom, cons, z_phys_nd, detJ_cc);

    const ColumnBudget before = compute_column_budget(geom, cons_before, *kessler.Qmoist_Ptr(0));
    const ColumnBudget after = compute_column_budget(geom, cons, *kessler.Qmoist_Ptr(0));
    const amrex::Real lhs = before.total_water_sum;
    const amrex::Real rhs = after.total_water_sum + after.rain_accum_sum;

    EXPECT_GT(rhs, lhs);
    EXPECT_GT(std::abs(lhs - rhs), property_accumulation_tol(4, lhs));
}

// Motivation: Expected behavior:
//   In a flat column with zero top precipitation flux, total column water plus
//   surface rain accumulation should be conserved.
// Observed current behavior:
//   Current development produces more final column water plus rain accumulation
//   than the initial column water for this deterministic column fixture.
// Minimal reproducer:
//   A 1x1x4 flat column with top-cell qp == 0 and nonzero interior/bottom qp.
// Why disabled:
//   This is a suspected production bug and must not be weakened in a tests-only
//   prompt.
// Enable trigger:
//   Production should satisfy the public-flow rain-budget contract for the
//   deterministic column fixture.
TEST(KesslerPhysicalProperties, DISABLED_KesslerPublicFlow_ColumnWaterBudgetIncludesSurfaceRain)
{
    const amrex::Geometry geom = make_geometry(1, 1, 4);
    amrex::BoxArray ba(geom.Domain());
    amrex::DistributionMapping dm(ba);
    amrex::MultiFab cons(ba, dm, RhoQ3_comp + 1, 1);
    cons.setVal(amrex::Real(0.0));
    fill_conserved_column_state_portable(cons);
    amrex::MultiFab cons_before(ba, dm, RhoQ3_comp + 1, 1);
    copy_multifab(cons, cons_before);

    std::unique_ptr<amrex::MultiFab> z_phys_nd;
    std::unique_ptr<amrex::MultiFab> detJ_cc;
    Kessler kessler;
    SolverChoice sc = make_solver_choice(MoistureType::Kessler, false);
    run_kessler_public_flow(kessler, sc, geom, cons, z_phys_nd, detJ_cc);

    const ColumnBudget before = compute_column_budget(geom, cons_before, *kessler.Qmoist_Ptr(0));
    const ColumnBudget after = compute_column_budget(geom, cons, *kessler.Qmoist_Ptr(0));
    const amrex::Real lhs = before.total_water_sum;
    const amrex::Real rhs = after.total_water_sum + after.rain_accum_sum;

    EXPECT_NEAR(lhs, rhs, property_accumulation_tol(4, lhs));
}

// Motivation: Expected behavior:
//   With detJ weighting present, initial sum detJ*rho*(qv+qc+qp)*dz should
//   equal final detJ-weighted column water plus surface rain.
// Observed current behavior:
//   detJ-weighted rain-budget semantics are not yet documented by production.
// Minimal reproducer:
//   A 1x1x4 column with deterministic detJ values and zero top qp.
// Why disabled:
//   This remains a documentation-contract question pending physics-owner
//   confirmation.
// Enable trigger:
//   Document and confirm the detJ-weighted rain-budget convention.
TEST(KesslerPhysicalProperties, DISABLED_KesslerPublicFlow_DetJWeightedColumnWaterBudgetIncludesSurfaceRain)
{
    const amrex::Geometry geom = make_geometry(1, 1, 4);
    amrex::BoxArray ba(geom.Domain());
    amrex::DistributionMapping dm(ba);
    amrex::MultiFab cons(ba, dm, RhoQ3_comp + 1, 1);
    cons.setVal(amrex::Real(0.0));
    fill_conserved_column_state_portable(cons);
    amrex::MultiFab cons_before(ba, dm, RhoQ3_comp + 1, 1);
    copy_multifab(cons, cons_before);

    std::unique_ptr<amrex::MultiFab> z_phys_nd;
    std::unique_ptr<amrex::MultiFab> detJ_cc = std::make_unique<amrex::MultiFab>(ba, dm, 1, 1);
    detJ_cc->setVal(amrex::Real(1.0));
    fill_detj_portable(*detJ_cc);

    Kessler kessler;
    SolverChoice sc = make_solver_choice(MoistureType::Kessler, false);
    run_kessler_public_flow(kessler, sc, geom, cons, z_phys_nd, detJ_cc);

    const ColumnBudget before = compute_column_budget(geom, cons_before, *kessler.Qmoist_Ptr(0), detJ_cc.get());
    const ColumnBudget after = compute_column_budget(geom, cons, *kessler.Qmoist_Ptr(0), detJ_cc.get());

    EXPECT_NEAR(before.total_water_sum,
                after.total_water_sum + after.rain_accum_sum,
                property_accumulation_tol(4, before.total_water_sum));
}

// Motivation: Kessler_NoRain through the public flow should leave rain water
// and rain accumulation untouched. This protects the distinction between the
// NoRain path and the full-rain path.
TEST(KesslerPhysicalProperties, KesslerNoRain_PublicFlowDoesNotModifyRainWaterOrRainAccum)
{
    const amrex::Geometry geom = make_geometry(1, 1, 4);
    amrex::BoxArray ba(geom.Domain());
    amrex::DistributionMapping dm(ba);
    amrex::MultiFab cons(ba, dm, RhoQ3_comp + 1, 1);
    amrex::MultiFab cons_before(ba, dm, RhoQ3_comp + 1, 1);
    amrex::MultiFab err(ba, dm, NumRainStateErrComps, 0);
    cons.setVal(amrex::Real(0.0));
    cons_before.setVal(amrex::Real(0.0));
    err.setVal(amrex::Real(0.0));
    fill_conserved_column_state_portable(cons);
    copy_multifab(cons, cons_before);

    std::unique_ptr<amrex::MultiFab> z_phys_nd;
    std::unique_ptr<amrex::MultiFab> detJ_cc;
    Kessler kessler;
    SolverChoice sc = make_solver_choice(MoistureType::Kessler_NoRain, true);

    initialize_kessler(kessler, sc, geom, cons, z_phys_nd, detJ_cc);
    run_and_sync([&] { kessler.Copy_State_to_Micro(cons); });
    amrex::MultiFab rain_before(ba, dm, 1, 1);
    copy_multifab(*kessler.Qmoist_Ptr(0), rain_before);
    run_and_sync([&] {
        kessler.Advance(kDefaultDt, sc);
        kessler.Copy_Micro_to_State(cons);
    });
    compute_rain_state_difference(cons, cons_before, *kessler.Qmoist_Ptr(0), rain_before, err);

    EXPECT_LE(err.max(RainStateErrQp), formula_abs_tol(cons_before.max(RhoQ3_comp)));
    EXPECT_LE(err.max(RainStateErrRainAccum), exact_zero_or_near_zero_tol());
}

// Motivation: Full Kessler with SHOC condensation disabled still advances
// warm-rain source terms and sedimentation. This protects the distinction
// between `Kessler && !do_cond` and the NoRain early-return path.
TEST(KesslerPhysicalProperties, KesslerFullRain_DoCondFalseStillRunsRainPath)
{
    const amrex::Geometry geom = make_geometry(1, 1, 4);
    amrex::BoxArray ba(geom.Domain());
    amrex::DistributionMapping dm(ba);
    amrex::MultiFab cons(ba, dm, RhoQ3_comp + 1, 1);
    cons.setVal(amrex::Real(0.0));
    fill_do_cond_false_rain_path_state_portable(cons);
    amrex::MultiFab cons_before(ba, dm, RhoQ3_comp + 1, 1);
    copy_multifab(cons, cons_before);

    std::unique_ptr<amrex::MultiFab> z_phys_nd;
    std::unique_ptr<amrex::MultiFab> detJ_cc;
    Kessler kessler;
    SolverChoice sc = make_solver_choice(MoistureType::Kessler, true);
    run_kessler_public_flow(kessler, sc, geom, cons, z_phys_nd, detJ_cc);

    const amrex::Real qv_before = cons_before.max(RhoQ1_comp);
    const amrex::Real qv_after = cons.max(RhoQ1_comp);
    const amrex::Real qc_before = cons_before.max(RhoQ2_comp);
    const amrex::Real qc_after = cons.max(RhoQ2_comp);
    const amrex::Real rain_accum_after = kessler.Qmoist_Ptr(0)->max(0);

    EXPECT_NEAR(qv_after, qv_before, property_accumulation_tol(4, qv_before));
    EXPECT_LT(qc_after, qc_before);
    EXPECT_GT(rain_accum_after, amrex::Real(0.0));
}