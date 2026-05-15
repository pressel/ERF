#include "ERF_Constants.H"
#include "ERF_ShocImplicit.H"
#include "ERF_ShocPDF.H"
#include "ERF_ShocStructure.H"
#include "ERF_ShocTKE.H"
#include "ERF_ShocMoments.H"
#include "ERF_ShocTestUtils.H"

#include <gtest/gtest.h>

namespace
{
amrex::Real
column_moist_energy (const ShocColumnData& col)
{
    const auto rho = col.rho.const_array();
    const auto dz = col.dz.const_array();
    const auto tabs = col.tabs.const_array();
    const auto zt = col.zt.const_array();
    const auto qv = col.qv.const_array();
    const auto qc = col.qc.const_array();
    const auto qi = col.qi.const_array();
    const auto u = col.u.const_array();
    const auto v = col.v.const_array();
    const auto tke = col.tke.const_array();
    amrex::Real total = 0.0;
    for (int k = 0; k < col.layout.nlev; ++k) {
        const amrex::Real mass = rho(0,k,0) * dz(0,k,0);
        total += mass
               * (Cp_d * tabs(0,k,0)
                  + CONST_GRAV * zt(0,k,0)
                  + amrex::Real(0.5) * (u(0,k,0) * u(0,k,0) + v(0,k,0) * v(0,k,0))
                  + tke(0,k,0)
                  + L_v * (qv(0,k,0) + qc(0,k,0))
                  + (L_v + amrex::Real(3.34e5)) * qi(0,k,0));
    }
    return total;
}
}

TEST(ShocPhysical, ColumnHeatBudgetTracksSurfaceFlux)
{
    auto col = shoc_test::make_column(6);
    ShocRuntimeOptions opts;

    auto tk = col.tk.array();
    auto tkh = col.tkh.array();
    auto exner = col.exner.array();
    auto thetal = col.thetal.array();
    auto theta = col.theta.array();
    auto tabs = col.tabs.array();
    auto host_dse = col.host_dse.array();
    auto qv = col.qv.array();
    auto qc = col.qc.array();
    auto qi = col.qi.array();
    auto qw = col.qw.array();
    for (int k = 0; k < col.layout.nlev; ++k) {
        tk(0,k,0) = 0.0;
        tkh(0,k,0) = 0.0;
        exner(0,k,0) = 1.0;
        thetal(0,k,0) = 300.0;
        theta(0,k,0) = 300.0;
        tabs(0,k,0) = 300.0;
        qv(0,k,0) = 0.008;
        qc(0,k,0) = 0.0;
        qi(0,k,0) = 0.0;
        qw(0,k,0) = qv(0,k,0);
        host_dse(0,k,0) = Cp_d * tabs(0,k,0) + CONST_GRAV * col.zt.const_array()(0,k,0);
        col.tke_tend.array()(0,k,0) = 0.0;
    }
    shoc::set_fab_val(col.surf_sens_flux, 0.02, shoc::InitRunOn::Host);
    shoc::set_fab_val(col.surf_lat_flux, 0.0, shoc::InitRunOn::Host);
    shoc::set_fab_val(col.surf_tau_u, 0.0, shoc::InitRunOn::Host);
    shoc::set_fab_val(col.surf_tau_v, 0.0, shoc::InitRunOn::Host);

    const amrex::Real before = column_moist_energy(col);
    const amrex::Real dt = 10.0;
    const amrex::Real rho_sfc = col.rho.const_array()(0,0,0);

    ShocImplicit::update_prognostics(col, opts, dt);

    const amrex::Real after = column_moist_energy(col);
    const amrex::Real expected = dt * rho_sfc * Cp_d * 0.02;
    EXPECT_NEAR(after - before, expected, 5.0e-9 * std::max(amrex::Real(1.0), std::abs(expected)));
}

TEST(ShocPhysical, StrongerSurfaceHeatingRaisesMeanThetaMore)
{
    auto weak = shoc_test::make_column(6);
    auto strong = shoc_test::make_column(6);
    ShocRuntimeOptions opts;

    for (auto* col : {&weak, &strong}) {
        auto tk = col->tk.array();
        auto tkh = col->tkh.array();
        auto exner = col->exner.array();
        for (int k = 0; k < col->layout.nlev; ++k) {
            tk(0,k,0) = 1.0;
            tkh(0,k,0) = 1.0;
            exner(0,k,0) = 1.0;
            col->tke_tend.array()(0,k,0) = 0.0;
        }
    }

    shoc::set_fab_val(weak.surf_sens_flux, 0.01, shoc::InitRunOn::Host);
    shoc::set_fab_val(strong.surf_sens_flux, 0.03, shoc::InitRunOn::Host);

    const amrex::Real weak_before = weak.thetal.const_array()(0,0,0);
    const amrex::Real strong_before = strong.thetal.const_array()(0,0,0);

    ShocImplicit::update_prognostics(weak, opts, 10.0);
    ShocImplicit::update_prognostics(strong, opts, 10.0);

    const amrex::Real weak_delta = weak.thetal.const_array()(0,0,0) - weak_before;
    const amrex::Real strong_delta = strong.thetal.const_array()(0,0,0) - strong_before;
    EXPECT_GT(strong_delta, weak_delta);
}

TEST(ShocPhysical, UnstableSurfaceForcingDeepensPblMoreThanStableForcing)
{
    auto stable = shoc_test::make_column(8);
    auto unstable = shoc_test::make_column(8);

    shoc::set_fab_val(stable.surf_sens_flux, -0.01, shoc::InitRunOn::Host);
    shoc::set_fab_val(unstable.surf_sens_flux, 0.02, shoc::InitRunOn::Host);
    shoc::set_fab_val(stable.surf_lat_flux, 0.0, shoc::InitRunOn::Host);
    shoc::set_fab_val(unstable.surf_lat_flux, 0.0, shoc::InitRunOn::Host);

    ShocStructure::diagnose_surface_layer(stable);
    ShocStructure::diagnose_pblh(stable);
    ShocStructure::diagnose_surface_layer(unstable);
    ShocStructure::diagnose_pblh(unstable);

    EXPECT_GE(unstable.pblh.const_array()(0,0,0), stable.pblh.const_array()(0,0,0));
}

TEST(ShocPhysical, TwoSmallStepsTrackOneLargeStep)
{
    auto one_step = shoc_test::make_column(5);
    auto two_step = shoc_test::make_column(5);
    ShocRuntimeOptions opts;

    for (auto* col : {&one_step, &two_step}) {
        auto tk = col->tk.array();
        auto tkh = col->tkh.array();
        auto exner = col->exner.array();
        for (int k = 0; k < col->layout.nlev; ++k) {
            tk(0,k,0) = 1.0;
            tkh(0,k,0) = 1.0;
            exner(0,k,0) = 1.0;
            col->tke_tend.array()(0,k,0) = 0.0;
        }
        shoc::set_fab_val(col->surf_sens_flux, 0.02, shoc::InitRunOn::Host);
        shoc::set_fab_val(col->surf_lat_flux, 5.0e-5, shoc::InitRunOn::Host);
    }

    ShocImplicit::update_prognostics(one_step, opts, 10.0);
    ShocImplicit::update_prognostics(two_step, opts, 5.0);
    ShocImplicit::update_prognostics(two_step, opts, 5.0);

    const auto th_one = one_step.thetal.const_array();
    const auto th_two = two_step.thetal.const_array();
    const auto qw_one = one_step.qw.const_array();
    const auto qw_two = two_step.qw.const_array();

    for (int k = 0; k < one_step.layout.nlev; ++k) {
        EXPECT_NEAR(th_one(0,k,0), th_two(0,k,0), 5.0e-3);
        EXPECT_NEAR(qw_one(0,k,0), qw_two(0,k,0), 5.0e-5);
    }
}

TEST(ShocPhysical, SubfreezingCondensatePartitionsIntoQi)
{
    auto col = shoc_test::make_column(5);
    ShocRuntimeOptions opts;

    auto exner = col.exner.array();
    auto thetal = col.thetal.array();
    auto qw = col.qw.array();
    auto qv = col.qv.array();
    auto qc = col.qc.array();
    auto qi = col.qi.array();
    auto tabs = col.tabs.array();
    auto tk = col.tk.array();
    auto tkh = col.tkh.array();

    for (int k = 0; k < col.layout.nlev; ++k) {
        exner(0,k,0) = 1.0;
        thetal(0,k,0) = 265.0;
        tabs(0,k,0) = 265.0;
        qw(0,k,0) = 0.004;
        qv(0,k,0) = 0.003;
        qc(0,k,0) = 0.0;
        qi(0,k,0) = 1.0e-3;
        tk(0,k,0) = 0.1;
        tkh(0,k,0) = 0.1;
        col.tke_tend.array()(0,k,0) = 0.0;
    }

    ShocImplicit::update_prognostics(col, opts, 10.0);

    const auto qc_new = col.qc.const_array();
    const auto qi_new = col.qi.const_array();
    for (int k = 0; k < col.layout.nlev; ++k) {
        EXPECT_GE(qi_new(0,k,0), qc_new(0,k,0));
    }
}

TEST(ShocPhysical, TerrainLikeVerticalGridStaysFinite)
{
    auto col = shoc_test::make_column(6);
    ShocRuntimeOptions opts;
    auto zi = col.zi.array();
    auto zt = col.zt.array();
    auto dz = col.dz.array();

    for (int k = 0; k <= col.layout.nlev; ++k) {
        zi(0,k,0) = 15.0 * k + 2.5 * k * k;
        if (k < col.layout.nlev) {
            const amrex::Real zlo = zi(0,k,0);
            const amrex::Real zhi = zi(0,k+1,0);
            zt(0,k,0) = 0.5 * (zlo + zhi);
            dz(0,k,0) = zhi - zlo;
        }
    }

    ShocStructure::diagnose_surface_layer(col);
    ShocStructure::diagnose_pblh(col);
    ShocStructure::diagnose_length_and_brunt(col, opts, 300.0, 500.0);
    ShocTKE::diagnose_tke_and_diffusivities(col, opts, 300.0);
    ShocMoments::diagnose_moments(col, opts);
    ShocPDF::diagnose_pdf(col, opts, 10.0);

    const auto mix = col.shoc_mix.const_array();
    const auto brunt = col.brunt.const_array();
    const auto cldfrac = col.shoc_cldfrac.const_array();
    for (int k = 0; k < col.layout.nlev; ++k) {
        EXPECT_TRUE(std::isfinite(mix(0,k,0)));
        EXPECT_TRUE(std::isfinite(brunt(0,k,0)));
        EXPECT_TRUE(std::isfinite(cldfrac(0,k,0)));
    }
}
