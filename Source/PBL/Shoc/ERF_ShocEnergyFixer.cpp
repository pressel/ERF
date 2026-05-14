#include "ERF_ShocEnergyFixer.H"

#include "ERF_Constants.H"

#include <algorithm>

using namespace amrex;

namespace
{
    constexpr Real k_shoc_min_tke = 4.0e-4;
    constexpr Real k_shoc_lat_ice = 3.34e5;
    constexpr Real k_shoc_min_temp = 180.0;
    constexpr Real k_shoc_freezing_temp = 273.15;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Real shoc_temperature (Real thetal, Real ql, Real exner)
    {
        return amrex::max(k_shoc_min_temp,
                          thetal * amrex::max(exner, 1.0e-12_rt) + (L_v / Cp_d) * amrex::max(0.0_rt, ql));
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Real moist_energy (Real tabs, Real z, Real qv, Real qc, Real qi,
                       Real u, Real v, Real tke)
    {
        const Real ql = std::max(0.0_rt, qc + qi);
        return Cp_d * tabs
             + CONST_GRAV * z
             + 0.5 * (u * u + v * v)
             + tke
             + (L_v + k_shoc_lat_ice) * qv
             + k_shoc_lat_ice * ql;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void reconstruct_pdf_state (Real thetal,
                                Real qw,
                                Real exner,
                                Real qi_seed,
                                Real pdf_ql,
                                Real& tabs,
                                Real& qv,
                                Real& qc,
                                Real& qi)
    {
        const Real ql_total = shoc_clamp(pdf_ql, 0.0_rt, amrex::max(0.0_rt, qw));
        tabs = shoc_temperature(thetal, ql_total, exner);

        const bool use_ice = (tabs < k_shoc_freezing_temp && qi_seed > 0.0_rt);
        qc = use_ice ? 0.0_rt : ql_total;
        qi = use_ice ? ql_total : 0.0_rt;
        qv = amrex::max(0.0_rt, qw - amrex::max(0.0_rt, ql_total));
    }
}

int
ShocEnergyFixer::diagnose_active_top (const Vector<Real>& tke)
{
    int shoc_top = -1;
    for (int k = static_cast<int>(tke.size()) - 1; k >= 0; --k) {
        if (tke[k] > k_shoc_min_tke) {
            shoc_top = k;
            break;
        }
    }
    return shoc_top;
}

AMREX_GPU_HOST_DEVICE
int
ShocEnergyFixer::diagnose_active_top (const Array4<const Real>& tke,
                                      int ic,
                                      int nlev)
{
    int shoc_top = -1;
    for (int k = nlev - 1; k >= 0; --k) {
        if (tke(ic,k,0) > k_shoc_min_tke) {
            shoc_top = k;
            break;
        }
    }
    return shoc_top;
}

void
ShocEnergyFixer::apply_column (const ShocColumnData& col,
                               int ic,
                               Real dt,
                               const Vector<Real>& thl_old,
                               const Vector<Real>& qv_old,
                               const Vector<Real>& qc_old,
                               const Vector<Real>& qi_old,
                               const Vector<Real>& u_old,
                               const Vector<Real>& v_old,
                               const Vector<Real>& tke_old,
                               Vector<Real>& thl_new,
                               const Vector<Real>& qv_new,
                               const Vector<Real>& qc_new,
                               const Vector<Real>& qi_new,
                               const Vector<Real>& u_new,
                               const Vector<Real>& v_new,
                               const Vector<Real>& tke_new)
{
    const auto rho = col.rho.const_array();
    const auto dz = col.dz.const_array();
    const auto zt = col.zt.const_array();
    const auto exner = col.exner.const_array();
    const auto surf_sens_flux = col.surf_sens_flux.const_array();
    const auto surf_lat_flux = col.surf_lat_flux.const_array();

    const int shoc_top = diagnose_active_top(tke_new);
    if (shoc_top < 0) return;

    Real energy_before = 0.0;
    Real energy_after = 0.0;
    Real air_mass = 0.0;

    for (int k = 0; k <= shoc_top; ++k) {
        const Real mass = rho(ic,k,0) * dz(ic,k,0);
        const Real tabs_old = shoc_temperature(thl_old[k], qc_old[k] + qi_old[k], exner(ic,k,0));
        const Real tabs_new = shoc_temperature(thl_new[k], qc_new[k] + qi_new[k], exner(ic,k,0));

        energy_before += mass * moist_energy(tabs_old, zt(ic,k,0),
                                             qv_old[k], qc_old[k], qi_old[k],
                                             u_old[k], v_old[k], tke_old[k]);
        energy_after += mass * moist_energy(tabs_new, zt(ic,k,0),
                                            qv_new[k], qc_new[k], qi_new[k],
                                            u_new[k], v_new[k], tke_new[k]);
        air_mass += mass;
    }

    if (air_mass <= 0.0) return;

    const Real latent_flux_coeff = L_v + k_shoc_lat_ice;
    const Real energy_target = energy_before
                             + dt * rho(ic,0,0)
                             * (Cp_d * exner(ic,0,0) * surf_sens_flux(ic,0,0)
                                + latent_flux_coeff * surf_lat_flux(ic,0,0));
    const Real delta_tabs = (energy_target - energy_after) / (Cp_d * air_mass);

    for (int k = 0; k <= shoc_top; ++k) {
        thl_new[k] += delta_tabs / std::max(exner(ic,k,0), 1.0e-12);
    }
}

AMREX_GPU_HOST_DEVICE
void
ShocEnergyFixer::apply_column_in_place (ShocColumnData& col,
                                        int ic,
                                        Real dt)
{
    const auto rho = col.rho.const_array();
    const auto dz = col.dz.const_array();
    const auto zt = col.zt.const_array();
    const auto exner = col.exner.const_array();
    const auto surf_sens_flux = col.surf_sens_flux.const_array();
    const auto surf_lat_flux = col.surf_lat_flux.const_array();
    const auto thetal_base = col.thetal_base.const_array();
    const auto qv_base = col.qv_base.const_array();
    const auto qc_base = col.qc_base.const_array();
    const auto qi_base = col.qi_base.const_array();
    const auto u_base = col.u_base.const_array();
    const auto v_base = col.v_base.const_array();
    const auto tke_base = col.tke_base_state.const_array();
    auto thetal = col.thetal.array();
    const auto qw = col.qw.const_array();
    const auto u = col.u.const_array();
    const auto v = col.v.const_array();
    const auto tke = col.tke.const_array();
    const auto shoc_ql = col.shoc_ql.const_array();
    const int nlev = col.layout.nlev;

    const int shoc_top = diagnose_active_top(tke, ic, nlev);
    if (shoc_top < 0) return;

    Real energy_before = 0.0_rt;
    Real energy_after = 0.0_rt;
    Real air_mass = 0.0_rt;

    for (int k = 0; k <= shoc_top; ++k) {
        const Real mass = rho(ic,k,0) * dz(ic,k,0);
        const Real tabs_old = shoc_temperature(thetal_base(ic,k,0), qc_base(ic,k,0) + qi_base(ic,k,0), exner(ic,k,0));

        Real tabs_new = 0.0_rt;
        Real qv_new = 0.0_rt;
        Real qc_new = 0.0_rt;
        Real qi_new = 0.0_rt;
        reconstruct_pdf_state(thetal(ic,k,0), qw(ic,k,0), exner(ic,k,0),
                              qi_base(ic,k,0), shoc_ql(ic,k,0),
                              tabs_new, qv_new, qc_new, qi_new);

        energy_before += mass * moist_energy(tabs_old, zt(ic,k,0),
                                             qv_base(ic,k,0), qc_base(ic,k,0), qi_base(ic,k,0),
                                             u_base(ic,k,0), v_base(ic,k,0), tke_base(ic,k,0));
        energy_after += mass * moist_energy(tabs_new, zt(ic,k,0),
                                            qv_new, qc_new, qi_new,
                                            u(ic,k,0), v(ic,k,0), tke(ic,k,0));
        air_mass += mass;
    }

    if (air_mass <= 0.0_rt) return;

    const Real latent_flux_coeff = L_v + k_shoc_lat_ice;
    const Real energy_target = energy_before
                             + dt * rho(ic,0,0)
                             * (Cp_d * exner(ic,0,0) * surf_sens_flux(ic,0,0)
                                + latent_flux_coeff * surf_lat_flux(ic,0,0));
    const Real delta_tabs = (energy_target - energy_after) / (Cp_d * air_mass);

    for (int k = 0; k <= shoc_top; ++k) {
        thetal(ic,k,0) += delta_tabs / amrex::max(exner(ic,k,0), 1.0e-12_rt);
    }
}
