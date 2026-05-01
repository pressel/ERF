#include "ERF_ShocEnergyFixer.H"

#include "ERF_Constants.H"

#include <algorithm>

using namespace amrex;

namespace
{
    constexpr Real k_shoc_min_tke = 4.0e-4;
    constexpr Real k_shoc_lat_ice = 3.34e5;

    AMREX_FORCE_INLINE
    Real shoc_temperature (Real thetal, Real ql, Real exner)
    {
        return thetal * std::max(exner, 1.0e-12) + (L_v / Cp_d) * std::max(0.0_rt, ql);
    }

    AMREX_FORCE_INLINE
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
    const int nlev = col.layout.nlev;

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
