#ifndef ERF_INIT_DYCOMS_H_
#define ERF_INIT_DYCOMS_H_

#include <prob_parm.H>

#include <ERF_DataStruct.H>
#include <ERF_EOS.H>
#include <ERF_MicrophysicsUtils.H>
#include <ERF_TerrainMetrics.H>
#include <AMReX_Algorithm.H>

#include <cmath>

#include <AMReX_GpuQualifiers.H>
#include <AMReX_Array.H>
#include <AMReX_Math.H>

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real
dycoms_theta_l (const ProbParm& parms, amrex::Real z)
{
    // RF02 specification: sharp discontinuity at zi (no smooth transition)
    // Below zi: constant BL value θ_l = 288.3 K
    // Above zi: free troposphere profile θ_l = 295 + (z-zi)^(1/3) K
    
    if (z < parms.zi) {
        return parms.theta_l_bl;  // 288.3 K below zi
    } else {
        const amrex::Real delta = z - parms.zi;
        return parms.theta_l_free_base + std::pow(delta, amrex::Real(1.0/3.0));
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real
dycoms_qt (const ProbParm& parms, amrex::Real z)
{
    // RF02 specification: sharp discontinuity at zi (no smooth transition)
    // Below zi: constant BL value qt = 9.45 g/kg
    // Above zi: exponential decay qt = 2 + 3*exp(-(z-zi)/500) g/kg
    //   At zi: qt = 2 + 3 = 5 g/kg (jump from 9.45 → 5 g/kg)
    //   As z→∞: qt → 2 g/kg
    
    if (z < parms.zi) {
        return parms.qt_bl;  // 9.45e-3 kg/kg below zi
    } else {
        const amrex::Real delta = z - parms.zi;
        return parms.qt_free_asymptote + 
               parms.qt_free_excess * std::exp(-delta / parms.qt_free_scale);
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real
dycoms_u (const ProbParm& /*parms*/, amrex::Real z)
{
    return 3.0 + 4.3e-3 * z;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real
dycoms_v (const ProbParm& /*parms*/, amrex::Real z)
{
    return -9.0 + 5.6e-3 * z;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real
dycoms_cell_height (int i, int j, int k,
                    const amrex::Array4<const amrex::Real>& z_nd,
                    const amrex::GeometryData& geomdata,
                    bool has_terrain)
{
    if (has_terrain) {
        return Compute_Zrel_AtCellCenter(i, j, k, z_nd);
    }
    return geomdata.ProbLo(2) + (k + 0.5) * geomdata.CellSize(2);
}

/**
 * DyCOMS-II RF02 Moist State Solver with Linearized θ_ℓ
 * ======================================================
 * 
 * Solves for (θ, T, qᵥ, qₗ) given (θ_ℓ_lin, qₜ, p₀) using the RF02 specification:
 * 
 *   θ = θ_ℓ_lin + (Lᵥ/(cₚ·π₀)) · qₗ
 * 
 * where π₀(z) = (p₀(z)/p₀)^(Rₐ/cₚ) is the Exner function held FIXED at the base-state
 * pressure p₀(z).
 * 
 * Supersaturation: Enforces S=0 by default (qᵥ = min(qₜ, qₛₐₜ(T,p₀))).
 * 
 * Usage for Anelastic:
 * - Hold (p₀, π₀, ρ₀) FIXED during saturation adjustment.
 * - Do NOT hold T fixed; compute T = θ·π₀ after solving.
 * - Initialize with ρ' = 0 (thermo fields consistent with ρ₀).
 * 
 * Usage for Compressible:
 * - Integrate p(z) hydrostatically using the resulting moist density.
 * 
 * Newton Iteration:
 * For saturated parcels, solve g(T) = T - π₀·θ_ℓ_lin - (Lᵥ/cₚ)·(qₜ - qₛₐₜ(T,p₀)) = 0
 * with g'(T) = 1 + (Lᵥ/cₚ)·∂qₛₐₜ/∂T.
 * Typically converges in 2-6 iterations.
 * 
 * @param[in]  pressure  Base-state pressure p₀(z) [Pa]
 * @param[in]  theta_l   Linearized liquid-water potential temperature θ_ℓ_lin [K]
 * @param[in]  qt        Total water mixing ratio qₜ = qᵥ + qₗ [kg/kg]
 * @param[in]  rdOcp     Gas constant ratio Rₐ/cₚ [dimensionless]
 * @param[out] rho       Moist air density [kg/m³]
 * @param[out] theta     Potential temperature θ [K]
 * @param[out] temp      Temperature T [K]
 * @param[out] qv        Water vapor mixing ratio qᵥ [kg/kg]
 * @param[out] qc        Cloud liquid water mixing ratio qₗ [kg/kg]
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void dycoms_compute_moist_state (amrex::Real pressure,
                                 amrex::Real theta_l,
                                 amrex::Real qt,
                                 amrex::Real rdOcp,
                                 amrex::Real& rho,
                                 amrex::Real& theta,
                                 amrex::Real& temp,
                                 amrex::Real& qv,
                                 amrex::Real& qc)
{
    using namespace amrex;

    // Convert pressure to mb for saturation routines (which expect hPa/mb)
    const Real p_mb = 0.01_rt * pressure;  // Pa → mb
    
    // Compute and HOLD FIXED the Exner function π₀ at base-state pressure
    const Real pi0 = getExnergivenP(pressure, rdOcp);
    
    // =====================================================================
    // Step 1: Quick unsaturated check
    // =====================================================================
    // If unsaturated, θ_ℓ = θ (no liquid) so T₀ = π₀·θ_ℓ_lin
    const Real T0 = pi0 * theta_l;
    Real qsat_0;
    erf_qsatw(T0, p_mb, qsat_0);
    
    if (qt <= qsat_0) {
        // Unsaturated: no iteration needed
        qv    = qt;
        qc    = 0.0_rt;
        theta = theta_l;  // θ = θ_ℓ when qₗ=0
        temp  = T0;
        rho   = getRhogivenTandPress(T0, pressure, qv);
        return;
    }

    // =====================================================================
    // Step 2: Saturated solve with linearized θ_ℓ at FIXED π₀
    // =====================================================================
    // Solve: g(T) = T - π₀·θ_ℓ_lin - (Lᵥ/cₚ)·(qₜ - qₛₐₜ(T, p₀)) = 0
    // with derivative g'(T) = 1 + (Lᵥ/cₚ)·∂qₛₐₜ/∂T
    
    Real T = T0;  // Start from unsaturated estimate

    for (int it = 0; it < 12; ++it) {
        Real qsat_iter, dqsat_dT;
        erf_qsatw   (T, p_mb, qsat_iter);
        erf_dtqsatw (T, p_mb, dqsat_dT);

        const Real g  = T - pi0 * theta_l - (lat_vap / Cp_d) * (qt - qsat_iter);
        const Real gp = 1.0_rt + (lat_vap / Cp_d) * dqsat_dT;

        Real dT = -g / amrex::max(gp, 1.0e-12_rt);
        
        // Clamp step size for robustness
        dT = amrex::min(amrex::max(dT, -5.0_rt), 5.0_rt);
        T += dT;

        // Converged when |ΔT| < 0.1 mK
        if (amrex::Math::abs(dT) < 1.0e-4_rt) break;
    }

    // =====================================================================
    // Step 3: Finalize with S=0 (supersaturation removed)
    // =====================================================================
    Real qsat_final;
    erf_qsatw(T, p_mb, qsat_final);

    qv = amrex::min(qt, qsat_final);  // Enforce S=0: any excess → condensate
    qc = amrex::max(0.0_rt, qt - qv);

    // RF02 linearized θ_ℓ relation: θ = θ_ℓ_lin + (Lᵥ/(cₚ·π₀))·qₗ
    theta = theta_l + (lat_vap / (Cp_d * pi0)) * qc;
    
    // Temperature consistent with θ and π₀
    temp = T;  // Already solved to satisfy T = π₀·θ_ℓ_lin + (Lᵥ/cₚ)·qₗ
    
    // Moist density using ideal gas law with vapor mixing ratio
    rho = getRhogivenTandPress(T, pressure, qv);
}

#endif
