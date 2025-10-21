#include "ERF_RadiationDyCOMS.H"

#include <ERF_IndexDefines.H>
#include <ERF_Constants.H>
#include <ERF_EOS.H>
#include <ERF_TerrainMetrics.H>

#include <AMReX_Geometry.H>
#include <AMReX_BoxArray.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Gpu.H>
#include <AMReX_Reduce.H>
#include <AMReX_ParallelDescriptor.H>

#include <cmath>
#include <iomanip>

using namespace amrex;

RadiationDyCOMS::RadiationDyCOMS (const SolverChoice& sc)
    : m_solver_choice(sc),
      m_F0(70.0),
      m_F1(22.0),
      m_kappa(85.0),
      m_qc_threshold(1.0e-7),
      m_a(1.0),
      m_rho_i(1.12),         // RF02 spec: 1.12 kg/m³ (inversion density)
      m_divergence(3.75e-6), // RF02 spec: 3.75e-6 s⁻¹ (large-scale divergence)
      m_zi(795.0),           // RF02 spec: 795 m (canonical inversion height)
      m_qt_threshold(0.008), // RF02 spec: 8 g/kg total water threshold for zi diagnostic
      m_use_local_zi(true)   // Use qt-based zi diagnostic by default
{
    ParmParse pp("prob.dycoms");
    pp.query("rad_F0", m_F0);
    pp.query("rad_F1", m_F1);
    pp.query("rad_kappa", m_kappa);
    pp.query("rad_qc_threshold", m_qc_threshold);
    pp.query("rad_a", m_a);
    pp.query("rho_i", m_rho_i);
    pp.query("divergence", m_divergence);
    pp.query("zi", m_zi);
    pp.query("rad_qt_threshold", m_qt_threshold);

    int use_local = m_use_local_zi ? 1 : 0;
    pp.query("rad_use_local_zi", use_local);
    m_use_local_zi = (use_local != 0);

    setupDataLog();
}

void
RadiationDyCOMS::Init (const Geometry& geom,
                       const BoxArray& ba,
                       MultiFab* cons_in)
{
    m_geom = geom;
    m_ba   = ba;
    m_initialized = true;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(cons_in != nullptr,
                                     "RadiationDyCOMS requires valid state MultiFab");

    // CRITICAL: RadiationDyCOMS requires full-column boxes (no z-tiling)
    // for correct optical depth computation via prefix/suffix sums.
    // Check that every box in the BoxArray spans the full vertical domain.
    const Box& domain = geom.Domain();
    const int domain_k_lo = domain.smallEnd(2);
    const int domain_k_hi = domain.bigEnd(2);

    for (int i = 0; i < ba.size(); ++i) {
        const Box& bx = ba[i];
        if (bx.smallEnd(2) != domain_k_lo || bx.bigEnd(2) != domain_k_hi) {
            amrex::Abort("RadiationDyCOMS ERROR: Box " + std::to_string(i) +
                        " does not span full vertical domain.\n" +
                        "  Box k-range: [" + std::to_string(bx.smallEnd(2)) + ", " +
                        std::to_string(bx.bigEnd(2)) + "]\n" +
                        "  Domain k-range: [" + std::to_string(domain_k_lo) + ", " +
                        std::to_string(domain_k_hi) + "]\n" +
                        "  RadiationDyCOMS requires no z-tiling for full-column optical depth.\n" +
                        "  Set fabarray.mfiter_tile_size = 1024000 2 1024000 in inputs to disable z-tiling.");
        }
    }
}

void
RadiationDyCOMS::Run (int& level,
                      int& /*step*/,
                      Real& /*time*/,
                      const Real& /*dt*/,
                      const BoxArray& /*ba*/,
                      Geometry& geom,
                      MultiFab* cons_in,
                      MultiFab* /*lsm_fluxes*/,
                      MultiFab* /*lsm_zenith*/,
                      Vector<MultiFab*>& /*lsm_input_ptrs*/,
                      Vector<MultiFab*>& /*lsm_output_ptrs*/,
                      MultiFab* qheating_rates,
                      MultiFab* z_phys_nd,
                      MultiFab* /*lat*/,
                      MultiFab* /*lon*/)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_initialized,
                                     "RadiationDyCOMS::Init must be called before Run");
    AMREX_ALWAYS_ASSERT(cons_in != nullptr);
    AMREX_ALWAYS_ASSERT(qheating_rates != nullptr);

    // Ensure the heating tendencies start from zero each call
    qheating_rates->setVal(0.0);

    m_geom = geom;
    m_last_heating = qheating_rates;
    m_last_level   = level;

    // Create a MultiFab to store inversion heights zi(i,j) for diagnostics
    // One component, no ghost cells. Note: This is a 3D MultiFab but zi is
    // constant in the vertical (one value per horizontal column), so we fill
    // all k-levels with the same value to ensure min/max reductions work correctly.
    MultiFab zi_field(cons_in->boxArray(), cons_in->DistributionMap(), 1, 0);
    zi_field.setVal(m_zi); // Initialize with fallback value

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(*cons_in, /*do_tiling=*/false); mfi.isValid(); ++mfi) {
        compute_radiative_tendency(mfi, cons_in, qheating_rates, z_phys_nd, geom, &zi_field);
    }

    // Compute min/max inversion height across all columns
    Real zi_min = zi_field.min(0);
    Real zi_max = zi_field.max(0);

    if (ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "Inversion height range: zi_min = " << zi_min
                       << " m, zi_max = " << zi_max << " m\n";
    }
}

void
RadiationDyCOMS::compute_radiative_tendency (const MFIter& mfi,
                                             MultiFab* cons_in,
                                             MultiFab* qheating_rates,
                                             MultiFab* z_phys_nd,
                                             const Geometry& geom,
                                             MultiFab* zi_field)
{
    /*
     * DyCOMS-II RF02 Longwave Radiation Parameterization (Stevens et al. 2005)
     * ==========================================================================
     *
     * Implements Beer's-law cooling with divergence-weighted additive term above
     * the inversion following RF02 specification.
     *
     * Key Features:
     * - Full-column optical depth computation via prefix/suffix sums
     * - Per-column diagnostic inversion height zi(x,y) from qt = 8 g/kg isosurface
     * - Subsidence heating term with correct units (includes divergence D)
     * - Thresholded cloud liquid water (qc) to avoid spurious optical depth
     * - Debug checks for magnitude and unit consistency
     *
     * Physical Constants (RF02 spec):
     * - F0 = 70 W/m² (downward flux from above)
     * - F1 = 22 W/m² (upward flux from surface)
     * - κ = 85 m²/kg (LW absorption coefficient for cloud water)
     * - qc_threshold = 1e-7 kg/kg (ignore qc below this in τ computation)
     * - a = 1 K·m^(-1/3) (subsidence warming coefficient)
     * - ρi = 1.12 kg/m³ (air density at inversion)
     * - D = 3.76e-6 s⁻¹ (large-scale divergence - CRITICAL for correct units)
     * - zi = 795 m (fallback inversion height from RF02 canonical value)
     * - qt_threshold = 0.008 kg/kg (8 g/kg - total water threshold for zi diagnostic)
     * - Cp_d = 1004.5 J/(kg·K) (dry air specific heat)
     *
     * Optical Depth: τ(z₁, z₂) = ∫[z₁,z₂] κ ρ qₗ dz [dimensionless]
     *
     * Beer's-law Fluxes at face z:
     *   F(z) = F0·exp(-τ(z,∞)) + F1·exp(-τ(0,z))  [W/m²]
     *
     * Additive subsidence term (z > zi, includes D for correct units):
     *   F_add(z) = a·ρi·Cp_d·D·[¼(z-zi)^(4/3) + zi·(z-zi)^(1/3)]  [W/m²]
     *
     * Heating Rate:
     *   dT/dt = -(F_top - F_bottom) / (ρ·Cp_d·Δz)  [K/s]
     *   dθ/dt = dT/dt / Π  [K/s]  where Π = (p/p0)^(R/Cp) is Exner function
     *
     * Expected Magnitudes:
     * - Peak in-cloud cooling: ~2-6 K/day (≈ 2.3e-5 to 6.9e-5 K/s)
     * - Additive term contribution: O(1-10) W/m² near inversion
     * - Column optical depth: O(0.3-2) depending on LWP
     *
     * Inversion Height Diagnostic:
     * - zi is diagnosed from qt = 8 g/kg isosurface (RF02 spec)
     * - Scans vertically to find where qt crosses threshold
     * - Linear interpolation between cell centers for sub-grid accuracy
     * - Fallback to configured zi if no crossing found (clear or fully moist column)
     * - Toggle: prob.dycoms.rad_use_local_zi (default = 1)
     *
     * MPI/Tiling Requirements:
     * - Requires full-column boxes (no z-tiling) for correct optical depth
     * - Checked in Init() with clear error message if violated
     */

    const Array4<const Real>& state = cons_in->const_array(mfi);
    Array4<Real> heating            = qheating_rates->array(mfi);
    const Array4<const Real> z_nd   = (z_phys_nd) ? z_phys_nd->const_array(mfi)
                                                  : Array4<const Real>{};
    Array4<Real> zi_arr             = (zi_field) ? zi_field->array(mfi)
                                                  : Array4<Real>{};

    const GeometryData geomdata = geom.data();
    const Real* prob_lo = geom.ProbLo();

    const bool has_z = static_cast<bool>(z_phys_nd);

        const Box& bx    = mfi.tilebox();
        const int k_lo   = bx.smallEnd(2);
        const int k_hi   = bx.bigEnd(2);

        // Assert tile spans full vertical
        const Box& dom = geom.Domain();
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            k_lo == dom.smallEnd(2) && k_hi == dom.bigEnd(2),
            "RadiationDyCOMS: tilebox does not span full vertical; set fabarray.mfiter_tile_size = 1024000 1024000 1000000"
        );

    const bool use_moisture = (m_solver_choice.moisture_type != MoistureType::None);
    const MoistureComponentIndices midx = m_solver_choice.moisture_indices;

    // Local copies for device capture (CUDA-Conservative pattern)
    const Real F0           = m_F0;
    const Real F1           = m_F1;
    const Real kappa        = m_kappa;
    const Real qc_threshold = m_qc_threshold;
    const Real a_coef       = m_a;
    const Real rho_i        = m_rho_i;
    const Real divergence   = m_divergence;
    const Real zi_fallback  = m_zi;
    const Real qt_threshold = m_qt_threshold;
    const bool use_local_zi = m_use_local_zi;

    // Maximum number of vertical levels (assuming reasonable domain size)
    // If domain is larger, increase this or switch to Arena allocation
    // Use preprocessor macros for MSVC compatibility in device code
#ifndef ERF_MAX_NZ_FACES
#define ERF_MAX_NZ_FACES 256
#endif
#ifndef ERF_MAX_NZ_CELLS
#define ERF_MAX_NZ_CELLS 255
#endif

    // Loop over horizontal (i,j) columns in the tile
    // For each column, do serial k-loop to build prefix/suffix sums

    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
    {
        // Only process once per (i,j) column at the lowest k in the tile
        if (k != k_lo) return;

        const int NZ = k_hi - k_lo + 1;
        const int NF = NZ + 1; // number of faces (including domain boundaries)

        AMREX_ASSERT_WITH_MESSAGE(NF <= ERF_MAX_NZ_FACES, "Increase ERF_MAX_NZ_FACES for this domain");

        // Stack arrays for this column
        Real zf[ERF_MAX_NZ_FACES];  // z-coordinate at faces [m]
        Real z_cc[ERF_MAX_NZ_CELLS]; // z-coordinate at cell centers [m]
        Real dz[ERF_MAX_NZ_CELLS];  // cell thickness [m]
        Real tau_from_bottom[ERF_MAX_NZ_FACES]; // τ(0, z_face) [dimensionless]
        Real tau_from_top[ERF_MAX_NZ_FACES];    // τ(z_face, ∞) [dimensionless]

        // =====================================================================
        // Step 1: Build z-face, z-center, and dz arrays (cache geometry)
        // =====================================================================
        for (int kf = 0; kf < NF; ++kf) {
            const int k_face = k_lo + kf;
            if (has_z) {
                zf[kf] = Compute_Z_AtWFace(i, j, k_face, z_nd);
            } else {
                zf[kf] = prob_lo[2] + k_face * geomdata.CellSize(2);
            }
        }

        for (int kc = 0; kc < NZ; ++kc) {
            dz[kc] = std::max(zf[kc+1] - zf[kc], 1.0e-6_rt);
            z_cc[kc] = 0.5_rt * (zf[kc] + zf[kc+1]); // cell center height
        }

        // =====================================================================
        // Step 2: Build prefix sum τ(0, z_face) from bottom up
        // =====================================================================
        tau_from_bottom[0] = 0.0_rt; // surface
        for (int kc = 0; kc < NZ; ++kc) {
            const int k_cell = k_lo + kc;
            const Real rho_k = state(i, j, k_cell, Rho_comp);

            Real ql_k = 0.0_rt;
            if (use_moisture && midx.qc >= 0) {
                const Real qc_k = state(i, j, k_cell, midx.qc);
                ql_k = qc_k / std::max(rho_k, 1.0e-12_rt);
            }
            ql_k = std::max(ql_k, 0.0_rt);

            // Apply threshold to suppress noise
            if (ql_k <= qc_threshold) {
                ql_k = 0.0_rt;
            }

            const Real tau_inc = kappa * rho_k * ql_k * dz[kc];

#ifdef ERF_DEBUG
            // Sanity check: optical depth increment must be non-negative and dimensionless
            AMREX_ASSERT_WITH_MESSAGE(tau_inc >= 0.0_rt,
                "Negative optical depth increment detected");
#endif

            tau_from_bottom[kc+1] = tau_from_bottom[kc] + tau_inc;
        }

        // =====================================================================
        // Step 3: Build suffix sum τ(z_face, ∞) from top down
        // =====================================================================
        tau_from_top[NF-1] = 0.0_rt; // top of domain
        for (int kc = NZ-1; kc >= 0; --kc) {
            const int k_cell = k_lo + kc;
            const Real rho_k = state(i, j, k_cell, Rho_comp);

            Real ql_k = 0.0_rt;
            if (use_moisture && midx.qc >= 0) {
                const Real qc_k = state(i, j, k_cell, midx.qc);
                ql_k = qc_k / std::max(rho_k, 1.0e-12_rt);
            }
            ql_k = std::max(ql_k, 0.0_rt);

            // Apply threshold to suppress noise
            if (ql_k <= qc_threshold) {
                ql_k = 0.0_rt;
            }

            const Real tau_inc = kappa * rho_k * ql_k * dz[kc];
            tau_from_top[kc] = tau_from_top[kc+1] + tau_inc;
        }

        // =====================================================================
        // Step 4: Diagnostic inversion height zi(i,j) from qt = 8 g/kg isosurface
        // =====================================================================
        // RF02 specification: zi is defined where total water qt crosses 8 g/kg.
        // We scan vertically to find where qt[k] >= qt_threshold and qt[k+1] < qt_threshold,
        // then linearly interpolate between cell centers to get sub-grid accuracy.
        //
        // Fallback behavior:
        // - If all qt >= threshold: zi = topmost cell center (H=0 everywhere)
        // - If all qt < threshold: zi = lowest cell center (H=1 everywhere)
        // - If no moisture or toggle off: zi = configured fallback value

        Real zi_local = zi_fallback;

        if (use_local_zi && use_moisture && midx.qv >= 0 && midx.qc >= 0) {
            // Compute total water qt = qv + qc at each cell center
            bool found_crossing = false;

            // Scan from bottom up to find where qt crosses below threshold
            for (int kc = 0; kc < NZ - 1; ++kc) {
                const int k_cell = k_lo + kc;
                const int k_above = k_lo + kc + 1;

                const Real rho_k = state(i, j, k_cell, Rho_comp);
                const Real qv_k = state(i, j, k_cell, midx.qv) / std::max(rho_k, 1.0e-12_rt);
                const Real qc_k = state(i, j, k_cell, midx.qc) / std::max(rho_k, 1.0e-12_rt);
                const Real qt_k = qv_k + qc_k;

                const Real rho_above = state(i, j, k_above, Rho_comp);
                const Real qv_above = state(i, j, k_above, midx.qv) / std::max(rho_above, 1.0e-12_rt);
                const Real qc_above = state(i, j, k_above, midx.qc) / std::max(rho_above, 1.0e-12_rt);
                const Real qt_above = qv_above + qc_above;

                // Check if we cross from moist (qt >= threshold) to dry (qt < threshold)
                if (qt_k >= qt_threshold && qt_above < qt_threshold) {
                    // Linear interpolation between cell centers to find crossing height
                    Real alpha = (qt_threshold - qt_k) / std::max(qt_above - qt_k, 1.0e-12_rt);
                    alpha = std::min(1.0_rt, std::max(0.0_rt, alpha));
                    zi_local = z_cc[kc] + alpha * (z_cc[kc+1] - z_cc[kc]);
                    found_crossing = true;
                    break;
                }
            }

            // Handle edge cases
            if (!found_crossing) {
                // Check if entire column is moist or dry
                const int k_bottom = k_lo;
                const Real rho_bot = state(i, j, k_bottom, Rho_comp);
                const Real qv_bot = state(i, j, k_bottom, midx.qv) / std::max(rho_bot, 1.0e-12_rt);
                const Real qc_bot = state(i, j, k_bottom, midx.qc) / std::max(rho_bot, 1.0e-12_rt);
                const Real qt_bot = qv_bot + qc_bot;

                if (qt_bot < qt_threshold) {
                    // Entire column is dry: set zi to bottom (H=1 everywhere)
                    zi_local = z_cc[0];
                } else {
                    // Entire column is moist: set zi to top (H=0 everywhere)
                    zi_local = z_cc[NZ-1];
                }
            }
        }

        // Store zi_local in the zi_field for diagnostics (if provided)
        // Fill all k-levels in this column with the same zi value since it's a 3D MultiFab
        if (zi_field) {
            for (int kc = 0; kc < NZ; ++kc) {
                const int k_cell = k_lo + kc;
                zi_arr(i, j, k_cell, 0) = zi_local;
            }
        }

        // =====================================================================
        // Step 5: Compute heating for each cell in the column
        // =====================================================================
        for (int kc = 0; kc < NZ; ++kc) {
            const int k_cell = k_lo + kc;

            const Real rho      = state(i, j, k_cell, Rho_comp);
            const Real rhotheta = state(i, j, k_cell, RhoTheta_comp);

            Real qv = 0.0_rt;
            if (use_moisture && midx.qv >= 0) {
                qv = state(i, j, k_cell, midx.qv) / std::max(rho, 1.0e-12_rt);
            }

            const Real pres  = getPgivenRTh(rhotheta, qv);
            const Real exner = getExnergivenP(pres, R_d / Cp_d);

            // Face indices and heights
            const int kb = kc;       // bottom face index
            const int kt = kc + 1;   // top face index
            const Real z_b = zf[kb]; // bottom face height [m]
            const Real z_t = zf[kt]; // top face height [m]

            // Optical depths from prefix/suffix sums
            const Real Qb = tau_from_bottom[kb];  // τ(0, z_bottom)
            const Real Qt = tau_from_bottom[kt];  // τ(0, z_top)
            const Real Qinf_b = tau_from_top[kb]; // τ(z_bottom, ∞)
            const Real Qinf_t = tau_from_top[kt]; // τ(z_top, ∞)

            // Beer's-law fluxes at faces [W/m²]
            Real F_b = F0 * std::exp(-Qinf_b) + F1 * std::exp(-Qb);
            Real F_t = F0 * std::exp(-Qinf_t) + F1 * std::exp(-Qt);

            // Additive subsidence term (Heaviside gate at zi)
            // CRITICAL: Must include divergence D for correct units (W/m²)
            if (z_b >= zi_local) {
                const Real eps_m   = 1.0_rt; // small length [m]
                const Real d_eff   = std::max(z_b - zi_local, 0.0_rt) + eps_m;
                const Real delta13 = std::cbrt(d_eff);
                const Real delta43 = d_eff * delta13;
                const Real F_add_b = a_coef * rho_i * Cp_d * divergence *
                                    (0.25_rt * delta43 + zi_local * delta13);
                F_b += F_add_b;

#ifdef ERF_DEBUG
                // Sanity check: additive term should be O(1-10) W/m² near inversion
                // Allow up to 50 W/m² for robustness (can be large far above inversion)
                if (std::abs(F_add_b) > 50.0_rt && d_eff < 500.0_rt) {
                    printf("WARNING: Large F_add_b=%.2f W/m² at z=%.1f (d_eff=%.1f, zi=%.1f)\n",
                           F_add_b, z_b, d_eff, zi_local);
                }
#endif
            }

            if (z_t >= zi_local) {
                const Real eps_m   = 1.0_rt; // small length [m]
                const Real d_eff   = std::max(z_t - zi_local, 0.0_rt) + eps_m;
                const Real delta13 = std::cbrt(d_eff);
                const Real delta43 = d_eff * delta13;
                const Real F_add_t = a_coef * rho_i * Cp_d * divergence *
                                    (0.25_rt * delta43 + zi_local * delta13);
                F_t += F_add_t;

#ifdef ERF_DEBUG
                if (std::abs(F_add_t) > 50.0_rt && d_eff < 500.0_rt) {
                    printf("WARNING: Large F_add_t=%.2f W/m² at z=%.1f (d_eff=%.1f, zi=%.1f)\n",
                           F_add_t, z_t, d_eff, zi_local);
                }
#endif
            }

            // Heating rate computation
            // dT/dt = -∇·F / (ρ·cp) = -(F_top - F_bottom) / (ρ·cp·Δz)
            const Real denom = rho * Cp_d * dz[kc];
            const Real dTdt   = -(F_t - F_b) / std::max(denom, 1.0e-12_rt);
            const Real dthetadt = dTdt / std::max(exner, 1.0e-12_rt);

#ifdef ERF_DEBUG
            // Sanity check: heating rate should be reasonable (< ~86 K/day = 1e-3 K/s)
            AMREX_ASSERT_WITH_MESSAGE(std::abs(dthetadt) < 1.0e-3_rt,
                "Heating rate exceeds 1e-3 K/s (unrealistic)");

            // Top-layer diagnostic (print top 5 layers for first column)
            if (i == bx.smallEnd(0) && j == bx.smallEnd(1) && kc >= NZ - 5) {
                printf("LW top k=%d/%d z=%.1f Qinf_b=%.3f Qinf_t=%.3f F_b=%.2f F_t=%.2f dθ/dt=%.3e K/s\n",
                       k_cell, k_hi, 0.5_rt * (z_b + z_t), Qinf_b, Qinf_t, F_b, F_t, dthetadt);
            }
#endif

            // Store heating tendency in θ slot (component 1)
            heating(i, j, k_cell, 0) = 0.0_rt;
            heating(i, j, k_cell, 1) = dthetadt;
        }
    });
}

void
RadiationDyCOMS::WriteDataLog (const Real& time)
{
    if (!hasDatalog() || m_last_heating == nullptr) {
        return;
    }

    ReduceOps<ReduceOpSum> reduce_op;
    ReduceData<Real> reduce_data(reduce_op);
    Long ncell = 0;

    for (MFIter mfi(*m_last_heating, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        const Array4<const Real> heating = m_last_heating->const_array(mfi);

        reduce_op.eval(bx, reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> GpuTuple<Real>
        {
            return {heating(i,j,k,1)};
        });

        ncell += bx.numPts();
    }

    Real lw_sum = amrex::get<0>(reduce_data.value());
    ParallelDescriptor::ReduceRealSum(lw_sum);
    ParallelDescriptor::ReduceLongSum(ncell);

    if (ParallelDescriptor::IOProcessor() && ncell > 0) {
        (*datalog) << std::setprecision(10) << time
                   << " " << lw_sum/static_cast<Real>(ncell) << "\n";
        datalog->flush();
    }
}
