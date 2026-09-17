#include <AMReX_Reduce.H>
#include <AMReX_GpuContainers.H>
#include "ERF_Constants.H"

#include <ERF_EOS.H>
#include <ERF_TimestepUtils.H>
#include <ERF.H>
#include "Diffusion/ERF_CloudChamberWallFlux.H"
#include "TimeIntegration/ERF_CloudChamberWallDtGuard.H"

using namespace amrex;

/**
 * Function that calls estTimeStep for each level
 *
 */
void
ERF::ComputeDt (int step, double cur_time_d)
{
    Vector<double> dt_tmp(finest_level+1);

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        dt_tmp[lev] = estTimeStep(lev, dt_mri_ratio[lev]);
        if (solverChoice.moisture_type == MoistureType::SBM) {
            dt_tmp[lev] = std::min(dt_tmp[lev], sbm_admissible_timestep(lev));
        }
    }

    ParallelDescriptor::ReduceRealMin(&dt_tmp[0], dt_tmp.size());

    double dt_0 = dt_tmp[0];
    int n_factor = 1;
    for (int lev = 0; lev <= finest_level; ++lev) {
        dt_tmp[lev] = amrex::min(dt_tmp[lev], static_cast<double>(change_max*dt[lev]));
        n_factor *= nsubsteps[lev];
        dt_0 = std::min(dt_0, static_cast<double>(n_factor*dt_tmp[lev]));

    }
    // Limit level 0 time step if requested
    if (step == 0) {
        dt_0 *= init_shrink;
        if (verbose && init_shrink != one) {
            Print() << "Timestep 0: shrink level 0 initial dt by " << init_shrink << std::endl;
        }
    }
    //
    // Limit dt by the value of stop_time.
    // Recall that stop_time is total time, but t_new is elapsed time,
    //     so we must add start_time to t_new
    //
    const double eps = 1.e-3*dt_0;
    if (cur_time_d + dt_0 > (stop_time - start_time) - eps) {
        dt_0 = (stop_time - start_time) - cur_time_d;
    }

    dt[0] = dt_0;
    for (int lev = 1; lev <= finest_level; ++lev) {
        dt[lev] = dt[lev-1] / nsubsteps[lev];
    }
}

double ERF::sbm_admissible_timestep(const int level) const
{
    if (solverChoice.moisture_type != MoistureType::SBM || level < 0 ||
        level > finest_level) {
        return std::numeric_limits<double>::max();
    }

    // P2 is qualified only for static Cartesian geometry.  Metric factors,
    // EB fractions, and moving terrain would change both the carrier face
    // construction and A_f/(V_i d_if), so those configurations fail closed
    // at the host coupling boundary.
    if (solverChoice.terrain_type != TerrainType::None) return 0.0;

    const MultiFab& state = vars_new[level][Vars::cons];
    const MultiFab& xvel = vars_new[level][Vars::xvel];
    const MultiFab& yvel = vars_new[level][Vars::yvel];
    const MultiFab& zvel = vars_new[level][Vars::zvel];
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        if (state.nGrowVect()[dir] < 1) return 0.0;
    }

    // One fixed reduction over the host state returns the actual variable-
    // density low-order demand components.  In static Cartesian geometry
    // ERF's VelocityToMomentum construction is
    //   mdot_x(i)=u(i)*0.5*(rho(i)+rho(i-1)),
    // and analogously in y/z.  The donor demand for a cell is the outward
    // positive part of those face mass fluxes divided by rho_anchor*V, plus
    // rho_face*K*A/(V*d) for every face.  No spectral-bin loop is involved.
    ReduceOps<ReduceOpMax, ReduceOpMax, ReduceOpMax> reduce_op;
    ReduceData<Real, Real, Real> reduce_data(reduce_op);
    for (MFIter mfi(state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        const auto s = state.const_array(mfi);
        const auto u = xvel.const_array(mfi);
        const auto v = yvel.const_array(mfi);
        const auto w = zvel.const_array(mfi);
        const Real dxi = geom[level].InvCellSize(0);
        const Real dyi = geom[level].InvCellSize(1);
        const Real dzi = geom[level].InvCellSize(2);
        const Real K = solverChoice.sbm_diffusion_coeff;
        reduce_op.eval(bx, reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                -> GpuTuple<Real,Real,Real> {
                const Real rho_i = s(i,j,k,Rho_comp);
                const Real rho_xlo = Real(0.5) *
                    (rho_i + s(i-1,j,k,Rho_comp));
                const Real rho_xhi = Real(0.5) *
                    (rho_i + s(i+1,j,k,Rho_comp));
                const Real rho_ylo = Real(0.5) *
                    (rho_i + s(i,j-1,k,Rho_comp));
                const Real rho_yhi = Real(0.5) *
                    (rho_i + s(i,j+1,k,Rho_comp));
                const Real rho_zlo = Real(0.5) *
                    (rho_i + s(i,j,k-1,Rho_comp));
                const Real rho_zhi = Real(0.5) *
                    (rho_i + s(i,j,k+1,Rho_comp));
                const bool invalid_density =
                    rho_i <= Real(0.0) || rho_xlo <= Real(0.0) ||
                    rho_xhi <= Real(0.0) || rho_ylo <= Real(0.0) ||
                    rho_yhi <= Real(0.0) || rho_zlo <= Real(0.0) ||
                    rho_zhi <= Real(0.0) ||
                    amrex::isnan(rho_i) || amrex::isinf(rho_i) ||
                    amrex::isnan(rho_xlo) || amrex::isinf(rho_xlo) ||
                    amrex::isnan(rho_xhi) || amrex::isinf(rho_xhi) ||
                    amrex::isnan(rho_ylo) || amrex::isinf(rho_ylo) ||
                    amrex::isnan(rho_yhi) || amrex::isinf(rho_yhi) ||
                    amrex::isnan(rho_zlo) || amrex::isinf(rho_zlo) ||
                    amrex::isnan(rho_zhi) || amrex::isinf(rho_zhi) ||
                    amrex::isnan(K) || amrex::isinf(K) || K < Real(0.0);
                const bool invalid_velocity =
                    amrex::isnan(u(i,j,k)) || amrex::isinf(u(i,j,k)) ||
                    amrex::isnan(u(i+1,j,k)) || amrex::isinf(u(i+1,j,k)) ||
                    amrex::isnan(v(i,j,k)) || amrex::isinf(v(i,j,k)) ||
                    amrex::isnan(v(i,j+1,k)) || amrex::isinf(v(i,j+1,k)) ||
                    amrex::isnan(w(i,j,k)) || amrex::isinf(w(i,j,k)) ||
                    amrex::isnan(w(i,j,k+1)) || amrex::isinf(w(i,j,k+1));
                if (invalid_density || invalid_velocity) {
                    const Real fail_closed = std::numeric_limits<Real>::max();
                    return {fail_closed, fail_closed, fail_closed};
                }
                const Real mxlo = u(i,j,k) * rho_xlo;
                const Real mxhi = u(i+1,j,k) * rho_xhi;
                const Real mylo = v(i,j,k) * rho_ylo;
                const Real myhi = v(i,j+1,k) * rho_yhi;
                const Real mzlo = w(i,j,k) * rho_zlo;
                const Real mzhi = w(i,j,k+1) * rho_zhi;
                const Real advective_rate =
                    (amrex::max(-mxlo, Real(0.0)) + amrex::max(mxhi, Real(0.0))) * dxi / rho_i +
                    (amrex::max(-mylo, Real(0.0)) + amrex::max(myhi, Real(0.0))) * dyi / rho_i +
                    (amrex::max(-mzlo, Real(0.0)) + amrex::max(mzhi, Real(0.0))) * dzi / rho_i;
                const Real diffusive_rate = K * (
                    (rho_xlo + rho_xhi) * dxi * dxi +
                    (rho_ylo + rho_yhi) * dyi * dyi +
                    (rho_zlo + rho_zhi) * dzi * dzi) / rho_i;
                return {advective_rate, diffusive_rate,
                        advective_rate + diffusive_rate};
            });
    }

    const GpuTuple<Real,Real,Real> local = reduce_data.value(reduce_op);
    double rates[3] = {static_cast<double>(amrex::get<0>(local)),
                       static_cast<double>(amrex::get<1>(local)),
                       static_cast<double>(amrex::get<2>(local))};
    ParallelDescriptor::ReduceRealMax(rates, 3);
    const double advective_rate = rates[0];
    const double diffusive_rate = rates[1];
    const double max_rate = rates[2];
    const double mathematical_bound = max_rate > 0.0 &&
        std::isfinite(max_rate) ? 1.0 / max_rate :
        (max_rate == 0.0 ? std::numeric_limits<double>::max() : 0.0);
    const double bound = ::erf_sbm::admissible_host_timestep(advective_rate,
                                                              diffusive_rate);
    if (verbose > 1 && ParallelDescriptor::IOProcessor()) {
        Print() << "SBM host stability bound at level " << level << " = " << bound
                << " (advective_rate=" << advective_rate
                << ", diffusive_rate=" << diffusive_rate
                << ", diffusion=" << solverChoice.sbm_diffusion_coeff << ")"
                << " host_carrier=reconstructed_u_rho\n";
        Print() << "SBM host low-order demand maximum at level " << level
                << " = " << max_rate
                << " (mathematical_bound=" << mathematical_bound
                << ", host_safety_factor=0.5)\n";
    }
    return bound;
}

/**
 * Function that calls estTimeStep for each level
 *
 * @param[in] level level of refinement (coarsest level i 0)
 * @param[out] dt_fast_ratio ratio of slow to fast time step
 */
double
ERF::estTimeStep (int level, long& dt_fast_ratio) const
{
    BL_PROFILE("ERF::estTimeStep()");

    // Terrain aware (T) and terrain unaware (N) time step estimates.
    double estdt_comp_T = bogus_large_value;
    double estdt_comp_N = bogus_large_value;
    double estdt_lowM_T = bogus_large_value;
    double estdt_lowM_N = bogus_large_value;

    // We intentionally use the level 0 domain to compute whether to use this direction in the dt calculation
    const int nxc = geom[0].Domain().length(0);
    const int nyc = geom[0].Domain().length(1);

    auto const dxinv = geom[level].InvCellSizeArray();
    auto dxinv_EB = dxinv; dxinv_EB[2] = one / dz_min[level];

    MultiFab const& S_new = vars_new[level][Vars::cons];

    // Keep the thermodynamic samples alongside the cell-centered velocity so
    // the wall-rate reduction can call the same pointwise MOST evaluator as
    // production wall transfer without allocating a second global temporary.
    MultiFab ccvel_N(grids[level],dmap[level],7,0);
    MultiFab ccvel_T(grids[level],dmap[level],3,0);

    int klo = geom[level].Domain().smallEnd(2);
    int khi = geom[level].Domain().bigEnd(2);
    MultiFab omega(convert(grids[level],IntVect(0,0,1)),dmap[level],1,0);
    for (MFIter mfi(vars_new[level][Vars::zvel]); mfi.isValid(); ++mfi)
    {
        Box vbx = mfi.validbox();

        const Array4<      Real>& omega_arr = omega.array(mfi);
        const Array4<const Real>& u_arr     = vars_new[level][IntVars::xmom].const_array(mfi);
        const Array4<const Real>& v_arr     = vars_new[level][IntVars::ymom].const_array(mfi);
        const Array4<const Real>& w_arr     = vars_new[level][IntVars::zmom].const_array(mfi);

        const Array4<const Real>& z_nd_arr  = z_phys_nd[level]->const_array(mfi);

        const Array4<const Real>& mf_ux     = mapfac[level][MapFacType::u_x]->const_array(mfi);
        const Array4<const Real>& mf_vy     = mapfac[level][MapFacType::v_y]->const_array(mfi);

        ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            if (k==klo || k==(khi+1)) {
                omega_arr(i,j,k) = zero;
            } else {
                omega_arr(i,j,k) = OmegaFromW(i,j,k,w_arr(i,j,k),
                                              u_arr,v_arr,mf_ux,mf_vy,
                                              z_nd_arr,dxinv);
            }
        });
    }

    average_face_to_cellcenter(ccvel_T,0,
                               Array<const MultiFab*,3>{&vars_new[level][Vars::xvel],
                                                        &vars_new[level][Vars::yvel],
                                                        &omega});
    average_face_to_cellcenter(ccvel_N,0,
                               Array<const MultiFab*,3>{&vars_new[level][Vars::xvel],
                                                        &vars_new[level][Vars::yvel],
                                                        &vars_new[level][Vars::zvel],});

    const bool chamber_cloudy = cloud_chamber_config.active &&
        cloud_chamber_config.cloudy;
    const Real rdOcp = solverChoice.rdOcp;
    const MultiFab& chamber_base_state = base_state[level];
    for (MFIter mfi(S_new); mfi.isValid(); ++mfi) {
        const Array4<const Real> state = S_new.const_array(mfi);
        const Array4<const Real> base = chamber_base_state.const_array(mfi);
        const Array4<Real> velocity = ccvel_N.array(mfi);
        const Box bx = mfi.validbox();
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            const Real rho = state(i,j,k,Rho_comp);
            velocity(i,j,k,3) = state(i,j,k,RhoTheta_comp) / rho;
            velocity(i,j,k,4) = chamber_cloudy ?
                state(i,j,k,RhoQ1_comp) / rho : Real(0.0);
            velocity(i,j,k,5) = base(i,j,k,BaseState::p0_comp);
            velocity(i,j,k,6) = rdOcp;
        });
    }

    bool l_substepping = (solverChoice.substepping_type[level] == SubsteppingType::Implicit);
    int  l_anelastic   = solverChoice.anelastic[level];

    bool l_comp_substepping_diag = (verbose && l_substepping && !l_anelastic && solverChoice.substepping_diag);

    Real estdt_comp_inv_N, estdt_comp_inv_T;
    Real estdt_lowM_inv_N, estdt_lowM_inv_T;
    Real estdt_vert_comp_inv, estdt_vert_lowM_inv;

    const MultiFab& z_nd_mf = *z_phys_nd[level];

    if (l_substepping && (nxc==1) && (nyc==1)) {
        // SCM -- should not depend on dx or dy; force minimum number of substeps
        estdt_comp_inv_T = std::numeric_limits<Real>::min();
        estdt_comp_inv_N = estdt_comp_inv_T;
    }
    else if (solverChoice.terrain_type == TerrainType::EB)
    {
        const eb_& eb_lev = get_eb(level);
        const MultiFab& detJ = (eb_lev.get_const_factory())->getVolFrac();

        estdt_comp_inv_N = ReduceMax(S_new, ccvel_N, detJ, 0,
        [=] AMREX_GPU_HOST_DEVICE (Box const& b,
                                   Array4<Real const> const& s,
                                   Array4<Real const> const& u,
                                   Array4<Real const> const& vf) -> Real
        {
           Real new_comp_dt = -bogus_large_value;
           amrex::Loop(b, [=,&new_comp_dt] (int i, int j, int k) noexcept
           {
               if (vf(i,j,k) > zero)
               {
                   const Real rho      = s(i, j, k, Rho_comp);
                   const Real rhotheta = s(i, j, k, RhoTheta_comp);

                   // NOTE: even when moisture is present,
                   //       we only use the partial pressure of the dry air
                   //       to compute the soundspeed
                   Real pressure = getPgivenRTh(rhotheta);
                   Real c = std::sqrt(Gamma * pressure / rho);

                   // If we are doing implicit acoustic substepping, then the z-direction does not contribute
                   //    to the computation of the time step
                   if (l_substepping) {
                       if ((nxc > 1) && (nyc==1)) {
                           // 2-D in x-z
                           new_comp_dt = amrex::max(((amrex::Math::abs(u(i,j,k,0))+c)*dxinv_EB[0]), new_comp_dt);
                       } else if ((nyc > 1) && (nxc==1)) {
                           // 2-D in y-z
                           new_comp_dt = amrex::max(((amrex::Math::abs(u(i,j,k,1))+c)*dxinv_EB[1]), new_comp_dt);
                       } else {
                           // 3-D
                           new_comp_dt = amrex::max(((amrex::Math::abs(u(i,j,k,0))+c)*dxinv_EB[0]),
                                                    ((amrex::Math::abs(u(i,j,k,1))+c)*dxinv_EB[1]), new_comp_dt);
                       }

                   // If we are not doing implicit acoustic substepping, then the z-direction contributes
                   //    to the computation of the time step
                   } else {
                       if (nxc > 1 && nyc > 1) {
                           new_comp_dt = amrex::max(((amrex::Math::abs(u(i,j,k,0))+c)*dxinv_EB[0]),
                                                    ((amrex::Math::abs(u(i,j,k,1))+c)*dxinv_EB[1]),
                                                    ((amrex::Math::abs(u(i,j,k,2))+c)*dxinv_EB[2]), new_comp_dt);
                       } else if (nxc > 1) {
                           new_comp_dt = amrex::max(((amrex::Math::abs(u(i,j,k,0))+c)*dxinv_EB[0]),
                                                    ((amrex::Math::abs(u(i,j,k,2))+c)*dxinv_EB[2]), new_comp_dt);
                       } else if (nyc > 1) {
                           new_comp_dt = amrex::max(((amrex::Math::abs(u(i,j,k,1))+c)*dxinv_EB[1]),
                                                    ((amrex::Math::abs(u(i,j,k,2))+c)*dxinv_EB[2]), new_comp_dt);
                       } else {
                           new_comp_dt = amrex::max(((amrex::Math::abs(u(i,j,k,2))+c)*dxinv_EB[2]), new_comp_dt);
                       }

                   }
               }
           });
           return new_comp_dt;
       });

        // The metric terms do not exist for EB, so the terrain aware
        // estimate is identical to the terrain unaware estimate
        estdt_comp_inv_T = estdt_comp_inv_N;

    } else {
        // One pass over the data returning both the terrain aware (T) estimate,
        // and the terrain unaware (N) estimate
        ReduceOps<ReduceOpMax,ReduceOpMax> reduce_op;
        ReduceData<Real,Real>              reduce_data(reduce_op);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(S_new,TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();

            const Array4<const Real>& s    = S_new.const_array(mfi);
            const Array4<const Real>& u_T  = ccvel_T.const_array(mfi);
            const Array4<const Real>& u_N  = ccvel_N.const_array(mfi);
            const Array4<const Real>& z_nd = z_nd_mf.const_array(mfi);

            reduce_op.eval(bx, reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> GpuTuple<Real,Real>
            {
                const Real rho      = s(i,j,k,Rho_comp);
                const Real rhotheta = s(i,j,k,RhoTheta_comp);

                // NOTE: even when moisture is present,
                //       we only use the partial pressure of the dry air
                //       to compute the soundspeed
                Real pressure = getPgivenRTh(rhotheta);
                Real c = std::sqrt(Gamma * pressure / rho);

                Real inv_dt_T = Compute_InvDt_Compressible(i,j,k, c,
                                    u_T(i,j,k,0), u_T(i,j,k,1), u_T(i,j,k,2),
                                    z_nd, dxinv, true, l_substepping, nxc, nyc);

                Real inv_dt_N = Compute_InvDt_Compressible(i,j,k, c,
                                    u_N(i,j,k,0), u_N(i,j,k,1), u_N(i,j,k,2),
                                    z_nd, dxinv, false, l_substepping, nxc, nyc);

                return {inv_dt_T, inv_dt_N};
            });
        }

        GpuTuple<Real,Real> hv = reduce_data.value(reduce_op);
        estdt_comp_inv_T       = amrex::get<0>(hv);
        estdt_comp_inv_N       = amrex::get<1>(hv);
    } // not EB

    {
        Real comp_inv[2] = {estdt_comp_inv_T, estdt_comp_inv_N};
        ParallelDescriptor::ReduceRealMax(comp_inv,2);
        estdt_comp_inv_T = comp_inv[0];
        estdt_comp_inv_N = comp_inv[1];
    }

    // Globally empty level -> ReduceMax = lowest(); treat level as non-constraining.
    estdt_comp_T = (estdt_comp_inv_T > zero) ? (cfl / estdt_comp_inv_T) : bogus_large_value;
    estdt_comp_N = (estdt_comp_inv_N > zero) ? (cfl / estdt_comp_inv_N) : bogus_large_value;

    //
    // Anelastic (low Mach) estimate -- purely advective, again terrain aware
    // and terrain unaware
    //
    if (solverChoice.terrain_type == TerrainType::EB)
    {
        estdt_lowM_inv_N = ReduceMax(ccvel_N, z_nd_mf, 0,
        [=] AMREX_GPU_HOST_DEVICE (Box const& b,
                                   Array4<Real const> const& u,
                                   Array4<Real const> const& z_nd) -> Real
        {
            Real new_lm_dt = -bogus_large_value;
            Loop(b, [=,&new_lm_dt] (int i, int j, int k) noexcept
            {
                Real inv_dt_lowM_N = Compute_InvDt_Anelastic(i,j,k,
                                             u(i,j,k,0), u(i,j,k,1), u(i,j,k,2),
                                             z_nd, dxinv_EB, false);
                new_lm_dt = amrex::max(inv_dt_lowM_N, new_lm_dt);
            });
            return new_lm_dt;
        });

        // The metric terms do not exist for EB
        estdt_lowM_inv_T = estdt_lowM_inv_N;

    } else {

        ReduceOps<ReduceOpMax,ReduceOpMax> reduce_op;
        ReduceData<Real,Real>              reduce_data(reduce_op);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(ccvel_T,TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();

            const Array4<const Real>& u_T  = ccvel_T.const_array(mfi);
            const Array4<const Real>& u_N  = ccvel_N.const_array(mfi);
            const Array4<const Real>& z_nd = z_nd_mf.const_array(mfi);

            reduce_op.eval(bx, reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> GpuTuple<Real,Real>
            {
                Real inv_dt_T = Compute_InvDt_Anelastic(i,j,k,
                                    u_T(i,j,k,0), u_T(i,j,k,1), u_T(i,j,k,2),
                                    z_nd, dxinv, true);

                Real inv_dt_N = Compute_InvDt_Anelastic(i,j,k,
                                    u_N(i,j,k,0), u_N(i,j,k,1), u_N(i,j,k,2),
                                    z_nd, dxinv, false);

                return {inv_dt_T, inv_dt_N};
            });
        }

        GpuTuple<Real,Real> hv = reduce_data.value(reduce_op);
        estdt_lowM_inv_T       = amrex::get<0>(hv);
        estdt_lowM_inv_N       = amrex::get<1>(hv);
    }

    {
        Real lowM_inv[2] = {estdt_lowM_inv_T, estdt_lowM_inv_N};
        ParallelDescriptor::ReduceRealMax(lowM_inv,2);
        estdt_lowM_inv_T = lowM_inv[0];
        estdt_lowM_inv_N = lowM_inv[1];
    }

    if (estdt_lowM_inv_T > zero) { estdt_lowM_T = cfl / estdt_lowM_inv_T; }
    if (estdt_lowM_inv_N > zero) { estdt_lowM_N = cfl / estdt_lowM_inv_N; }

     Real max_wall_rate = Real(0.0);
     Real estdt_wall = bogus_large_value;
     if (cloud_chamber_config.active &&
         cloud_chamber_config.physical_initialization &&
         cloud_chamber_config.has_wall_rate_channel()) {
         const auto walls = cloud_chamber_config.wall_boundary();
         const Box domain = geom[level].Domain();
         max_wall_rate = ReduceMax(ccvel_N, 0,
         [=] AMREX_GPU_HOST_DEVICE (Box const& b,
                                    Array4<Real const> const& velocity) -> Real
         {
             Real rate = Real(0.0);
             amrex::Loop(b, [=,&rate] (int i, int j, int k) noexcept
             {
                 // A free staggered component can receive tangential traction
                 // from every active perpendicular wall at an edge/corner.
                 // Fixed and neutral coefficients use the exact local
                 // velocity-parallel Jacobian row-sum factor.  MOST evaluates
                 // its current coefficients once per wall/cell and uses that
                 // state as a frozen-coefficient local rate estimate; it is
                 // not a nonlinear MOST Jacobian bound.
                 Real momentum_rate = Real(0.0);
                 amrex::GpuArray<Real, AMREX_SPACEDIM> low_momentum_rates{};
                 amrex::GpuArray<Real, AMREX_SPACEDIM> high_momentum_rates{};
                 // Evaluate each encountered wall/cell state once.  The
                 // component row sums below only compose these retained
                 // per-face rates; they must not repeat the MOST solve.
                 for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                     const bool low = (dir == 0 ? i == domain.smallEnd(0) :
                                       (dir == 1 ? j == domain.smallEnd(1) :
                                                    k == domain.smallEnd(2)));
                     const bool high = (dir == 0 ? i == domain.bigEnd(0) :
                                        (dir == 1 ? j == domain.bigEnd(1) :
                                                     k == domain.bigEnd(2)));
                     if (low) {
                         const auto& wall = walls[2*dir];
                         if (erf_cloud_chamber_wall_flux::
                             wall_rate_requires_tangential_speed(wall)) {
                             const Real U_t =
                                 erf_cloud_chamber_wall_flux::
                                 tangential_speed_cell_centered(
                                     dir, i, j, k, velocity, wall);
                             const auto runtime =
                                 erf_cloud_chamber_wall_flux::most_wall_coefficients(
                                     wall, velocity(i,j,k,3), velocity(i,j,k,4),
                                     velocity(i,j,k,5), velocity(i,j,k,6), U_t,
                                     Real(0.5) / dxinv[dir],
                                     dir == 2 ? 1 : 0);
                             rate = amrex::max(rate,
                                 erf_cloud_chamber_wall_flux::wall_rate_for_face(
                                     wall, U_t, dxinv[dir], runtime));
                             low_momentum_rates[dir] =
                                 erf_cloud_chamber_wall_flux::momentum_rate_for_face(
                                     wall, U_t, dxinv[dir], runtime);
                         }
                     }
                     if (high) {
                         const auto& wall = walls[2*dir+1];
                         if (erf_cloud_chamber_wall_flux::
                             wall_rate_requires_tangential_speed(wall)) {
                             const Real U_t =
                                 erf_cloud_chamber_wall_flux::
                                 tangential_speed_cell_centered(
                                     dir, i, j, k, velocity, wall);
                             const auto runtime =
                                 erf_cloud_chamber_wall_flux::most_wall_coefficients(
                                     wall, velocity(i,j,k,3), velocity(i,j,k,4),
                                     velocity(i,j,k,5), velocity(i,j,k,6), U_t,
                                     Real(0.5) / dxinv[dir],
                                     dir == 2 ? -1 : 0);
                             rate = amrex::max(rate,
                                 erf_cloud_chamber_wall_flux::wall_rate_for_face(
                                     wall, U_t, dxinv[dir], runtime));
                             high_momentum_rates[dir] =
                                 erf_cloud_chamber_wall_flux::momentum_rate_for_face(
                                     wall, U_t, dxinv[dir], runtime);
                         }
                     }
                 }
                 for (int component = 0; component < AMREX_SPACEDIM; ++component) {
                     const Real component_rate =
                         erf_cloud_chamber_wall_flux::momentum_row_sum_rate(
                             component, low_momentum_rates, high_momentum_rates);
                     momentum_rate = amrex::max(momentum_rate, component_rate);
                 }
                 rate = amrex::max(rate, momentum_rate);
             });
             return rate;
         });
         ParallelDescriptor::ReduceRealMax(max_wall_rate);
         if (max_wall_rate > Real(0.0)) {
             estdt_wall = erf_cloud_chamber_wall_flux::wall_dt_from_max_rate(max_wall_rate);
         }
     }
     // Wall-rate kernels and reductions use amrex::Real, while ERF's host
     // timestep estimates are double even in ERF_PRECISION=SINGLE builds.
     const double estdt_wall_host = static_cast<double>(estdt_wall);
     estdt_comp_T = std::min(estdt_comp_T, estdt_wall_host);
     estdt_comp_N = std::min(estdt_comp_N, estdt_wall_host);
     estdt_lowM_T = std::min(estdt_lowM_T, estdt_wall_host);
     estdt_lowM_N = std::min(estdt_lowM_N, estdt_wall_host);

     const double fixed_dt_level = static_cast<double>(fixed_dt[level]);
     erf_cloud_chamber_wall_dt_guard::enforce_fixed_dt_limit(
         level, fixed_dt_level, estdt_wall_host, max_wall_rate);

     // Additional vertical diagnostics
     if (l_comp_substepping_diag) {
         estdt_vert_comp_inv = ReduceMax(S_new, ccvel_T, z_nd_mf, 0,
         [=] AMREX_GPU_HOST_DEVICE (Box const& b,
                                    Array4<Real const> const& s,
                                    Array4<Real const> const& u,
                                    Array4<Real const> const& z_nd) -> Real
         {
             Real new_comp_dt = -bogus_large_value;
             amrex::Loop(b, [=,&new_comp_dt] (int i, int j, int k) noexcept
             {
                 {
                     const Real rho      = s(i, j, k, Rho_comp);
                     const Real rhotheta = s(i, j, k, RhoTheta_comp);

                     // NOTE: even when moisture is present,
                     //       we only use the partial pressure of the dry air
                     //       to compute the soundspeed
                     Real pressure = getPgivenRTh(rhotheta);
                     Real c = std::sqrt(Gamma * pressure / rho);

                     Real h_zeta  = Compute_h_zeta_AtCellCenter(i,j,k,dxinv,z_nd);
                     Real idz_loc = dxinv[2] / h_zeta;

                     // Look at z-direction only
                     new_comp_dt = amrex::max((amrex::Math::abs(u(i,j,k,2)) + c) * idz_loc, new_comp_dt);
                 }
             });
             return new_comp_dt;
         });

         estdt_vert_lowM_inv = ReduceMax(ccvel_T, z_nd_mf, 0,
         [=] AMREX_GPU_HOST_DEVICE (Box const& b,
                                    Array4<Real const> const& u,
                                    Array4<Real const> const& z_nd) -> Real
         {
             Real new_lowM_dt = -bogus_large_value;
             amrex::Loop(b, [=,&new_lowM_dt] (int i, int j, int k) noexcept
             {
                 Real h_zeta  = Compute_h_zeta_AtCellCenter(i,j,k,dxinv,z_nd);
                 Real idz_loc = dxinv[2] / h_zeta;
                 new_lowM_dt = amrex::max((amrex::Math::abs(u(i,j,k,2))) * idz_loc, new_lowM_dt);
             });
             return new_lowM_dt;
         });

         ParallelDescriptor::ReduceRealMax(estdt_vert_comp_inv);
         ParallelDescriptor::ReduceRealMax(estdt_vert_lowM_inv);
     }

     if (verbose) {
         // Terrain aware   (T): includes h_xi, h_eta and h_zeta
         // Terrain unaware (N): no metric terms, dxinv[2] used for the vertical spacing
         if (fixed_dt[level] <= zero) {
             Print() << "Using cfl = " << cfl << " and dx/dy/dz_min = " <<
               one/dxinv[0] << " " << one/dxinv[1] << " " << dz_min[level] << std::endl;
             Print() << "Compressible dt at level " << level << ":  "
                     << estdt_comp_T << " (terrain aware)  "
                     << estdt_comp_N << " (terrain unaware)" << std::endl;
             if (estdt_lowM_inv_T > 0.0_rt) {
                 Print() << "Anelastic   dt at level " << level << ":  "
                         << estdt_lowM_T << " (terrain aware)  "
                         << estdt_lowM_N << " (terrain unaware)" << std::endl;
             } else {
                 Print() << "Anelastic dt at level " << level << ": undefined " << std::endl;
             }
         }

         if (fixed_dt[level] > zero) {
             Print() << "Based on cfl of one " << std::endl;
             Print() << "Compressible dt at level " << level << " would be:  "
                     << estdt_comp_T/cfl << " (terrain aware)  "
                     << estdt_comp_N/cfl << " (terrain unaware)" << std::endl;
             if (estdt_lowM_inv_T > zero) {
                 Print() << "Anelastic    dt at level " << level << " would be:  "
                         << estdt_lowM_T/cfl << " (terrain aware)  "
                         << estdt_lowM_N/cfl << " (terrain unaware)" << std::endl;
             } else {
                 Print() << "Anelastic    dt at level " << level << " would be undefined " << std::endl;
             }
             Print() << "Fixed dt at level " << level << "       is:  " << fixed_dt[level] << std::endl;
             if (fixed_fast_dt[level] > zero) {
                 Print() << "Fixed fast dt at level " << level << "       is:  " << fixed_fast_dt[level] << std::endl;
             }
         }
     }

     if (solverChoice.substepping_type[level] != SubsteppingType::None) {
         if (fixed_dt[level] > zero && fixed_fast_dt[level] > zero) {
             dt_fast_ratio = static_cast<long>( fixed_dt[level] / fixed_fast_dt[level] );
             if (dt_fast_ratio < 1) {
                 Abort("Invalid fixed_fast_dt: must be <= fixed_dt so mri_dt_ratio >= 1");
             }
         } else if (fixed_dt[level] > zero) {
             // Max CFL_c = one for substeps by default, but we enforce a min of 4 substeps
         auto dt_sub_max = (estdt_comp_T * (third/cfl) * sub_cfl);
             dt_fast_ratio = static_cast<long>( std::max(fixed_dt[level]/static_cast<double>(dt_sub_max), 4.0) );
         } else {
             // auto dt_sub_max = (estdt_comp_T/cfl * sub_cfl);
             // dt_fast_ratio = static_cast<long>( std::max(estdt_comp_T/dt_sub_max,Real(4.)) );
         dt_fast_ratio = static_cast<long>( std::max((cfl/third) / sub_cfl, Real(4.)) );
         }

         // Force time step ratio to be an even value
         if (solverChoice.force_stage1_single_substep) {
             if ( dt_fast_ratio%2 != 0) dt_fast_ratio += 1;
         } else {
             if ( dt_fast_ratio%6 != 0) {
                 Print() << "mri_dt_ratio = " << dt_fast_ratio
                         << " not divisible by 6 for N/3 substeps in stage 1" << std::endl;
                 dt_fast_ratio = static_cast<int>(std::ceil(dt_fast_ratio/Real(6.0)) * 6);
             }
         }

         if (verbose) {
             Print() << "smallest even ratio is: " << dt_fast_ratio << std::endl;
         }
     } // if substepping

     // Print out some extra diagnostics -- dt calcs are repeated so as to not
     // disrupt the overall code flow...
     if (l_comp_substepping_diag) {
         double dt_diag = (fixed_dt[level] > zero) ? fixed_dt[level] : static_cast<double>(estdt_comp_T);
         int  ns      = (fixed_mri_dt_ratio > zero) ? fixed_mri_dt_ratio : dt_fast_ratio;

         // horizontal acoustic CFL must be < 1 (fully explicit)
         // vertical   acoustic CFL may  be > 1
         Print() << "effective horiz,vert acoustic CFL with " << ns << " substeps : "
            << (dt_diag / ns) * estdt_comp_inv_T << " "
            << (dt_diag / ns) * estdt_vert_comp_inv << std::endl;

         // vertical advective CFL should be < 1, otherwise w-damping may be needed
         Print() << "effective vert advective CFL : "
            << dt_diag * estdt_vert_lowM_inv << std::endl;
     }

     if (fixed_dt[level] > zero) {
         return fixed_dt[level];
     } else {
         // Anelastic (substepping is not allowed)
         if (l_anelastic) {

            // Make sure that timestep is less than the dt_max
            estdt_lowM_T = std::min(estdt_lowM_T, dt_max);

            // On the first timestep enforce dt_max_initial
            if (istep[level] == 0) {
                return std::min(dt_max_initial, estdt_lowM_T);
            } else {
                return estdt_lowM_T;
            }


         // Compressible with or without substepping
         } else {
             return estdt_comp_T;
         }
     }
}
