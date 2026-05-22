#include <ERF_EOS.H>
#include <ERF_TileNoZ.H>

#include "ERF_DataStruct.H"
#include "ERF_Kessler.H"
#include "ERF_KesslerUtils.H"

using namespace amrex;

/**
 * Compute Precipitation-related Microphysics quantities.
 */
void Kessler::AdvanceKessler (const SolverChoice &solverChoice)
{
    AdvanceKesslerRefactored(solverChoice);
}

#ifdef ERF_ENABLE_KESSLER_BFB_REFERENCE
void Kessler::AdvanceKesslerLegacyForBFB (const SolverChoice &solverChoice)
{
    bool do_cond = m_do_cond;
    auto tabs    = mic_fab_vars[MicVar_Kess::tabs];
    auto domain  = m_geom.Domain();
    int i_lo = domain.smallEnd(0);
    int i_hi = domain.bigEnd(0);
    int j_lo = domain.smallEnd(1);
    int j_hi = domain.bigEnd(1);
    int k_lo = domain.smallEnd(2);
    int k_hi = domain.bigEnd(2);
    if (solverChoice.moisture_type == MoistureType::Kessler)
    {
        MultiFab fz;
        auto ba    = tabs->boxArray();
        auto dm    = tabs->DistributionMap();
        fz.define(convert(ba, IntVect(0,0,1)), dm, 1, 0); // No ghost cells

        Real dtn  = dt;
        Real coef = dtn/m_dzmin;

        for ( MFIter mfi(*tabs,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto qv_array    = mic_fab_vars[MicVar_Kess::qv]->array(mfi);
            auto qc_array    = mic_fab_vars[MicVar_Kess::qcl]->array(mfi);
            auto qp_array    = mic_fab_vars[MicVar_Kess::qp]->array(mfi);
            auto qt_array    = mic_fab_vars[MicVar_Kess::qt]->array(mfi);
            auto tabs_array  = mic_fab_vars[MicVar_Kess::tabs]->array(mfi);
            auto pres_array  = mic_fab_vars[MicVar_Kess::pres]->array(mfi);
            auto theta_array = mic_fab_vars[MicVar_Kess::theta]->array(mfi);
            auto rho_array   = mic_fab_vars[MicVar_Kess::rho]->array(mfi);

            auto tbx = mfi.tilebox();
            if (tbx.smallEnd(0) == i_lo) { tbx.growLo(0,-m_real_width); }
            if (tbx.bigEnd(0)   == i_hi) { tbx.growHi(0,-m_real_width); }
            if (tbx.smallEnd(1) == j_lo) { tbx.growLo(1,-m_real_width); }
            if (tbx.bigEnd(1)   == j_hi) { tbx.growHi(1,-m_real_width); }

            Real d_fac_cond = m_fac_cond;

            ParallelFor(tbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                qv_array(i,j,k) = std::max(Real(0), qv_array(i,j,k));
                qc_array(i,j,k) = std::max(Real(0), qc_array(i,j,k));
                qp_array(i,j,k) = std::max(Real(0), qp_array(i,j,k));

                Real qcc, auto_r, accrr;
                Real qsat, dtqsat;
                Real dq_clwater_to_rain, dq_rain_to_vapor, dq_clwater_to_vapor, dq_vapor_to_clwater;

                Real pressure = pres_array(i,j,k);
                erf_qsatw(tabs_array(i,j,k), pressure, qsat);
                erf_dtqsatw(tabs_array(i,j,k), pressure, dtqsat);

                if (qsat <= Real(0)) {
                    amrex::Warning("qsat computed as non-positive; setting to Real(0)!");
                    qsat = Real(0);
                }

                dq_clwater_to_rain  = Real(0);
                dq_rain_to_vapor    = Real(0);
                dq_vapor_to_clwater = Real(0);
                dq_clwater_to_vapor = Real(0);

                Real fac = (L_v/Cp_d)*dtqsat;

                if ( (qv_array(i,j,k) > qsat) && do_cond ) {
                    dq_vapor_to_clwater = std::min(qv_array(i,j,k), (qv_array(i,j,k)-qsat)/(Real(1) + fac));
                }

                if ( (qv_array(i,j,k) < qsat) && (qc_array(i,j,k) > Real(0)) && do_cond ) {
                    dq_clwater_to_vapor = std::min(qc_array(i,j,k), (qsat - qv_array(i,j,k))/(Real(1) + fac));
                }

                if (( qp_array(i,j,k) > Real(0)) && (qv_array(i,j,k) < qsat) ) {
                    Real C = Real(1.6) + Real(124.9)*std::pow(Real(0.001)*rho_array(i,j,k)*qp_array(i,j,k),Real(0.2046));
                    dq_rain_to_vapor = Real(1)/(Real(0.001)*rho_array(i,j,k))*(Real(1) - qv_array(i,j,k)/qsat)*C*std::pow(Real(0.001)*rho_array(i,j,k)*qp_array(i,j,k),Real(0.525))
                        /(Real(5.4e5) + Real(2.55e6)/(pressure*qsat))*dtn;
                    dq_rain_to_vapor = std::min({qp_array(i,j,k), dq_rain_to_vapor});
                }

                if (qc_array(i,j,k) > Real(0)) {
                    qcc = qc_array(i,j,k);

                    auto_r = Real(0);
                    if (qcc > qcw0) {
                        auto_r = alphaelq;
                    }

                    accrr = Real(0);
                    accrr = Real(2.2) * std::pow(qp_array(i,j,k) , Real(0.875));
                    dq_clwater_to_rain = dtn *(accrr*qcc + auto_r*(qcc - qcw0));
                    dq_clwater_to_rain = std::min(dq_clwater_to_rain, qc_array(i,j,k));
                }

                qv_array(i,j,k) += -dq_vapor_to_clwater + dq_clwater_to_vapor + dq_rain_to_vapor;
                qc_array(i,j,k) +=  dq_vapor_to_clwater - dq_clwater_to_vapor - dq_clwater_to_rain;
                qp_array(i,j,k) +=  dq_clwater_to_rain - dq_rain_to_vapor;

                Real theta_over_T = theta_array(i,j,k)/tabs_array(i,j,k);
                theta_array(i,j,k) += theta_over_T * d_fac_cond * (dq_vapor_to_clwater - dq_clwater_to_vapor - dq_rain_to_vapor);

                qv_array(i,j,k) = std::max(Real(0), qv_array(i,j,k));
                qc_array(i,j,k) = std::max(Real(0), qc_array(i,j,k));
                qp_array(i,j,k) = std::max(Real(0), qp_array(i,j,k));

                qt_array(i,j,k) = qv_array(i,j,k) + qc_array(i,j,k);
            });
        }

        for ( MFIter mfi(fz, TilingIfNotGPU()); mfi.isValid(); ++mfi ){
            auto rho_array = mic_fab_vars[MicVar_Kess::rho]->array(mfi);
            auto qp_array  = mic_fab_vars[MicVar_Kess::qp]->array(mfi);

            auto fz_array  = fz.array(mfi);
            const Box& tbz = mfi.tilebox();

            ParallelFor(tbz, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                Real rho_avg, qp_avg;

                if (k==k_lo) {
                    rho_avg = rho_array(i,j,k);
                    qp_avg  = qp_array(i,j,k);
                } else if (k==k_hi+1) {
                    rho_avg = rho_array(i,j,k-1);
                    qp_avg  = qp_array(i,j,k-1);
                } else {
                    rho_avg = myhalf*(rho_array(i,j,k-1) + rho_array(i,j,k));
                    qp_avg = myhalf*(qp_array(i,j,k-1)  + qp_array(i,j,k));
                }

                qp_avg = std::max(Real(0), qp_avg);

                Real V_terminal = Real(36.34)*std::pow(rho_avg*Real(0.001)*qp_avg, Real(0.1346))*std::pow(rho_avg/Real(1.16), -myhalf);
                fz_array(i,j,k) = rho_avg*V_terminal*qp_avg;
            });
        }

        Real wt_max;
        int n_substep;
        auto const& ma_fz_arr = fz.const_arrays();
        GpuTuple<Real> max = ParReduce(TypeList<ReduceOpMax>{},
                                       TypeList<Real>{},
                                       fz, IntVect(0),
                                       [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
                                       -> GpuTuple<Real>
                                       {
                                           return { ma_fz_arr[box_no](i,j,k) };
                                       });
        wt_max = get<0>(max) + std::numeric_limits<Real>::epsilon();
        n_substep = int( std::ceil(wt_max * coef / CFL_MAX) );
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(n_substep >= 1,
                                         "Kessler: Number of precipitation substeps must be greater than 0!");
        coef /= Real(n_substep);
        dtn  /= Real(n_substep);

        for (int nsub(0); nsub<n_substep; ++nsub) {
            for ( MFIter mfi(*tabs, TilingIfNotGPU()); mfi.isValid(); ++mfi ){
                auto rho_array = mic_fab_vars[MicVar_Kess::rho]->array(mfi);
                auto qp_array  = mic_fab_vars[MicVar_Kess::qp]->array(mfi);
                auto rain_accum_array = mic_fab_vars[MicVar_Kess::rain_accum]->array(mfi);
                auto fz_array  = fz.array(mfi);

                const auto dJ_array = (m_detJ_cc) ? m_detJ_cc->const_array(mfi) : Array4<const Real>{};

                const Box& tbx = mfi.tilebox();
                const Box& tbz = mfi.tilebox(IntVect(0,0,1),IntVect(0));

                ParallelFor(tbz, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                {
                    Real rho_avg, qp_avg;
                    if (k==k_lo) {
                        rho_avg = rho_array(i,j,k);
                        qp_avg  = qp_array(i,j,k);
                    } else if (k==k_hi+1) {
                        rho_avg = rho_array(i,j,k-1);
                        qp_avg  = qp_array(i,j,k-1);
                    } else {
                        rho_avg = myhalf*(rho_array(i,j,k-1) + rho_array(i,j,k));
                        qp_avg = myhalf*(qp_array(i,j,k-1)  + qp_array(i,j,k));
                    }

                    qp_avg = std::max(Real(0), qp_avg);

                    Real V_terminal = Real(36.34)*std::pow(rho_avg*Real(0.001)*qp_avg, Real(0.1346))*std::pow(rho_avg/Real(1.16), -myhalf);
                    fz_array(i,j,k) = rho_avg*V_terminal*qp_avg;

                    if(k==k_lo){
                        rain_accum_array(i,j,k) = rain_accum_array(i,j,k) + rho_avg*qp_avg*V_terminal*dtn/Real(1000.0)*Real(1000.0);
                    }
                });

                ParallelFor(tbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                {
                    Real dJinv = (dJ_array) ? Real(1)/dJ_array(i,j,k) : Real(1);

                    if(std::fabs(fz_array(i,j,k+1)) < 1e-14) fz_array(i,j,k+1) = Real(0);
                    if(std::fabs(fz_array(i,j,k  )) < 1e-14) fz_array(i,j,k  ) = Real(0);
                    Real dq_sed = dJinv * (Real(1)/rho_array(i,j,k)) * (fz_array(i,j,k+1) - fz_array(i,j,k)) * coef;
                    if(std::fabs(dq_sed) < 1e-14) dq_sed = Real(0);

                    qp_array(i,j,k) +=  dq_sed;
                    qp_array(i,j,k)  = std::max(Real(0), qp_array(i,j,k));
                });
            }
        }
    }

    if (solverChoice.moisture_type == MoistureType::Kessler_NoRain) {
        if (!do_cond) { return; }
        for ( MFIter mfi(*tabs,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto qv_array    = mic_fab_vars[MicVar_Kess::qv]->array(mfi);
            auto qc_array    = mic_fab_vars[MicVar_Kess::qcl]->array(mfi);
            auto qt_array    = mic_fab_vars[MicVar_Kess::qt]->array(mfi);
            auto tabs_array  = mic_fab_vars[MicVar_Kess::tabs]->array(mfi);
            auto theta_array = mic_fab_vars[MicVar_Kess::theta]->array(mfi);
            auto pres_array  = mic_fab_vars[MicVar_Kess::pres]->array(mfi);

            auto tbx = mfi.tilebox();
            if (tbx.smallEnd(0) == i_lo) { tbx.growLo(0,-m_real_width); }
            if (tbx.bigEnd(0)   == i_hi) { tbx.growHi(0,-m_real_width); }
            if (tbx.smallEnd(1) == j_lo) { tbx.growLo(1,-m_real_width); }
            if (tbx.bigEnd(1)   == j_hi) { tbx.growHi(1,-m_real_width); }

            Real d_fac_cond = m_fac_cond;

            ParallelFor(tbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                qc_array(i,j,k) = std::max(Real(0), qc_array(i,j,k));

                Real qsat, dtqsat;
                Real dq_clwater_to_vapor, dq_vapor_to_clwater;

                Real pressure = pres_array(i,j,k);
                erf_qsatw(tabs_array(i,j,k), pressure, qsat);
                erf_dtqsatw(tabs_array(i,j,k), pressure, dtqsat);

                dq_vapor_to_clwater = Real(0);
                dq_clwater_to_vapor = Real(0);

                Real fac = (L_v/Cp_d)*dtqsat;

                if (qv_array(i,j,k) > qsat){
                    dq_vapor_to_clwater = std::min(qv_array(i,j,k), (qv_array(i,j,k)-qsat)/(Real(1) + fac));
                }
                if (qv_array(i,j,k) < qsat && qc_array(i,j,k) > Real(0)){
                    dq_clwater_to_vapor = std::min(qc_array(i,j,k), (qsat - qv_array(i,j,k))/(Real(1) + fac));
                }

                qv_array(i,j,k) += -dq_vapor_to_clwater + dq_clwater_to_vapor;
                qc_array(i,j,k) +=  dq_vapor_to_clwater - dq_clwater_to_vapor;

                Real theta_over_T = theta_array(i,j,k)/tabs_array(i,j,k);

                theta_array(i,j,k) += theta_over_T * d_fac_cond * (dq_vapor_to_clwater - dq_clwater_to_vapor);

                qv_array(i,j,k) = std::max(Real(0), qv_array(i,j,k));
                qc_array(i,j,k) = std::max(Real(0), qc_array(i,j,k));

                qt_array(i,j,k) = qv_array(i,j,k) + qc_array(i,j,k);
            });
        }
    }
}
#endif

void Kessler::AdvanceKesslerRefactored (const SolverChoice &solverChoice)
{
    bool do_cond = m_do_cond;
    auto tabs    = mic_fab_vars[MicVar_Kess::tabs];
    auto domain  = m_geom.Domain();
    int i_lo = domain.smallEnd(0);
    int i_hi = domain.bigEnd(0);
    int j_lo = domain.smallEnd(1);
    int j_hi = domain.bigEnd(1);
    int k_lo = domain.smallEnd(2);
    int k_hi = domain.bigEnd(2);
    if (solverChoice.moisture_type == MoistureType::Kessler)
    {
        MultiFab fz;
        auto ba    = tabs->boxArray();
        auto dm    = tabs->DistributionMap();
        fz.define(convert(ba, IntVect(0,0,1)), dm, 1, 0); // No ghost cells

        Real dtn  = dt;
        Real coef = dtn/m_dzmin;

        for ( MFIter mfi(*tabs,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto qv_array    = mic_fab_vars[MicVar_Kess::qv]->array(mfi);
            auto qc_array    = mic_fab_vars[MicVar_Kess::qcl]->array(mfi);
            auto qp_array    = mic_fab_vars[MicVar_Kess::qp]->array(mfi);
            auto qt_array    = mic_fab_vars[MicVar_Kess::qt]->array(mfi);
            auto tabs_array  = mic_fab_vars[MicVar_Kess::tabs]->array(mfi);
            auto pres_array  = mic_fab_vars[MicVar_Kess::pres]->array(mfi);
            auto theta_array = mic_fab_vars[MicVar_Kess::theta]->array(mfi);
            auto rho_array   = mic_fab_vars[MicVar_Kess::rho]->array(mfi);

            auto tbx = mfi.tilebox();
            if (tbx.smallEnd(0) == i_lo) { tbx.growLo(0,-m_real_width); }
            if (tbx.bigEnd(0)   == i_hi) { tbx.growHi(0,-m_real_width); }
            if (tbx.smallEnd(1) == j_lo) { tbx.growLo(1,-m_real_width); }
            if (tbx.bigEnd(1)   == j_hi) { tbx.growHi(1,-m_real_width); }

            Real d_fac_cond = m_fac_cond;

            ParallelFor(tbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                qv_array(i,j,k) = std::max(Real(0), qv_array(i,j,k));
                qc_array(i,j,k) = std::max(Real(0), qc_array(i,j,k));
                qp_array(i,j,k) = std::max(Real(0), qp_array(i,j,k));

                Real qsat_local, dtqsat_local;
                Real pressure = pres_array(i,j,k);
                erf_qsatw(tabs_array(i,j,k), pressure, qsat_local);
                erf_dtqsatw(tabs_array(i,j,k), pressure, dtqsat_local);

                if (qsat_local <= Real(0)) {
                    amrex::Warning("qsat computed as non-positive; setting to Real(0)!");
                    qsat_local = Real(0);
                }

                const KesslerSourceTerms source_terms = kessler_warm_rain_sources(
                    qv_array(i,j,k), qc_array(i,j,k), qp_array(i,j,k), rho_array(i,j,k),
                    pressure, qsat_local, dtqsat_local, dtn, do_cond);

                qv_array(i,j,k) += -source_terms.dq_vapor_to_cloud
                                 +  source_terms.dq_cloud_to_vapor
                                 +  source_terms.dq_rain_to_vapor;
                qc_array(i,j,k) +=  source_terms.dq_vapor_to_cloud
                                 -  source_terms.dq_cloud_to_vapor
                                 -  source_terms.dq_cloud_to_rain;
                qp_array(i,j,k) +=  source_terms.dq_cloud_to_rain
                                 -  source_terms.dq_rain_to_vapor;

                Real theta_over_T = theta_array(i,j,k)/tabs_array(i,j,k);
                theta_array(i,j,k) += theta_over_T * d_fac_cond
                    * (source_terms.dq_vapor_to_cloud
                       - source_terms.dq_cloud_to_vapor
                       - source_terms.dq_rain_to_vapor);

                qv_array(i,j,k) = std::max(Real(0), qv_array(i,j,k));
                qc_array(i,j,k) = std::max(Real(0), qc_array(i,j,k));
                qp_array(i,j,k) = std::max(Real(0), qp_array(i,j,k));

                qt_array(i,j,k) = qv_array(i,j,k) + qc_array(i,j,k);
            });
        }

        for ( MFIter mfi(fz, TilingIfNotGPU()); mfi.isValid(); ++mfi ){
            auto rho_array = mic_fab_vars[MicVar_Kess::rho]->array(mfi);
            auto qp_array  = mic_fab_vars[MicVar_Kess::qp]->array(mfi);

            auto fz_array  = fz.array(mfi);
            const Box& tbz = mfi.tilebox();

            ParallelFor(tbz, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                Real rho_avg, qp_avg;

                if (k==k_lo) {
                    rho_avg = rho_array(i,j,k);
                    qp_avg  = qp_array(i,j,k);
                } else if (k==k_hi+1) {
                    rho_avg = rho_array(i,j,k-1);
                    qp_avg  = qp_array(i,j,k-1);
                } else {
                    rho_avg = myhalf*(rho_array(i,j,k-1) + rho_array(i,j,k));
                    qp_avg = myhalf*(qp_array(i,j,k-1)  + qp_array(i,j,k));
                }

                qp_avg = std::max(Real(0), qp_avg);

                const Real terminal_velocity = kessler_terminal_velocity(rho_avg, qp_avg);
                fz_array(i,j,k) = kessler_precip_flux(rho_avg, terminal_velocity, qp_avg);
            });
        }

        auto const& ma_fz_arr = fz.const_arrays();
        GpuTuple<Real> max = ParReduce(TypeList<ReduceOpMax>{},
                                       TypeList<Real>{},
                                       fz, IntVect(0),
                                       [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
                                       -> GpuTuple<Real>
                                       {
                                           return { ma_fz_arr[box_no](i,j,k) };
                                       });
        int n_substep = kessler_num_sedimentation_substeps(get<0>(max), dt, m_dzmin);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(n_substep >= 1,
                                         "Kessler: Number of precipitation substeps must be greater than 0!");
        coef /= Real(n_substep);
        dtn  /= Real(n_substep);

        for (int nsub(0); nsub<n_substep; ++nsub) {
            for ( MFIter mfi(*tabs, TilingIfNotGPU()); mfi.isValid(); ++mfi ){
                auto rho_array = mic_fab_vars[MicVar_Kess::rho]->array(mfi);
                auto qp_array  = mic_fab_vars[MicVar_Kess::qp]->array(mfi);
                auto rain_accum_array = mic_fab_vars[MicVar_Kess::rain_accum]->array(mfi);
                auto fz_array  = fz.array(mfi);

                const auto dJ_array = (m_detJ_cc) ? m_detJ_cc->const_array(mfi) : Array4<const Real>{};

                const Box& tbx = mfi.tilebox();
                const Box& tbz = mfi.tilebox(IntVect(0,0,1),IntVect(0));

                ParallelFor(tbz, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                {
                    Real rho_avg, qp_avg;
                    if (k==k_lo) {
                        rho_avg = rho_array(i,j,k);
                        qp_avg  = qp_array(i,j,k);
                    } else if (k==k_hi+1) {
                        rho_avg = rho_array(i,j,k-1);
                        qp_avg  = qp_array(i,j,k-1);
                    } else {
                        rho_avg = myhalf*(rho_array(i,j,k-1) + rho_array(i,j,k));
                        qp_avg = myhalf*(qp_array(i,j,k-1)  + qp_array(i,j,k));
                    }

                    qp_avg = std::max(Real(0), qp_avg);

                    const Real terminal_velocity = kessler_terminal_velocity(rho_avg, qp_avg);
                    fz_array(i,j,k) = kessler_precip_flux(rho_avg, terminal_velocity, qp_avg);

                    if(k==k_lo){
                        rain_accum_array(i,j,k) = rain_accum_array(i,j,k) + rho_avg*qp_avg*terminal_velocity*dtn/Real(1000.0)*Real(1000.0);
                    }
                });

                ParallelFor(tbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                {
                    Real dJinv = (dJ_array) ? Real(1)/dJ_array(i,j,k) : Real(1);

                    fz_array(i,j,k+1) = kessler_zero_small_value(fz_array(i,j,k+1));
                    fz_array(i,j,k  ) = kessler_zero_small_value(fz_array(i,j,k  ));
                    Real dq_sed = dJinv * (Real(1)/rho_array(i,j,k)) * (fz_array(i,j,k+1) - fz_array(i,j,k)) * coef;
                    dq_sed = kessler_zero_small_value(dq_sed);

                    qp_array(i,j,k) +=  dq_sed;
                    qp_array(i,j,k)  = std::max(Real(0), qp_array(i,j,k));
                });
            }
        }
    }

    if (solverChoice.moisture_type == MoistureType::Kessler_NoRain) {
        if (!do_cond) { return; }
        for ( MFIter mfi(*tabs,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto qv_array    = mic_fab_vars[MicVar_Kess::qv]->array(mfi);
            auto qc_array    = mic_fab_vars[MicVar_Kess::qcl]->array(mfi);
            auto qt_array    = mic_fab_vars[MicVar_Kess::qt]->array(mfi);
            auto tabs_array  = mic_fab_vars[MicVar_Kess::tabs]->array(mfi);
            auto theta_array = mic_fab_vars[MicVar_Kess::theta]->array(mfi);
            auto pres_array  = mic_fab_vars[MicVar_Kess::pres]->array(mfi);

            auto tbx = mfi.tilebox();
            if (tbx.smallEnd(0) == i_lo) { tbx.growLo(0,-m_real_width); }
            if (tbx.bigEnd(0)   == i_hi) { tbx.growHi(0,-m_real_width); }
            if (tbx.smallEnd(1) == j_lo) { tbx.growLo(1,-m_real_width); }
            if (tbx.bigEnd(1)   == j_hi) { tbx.growHi(1,-m_real_width); }

            Real d_fac_cond = m_fac_cond;

            ParallelFor(tbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                qc_array(i,j,k) = std::max(Real(0), qc_array(i,j,k));

                Real qsat, dtqsat;
                Real dq_clwater_to_vapor, dq_vapor_to_clwater;

                Real pressure = pres_array(i,j,k);
                erf_qsatw(tabs_array(i,j,k), pressure, qsat);
                erf_dtqsatw(tabs_array(i,j,k), pressure, dtqsat);

                dq_vapor_to_clwater = Real(0);
                dq_clwater_to_vapor = Real(0);

                Real fac = (L_v/Cp_d)*dtqsat;

                if (qv_array(i,j,k) > qsat){
                    dq_vapor_to_clwater = std::min(qv_array(i,j,k), (qv_array(i,j,k)-qsat)/(Real(1) + fac));
                }
                if (qv_array(i,j,k) < qsat && qc_array(i,j,k) > Real(0)){
                    dq_clwater_to_vapor = std::min(qc_array(i,j,k), (qsat - qv_array(i,j,k))/(Real(1) + fac));
                }

                qv_array(i,j,k) += -dq_vapor_to_clwater + dq_clwater_to_vapor;
                qc_array(i,j,k) +=  dq_vapor_to_clwater - dq_clwater_to_vapor;

                Real theta_over_T = theta_array(i,j,k)/tabs_array(i,j,k);
                theta_array(i,j,k) += theta_over_T * d_fac_cond * (dq_vapor_to_clwater - dq_clwater_to_vapor);

                qv_array(i,j,k) = std::max(Real(0), qv_array(i,j,k));
                qc_array(i,j,k) = std::max(Real(0), qc_array(i,j,k));

                qt_array(i,j,k) = qv_array(i,j,k) + qc_array(i,j,k);
            });
        }
    }
}
