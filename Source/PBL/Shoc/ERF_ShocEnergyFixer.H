#ifndef ERF_SHOC_ENERGY_FIXER_H_
#define ERF_SHOC_ENERGY_FIXER_H_

#include "ERF_ShocColumnData.H"

class ShocEnergyFixer
{
public:
    static int diagnose_active_top (const amrex::Vector<amrex::Real>& tke);

    AMREX_GPU_HOST_DEVICE
    static int diagnose_active_top (const amrex::Array4<const amrex::Real>& tke,
                                    int ic,
                                    int nlev);

    static void apply_column (const ShocColumnData& col,
                              int ic,
                              amrex::Real dt,
                              const amrex::Vector<amrex::Real>& thl_old,
                              const amrex::Vector<amrex::Real>& qv_old,
                              const amrex::Vector<amrex::Real>& qc_old,
                              const amrex::Vector<amrex::Real>& qi_old,
                              const amrex::Vector<amrex::Real>& u_old,
                              const amrex::Vector<amrex::Real>& v_old,
                              const amrex::Vector<amrex::Real>& tke_old,
                              amrex::Vector<amrex::Real>& thl_new,
                              const amrex::Vector<amrex::Real>& qv_new,
                              const amrex::Vector<amrex::Real>& qc_new,
                              const amrex::Vector<amrex::Real>& qi_new,
                              const amrex::Vector<amrex::Real>& u_new,
                              const amrex::Vector<amrex::Real>& v_new,
                              const amrex::Vector<amrex::Real>& tke_new);

    AMREX_GPU_HOST_DEVICE
    static void apply_column_in_place (ShocColumnData& col,
                                       int ic,
                                       amrex::Real dt);
};

#endif
