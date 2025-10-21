#ifndef ERF_RADIATION_DYCOMS_H
#define ERF_RADIATION_DYCOMS_H

#include <ERF_RadiationInterface.H>
#include <ERF_DataStruct.H>

#include <AMReX_Geometry.H>
#include <AMReX_BoxArray.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

/**
 * Simple longwave radiation model tailored to the DyCOMS-II RF02 case.
 *
 * Implements the Beer's-law cooling parameterization described in the
 * DyCOMS-II intercomparison specs.
 */
class RadiationDyCOMS : public IRadiation
{
public:
    explicit RadiationDyCOMS (const SolverChoice& sc);
    ~RadiationDyCOMS () override = default;

    void
    Init (const amrex::Geometry& geom,
          const amrex::BoxArray& ba,
          amrex::MultiFab* cons_in) override;

    void
    Run (int& level,
         int& step,
         amrex::Real& time,
         const amrex::Real& dt,
         const amrex::BoxArray& ba,
         amrex::Geometry& geom,
         amrex::MultiFab* cons_in,
         amrex::MultiFab* lsm_fluxes,
         amrex::MultiFab* lsm_zenith,
         amrex::Vector<amrex::MultiFab*>& lsm_input_ptrs,
         amrex::Vector<amrex::MultiFab*>& lsm_output_ptrs,
         amrex::MultiFab* qheating_rates,
         amrex::MultiFab* z_phys_nd,
         amrex::MultiFab* lat,
         amrex::MultiFab* lon) override;

    void
    WriteDataLog (const amrex::Real& time) override;

    void
    compute_radiative_tendency (const amrex::MFIter& mfi,
                                amrex::MultiFab* cons_in,
                                amrex::MultiFab* qheating_rates,
                                amrex::MultiFab* z_phys_nd,
                                const amrex::Geometry& geom,
                                amrex::MultiFab* zi_field = nullptr);

private:
    SolverChoice m_solver_choice;
    amrex::Geometry m_geom;
    amrex::BoxArray m_ba;

    amrex::Real m_F0;
    amrex::Real m_F1;
    amrex::Real m_kappa;
    amrex::Real m_qc_threshold;
    amrex::Real m_a;
    amrex::Real m_rho_i;
    amrex::Real m_divergence;
    amrex::Real m_zi;
    amrex::Real m_qt_threshold;
    bool m_use_local_zi;

    bool m_initialized = false;
    amrex::MultiFab* m_last_heating = nullptr;
    int m_last_level = 0;
};

#endif
