#ifndef ERF_PROB_H_
#define ERF_PROB_H_

#include <string>

#include "ERF_ProbCommon.H"
#include "prob_parm.H"
#include "ERF_InitDyCOMS.H"

class Problem : public ProblemBase
{
public:
    Problem();

#include "Prob/ERF_InitRayleighDamping.H"

    void init_custom_pert (
        const amrex::Box&  bx,
        const amrex::Box& xbx,
        const amrex::Box& ybx,
        const amrex::Box& zbx,
        amrex::Array4<amrex::Real const> const& state,
        amrex::Array4<amrex::Real      > const& state_pert,
        amrex::Array4<amrex::Real      > const& x_vel_pert,
        amrex::Array4<amrex::Real      > const& y_vel_pert,
        amrex::Array4<amrex::Real      > const& z_vel_pert,
        amrex::Array4<amrex::Real      > const& r_hse,
        amrex::Array4<amrex::Real      > const& p_hse,
        amrex::Array4<amrex::Real const> const& z_nd,
        amrex::Array4<amrex::Real const> const& z_cc,
        amrex::GeometryData const& geomdata,
        amrex::Array4<amrex::Real const> const& mf_m,
        amrex::Array4<amrex::Real const> const& mf_u,
        amrex::Array4<amrex::Real const> const& mf_v,
        const SolverChoice& sc,
        const int lev) override;

    void erf_init_dens_hse (amrex::MultiFab& rho_hse,
                            std::unique_ptr<amrex::MultiFab>& z_phys_nd,
                            std::unique_ptr<amrex::MultiFab>& z_phys_cc,
                            amrex::Geometry const& geom) override;

    void erf_init_dens_hse_moist (amrex::MultiFab& rho_hse,
                                  std::unique_ptr<amrex::MultiFab>& z_phys_nd,
                                  amrex::Geometry const& geom) override;

    amrex::Real getSurfacePressure () const override { return parms.p_surf; }

    void update_w_subsidence (
        const amrex::Real& time,
        amrex::Vector<amrex::Real>& wbar,
        amrex::Gpu::DeviceVector<amrex::Real>& d_wbar,
        const amrex::Geometry& geom,
        std::unique_ptr<amrex::MultiFab>& z_phys_nd) override;

    void update_geostrophic_profile (
        const amrex::Real& /*time*/,
        amrex::Vector<amrex::Real>& u_geos,
        amrex::Gpu::DeviceVector<amrex::Real>& d_u_geos,
        amrex::Vector<amrex::Real>& v_geos,
        amrex::Gpu::DeviceVector<amrex::Real>& d_v_geos,
        const amrex::Geometry& geom,
        std::unique_ptr<amrex::MultiFab>& z_phys_cc) override;

protected:
    std::string name () override { return "DyCOMS_RF02"; }

private:
    ProbParm parms;
    amrex::Vector<amrex::Real> zlevels;
};

#endif
