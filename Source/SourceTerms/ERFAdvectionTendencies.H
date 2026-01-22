#ifndef ERF_ADVECTION_TENDENCIES_H
#define ERF_ADVECTION_TENDENCIES_H

#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_Vector.H>
#include "ERForcingReader.H"

class ERFAdvectionTendencies {
public:
    ERFAdvectionTendencies(std::shared_ptr<ERForcingReader> reader);

    void compute_tendencies(const amrex::Real& time,
                            const amrex::Geometry& geom,
                            const amrex::MultiFab& z_phys_cc,
                            amrex::MultiFab& theta_rhs,
                            amrex::MultiFab& qv_rhs);

private:
    std::shared_ptr<ERForcingReader> m_reader;
};

#endif // ERF_ADVECTION_TENDENCIES_H
