#ifndef ERF_SUBSIDENCE_H
#define ERF_SUBSIDENCE_H

#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_Vector.H>
#include "ERForcingReader.H"

class ERFSubsidence {
public:
    ERFSubsidence(std::shared_ptr<ERForcingReader> reader);

    void compute_subsidence_tendency(const amrex::Real& time,
                                     const amrex::Geometry& geom,
                                     const amrex::MultiFab& z_phys_cc,
                                     amrex::MultiFab& theta_rhs);

private:
    std::shared_ptr<ERForcingReader> m_reader;
};

#endif // ERF_SUBSIDENCE_H
