#ifndef ERF_DIAGNOSTICS_FORCING_FILE_H
#define ERF_DIAGNOSTICS_FORCING_FILE_H

#include <string>
#include <vector>
#include <AMReX_Vector.H>
#include <AMReX_RealVect.H>
#include <AMReX_MultiFab.H>
#include "ERForcingReader.H"

class ERFDiagnosticsForcingFile {
public:
    ERFDiagnosticsForcingFile(std::shared_ptr<ERForcingReader> reader);

    // Run diagnostics
    void write_diagnostics(const amrex::Real& time,
                              const amrex::Geometry& geom,
                              const amrex::MultiFab& z_phys_cc,
                              const amrex::MultiFab& theta,
                              const amrex::MultiFab& qv,
                              const amrex::MultiFab& u,
                              const amrex::MultiFab& v);

    // Run initial diagnostics
    void initial_diagnostics(const amrex::Real& time,
                              const amrex::Geometry& geom,
                              const amrex::MultiFab& z_phys_cc,
                              const amrex::MultiFab& u,
                              const amrex::MultiFab& v,
                              const amrex::MultiFab& theta,
                              const amrex::MultiFab& qv);

private:
    std::shared_ptr<ERForcingReader> m_reader;
};

#endif // ERF_DIAGNOSTICS_FORCING_FILE_H
