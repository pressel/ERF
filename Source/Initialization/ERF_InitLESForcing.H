#ifndef ERF_INIT_LES_FORCING_H
#define ERF_INIT_LES_FORCING_H

#include <string>
#include <memory>
#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include "ERForcingReader.H"

class ERFInitLESForcing {
public:
    ERFInitLESForcing();

    void define(const std::string& filename);

    void initialize_domain(const amrex::Geometry& geom,
                           const amrex::MultiFab& z_phys_cc,
                           amrex::MultiFab& u_cc,
                           amrex::MultiFab& v_cc,
                           amrex::MultiFab& w_cc,
                           amrex::MultiFab& cons_cc,
                           amrex::Real start_time);

    std::shared_ptr<ERForcingReader> get_reader() { return m_reader; }

private:
    std::shared_ptr<ERForcingReader> m_reader;
};

#endif // ERF_INIT_LES_FORCING_H
