#include "ERF_SBMRestart.H"

#include <AMReX_MFIter.H>
#include <AMReX_MultiFabUtil.H>

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace erf_sbm {

std::string restart_schema(const SBMLayout& layout)
{
    return std::string("ERF-SBM-RESTART-M1-v1\n") +
           "layout=" + layout.schema_identity() + "\n" +
           "constraint-policy=nonnegative-bin-mass-and-moments-v1\n" +
           "projection=liquid-mass-sum-to-qc-qr-v1\n" +
           "representation=bin-mass-density-v1\n" +
           "transport=zero-transport-fixture-v1\n";
}

bool restart_schema_matches(const SBMLayout& layout, const std::string& persisted)
{
    return restart_schema(layout) == persisted;
}

bool restart_projection_matches(const amrex::MultiFab& spectrum,
                                const amrex::MultiFab& persisted_core,
                                const SBMBulkProjection& projection,
                                const int qc_component, const int qr_component,
                                const amrex::Real tolerance_scale)
{
    if (spectrum.boxArray() != persisted_core.boxArray() ||
        spectrum.DistributionMap() != persisted_core.DistributionMap() ||
        qc_component < 0 || qr_component < 0 ||
        qc_component >= persisted_core.nComp() || qr_component >= persisted_core.nComp() ||
        !(tolerance_scale >= amrex::Real(0.0))) {
        return false;
    }

    amrex::MultiFab expected(persisted_core.boxArray(), persisted_core.DistributionMap(), 2, 0);
    amrex::MultiFab stored(persisted_core.boxArray(), persisted_core.DistributionMap(), 2, 0);
    expected.setVal(amrex::Real(0.0));
    for (amrex::MFIter mfi(expected); mfi.isValid(); ++mfi) {
        projection.apply_to_core(mfi.validbox(), spectrum.const_array(mfi),
                                 expected.array(mfi), 0, 1);
    }
    amrex::MultiFab::Copy(stored, persisted_core, qc_component, 0, 1, 0);
    amrex::MultiFab::Copy(stored, persisted_core, qr_component, 1, 1, 0);

    const amrex::Real scale = std::max({expected.norm0(0), expected.norm0(1),
                                        stored.norm0(0), stored.norm0(1)});
    amrex::MultiFab::Subtract(expected, stored, 0, 0, 2, 0);
    const amrex::Real difference = std::max(expected.norm0(0), expected.norm0(1));
    const amrex::Real tolerance = tolerance_scale *
        std::numeric_limits<amrex::Real>::epsilon() *
        std::max(scale, std::numeric_limits<amrex::Real>::min());
    return difference <= tolerance;
}

} // namespace erf_sbm
