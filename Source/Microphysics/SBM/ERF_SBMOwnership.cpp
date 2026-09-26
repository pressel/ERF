#include "ERF_SBMOwnership.H"

#include <AMReX.H>

namespace erf_sbm {

const char* host_write_path_name(const HostWritePath path) noexcept
{
    switch (path) {
    case HostWritePath::Advection: return "advection";
    case HostWritePath::Diffusion: return "diffusion";
    case HostWritePath::Source: return "source";
    case HostWritePath::Positivity: return "positivity";
    case HostWritePath::Microphysics: return "microphysics";
    case HostWritePath::Wall: return "wall";
    case HostWritePath::Boundary: return "boundary";
    }
    return "unknown";
}

bool host_write_allowed(const bool sbm_active, const int component) noexcept
{
    return !sbm_active || (component != RhoQ2_comp && component != RhoQ3_comp);
}

void require_host_write_allowed(const bool sbm_active, const int component,
                               const HostWritePath path)
{
    if (!host_write_allowed(sbm_active, component)) {
        amrex::Abort(std::string("SBM owns compact qc/qr; native ") +
                     host_write_path_name(path) + " write denied for component " +
                     std::to_string(component));
    }
}

} // namespace erf_sbm
