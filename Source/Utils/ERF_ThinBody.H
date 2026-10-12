#ifndef ERF_THIN_BODY_H_
#define ERF_THIN_BODY_H_

#include <algorithm>
#include <sstream>

#include <AMReX_Box.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_iMultiFab.H>

namespace ERFThinBody {

/** Convert a volume integral to the corresponding spatial mean. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real
volume_weighted_mean (amrex::Real integral, amrex::Real volume) noexcept
{
    return integral / volume;
}

/** Distance from a coordinate to a closed interval. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real
gap_to_interval (amrex::Real point, amrex::Real low, amrex::Real high) noexcept
{
    return amrex::max(amrex::max(low - point, amrex::Real(0.0)), point - high);
}

/** Distance to the nearest copy of an interval in a periodic direction. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real
periodic_gap (amrex::Real point, amrex::Real low, amrex::Real high,
              bool periodic, amrex::Real period) noexcept
{
    amrex::Real gap = gap_to_interval(point, low, high);
    if (periodic) {
        gap = amrex::min(gap, gap_to_interval(point, low - period, high - period));
        gap = amrex::min(gap, gap_to_interval(point, low + period, high + period));
    }
    return gap;
}

/** Height of a ConstantDz cell centre above the physical zlo boundary. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real
constant_dz_ground_distance (int k, int klo, amrex::Real dz) noexcept
{
    return (static_cast<amrex::Real>(k - klo) + amrex::Real(0.5)) * dz;
}

/**
 * Add the equivalent periodic representation of a listed normal face when it
 * lies on a periodic seam. This keeps both valid nodal values impermeable.
 */
inline amrex::Vector<amrex::IntVect>
periodic_face_representations (const amrex::Vector<amrex::IntVect>& faces,
                               const amrex::Box& domain, int dir,
                               bool periodic)
{
    amrex::Vector<amrex::IntVect> result = faces;
    if (!periodic) {
        return result;
    }

    const amrex::Box face_domain = amrex::surroundingNodes(domain, dir);
    const int low = face_domain.smallEnd(dir);
    const int high = face_domain.bigEnd(dir);
    for (const amrex::IntVect& face : faces) {
        if (face[dir] == low || face[dir] == high) {
            amrex::IntVect equivalent = face;
            equivalent[dir] = (face[dir] == low) ? high : low;
            if (std::find(result.begin(), result.end(), equivalent) == result.end()) {
                result.push_back(equivalent);
            }
        }
    }
    return result;
}

/** Validate thin-face input indices against the orientation-specific domain. */
inline void
validate_face_indices (const amrex::Vector<amrex::IntVect>& faces,
                       const amrex::Box& domain,
                       int dir,
                       const char* input_key)
{
    const amrex::Box face_domain = amrex::surroundingNodes(domain, dir);
    for (const amrex::IntVect& face : faces) {
        if (!face_domain.contains(face)) {
            std::ostringstream message;
            message << "erf." << input_key << " contains face " << face
                    << " outside the valid " << "xyz"[dir] << "-face domain " << face_domain;
            amrex::Abort(message.str());
        }
    }
}

/**
 * Set every FAB representation of a requested nodal face to the zero-flux
 * value. The host list is copied once by the caller; this kernel is setup-only.
 */
template <typename FaceVector>
inline void
mask_faces (amrex::iMultiFab& mask, const FaceVector& faces_d, int nfaces)
{
    const amrex::IntVect* face_ptr = faces_d.data();
    for (amrex::MFIter mfi(mask); mfi.isValid(); ++mfi) {
        const amrex::Box valid = mfi.validbox();
        const amrex::IntVect lo = valid.smallEnd();
        const amrex::IntVect hi = valid.bigEnd();
        auto mask_arr = mask.array(mfi);
        amrex::ParallelFor(nfaces, [=] AMREX_GPU_DEVICE (int iface) noexcept {
            const int i = face_ptr[iface][0];
            const int j = face_ptr[iface][1];
            const int k = face_ptr[iface][2];
            if (i >= lo[0] && i <= hi[0] &&
                j >= lo[1] && j <= hi[1] &&
                k >= lo[2] && k <= hi[2]) {
                mask_arr(i,j,k) = 0;
            }
        });
    }
}

} // namespace ERFThinBody

#endif
