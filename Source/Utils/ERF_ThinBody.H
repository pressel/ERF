#ifndef ERF_THIN_BODY_H_
#define ERF_THIN_BODY_H_

#include <sstream>

#include <AMReX_Box.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_iMultiFab.H>

namespace ERFThinBody {

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
