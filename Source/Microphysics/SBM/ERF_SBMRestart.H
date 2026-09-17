#ifndef ERF_SBM_RESTART_H_
#define ERF_SBM_RESTART_H_

#include "ERF_SBMLayout.H"

#include <AMReX_REAL.H>

#include <string>

namespace erf_sbm {

inline constexpr const char* SBM_CONSTRAINT_POLICY_ID = "complete-groups-donor-support-v2";
inline constexpr const char* SBM_TRANSPORT_POLICY_ID = "WENO_Z3-group-FCT-v4";
inline constexpr const char* SBM_AMR_TRANSFER_POLICY_ID = "carrier-weighted-mixing-ratio-AMR-v1";
inline constexpr const char* SBM_BOUNDARY_POLICY_ID = "periodic+wall+outflow-no-inflow-v1";

struct SBMCheckpointSchema {
    std::string schema_version;
    std::string population_ids;
    std::string moment_modes;
    std::string grid_identity;
    std::string property_identity;
    std::string constraint_policy;
    std::string projection_identity;
    std::string transport_identity;
    std::string amr_transfer_policy;
    std::string numerical_policy;
    std::string boundary_policy;
};

[[nodiscard]] SBMCheckpointSchema
make_checkpoint_schema(const SBMLayout& layout,
                       const std::string& constraint_policy,
                       const std::string& transport_identity,
                       const std::string& numerical_policy,
                       const std::string& boundary_policy = SBM_BOUNDARY_POLICY_ID,
                       const std::string& amr_transfer_policy = SBM_AMR_TRANSFER_POLICY_ID);

[[nodiscard]] std::string compare_checkpoint_schema(const SBMCheckpointSchema& expected,
                                                    const SBMCheckpointSchema& actual);

void write_checkpoint_schema(const std::string& path, const SBMCheckpointSchema& schema);
[[nodiscard]] SBMCheckpointSchema read_checkpoint_schema(const std::string& path);

/** Scale-aware equality used for checkpointed bulk projections. */
[[nodiscard]] bool compare_projection(amrex::Real checkpointed,
                                      amrex::Real reconstructed,
                                      amrex::Real scale,
                                      int operation_count,
                                      amrex::Real* absolute_error = nullptr) noexcept;

} // namespace erf_sbm

#endif
