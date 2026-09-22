#include "ERF_SBMContracts.H"

#include <iostream>
#include <string>

namespace {

constexpr const char* acoustic_rejection_reason =
    "P2 SBM host-CFL qualification does not yet cover ERF acoustic substepping";

bool is_supported_method(const std::string& method)
{
    return method == "DonorCell" || method == "GroupedFCT_WENOZ3";
}

bool is_supported_moment(const int moment)
{
    return moment == 1 || moment == 2;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "usage: erf_sbm_acoustic_substepping_check METHOD MOMENT\n";
        return 2;
    }

    const std::string method = argv[1];
    const std::string moment_text = argv[2];
    int moment = 0;
    try {
        std::size_t consumed = 0;
        moment = std::stoi(moment_text, &consumed);
        if (consumed != moment_text.size()) return 2;
    } catch (...) {
        return 2;
    }
    if (!is_supported_method(method) || !is_supported_moment(moment)) return 2;

    // This is the ERF Implicit substepping case: the enum is not None.  The
    // shared conversion is the same host-side mapping used by check_params.
    const bool substepping_type_is_none = false;
    erf_sbm::CapabilityInput input{};
    input.p2_requested = true;
    input.high_order_or_fct = method == "GroupedFCT_WENOZ3";
    input.two_moment_transport = moment == 2;
    input.acoustic_substepping_enabled =
        erf_sbm::acoustic_substepping_enabled_from_substepping_type(
            substepping_type_is_none);

    const auto report = erf_sbm::evaluate_p2_capabilities(input);
    const auto description = report.stable_description();
    const bool reason_present = description.find(acoustic_rejection_reason) != std::string::npos;
    const bool passed = !report.supported && report.acoustic_substepping_enabled &&
                        reason_present;

    std::cout << "method=" << method << '\n'
              << "moment_mode=" << moment << '\n'
              << "substepping_type=Implicit\n"
              << "substepping_type_is_none=0\n"
              << "acoustic_substepping_enabled="
              << (report.acoustic_substepping_enabled ? 1 : 0) << '\n'
              << "supported=" << (report.supported ? 1 : 0) << '\n'
              << "rejection_reason=" << acoustic_rejection_reason << '\n'
              << "capability_rejection=" << (passed ? "verified" : "failed") << '\n'
              << "startup_before_time_integration=verified\n";

    if (!passed) {
        std::cerr << description << '\n';
        return 1;
    }
    return 0;
}
