#include <AMReX.H>

#include <gtest/gtest.h>

#include <cstring>
#include <iostream>

#if defined(AMREX_USE_CUDA)
#include <cuda_runtime_api.h>
#elif defined(AMREX_USE_HIP)
#include <hip/hip_runtime.h>
#endif

namespace
{
constexpr int erf_test_skip_code = 77;

bool
is_gtest_discovery (int argc, char** argv)
{
    for (int n = 1; n < argc; ++n) {
        if (std::strcmp(argv[n], "--gtest_list_tests") == 0) {
            return true;
        }
    }
    return false;
}

bool
gpu_runtime_available ()
{
#if defined(AMREX_USE_CUDA)
    int ndev = 0;
    const cudaError_t err = cudaGetDeviceCount(&ndev);
    if (err != cudaSuccess || ndev <= 0) {
        std::cerr << "Skipping SHOC CUDA unit tests: "
                  << cudaGetErrorString(err)
                  << " (device count = " << ndev << ")\n";
        return false;
    }
#elif defined(AMREX_USE_HIP)
    int ndev = 0;
    const hipError_t err = hipGetDeviceCount(&ndev);
    if (err != hipSuccess || ndev <= 0) {
        std::cerr << "Skipping SHOC HIP unit tests: "
                  << hipGetErrorString(err)
                  << " (device count = " << ndev << ")\n";
        return false;
    }
#endif
    return true;
}
}

int
main (int argc, char** argv)
{
    const bool discovery = is_gtest_discovery(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);

    if (discovery) {
        return RUN_ALL_TESTS();
    }

    if (!gpu_runtime_available()) {
        return erf_test_skip_code;
    }

    amrex::Initialize(argc, argv, false);
    const int rc = RUN_ALL_TESTS();
    amrex::Finalize();
    return rc;
}
