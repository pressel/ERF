#include <AMReX.H>

#include <gtest/gtest.h>

int
main (int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    amrex::Initialize(argc, argv, false);
    const int rc = RUN_ALL_TESTS();
    amrex::Finalize();
    return rc;
}
