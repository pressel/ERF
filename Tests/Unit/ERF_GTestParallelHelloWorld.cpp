#include <AMReX.H>
#include <AMReX_ParallelDescriptor.H>

#include <gtest/gtest.h>

#ifndef ERF_PARALLEL_TEST_NPROCS
#define ERF_PARALLEL_TEST_NPROCS 2
#endif

TEST(ParallelEnvironment, HelloWorld)
{
    const int my_rank = amrex::ParallelDescriptor::MyProc();
    const int n_ranks = amrex::ParallelDescriptor::NProcs();

    EXPECT_GE(my_rank, 0);
    EXPECT_LT(my_rank, n_ranks);

    EXPECT_EQ(n_ranks, ERF_PARALLEL_TEST_NPROCS);

    int local_val = 1;
    amrex::ParallelDescriptor::ReduceIntSum(local_val);

    EXPECT_EQ(local_val, n_ranks);
}