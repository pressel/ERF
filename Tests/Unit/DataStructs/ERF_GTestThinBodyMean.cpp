#include <gtest/gtest.h>

#include "ERF_ThinBody.H"

TEST(ThinBodyProjection, SingularRhsUsesVolumeWeightedMean)
{
    constexpr amrex::Real cell_volume = amrex::Real(2.0);
    constexpr amrex::Real rhs_values[] = {
        amrex::Real(2.0), amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0)
    };

    amrex::Real integral = amrex::Real(0.0);
    amrex::Real total_volume = amrex::Real(0.0);
    for (amrex::Real value : rhs_values) {
        integral += value * cell_volume;
        total_volume += cell_volume;
    }

    ASSERT_DOUBLE_EQ(integral, amrex::Real(4.0));
    ASSERT_DOUBLE_EQ(total_volume, amrex::Real(8.0));
    const amrex::Real mean = ERFThinBody::volume_weighted_mean(integral, total_volume);
    EXPECT_DOUBLE_EQ(mean, amrex::Real(0.5));

    amrex::Real corrected_integral = amrex::Real(0.0);
    amrex::Real legacy_unweighted_sum = amrex::Real(0.0);
    for (amrex::Real value : rhs_values) {
        corrected_integral += (value - mean) * cell_volume;
        legacy_unweighted_sum += value - integral;
    }
    EXPECT_DOUBLE_EQ(corrected_integral, amrex::Real(0.0));
    EXPECT_DOUBLE_EQ(legacy_unweighted_sum, amrex::Real(-14.0));
}
