#include <array>
#include <cmath>
#include <string>
#include <utility>

#include <AMReX_FArrayBox.H>
#include <AMReX_GpuContainers.H>

#include <gtest/gtest.h>

#include "ERF_GTestInterpolationCommon.H"

using namespace interpolation_test;

namespace {

inline amrex::Box centered_face_box ()
{
    return amrex::Box(amrex::IntVect(-4, 0, 0), amrex::IntVect(4, 0, 0));
}

void fill_centered_face_box (amrex::FArrayBox& qty,
                             const CenteredFaceStencil9& stencil)
{
    auto qty_arr = qty.array();
    qty_arr(-4, 0, 0, 0) = stencil.qm4;
    qty_arr(-3, 0, 0, 0) = stencil.qm3;
    qty_arr(-2, 0, 0, 0) = stencil.qm2;
    qty_arr(-1, 0, 0, 0) = stencil.qm1;
    qty_arr( 0, 0, 0, 0) = stencil.q0;
    qty_arr( 1, 0, 0, 0) = stencil.q1;
    qty_arr( 2, 0, 0, 0) = stencil.q2;
    qty_arr( 3, 0, 0, 0) = stencil.q3;
    qty_arr( 4, 0, 0, 0) = stencil.q4;
}

CenteredFaceStencil9 make_constant_stencil9 (const amrex::Real value)
{
    return CenteredFaceStencil9{value, value, value, value, value, value, value, value, value};
}

CenteredFaceStencil9 mirror_face_plus_half_stencil9 (const CenteredFaceStencil9& stencil)
{
    return CenteredFaceStencil9{
        amrex::Real(0.0),
        stencil.q4,
        stencil.q3,
        stencil.q2,
        stencil.q1,
        stencil.q0,
        stencil.qm1,
        stencil.qm2,
        stencil.qm3};
}

CenteredFaceStencil3 take_stencil3 (const CenteredFaceStencil9& stencil)
{
    return CenteredFaceStencil3{stencil.qm1, stencil.q0, stencil.q1};
}

CenteredFaceStencil5 take_stencil5 (const CenteredFaceStencil9& stencil)
{
    return CenteredFaceStencil5{stencil.qm2, stencil.qm1, stencil.q0, stencil.q1, stencil.q2};
}

CenteredFaceStencil7 take_stencil7 (const CenteredFaceStencil9& stencil)
{
    return CenteredFaceStencil7{stencil.qm3, stencil.qm2, stencil.qm1, stencil.q0,
                                stencil.q1, stencil.q2, stencil.q3};
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real evaluate_weno_x_device (const AdvType adv_type,
                                    const amrex::Array4<const amrex::Real>& qty,
                                    const int sigma) noexcept
{
    const amrex::Real upw = amrex::Real(sigma);
    amrex::Real value = amrex::Real(0.0);

    if (adv_type == AdvType::Weno_3) {
        WENO3 policy(qty, amrex::Real(1.0));
        policy.InterpolateInX(1, 0, 0, kQtyComp, value, upw);
    } else if (adv_type == AdvType::Weno_5) {
        WENO5 policy(qty, amrex::Real(1.0));
        policy.InterpolateInX(1, 0, 0, kQtyComp, value, upw);
    } else if (adv_type == AdvType::Weno_7) {
        WENO7 policy(qty, amrex::Real(1.0));
        policy.InterpolateInX(1, 0, 0, kQtyComp, value, upw);
    } else if (adv_type == AdvType::Weno_3Z) {
        WENO_Z3 policy(qty, amrex::Real(1.0));
        policy.InterpolateInX(1, 0, 0, kQtyComp, value, upw);
    } else if (adv_type == AdvType::Weno_3MZQ) {
        WENO_MZQ3 policy(qty, amrex::Real(1.0));
        policy.InterpolateInX(1, 0, 0, kQtyComp, value, upw);
    } else if (adv_type == AdvType::Weno_5Z) {
        WENO_Z5 policy(qty, amrex::Real(1.0));
        policy.InterpolateInX(1, 0, 0, kQtyComp, value, upw);
    } else {
        WENO_Z7 policy(qty, amrex::Real(1.0));
        policy.InterpolateInX(1, 0, 0, kQtyComp, value, upw);
    }

    return value;
}

amrex::Real evaluate_weno_x_on_device (const CenteredFaceStencil9& stencil,
                                       const AdvType adv_type,
                                       const int sigma)
{
    amrex::FArrayBox qty(centered_face_box(), 1);
    fill_centered_face_box(qty, stencil);
    const auto qty_arr = qty.const_array();
    amrex::Gpu::DeviceVector<amrex::Real> device_result(1, amrex::Real(0.0));
    amrex::Gpu::HostVector<amrex::Real> host_result(1, amrex::Real(0.0));
    amrex::Real* result_ptr = device_result.data();

    amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE (int) noexcept {
        result_ptr[0] = evaluate_weno_x_device(adv_type, qty_arr, sigma);
    });
    gpu_sync();

    amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                     device_result.begin(), device_result.end(), host_result.begin());
    return host_result[0];
}

// MUST MATCH: these raw tau helpers intentionally encode the WENO-Z tau
// definitions so the symmetry tests can select cases that exercise both raw
// tau signs before the scheme takes abs(tau).
amrex::Real raw_tau3 (const CenteredFaceStencil3& stencil)
{
    const amrex::Real b1 = (stencil.q0 - stencil.qm1) * (stencil.q0 - stencil.qm1);
    const amrex::Real b2 = (stencil.q1 - stencil.q0) * (stencil.q1 - stencil.q0);
    return b2 - b1;
}

amrex::Real raw_tau5 (const CenteredFaceStencil5& stencil)
{
    const amrex::Real c1 = amrex::Real(13.0) / amrex::Real(12.0);
    const amrex::Real b1 = c1 * (stencil.qm2 - amrex::Real(2.0) * stencil.qm1 + stencil.q0) *
                                (stencil.qm2 - amrex::Real(2.0) * stencil.qm1 + stencil.q0) +
                           amrex::Real(0.25) * (stencil.qm2 - amrex::Real(4.0) * stencil.qm1 + amrex::Real(3.0) * stencil.q0) *
                                                (stencil.qm2 - amrex::Real(4.0) * stencil.qm1 + amrex::Real(3.0) * stencil.q0);
    const amrex::Real b3 = c1 * (stencil.q0 - amrex::Real(2.0) * stencil.q1 + stencil.q2) *
                                (stencil.q0 - amrex::Real(2.0) * stencil.q1 + stencil.q2) +
                           amrex::Real(0.25) * (amrex::Real(3.0) * stencil.q0 - amrex::Real(4.0) * stencil.q1 + stencil.q2) *
                                                (amrex::Real(3.0) * stencil.q0 - amrex::Real(4.0) * stencil.q1 + stencil.q2);
    return b3 - b1;
}

amrex::Real raw_tau7 (const CenteredFaceStencil7& stencil)
{
    const amrex::Real b1 = ( stencil.qm3 * stencil.qm3 * amrex::Real(6649.)/amrex::Real(2880.0)
                           - stencil.qm3 * stencil.qm2 * amrex::Real(2623.)/amrex::Real(160.0)
                           + stencil.qm3 * stencil.qm1 * amrex::Real(9449.)/amrex::Real(480.0)
                           - stencil.qm3 * stencil.q0  * amrex::Real(11389.)/amrex::Real(1440.0)
                           + stencil.qm2 * stencil.qm2 * amrex::Real(28547.)/amrex::Real(960.0)
                           - stencil.qm2 * stencil.qm1 * amrex::Real(35047.)/amrex::Real(480.0)
                           + stencil.qm2 * stencil.q0  * amrex::Real(14369.)/amrex::Real(480.0)
                           + stencil.qm1 * stencil.qm1 * amrex::Real(44747.)/amrex::Real(960.0)
                           - stencil.qm1 * stencil.q0  * amrex::Real(6383.)/amrex::Real(160.0)
                           + stencil.q0  * stencil.q0  * amrex::Real(25729.)/amrex::Real(2880.0) );
    const amrex::Real b2 = ( stencil.qm2 * stencil.qm2 * amrex::Real(3169.0)/amrex::Real(2880.0)
                           - stencil.qm2 * stencil.qm1 * amrex::Real(3229.0)/amrex::Real(480.0)
                           + stencil.qm2 * stencil.q0  * amrex::Real(3169.0)/amrex::Real(480.0)
                           - stencil.qm2 * stencil.q1  * amrex::Real(2989.0)/amrex::Real(1440.0)
                           + stencil.qm1 * stencil.qm1 * amrex::Real(11147.0)/amrex::Real(960.0)
                           - stencil.qm1 * stencil.q0  * amrex::Real(11767.0)/amrex::Real(480.0)
                           + stencil.qm1 * stencil.q1  * amrex::Real(1283.0)/amrex::Real(160.0)
                           + stencil.q0  * stencil.q0  * amrex::Real(13667.0)/amrex::Real(960.0)
                           - stencil.q0  * stencil.q1  * amrex::Real(5069.0)/amrex::Real(480.0)
                           + stencil.q1  * stencil.q1  * amrex::Real(6649.0)/amrex::Real(2880.0) );
    const amrex::Real b3 = ( stencil.qm1 * stencil.qm1 * amrex::Real(6649.)/amrex::Real(2880.0)
                           - stencil.qm1 * stencil.q0  * amrex::Real(5069.)/amrex::Real(480.0)
                           + stencil.qm1 * stencil.q1  * amrex::Real(1283.)/amrex::Real(160.0)
                           - stencil.qm1 * stencil.q2  * amrex::Real(2989.)/amrex::Real(1440.0)
                           + stencil.q0  * stencil.q0  * amrex::Real(13667.)/amrex::Real(960.0)
                           - stencil.q0  * stencil.q1  * amrex::Real(11767.)/amrex::Real(480.0)
                           + stencil.q0  * stencil.q2  * amrex::Real(3169.)/amrex::Real(480.0)
                           + stencil.q1  * stencil.q1  * amrex::Real(11147.)/amrex::Real(960.0)
                           - stencil.q1  * stencil.q2  * amrex::Real(3229.)/amrex::Real(480.0)
                           + stencil.q2  * stencil.q2  * amrex::Real(3169.)/amrex::Real(2880.0) );
    const amrex::Real b4 = ( stencil.q0  * stencil.q0  * amrex::Real(25729.)/amrex::Real(2880.0)
                           - stencil.q0  * stencil.q1  * amrex::Real(6383.)/amrex::Real(160.0)
                           + stencil.q0  * stencil.q2  * amrex::Real(14369.)/amrex::Real(480.0)
                           - stencil.q0  * stencil.q3  * amrex::Real(11389.)/amrex::Real(1440.0)
                           + stencil.q1  * stencil.q1  * amrex::Real(44747.)/amrex::Real(960.0)
                           - stencil.q1  * stencil.q2  * amrex::Real(35047.)/amrex::Real(480.0)
                           + stencil.q1  * stencil.q3  * amrex::Real(9449.)/amrex::Real(480.0)
                           + stencil.q2  * stencil.q2  * amrex::Real(28547.)/amrex::Real(960.0)
                           - stencil.q2  * stencil.q3  * amrex::Real(2623.)/amrex::Real(160.0)
                           + stencil.q3  * stencil.q3  * amrex::Real(6649.)/amrex::Real(2880.0) );
    return b1 - b2 - b3 + b4;
}

std::string weno_trace (const AdvType adv_type,
                        const int sigma,
                        const int sample_index,
                        const amrex::Real actual,
                        const amrex::Real expected,
                        const amrex::Real factor)
{
    return "scheme=" + scheme_name(adv_type) +
           " sigma=" + std::to_string(sigma) +
           " sample=" + std::to_string(sample_index) +
           " actual=" + std::to_string(static_cast<double>(actual)) +
           " expected=" + std::to_string(static_cast<double>(expected)) +
           " normalized_error=" + std::to_string(static_cast<double>(normalized_error(actual, expected, factor)));
}

} // namespace

TEST(InterpolationWENOScalar, ClassicWenoPreservesConstantsAndReturnsFiniteValues)
{
    const std::array<AdvType, 3> schemes = {{AdvType::Weno_3, AdvType::Weno_5, AdvType::Weno_7}};
    const CenteredFaceStencil9 stencil = make_constant_stencil9(amrex::Real(1.75));

    for (const AdvType adv_type : schemes) {
        for (const int sigma : std::array<int, 2>{{-1, 1}}) {
            const amrex::Real actual = evaluate_weno_x_on_device(stencil, adv_type, sigma);
            EXPECT_TRUE(std::isfinite(actual));
            EXPECT_NEAR(actual, stencil.q0, scaled_tol(actual, stencil.q0, kWenoIdentityRelTol))
                << weno_trace(adv_type, sigma, 0, actual, stencil.q0, kWenoIdentityRelTol);
        }
    }
}

TEST(InterpolationWENOScalar, ClassicWenoIsExactForLinearFiniteVolumeData)
{
    const std::array<AdvType, 3> schemes = {{AdvType::Weno_3, AdvType::Weno_5, AdvType::Weno_7}};
    const CenteredFaceStencil9 stencil = make_centered_monomial_stencil9(1);
    const amrex::Real expected = centered_face_value_monomial(1, kFacePlusHalf);

    for (const AdvType adv_type : schemes) {
        for (const int sigma : std::array<int, 2>{{-1, 1}}) {
            const amrex::Real actual = evaluate_weno_x_on_device(stencil, adv_type, sigma);
            EXPECT_NEAR(actual, expected, scaled_tol(actual, expected, kWenoValueRelTol))
                << weno_trace(adv_type, sigma, 1, actual, expected, kWenoValueRelTol);
        }
    }
}

TEST(InterpolationWENOScalar, ClassicWenoOptimalFiniteVolumeReferencesMatchDesignDegree)
{
    for (int degree = 0; degree <= 2; ++degree) {
        const CenteredFaceStencil3 stencil = make_centered_monomial_stencil3(degree);
        const amrex::Real actual = weno3_fv_optimal_face_value(stencil);
        const amrex::Real expected = centered_face_value_monomial(degree, kFacePlusHalf);
        EXPECT_NEAR(actual, expected, scaled_tol(actual, expected, kPolynomialRelTol));
    }

    for (int degree = 0; degree <= 4; ++degree) {
        const CenteredFaceStencil5 stencil = make_centered_monomial_stencil5(degree);
        const amrex::Real actual = weno5_fv_optimal_face_value(stencil);
        const amrex::Real expected = centered_face_value_monomial(degree, kFacePlusHalf);
        EXPECT_NEAR(actual, expected, scaled_tol(actual, expected, kPolynomialRelTol));
    }

    for (int degree = 0; degree <= 6; ++degree) {
        const CenteredFaceStencil7 stencil = make_centered_monomial_stencil7(degree);
        const amrex::Real actual = weno7_fv_optimal_face_value(stencil);
        const amrex::Real expected = centered_face_value_monomial(degree, kFacePlusHalf);
        EXPECT_NEAR(actual, expected, scaled_tol(actual, expected, kPolynomialRelTol));
    }
}

TEST(InterpolationWENOScalar, WenoZVariantsPreserveConstantsAndLinearFields)
{
    const std::array<AdvType, 4> schemes = {{AdvType::Weno_3Z, AdvType::Weno_3MZQ,
                                             AdvType::Weno_5Z, AdvType::Weno_7Z}};
    const CenteredFaceStencil9 constant_stencil = make_constant_stencil9(amrex::Real(-0.875));
    const CenteredFaceStencil9 linear_stencil = make_centered_monomial_stencil9(1);
    const amrex::Real linear_expected = centered_face_value_monomial(1, kFacePlusHalf);

    for (const AdvType adv_type : schemes) {
        for (const int sigma : std::array<int, 2>{{-1, 1}}) {
            const amrex::Real constant_actual = evaluate_weno_x_on_device(constant_stencil, adv_type, sigma);
            EXPECT_TRUE(std::isfinite(constant_actual));
            EXPECT_NEAR(constant_actual, constant_stencil.q0,
                        scaled_tol(constant_actual, constant_stencil.q0, kWenoIdentityRelTol))
                << weno_trace(adv_type, sigma, 0, constant_actual, constant_stencil.q0, kWenoIdentityRelTol);

            const amrex::Real linear_actual = evaluate_weno_x_on_device(linear_stencil, adv_type, sigma);
            EXPECT_NEAR(linear_actual, linear_expected,
                        scaled_tol(linear_actual, linear_expected, kWenoValueRelTol))
                << weno_trace(adv_type, sigma, 1, linear_actual, linear_expected, kWenoValueRelTol);
        }
    }
}

TEST(InterpolationWENOScalar, WenoZOptimalFiniteVolumeReferencesMatchSmoothLimit)
{
    for (int degree = 0; degree <= 2; ++degree) {
        const CenteredFaceStencil3 stencil = make_centered_monomial_stencil3(degree);
        const amrex::Real actual = weno3_fv_optimal_face_value(stencil);
        const amrex::Real expected = centered_face_value_monomial(degree, kFacePlusHalf);
        EXPECT_NEAR(actual, expected, scaled_tol(actual, expected, kPolynomialRelTol));
    }

    for (int degree = 0; degree <= 4; ++degree) {
        const CenteredFaceStencil5 stencil = make_centered_monomial_stencil5(degree);
        const amrex::Real actual = weno5_fv_optimal_face_value(stencil);
        const amrex::Real expected = centered_face_value_monomial(degree, kFacePlusHalf);
        EXPECT_NEAR(actual, expected, scaled_tol(actual, expected, kPolynomialRelTol));
    }

    for (int degree = 0; degree <= 6; ++degree) {
        const CenteredFaceStencil7 stencil = make_centered_monomial_stencil7(degree);
        const amrex::Real actual = weno7_fv_optimal_face_value(stencil);
        const amrex::Real expected = centered_face_value_monomial(degree, kFacePlusHalf);
        EXPECT_NEAR(actual, expected, scaled_tol(actual, expected, kPolynomialRelTol));
    }
}

TEST(InterpolationWENOScalar, WenoZTauAbsoluteValueMirrorSymmetry)
{
    const CenteredFaceStencil9 z3_positive{amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0),
                                           amrex::Real(0.0), amrex::Real(-0.5), amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0)};
    const CenteredFaceStencil9 z3_negative{amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0), amrex::Real(-0.5),
                                           amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0)};
    const CenteredFaceStencil9 z5_positive{amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0),
                                           amrex::Real(0.0), amrex::Real(-0.5), amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0)};
    const CenteredFaceStencil9 z5_negative{amrex::Real(0.0), amrex::Real(0.0), amrex::Real(-0.5), amrex::Real(0.0),
                                           amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0)};
    const CenteredFaceStencil9 z7_positive{amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0),
                                           amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0), amrex::Real(-0.5), amrex::Real(0.0)};
    const CenteredFaceStencil9 z7_negative{amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0),
                                           amrex::Real(0.0), amrex::Real(-0.5), amrex::Real(-0.5), amrex::Real(0.0), amrex::Real(0.0)};

    EXPECT_GT(raw_tau3(take_stencil3(z3_positive)), amrex::Real(0.0));
    EXPECT_LT(raw_tau3(take_stencil3(z3_negative)), amrex::Real(0.0));
    EXPECT_GT(raw_tau5(take_stencil5(z5_positive)), amrex::Real(0.0));
    EXPECT_LT(raw_tau5(take_stencil5(z5_negative)), amrex::Real(0.0));
    EXPECT_GT(raw_tau7(take_stencil7(z7_positive)), amrex::Real(0.0));
    EXPECT_LT(raw_tau7(take_stencil7(z7_negative)), amrex::Real(0.0));

    const std::array<std::pair<AdvType, CenteredFaceStencil9>, 6> cases = {{
        {AdvType::Weno_3Z, z3_positive},
        {AdvType::Weno_3Z, z3_negative},
        {AdvType::Weno_5Z, z5_positive},
        {AdvType::Weno_5Z, z5_negative},
        {AdvType::Weno_7Z, z7_positive},
        {AdvType::Weno_7Z, z7_negative}}};

    for (int sample_index = 0; sample_index < static_cast<int>(cases.size()); ++sample_index) {
        const AdvType adv_type = cases[sample_index].first;
        const CenteredFaceStencil9& stencil = cases[sample_index].second;
        const CenteredFaceStencil9 mirrored = mirror_face_plus_half_stencil9(stencil);
        const amrex::Real positive = evaluate_weno_x_on_device(stencil, adv_type, 1);
        const amrex::Real negative_mirror = evaluate_weno_x_on_device(mirrored, adv_type, -1);

        EXPECT_NEAR(positive, negative_mirror,
                    scaled_tol(positive, negative_mirror, kWenoValueRelTol))
            << weno_trace(adv_type, 1, sample_index, positive, negative_mirror, kWenoValueRelTol);
    }
}