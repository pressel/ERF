#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_ParmParse.H>
#include <AMReX_RealBox.H>
#include <AMReX_iMultiFab.H>

#include <ERF.H>
#include <ERF_ThinBody.H>
#include <ERF_Utils.H>

#include <gtest/gtest.h>

namespace {

using amrex::Box;
using amrex::BoxArray;
using amrex::DistributionMapping;
using amrex::Geometry;
using amrex::IntVect;
using amrex::MultiFab;
using amrex::ParmParse;
using amrex::Real;
using amrex::RealBox;
using amrex::Vector;

std::string
new_prefix ()
{
    static int serial = 0;
    return "erf_advchoice_test_" + std::to_string(++serial);
}

void
set_all_schemes (SolverChoice& sc, AdvType scheme)
{
    sc.advChoice.dycore_horiz_adv_type = scheme;
    sc.advChoice.dycore_vert_adv_type = scheme;
    sc.advChoice.dryscal_horiz_adv_type = scheme;
    sc.advChoice.dryscal_vert_adv_type = scheme;
    sc.advChoice.moistscal_horiz_adv_type = scheme;
    sc.advChoice.moistscal_vert_adv_type = scheme;
}

Real
read_cell (const MultiFab& mf, const IntVect& cell)
{
    Real value = Real(0.0);
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
        if (mfi.validbox().contains(cell)) {
            const auto arr = mf.const_array(mfi);
            const Real* ptr = &arr(cell[0], cell[1], cell[2], 0);
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, ptr, ptr + 1, &value);
            break;
        }
    }
    return value;
}

Geometry
make_geometry (const Box& domain, const RealBox& real_box)
{
    const amrex::Array<int, AMREX_SPACEDIM> periodicity{0, 0, 0};
    return Geometry(domain, real_box, amrex::CoordSys::cartesian, periodicity);
}

TEST(AdvChoiceParser, DefaultsAliasesAndAllSixFractions)
{
    const std::string prefix = new_prefix();
    ParmParse pp(prefix);
    pp.add("dycore_horiz_adv_type", std::string("Upwind_3rd"));
    pp.add("dycore_horiz_upw_frac", Real(0.0));
    pp.add("dycore_vert_adv_type", std::string("Blended_3rd4th"));
    pp.add("dycore_vert_upw_frac", Real(0.25));
    pp.add("dryscal_horiz_adv_type", std::string("Upwind_5th"));
    pp.add("dryscal_horiz_upw_frac", Real(0.5));
    pp.add("dryscal_vert_adv_type", std::string("Blended_5th6th"));
    pp.add("dryscal_vert_upw_frac", Real(0.75));
    pp.add("moistscal_horiz_adv_type", std::string("Blended_3rd4th"));
    pp.add("moistscal_horiz_upw_frac", Real(1.0));
    pp.add("moistscal_vert_adv_type", std::string("Upwind_3rd"));
    pp.add("moistscal_vert_upw_frac", Real(0.125));

    AdvChoice choice;
    choice.init_params(prefix);
    EXPECT_EQ(choice.dycore_horiz_adv_type, AdvType::Upwind_3rd);
    EXPECT_EQ(choice.dycore_vert_adv_type, AdvType::Upwind_3rd);
    EXPECT_EQ(choice.dryscal_horiz_adv_type, AdvType::Upwind_5th);
    EXPECT_EQ(choice.dryscal_vert_adv_type, AdvType::Upwind_5th);
    EXPECT_EQ(choice.moistscal_horiz_adv_type, AdvType::Upwind_3rd);
    EXPECT_EQ(choice.moistscal_vert_adv_type, AdvType::Upwind_3rd);
    EXPECT_DOUBLE_EQ(choice.dycore_horiz_upw_frac, Real(0.0));
    EXPECT_DOUBLE_EQ(choice.dycore_vert_upw_frac, Real(0.25));
    EXPECT_DOUBLE_EQ(choice.dryscal_horiz_upw_frac, Real(0.5));
    EXPECT_DOUBLE_EQ(choice.dryscal_vert_upw_frac, Real(0.75));
    EXPECT_DOUBLE_EQ(choice.moistscal_horiz_upw_frac, Real(1.0));
    EXPECT_DOUBLE_EQ(choice.moistscal_vert_upw_frac, Real(0.125));

    const std::string defaults_prefix = new_prefix();
    AdvChoice defaults;
    defaults.init_params(defaults_prefix);
    EXPECT_FALSE(defaults.use_efficient_advection);
    EXPECT_EQ(defaults.dycore_horiz_adv_type, AdvType::Upwind_3rd);
    EXPECT_EQ(defaults.dryscal_vert_adv_type, AdvType::Upwind_3rd);
    EXPECT_EQ(defaults.moistscal_horiz_adv_type, AdvType::Weno_3);
    EXPECT_EQ(defaults.moistscal_vert_adv_type, AdvType::Weno_3);
    EXPECT_DOUBLE_EQ(defaults.moistscal_vert_upw_frac, Real(1.0));
}

TEST(AdvChoiceParser, ReinitializationResetsFractionsFlagsAndFaceLists)
{
    AdvChoice choice;
    const std::string configured_prefix = new_prefix();
    ParmParse configured(configured_prefix);
    configured.add("use_efficient_advection", true);
    configured.add("dycore_horiz_adv_type", std::string("Upwind_3rd"));
    configured.add("dycore_horiz_upw_frac", Real(0.25));
    configured.addarr("zero_xflux_faces", Vector<IntVect>{IntVect(1,2,3), IntVect(1,2,3)});
    choice.init_params(configured_prefix);
    ASSERT_TRUE(choice.use_efficient_advection);
    ASSERT_EQ(choice.zero_xflux.size(), 1u);
    ASSERT_TRUE(choice.have_zero_flux_faces);

    choice.init_params(new_prefix());
    EXPECT_FALSE(choice.use_efficient_advection);
    EXPECT_DOUBLE_EQ(choice.dycore_horiz_upw_frac, Real(1.0));
    EXPECT_TRUE(choice.zero_xflux.empty());
    EXPECT_TRUE(choice.zero_yflux.empty());
    EXPECT_TRUE(choice.zero_zflux.empty());
    EXPECT_FALSE(choice.have_zero_flux_faces);
    EXPECT_EQ(choice.dycore_horiz_adv_type, AdvType::Upwind_3rd);
}

TEST(AdvChoiceParser, DisplayUsesStoredStateWithoutMutatingParmParse)
{
    const std::string prefix = new_prefix();
    ParmParse pp(prefix);
    AdvChoice choice;
    choice.init_params(prefix);
    choice.dycore_horiz_adv_type = AdvType::Upwind_3rd;
    choice.dycore_horiz_upw_frac = Real(0.25);
    EXPECT_FALSE(pp.contains("dycore_horiz_adv_type"));
    choice.display(prefix);
    EXPECT_FALSE(pp.contains("dycore_horiz_adv_type"));
}

TEST(AdvChoiceParser, RejectsBlankUnknownUnusedAndNonfiniteFractions)
{
    {
        const std::string prefix = new_prefix();
        ParmParse pp(prefix);
        pp.add("dycore_horiz_adv_type", std::string(""));
        EXPECT_DEATH(AdvChoice{}.init_params(prefix), "explicitly empty");
    }
    {
        const std::string prefix = new_prefix();
        ParmParse pp(prefix);
        pp.add("dycore_horiz_adv_type", std::string("TypoScheme"));
        EXPECT_DEATH(AdvChoice{}.init_params(prefix), "unsupported value");
    }
    {
        const std::string prefix = new_prefix();
        ParmParse pp(prefix);
        pp.add("moistscal_horiz_upw_frac", Real(0.25));
        EXPECT_DEATH(AdvChoice{}.init_params(prefix), "unused by the selected");
    }
    {
        const std::string prefix = new_prefix();
        ParmParse pp(prefix);
        pp.add("moistscal_horiz_upw_frac", std::numeric_limits<Real>::quiet_NaN());
        EXPECT_DEATH(AdvChoice{}.init_params(prefix), "must be finite and in \\[0,1\\]");
    }

    const std::array<Real, 5> invalid_values{{
        Real(-0.1), Real(1.1), std::numeric_limits<Real>::quiet_NaN(),
        std::numeric_limits<Real>::infinity(), -std::numeric_limits<Real>::infinity()}};
    for (Real invalid : invalid_values) {
        const std::string prefix = new_prefix();
        ParmParse pp(prefix);
        pp.add("dryscal_horiz_adv_type", std::string("Upwind_3rd"));
        pp.add("dryscal_horiz_upw_frac", invalid);
        EXPECT_DEATH(AdvChoice{}.init_params(prefix), "must be finite and in \\[0,1\\]");
    }
}

TEST(AdvChoiceValidation, ScalarWenoPairingAndEBActiveFamilies)
{
    AdvChoice valid;
    valid.dryscal_horiz_adv_type = AdvType::Weno_5;
    valid.dryscal_vert_adv_type = AdvType::Weno_5;
    valid.validate_active_configuration(false, false);
    valid.dryscal_horiz_adv_type = AdvType::Centered_4th;
    valid.dryscal_vert_adv_type = AdvType::Upwind_3rd;
    valid.validate_active_configuration(false, false);
    valid.dycore_horiz_adv_type = AdvType::Weno_7;
    valid.dycore_vert_adv_type = AdvType::Weno_7;
    EXPECT_NO_THROW(valid.validate_active_configuration(false, false));
    valid.dycore_vert_adv_type = AdvType::Weno_7Z;
    EXPECT_DEATH(valid.validate_active_configuration(false, false),
                 "dycore_horiz_adv_type = WENO7.*dycore_vert_adv_type = WENOZ7");

    AdvChoice mismatch;
    mismatch.dryscal_horiz_adv_type = AdvType::Weno_5;
    mismatch.dryscal_vert_adv_type = AdvType::Centered_2nd;
    EXPECT_DEATH(mismatch.validate_active_configuration(false, false), "must select the same WENO");
    mismatch.dryscal_horiz_adv_type = AdvType::Centered_2nd;
    mismatch.dryscal_vert_adv_type = AdvType::Weno_5;
    EXPECT_DEATH(mismatch.validate_active_configuration(false, false), "must select the same WENO");
    mismatch.dryscal_horiz_adv_type = AdvType::Weno_5;
    mismatch.dryscal_vert_adv_type = AdvType::Weno_5Z;
    EXPECT_DEATH(mismatch.validate_active_configuration(false, false), "must select the same WENO");

    AdvChoice eb_dry;
    EXPECT_NO_THROW(eb_dry.validate_active_configuration(true, false));
    EXPECT_DEATH(eb_dry.validate_active_configuration(true, true), "moistscal_horiz_adv_type = WENO3");
    eb_dry.moistscal_horiz_adv_type = AdvType::Upwind_3rd;
    eb_dry.moistscal_vert_adv_type = AdvType::Centered_2nd;
    EXPECT_NO_THROW(eb_dry.validate_active_configuration(true, true));
    eb_dry.dycore_horiz_adv_type = AdvType::Weno_5;
    EXPECT_DEATH(eb_dry.validate_active_configuration(true, false), "dycore_horiz_adv_type = WENO5");
}

TEST(AdvChoiceHalo, MaximumStencilReachCoversAllActiveDirections)
{
    SolverChoice sc;
    sc.moisture_type = MoistureType::Kessler;
    sc.terrain_type = TerrainType::None;
    sc.use_num_diff = false;
    set_all_schemes(sc, AdvType::Centered_2nd);
    const auto ngrow = [&sc]() {
        return AdvChoice::required_ghost_cells(sc.advChoice, sc.use_num_diff,
                                               sc.terrain_type == TerrainType::EB,
                                               sc.moisture_type != MoistureType::None);
    };
    EXPECT_EQ(ngrow(), 2);

    std::array<AdvType*, 6> schemes{{
        &sc.advChoice.dycore_horiz_adv_type, &sc.advChoice.dycore_vert_adv_type,
        &sc.advChoice.dryscal_horiz_adv_type, &sc.advChoice.dryscal_vert_adv_type,
        &sc.advChoice.moistscal_horiz_adv_type, &sc.advChoice.moistscal_vert_adv_type}};
    for (AdvType* scheme : schemes) {
        *scheme = AdvType::Weno_7;
        EXPECT_EQ(ngrow(), 4);
        *scheme = AdvType::Weno_7Z;
        EXPECT_EQ(ngrow(), 4);
        *scheme = AdvType::Centered_2nd;
    }

    sc.advChoice.dycore_horiz_adv_type = AdvType::Upwind_5th;
    sc.advChoice.dycore_vert_adv_type = AdvType::Weno_7Z;
    EXPECT_EQ(ngrow(), 4);
    sc.advChoice.dycore_horiz_adv_type = AdvType::Weno_7Z;
    sc.advChoice.dycore_vert_adv_type = AdvType::Upwind_5th;
    EXPECT_EQ(ngrow(), 4);

    sc.terrain_type = TerrainType::EB;
    set_all_schemes(sc, AdvType::Centered_2nd);
    EXPECT_EQ(ngrow(), 4);
    sc.terrain_type = TerrainType::None;
    sc.use_num_diff = true;
    EXPECT_EQ(ngrow(), 3);

    sc.moisture_type = MoistureType::None;
    sc.advChoice.moistscal_horiz_adv_type = AdvType::Weno_7;
    sc.advChoice.moistscal_vert_adv_type = AdvType::Weno_7Z;
    EXPECT_EQ(ngrow(), 3);
}

TEST(ThinBodyGeometry, ExactFiniteFaceDistanceUsesPhysicalOriginAndIndexOffset)
{
    const IntVect dom_lo(-3, 5, 7);
    const Box domain(dom_lo, IntVect(-2, 6, 8));
    const RealBox real_box(Real(10.0), Real(20.0), Real(30.0),
                           Real(12.0), Real(21.0), Real(32.0));
    const Geometry geom = make_geometry(domain, real_box);
    BoxArray ba(domain);
    DistributionMapping dm(ba);
    auto distance = std::make_unique<MultiFab>(ba, dm, 1, 0);
    distance->setVal(Real(10.0));

    Vector<IntVect> xfaces{IntVect(-3, 5, 7)};
    Vector<IntVect> yfaces;
    Vector<IntVect> zfaces;
    thinbody_wall_dist(distance, xfaces, yfaces, zfaces, geom, true);

    // Cell center: (10.5,20.25,31.5); face: x=10, y=[20,20.5], z=[30,31].
    const IntVect target(-3, 5, 8);
    EXPECT_NEAR(read_cell(*distance, target), std::sqrt(Real(0.5)),
                Real(8.0) * std::numeric_limits<Real>::epsilon());
    EXPECT_NEAR(read_cell(*distance, IntVect(-3, 5, 7)), Real(0.5),
                Real(8.0) * std::numeric_limits<Real>::epsilon());
    EXPECT_NEAR(read_cell(*distance, IntVect(-3, 6, 8)), Real(0.75),
                Real(8.0) * std::numeric_limits<Real>::epsilon());

    // An interior x-normal face has cells on both sides at equal distance.
    xfaces.assign(1, IntVect(-2, 5, 7));
    distance->setVal(Real(10.0));
    thinbody_wall_dist(distance, xfaces, yfaces, zfaces, geom, true);
    EXPECT_NEAR(read_cell(*distance, IntVect(-3, 5, 7)), Real(0.5),
                Real(8.0) * std::numeric_limits<Real>::epsilon());
    EXPECT_NEAR(read_cell(*distance, IntVect(-2, 5, 7)), Real(0.5),
                Real(8.0) * std::numeric_limits<Real>::epsilon());

    xfaces.clear();
    yfaces.push_back(IntVect(-3, 5, 7));
    distance->setVal(Real(10.0));
    thinbody_wall_dist(distance, xfaces, yfaces, zfaces, geom, true);
    EXPECT_NEAR(read_cell(*distance, IntVect(-2, 5, 8)), Real(0.75),
                Real(8.0) * std::numeric_limits<Real>::epsilon());

    // A y-normal face also gives the exact normal distance on either side.
    yfaces.assign(1, IntVect(-3, 6, 7));
    distance->setVal(Real(10.0));
    thinbody_wall_dist(distance, xfaces, yfaces, zfaces, geom, true);
    EXPECT_NEAR(read_cell(*distance, IntVect(-3, 5, 7)), Real(0.25),
                Real(8.0) * std::numeric_limits<Real>::epsilon());
    EXPECT_NEAR(read_cell(*distance, IntVect(-3, 6, 7)), Real(0.25),
                Real(8.0) * std::numeric_limits<Real>::epsilon());

    yfaces.clear();
    zfaces.push_back(IntVect(-3, 5, 7));
    distance->setVal(Real(10.0));
    thinbody_wall_dist(distance, xfaces, yfaces, zfaces, geom, true);
    EXPECT_NEAR(read_cell(*distance, IntVect(-2, 5, 7)), std::sqrt(Real(0.5)),
                Real(8.0) * std::numeric_limits<Real>::epsilon());

    // A z-normal face is checked on both sides, too.
    zfaces.assign(1, IntVect(-3, 5, 8));
    distance->setVal(Real(10.0));
    thinbody_wall_dist(distance, xfaces, yfaces, zfaces, geom, true);
    EXPECT_NEAR(read_cell(*distance, IntVect(-3, 5, 7)), Real(0.5),
                Real(8.0) * std::numeric_limits<Real>::epsilon());
    EXPECT_NEAR(read_cell(*distance, IntVect(-3, 5, 8)), Real(0.5),
                Real(8.0) * std::numeric_limits<Real>::epsilon());

    // The nearest among multiple faces wins; a pre-existing smaller ground
    // distance is still preserved.
    zfaces.clear();
    xfaces = {IntVect(-3, 5, 7), IntVect(-2, 5, 7)};
    distance->setVal(Real(10.0));
    thinbody_wall_dist(distance, xfaces, yfaces, zfaces, geom, true);
    EXPECT_NEAR(read_cell(*distance, IntVect(-2, 5, 7)), Real(0.5),
                Real(8.0) * std::numeric_limits<Real>::epsilon());

    xfaces.assign(1, IntVect(-3, 5, 7));
    distance->setVal(Real(0.25));
    thinbody_wall_dist(distance, xfaces, yfaces, zfaces, geom, true);
    EXPECT_DOUBLE_EQ(read_cell(*distance, target), Real(0.25));
}

TEST(ThinBodyMasks, ValidatesOrientationSpecificFaceDomains)
{
    const Box domain(IntVect(0,0,0), IntVect(1,1,1));
    const Vector<IntVect> valid_x{
        IntVect(0,0,0), IntVect(1,0,0), IntVect(2,0,0)};
    const Vector<IntVect> valid_y{
        IntVect(0,0,0), IntVect(0,1,0), IntVect(0,2,0)};
    const Vector<IntVect> valid_z{
        IntVect(0,0,0), IntVect(0,0,1), IntVect(0,0,2)};
    EXPECT_NO_THROW(ERFThinBody::validate_face_indices(valid_x, domain, 0, "zero_xflux_faces"));
    EXPECT_NO_THROW(ERFThinBody::validate_face_indices(valid_y, domain, 1, "zero_yflux_faces"));
    EXPECT_NO_THROW(ERFThinBody::validate_face_indices(valid_z, domain, 2, "zero_zflux_faces"));
    EXPECT_DEATH(ERFThinBody::validate_face_indices(
                     Vector<IntVect>{IntVect(3,0,0)}, domain, 0, "zero_xflux_faces"),
                 "zero_xflux_faces contains face");
    EXPECT_DEATH(ERFThinBody::validate_face_indices(
                     Vector<IntVect>{IntVect(0,2,0)}, domain, 0, "zero_xflux_faces"),
                 "outside the valid x-face domain");
    EXPECT_DEATH(ERFThinBody::validate_face_indices(
                     Vector<IntVect>{IntVect(0,3,0)}, domain, 1, "zero_yflux_faces"),
                 "outside the valid y-face domain");
    EXPECT_DEATH(ERFThinBody::validate_face_indices(
                     Vector<IntVect>{IntVect(0,0,3)}, domain, 2, "zero_zflux_faces"),
                 "outside the valid z-face domain");
}

TEST(ThinBodyMasks, MasksEveryNodalFabRepresentation)
{
    Vector<Box> boxes{Box(IntVect(0,0,0), IntVect(0,1,1)),
                      Box(IntVect(1,0,0), IntVect(1,1,1))};
    BoxArray ba(boxes.data(), static_cast<int>(boxes.size()));
    ba.surroundingNodes(0);
    DistributionMapping dm(ba);
    amrex::iMultiFab mask(ba, dm, 1, 0);
    mask.setVal(1);
    Vector<IntVect> faces{IntVect(0,0,0), IntVect(1,0,0), IntVect(2,0,0)};
    amrex::Gpu::DeviceVector<IntVect> faces_d(faces.size());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, faces.begin(), faces.end(), faces_d.begin());
    ERFThinBody::mask_faces(mask, faces_d, static_cast<int>(faces_d.size()));
    amrex::Gpu::streamSynchronize();

    int local_zero_representations = 0;
    for (amrex::MFIter mfi(mask); mfi.isValid(); ++mfi) {
        if (mfi.validbox().contains(IntVect(1,0,0))) {
            const auto arr = mask.const_array(mfi);
            const int* ptr = &arr(1,0,0,0);
            int value = 1;
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, ptr, ptr + 1, &value);
            EXPECT_EQ(value, 0);
            ++local_zero_representations;
        }
    }
    amrex::ParallelDescriptor::ReduceIntSum(local_zero_representations);
    EXPECT_EQ(local_zero_representations, 2);
}

} // namespace
