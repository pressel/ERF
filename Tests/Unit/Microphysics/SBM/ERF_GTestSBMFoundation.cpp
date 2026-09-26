#include <gtest/gtest.h>

#include <AMReX_BoxArray.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_VisMF.H>

#include "ERF_SBMConstraintGroups.H"
#include "ERF_SBMBulkProjection.H"
#include "ERF_SBMOwnership.H"
#include "ERF_SBMRestart.H"
#include "ERF_SBMStateManager.H"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using amrex::Box;
using amrex::BoxArray;
using amrex::DistributionMapping;
using amrex::MultiFab;
using amrex::Real;

erf_sbm::SpectralGridSpec make_grid(const int nbins, const Real edge_scale = Real(1.0))
{
    erf_sbm::SpectralGridSpec grid;
    grid.coordinate_kind = erf_sbm::CoordinateKind::Mass;
    grid.coordinate_units = "kg";
    grid.edges.resize(static_cast<std::size_t>(nbins + 1));
    grid.pivots.resize(static_cast<std::size_t>(nbins));
    for (int b = 0; b <= nbins; ++b) {
        grid.edges[static_cast<std::size_t>(b)] = edge_scale * Real(b + 1);
    }
    for (int b = 0; b < nbins; ++b) {
        grid.pivots[static_cast<std::size_t>(b)] =
            (grid.edges[static_cast<std::size_t>(b)] + grid.edges[static_cast<std::size_t>(b + 1)]) / Real(2.0);
    }
    return grid;
}

erf_sbm::SpectralPopulationSpec make_population(
    const int population_id, const int nbins,
    const erf_sbm::MomentMode moment = erf_sbm::MomentMode::OneMoment,
    const Real edge_scale = Real(1.0))
{
    erf_sbm::SpectralPopulationSpec population;
    population.population_id = population_id;
    population.semantic_id = "population_" + std::to_string(population_id);
    population.phase = population_id == 0 ? erf_sbm::PopulationPhase::Liquid :
                                            erf_sbm::PopulationPhase::Aerosol;
    population.grid = make_grid(nbins, edge_scale);
    population.moment_mode = moment;
    return population;
}

erf_sbm::AttachedPropertyDescriptor make_property(const std::string& name,
                                                  const int carrier_population)
{
    erf_sbm::AttachedPropertyDescriptor property;
    property.name = name;
    property.semantic_id = name + ".semantic";
    property.units = "kg m^-3";
    property.carrier_population = carrier_population;
    property.kind = erf_sbm::PropertyKind::ExtensiveMass;
    property.support = erf_sbm::SupportRequirement::None;
    property.remap_policy = erf_sbm::PropertyRemapPolicy::CarrierBinConservative;
    property.transported = true;
    property.support_min = Real(0.0);
    return property;
}

erf_sbm::SBMLayout make_layout(
    const int nbins = 4,
    const erf_sbm::MomentMode moment = erf_sbm::MomentMode::OneMoment,
    const int split = -1,
    const Real edge_scale = Real(1.0),
    std::vector<erf_sbm::AttachedPropertyDescriptor> properties = {})
{
    erf_sbm::SBMLayoutSpec spec;
    spec.populations.push_back(make_population(0, nbins, moment, edge_scale));
    spec.liquid_projection = {0, split < 0 ? nbins / 2 : split};
    spec.attached_properties = std::move(properties);
    return erf_sbm::SBMLayout(std::move(spec));
}

BoxArray make_boxes()
{
    return BoxArray(Box(amrex::IntVect(0), amrex::IntVect(1)));
}

Real max_component_norm(const MultiFab& mf, const int ncomp)
{
    Real result = Real(0.0);
    for (int comp = 0; comp < ncomp; ++comp) result = std::max(result, mf.norm0(comp));
    return result;
}

TEST(SBMFoundation, RuntimeLayoutAndMomentOffsets)
{
    const auto one_moment = make_layout(5);
    ASSERT_EQ(one_moment.ncomp(), 5);
    EXPECT_EQ(one_moment.populations()[0].mass_offset, 0);
    EXPECT_EQ(one_moment.populations()[0].number_offset, -1);

    const auto two_moment = make_layout(5, erf_sbm::MomentMode::TwoMoment);
    ASSERT_EQ(two_moment.ncomp(), 10);
    EXPECT_EQ(two_moment.populations()[0].mass_offset, 0);
    EXPECT_EQ(two_moment.populations()[0].number_offset, 5);
    EXPECT_NE(one_moment.schema_identity(), two_moment.schema_identity());

    erf_sbm::SpectralGridSpec invalid;
    EXPECT_FALSE(erf_sbm::SpectralGrid::validate(invalid).valid);
    EXPECT_THROW(erf_sbm::SBMLayout([&]() {
        erf_sbm::SBMLayoutSpec spec;
        auto population = make_population(0, 1);
        population.grid.edges.clear();
        spec.populations.push_back(std::move(population));
        spec.liquid_projection = {0, 1};
        return spec;
    }()), std::invalid_argument);
}

TEST(SBMFoundation, ConstraintGroupsAreAtomicAndRuntimeSized)
{
    erf_sbm::SBMLayoutSpec spec;
    spec.populations.push_back(make_population(0, 2));
    spec.populations.push_back(make_population(1, 5, erf_sbm::MomentMode::TwoMoment));
    spec.liquid_projection = {0, 1};
    const erf_sbm::SBMLayout layout(std::move(spec));
    const auto groups = erf_sbm::make_constraint_groups(layout);
    ASSERT_EQ(groups.size(), 7u);
    EXPECT_EQ(groups[0].members.size(), 1u);
    EXPECT_EQ(groups[2].population_id, 1);
    EXPECT_EQ(groups[2].bin, 0);
    EXPECT_EQ(groups[2].members.size(), 2u);
    EXPECT_EQ(groups[2].transport_members, groups[2].members);

    // LinearConstraint stores an arbitrary number of homogeneous terms; M1
    // semantic evaluation must not impose a two-term device-flattener limit.
    erf_sbm::ConstraintGroup synthetic;
    synthetic.members = {0, 1, 2};
    synthetic.constraints.push_back({"three_term_homogeneous",
        {{0, Real(1.0)}, {1, Real(2.0)}, {2, Real(-1.0)}}});
    Real margin = Real(0.0);
    EXPECT_TRUE(synthetic.admissible({Real(1.0), Real(2.0), Real(1.0)}, &margin));
    EXPECT_DOUBLE_EQ(margin, Real(4.0));
    EXPECT_FALSE(synthetic.admissible({Real(0.0), Real(0.0), Real(1.0)}));
}

TEST(SBMFoundation, FixedBulkProjectionIsLinearAndNonMutating)
{
    const auto layout = make_layout(4, erf_sbm::MomentMode::OneMoment, 2);
    const erf_sbm::SBMBulkProjection projection(layout);
    std::vector<Real> x{Real(1.0), Real(2.0), Real(3.0), Real(4.0)};
    const auto original = x;
    const auto projected = projection.apply(x);
    EXPECT_DOUBLE_EQ(projected.qc, Real(3.0));
    EXPECT_DOUBLE_EQ(projected.qr, Real(7.0));
    EXPECT_EQ(x, original);

    const std::vector<Real> y{Real(4.0), Real(1.0), Real(2.0), Real(3.0)};
    const Real a = Real(1.25);
    const Real b = Real(0.5);
    std::vector<Real> combination(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) combination[i] = a*x[i] + b*y[i];
    const auto lhs = projection.apply(combination);
    const auto px = projection.apply(x);
    const auto py = projection.apply(y);
    EXPECT_NEAR(lhs.qc, a*px.qc + b*py.qc, Real(8.0)*std::numeric_limits<Real>::epsilon());
    EXPECT_NEAR(lhs.qr, a*px.qr + b*py.qr, Real(8.0)*std::numeric_limits<Real>::epsilon());
    EXPECT_DOUBLE_EQ(projection.apply(std::vector<Real>(4, Real(0.0))).qc, Real(0.0));
    EXPECT_DOUBLE_EQ(projection.apply(std::vector<Real>(4, Real(0.0))).qr, Real(0.0));
}

TEST(SBMFoundation, OwnershipKeepsVaporHostOwnedAndDeniesCompactLiquidWrites)
{
    using erf_sbm::HostWritePath;
    const std::vector<HostWritePath> paths{
        HostWritePath::Advection, HostWritePath::Diffusion, HostWritePath::Source,
        HostWritePath::Positivity, HostWritePath::Microphysics,
        HostWritePath::Wall, HostWritePath::Boundary};
    for (const auto path : paths) {
        EXPECT_TRUE(erf_sbm::host_write_allowed(false, RhoQ2_comp));
        EXPECT_TRUE(erf_sbm::host_write_allowed(false, RhoQ3_comp));
        EXPECT_FALSE(erf_sbm::host_write_allowed(true, RhoQ2_comp))
            << erf_sbm::host_write_path_name(path);
        EXPECT_FALSE(erf_sbm::host_write_allowed(true, RhoQ3_comp))
            << erf_sbm::host_write_path_name(path);
        EXPECT_TRUE(erf_sbm::host_write_allowed(true, RhoQ1_comp));
    }
}

TEST(SBMFoundation, ZeroAndNonzeroFixtureStatesRemainIdentityAndProject)
{
    const auto layout = make_layout();
    const BoxArray boxes = make_boxes();
    const DistributionMapping mapping(boxes);
    erf_sbm::SBMStateManager manager(layout, 1);
    manager.define(0, boxes, mapping);

    MultiFab core(boxes, mapping, RhoQ1_comp + 3, 0);
    core.setVal(Real(9.0));
    manager.project_to_core(0, core, RhoQ2_comp, RhoQ3_comp);
    EXPECT_DOUBLE_EQ(max_component_norm(manager.state(0), layout.ncomp()), Real(0.0));
    EXPECT_DOUBLE_EQ(core.norm0(RhoQ2_comp), Real(0.0));
    EXPECT_DOUBLE_EQ(core.norm0(RhoQ3_comp), Real(0.0));

    for (int bin = 0; bin < 4; ++bin) {
        manager.state(0).setVal(Real(bin + 1), bin, 1, 0);
    }
    MultiFab before(boxes, mapping, layout.ncomp(), 0);
    MultiFab::Copy(before, manager.state(0), 0, 0, layout.ncomp(), 0);
    manager.project_to_core(0, core, RhoQ2_comp, RhoQ3_comp);

    MultiFab difference(boxes, mapping, layout.ncomp(), 0);
    MultiFab::Copy(difference, manager.state(0), 0, 0, layout.ncomp(), 0);
    MultiFab::Subtract(difference, before, 0, 0, layout.ncomp(), 0);
    EXPECT_DOUBLE_EQ(max_component_norm(difference, layout.ncomp()), Real(0.0));
    EXPECT_DOUBLE_EQ(core.norm0(RhoQ2_comp), Real(3.0));
    EXPECT_DOUBLE_EQ(core.norm0(RhoQ3_comp), Real(7.0));
}

TEST(SBMFoundation, ManagerDefinesAndDestroysMultipleLevelsWithoutStaleState)
{
    const auto layout = make_layout(3, erf_sbm::MomentMode::TwoMoment);
    const BoxArray boxes = make_boxes();
    const DistributionMapping mapping(boxes);
    erf_sbm::SBMStateManager manager(layout, 2);
    manager.define(0, boxes, mapping);
    manager.define(1, boxes, mapping);
    EXPECT_TRUE(manager.is_defined(0));
    EXPECT_TRUE(manager.is_defined(1));
    manager.state(1).setVal(Real(4.0));
    manager.destroy(1);
    EXPECT_FALSE(manager.is_defined(1));
    EXPECT_THROW(static_cast<void>(manager.state(1)), std::logic_error);
    manager.define(1, boxes, mapping);
    EXPECT_DOUBLE_EQ(max_component_norm(manager.state(1), layout.ncomp()), Real(0.0));
    manager.destroy(0);
    manager.destroy(1);
    EXPECT_FALSE(manager.is_defined(0));
    EXPECT_FALSE(manager.is_defined(1));
}

TEST(SBMFoundation, RestartSchemaAndSpectrumRoundTripAreExact)
{
    auto properties = std::vector<erf_sbm::AttachedPropertyDescriptor>{
        make_property("aerosol_number", 0), make_property("solute_mass", 0)};
    const auto layout = make_layout(4, erf_sbm::MomentMode::TwoMoment, 2,
                                    Real(1.0), properties);
    const BoxArray boxes = make_boxes();
    const DistributionMapping mapping(boxes);
    erf_sbm::SBMStateManager manager(layout, 1);
    manager.define(0, boxes, mapping);
    for (int comp = 0; comp < layout.ncomp(); ++comp) {
        manager.state(0).setVal(Real(comp + 1) / Real(8.0), comp, 1, 0);
    }

    const auto unique = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const std::filesystem::path directory = std::filesystem::temp_directory_path() /
        ("erf_sbm_restart_" + std::to_string(unique));
    std::filesystem::create_directories(directory);
    const std::string prefix = (directory / "spectrum").string();
    amrex::VisMF::Write(manager.state(0), prefix);
    const std::string schema = erf_sbm::restart_schema(layout);
    {
        std::ofstream output(directory / "SBM_Schema", std::ios::binary);
        output << schema;
    }
    MultiFab restored(boxes, mapping, layout.ncomp(), 0);
    amrex::VisMF::Read(restored, prefix);
    MultiFab difference(boxes, mapping, layout.ncomp(), 0);
    MultiFab::Copy(difference, manager.state(0), 0, 0, layout.ncomp(), 0);
    MultiFab::Subtract(difference, restored, 0, 0, layout.ncomp(), 0);
    EXPECT_DOUBLE_EQ(max_component_norm(difference, layout.ncomp()), Real(0.0));
    EXPECT_TRUE(erf_sbm::restart_schema_matches(layout, schema));

    std::filesystem::remove_all(directory);
}

TEST(SBMFoundation, RestartSchemaRejectsChangedScientificMeaning)
{
    const auto original = make_layout();
    const auto changed_grid = make_layout(4, erf_sbm::MomentMode::OneMoment, 2, Real(1.1));
    const auto changed_moment = make_layout(4, erf_sbm::MomentMode::TwoMoment);
    const auto changed_projection = make_layout(4, erf_sbm::MomentMode::OneMoment, 1);

    auto properties_ab = std::vector<erf_sbm::AttachedPropertyDescriptor>{
        make_property("property_a", 0), make_property("property_b", 0)};
    auto properties_ba = properties_ab;
    std::reverse(properties_ba.begin(), properties_ba.end());
    const auto ordered_properties = make_layout(4, erf_sbm::MomentMode::OneMoment,
                                               2, Real(1.0), properties_ab);
    const auto reversed_properties = make_layout(4, erf_sbm::MomentMode::OneMoment,
                                                2, Real(1.0), properties_ba);
    const std::string schema = erf_sbm::restart_schema(original);
    EXPECT_TRUE(erf_sbm::restart_schema_matches(original, schema));
    EXPECT_FALSE(erf_sbm::restart_schema_matches(changed_grid, schema));
    EXPECT_FALSE(erf_sbm::restart_schema_matches(changed_moment, schema));
    EXPECT_FALSE(erf_sbm::restart_schema_matches(changed_projection, schema));
    EXPECT_FALSE(erf_sbm::restart_schema_matches(ordered_properties, schema));
    EXPECT_FALSE(erf_sbm::restart_schema_matches(reversed_properties,
                                                 erf_sbm::restart_schema(ordered_properties)));
}

TEST(SBMFoundation, CorruptCompactRestartIsRejectedBeforeProjectionCanRepairIt)
{
    const auto layout = make_layout();
    const BoxArray boxes = make_boxes();
    const DistributionMapping mapping(boxes);
    erf_sbm::SBMStateManager manager(layout, 1);
    manager.define(0, boxes, mapping);
    for (int bin = 0; bin < 4; ++bin) manager.state(0).setVal(Real(bin + 1), bin, 1, 0);

    MultiFab core(boxes, mapping, RhoQ1_comp + 3, 0);
    core.setVal(Real(0.0));
    const erf_sbm::SBMBulkProjection projection(layout);
    manager.project_to_core(0, core, RhoQ2_comp, RhoQ3_comp);
    ASSERT_TRUE(erf_sbm::restart_projection_matches(
        manager.state(0), core, projection, RhoQ2_comp, RhoQ3_comp));

    core.setVal(Real(99.0), RhoQ2_comp, 1, 0);
    const Real corrupted_value = core.norm0(RhoQ2_comp);
    EXPECT_FALSE(erf_sbm::restart_projection_matches(
        manager.state(0), core, projection, RhoQ2_comp, RhoQ3_comp));
    EXPECT_DOUBLE_EQ(core.norm0(RhoQ2_comp), corrupted_value);
}

} // namespace
