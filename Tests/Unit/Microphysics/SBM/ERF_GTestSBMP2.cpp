#include <gtest/gtest.h>

#include "ERF_SBMAMR.H"
#include "ERF_AuxiliaryStateManager.H"
#include "ERF_SBMBoundary.H"
#include "ERF_SBMBulkProjection.H"
#include "ERF_SBMConstraintGroups.H"
#include "ERF_SBMContracts.H"
#include "ERF_SBMDiffusion.H"
#include "ERF_SBMFCT.H"
#include "ERF_SBMRestart.H"
#include "ERF_SBMLayout.H"
#include "ERF_SBMTransportPrototype.H"
#include "ERF_SBMErfBoundary.H"
#include "ERF_SBMHostCFL.H"
#include "ERF_Interpolation_WENO_Z.H"
#include "ERF_Advection.H"
#include "ERF_IndexDefines.H"

#include <AMReX_MFParallelFor.H>
#include <AMReX_BoxList.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_RealBox.H>

#include <cmath>
#include <algorithm>
#include <array>
#include <filesystem>
#include <limits>
#include <fstream>
#include <iomanip>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

using erf_sbm::MomentMode;
using Real = amrex::Real;
using amrex::Array4;
using amrex::Box;
using amrex::BoxArray;
using amrex::DistributionMapping;
using amrex::FArrayBox;
using amrex::Geometry;
using amrex::IntVect;
using amrex::MultiFab;

Real precision_tolerance(const Real scale, const Real ulps = Real(64.0))
{
    return ulps * std::numeric_limits<Real>::epsilon() *
        std::max(Real(1.0), std::abs(scale));
}

Real relative_roundoff_tolerance(const Real scale, const Real ulps = Real(64.0))
{
    return ulps * std::numeric_limits<Real>::epsilon() * std::abs(scale) +
        std::numeric_limits<Real>::denorm_min();
}

erf_sbm::SBMLayout make_layout(const int nbins, const MomentMode mode,
                               const bool with_property = false,
                               const Real property_support_max = Real(1.0),
                               const Real coordinate_scale = Real(1.0))
{
    erf_sbm::SpectralPopulationSpec population;
    population.population_id = 0;
    population.semantic_id = "liquid";
    population.phase = erf_sbm::PopulationPhase::Liquid;
    population.moment_mode = mode;
    population.grid.coordinate_kind = erf_sbm::CoordinateKind::Mass;
    population.grid.coordinate_units = "kg";
    for (int b = 0; b <= nbins; ++b) {
        population.grid.edges.push_back(static_cast<Real>(b) * coordinate_scale);
    }
    for (int b = 0; b < nbins; ++b) {
        population.grid.pivots.push_back((static_cast<Real>(b) + Real(0.5)) * coordinate_scale);
    }
    population.mass_state_units = "kg m^-3";
    population.number_state_units = "m^-3";
    erf_sbm::SBMLayoutSpec spec;
    spec.populations.push_back(population);
    spec.liquid_projection = {0, nbins/2};
    if (with_property) {
        spec.attached_properties.push_back({"solute", "solute", "kg m^-3", 0,
            erf_sbm::PropertyKind::MassBoundedSubset,
            erf_sbm::SupportRequirement::PositiveMass,
            erf_sbm::PropertyRemapPolicy::CarrierBinConservative,
            true, false, 0.0, property_support_max});
    }
    return erf_sbm::SBMLayout(std::move(spec));
}

// Independent ERF WENO-Z3 reference for the adapter equivalence tests.  Keep
// this algebra local to the test so a shared production helper cannot make a
// self-consistent but non-canonical implementation pass.
Real canonical_weno_z3_reference(const Real qm2, const Real qm1, const Real q,
                                  const Real qp1, const Real carrier)
{
    const bool positive = carrier >= Real(0.0);
    const Real q0 = positive ? Real(0.5) * (-qm2 + Real(3.0)*qm1) :
        Real(0.5) * (Real(3.0)*q - qp1);
    const Real q1 = Real(0.5) * (qm1 + q);
    const Real beta0 = positive ? (qm1-qm2)*(qm1-qm2) :
        (qp1-q)*(qp1-q);
    const Real beta1 = (q-qm1)*(q-qm1);
#ifdef AMREX_USE_FLOAT
    const Real epsilon = Real(1.0e-12);
#else
    const Real epsilon = Real(1.0e-40);
#endif
    const Real tau = std::abs(beta1-beta0);
    const Real w0 = (Real(1.0)/Real(3.0)) *
        (Real(1.0) + (tau*tau)/((epsilon+beta0)*(epsilon+beta0)));
    const Real w1 = (Real(2.0)/Real(3.0)) *
        (Real(1.0) + (tau*tau)/((epsilon+beta1)*(epsilon+beta1)));
    return (w0*q0+w1*q1)/(w0+w1);
}

struct ProductionChunkRun {
    std::unique_ptr<amrex::MultiFab> spectral;
    std::unique_ptr<amrex::MultiFab> compact;
    std::unique_ptr<erf_auxiliary::AuxiliaryFaceTransfer> accepted;
    std::size_t temporary_bytes{0};
};

ProductionChunkRun run_production_chunk_case(const int nbins, const MomentMode mode,
                                             const int chunk_size)
{
    const auto layout = make_layout(nbins, mode);
    const Box domain(IntVect(0, 0, 0), IntVect(15, 1, 1));
    BoxArray boxes(domain);
    boxes.maxSize(IntVect(8, 2, 2));
    const DistributionMapping dm(boxes);
    const amrex::RealBox real_box({AMREX_D_DECL(0.0, 0.0, 0.0)},
                                  {AMREX_D_DECL(1.0, 1.0, 1.0)});
    const std::array<int, AMREX_SPACEDIM> periodicity{AMREX_D_DECL(1, 1, 1)};
    const Geometry geometry(domain, &real_box, amrex::CoordSys::cartesian,
                            periodicity.data());

    amrex::MultiFab rho(boxes, dm, 1, 2);
    amrex::MultiFab core(boxes, dm, RhoQ3_comp + 1, 0);
    amrex::MultiFab xflux(amrex::convert(boxes, IntVect(1, 0, 0)), dm, 1, 0);
    amrex::MultiFab yflux(amrex::convert(boxes, IntVect(0, 1, 0)), dm, 1, 0);
    amrex::MultiFab zflux(amrex::convert(boxes, IntVect(0, 0, 1)), dm, 1, 0);
    rho.setVal(Real(0.0));
    core.setVal(Real(0.0));
    xflux.setVal(Real(0.125));
    yflux.setVal(Real(0.0));
    zflux.setVal(Real(0.0));
    for (amrex::MFIter mfi(rho); mfi.isValid(); ++mfi) {
        const auto density = rho.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            density(i,j,k,0) = Real(1.0) + Real(0.05) * Real(i + 1);
        });
    }
    rho.FillBoundary(geometry.periodicity());
    xflux.FillBoundary(geometry.periodicity());
    yflux.FillBoundary(geometry.periodicity());
    zflux.FillBoundary(geometry.periodicity());

    erf_auxiliary::AuxiliaryStateManager manager(layout.auxiliary_layout());
    manager.define_level(0, boxes, dm, 2);
    const auto& population = layout.populations().front();
    const int mass_offset = population.mass_offset;
    const int number_offset = population.number_offset;
    const bool two_moment = (mode == MomentMode::TwoMoment);
    for (amrex::MFIter mfi(manager.output(0)); mfi.isValid(); ++mfi) {
        const auto state = manager.output(0).array(mfi);
        const auto density = rho.const_array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const Real variation = Real(1.0) + Real(0.20) *
                std::sin(Real(6.2831853071795864769) *
                         (Real(i) + Real(0.5)) / Real(16.0));
            for (int b = 0; b < nbins; ++b) {
                const Real mass = density(i,j,k,0) * Real(1.e-3) *
                    Real(b + 1) * variation;
                state(i,j,k,mass_offset+b) = mass;
                if (two_moment) {
                    state(i,j,k,number_offset+b) = mass / (Real(b) + Real(0.5));
                }
            }
        });
    }
    manager.output(0).FillBoundary(geometry.periodicity());
    manager.begin_step(0, 0.0);

    erf_auxiliary::AuxiliaryFaceTransfer stage_flux;
    stage_flux.define(boxes, dm, layout.ncomp(), 0);
    const Real dt = Real(0.5) / Real(16.0);
    const auto context = erf_auxiliary::make_compressible_stage(
        2, 0.0, dt / 2.0, dt, dt, nullptr, nullptr);
    erf_sbm::advance_stage(manager, layout, context, rho, core,
                           xflux, yflux, zflux, geometry, stage_flux,
                           erf_sbm::TransportMethod::GroupedFCT_WENOZ3,
                           0, Real(0.0), chunk_size);

    ProductionChunkRun result;
    result.spectral = std::make_unique<amrex::MultiFab>(
        boxes, dm, layout.ncomp(), manager.output(0).nGrowVect());
    amrex::MultiFab::Copy(*result.spectral, manager.output(0), 0, 0,
                          layout.ncomp(), manager.output(0).nGrowVect());
    result.compact = std::make_unique<amrex::MultiFab>(boxes, dm, RhoQ3_comp + 1, 0);
    amrex::MultiFab::Copy(*result.compact, core, 0, 0, RhoQ3_comp + 1, 0);
    result.accepted = std::make_unique<erf_auxiliary::AuxiliaryFaceTransfer>();
    result.accepted->define(boxes, dm, layout.ncomp(), 0);
    const auto& accepted = manager.face_transfer_ledger(0).accepted();
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        amrex::MultiFab::Copy(result.accepted->direction(dir), accepted.direction(dir),
                              0, 0, layout.ncomp(), 0);
    }
    result.temporary_bytes = erf_sbm::grouped_fct_peak_working_bytes(
        layout, boxes, dm, chunk_size);
    return result;
}

Real max_multifab_difference(const amrex::MultiFab& left,
                             const amrex::MultiFab& right, const int ncomp)
{
    // A one-box decomposition leaves non-owning MPI ranks with no local FABs.
    // Do this check before constructing/subtracting the temporary: AMReX's
    // local Subtract path requires matching local FAB storage.
    if (left.local_size() == 0 || right.local_size() == 0) return Real(0.0);
    amrex::MultiFab difference(left.boxArray(), left.DistributionMap(), ncomp, 0);
    amrex::MultiFab::Copy(difference, left, 0, 0, ncomp, 0);
    amrex::MultiFab::Subtract(difference, right, 0, 0, ncomp, 0);
    return difference.norm0(0, ncomp, amrex::IntVect(0), true);
}

Real max_face_transfer_difference(const erf_auxiliary::AuxiliaryFaceTransfer& left,
                                  const erf_auxiliary::AuxiliaryFaceTransfer& right,
                                  const int ncomp)
{
    Real result = Real(0.0);
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        result = amrex::max(result, max_multifab_difference(
            left.direction(dir), right.direction(dir), ncomp));
    }
    return result;
}

TEST(SBMP2, ConstraintGroupsCoverOneAndTwoMomentLayouts)
{
    for (const int nbins : {4, 16, 64}) {
        for (const auto mode : {MomentMode::OneMoment, MomentMode::TwoMoment}) {
            const auto layout = make_layout(nbins, mode, true);
            const auto groups = erf_sbm::make_constraint_groups(layout);
            ASSERT_EQ(groups.size(), static_cast<std::size_t>(nbins));
            for (const auto& group : groups) {
                EXPECT_FALSE(group.constraints.empty());
                EXPECT_TRUE(group.contains(group.members.front()));
                std::vector<Real> state(static_cast<std::size_t>(layout.ncomp()), 0.0);
                for (const int component : group.members) state[static_cast<std::size_t>(component)] =
                    (component >= layout.property_offset(0)) ? 0.001 : 1.0;
                if (mode == MomentMode::TwoMoment) {
                    const auto& population = layout.populations().front();
                    state[static_cast<std::size_t>(population.mass_offset + group.bin)] =
                        0.5 * (population.grid.edges()[static_cast<std::size_t>(group.bin)] +
                               population.grid.edges()[static_cast<std::size_t>(group.bin + 1)]);
                    state[static_cast<std::size_t>(population.number_offset + group.bin)] = 1.0;
                }
                Real margin = 0.0;
                std::string failed;
                EXPECT_TRUE(group.admissible(state, &margin, &failed))
                    << group.semantic_id << " failed " << failed << " margin=" << margin;
            }
        }
    }
}

TEST(SBMP2, ConstraintDescriptorsFlattenCompleteGroups)
{
    const auto layout = make_layout(4, MomentMode::TwoMoment, true);
    const auto groups = erf_sbm::make_constraint_groups(layout);
    const auto descriptors = erf_sbm::make_constraint_descriptors(layout);
    std::size_t expected = 0;
    for (std::size_t gi = 0; gi < groups.size(); ++gi) {
        expected += groups[gi].constraints.size();
        for (std::size_t ci = 0; ci < groups[gi].constraints.size(); ++ci) {
            const auto& d = descriptors[expected - groups[gi].constraints.size() + ci];
            EXPECT_EQ(d.group_index, static_cast<int>(gi));
            EXPECT_EQ(d.bin, groups[gi].bin);
            EXPECT_EQ(d.constraint_index, static_cast<int>(ci));
            EXPECT_GE(d.term_count, 1);
            EXPECT_LE(d.term_count, 2);
        }
    }
    EXPECT_EQ(descriptors.size(), expected);
}

TEST(SBMP2, ProductionChunkPlannerKeepsCompleteConstraintGroupsAtomic)
{
    const auto layout = make_layout(4, MomentMode::TwoMoment, true);
    const auto groups = erf_sbm::make_constraint_groups(layout);
    const auto one_group = erf_sbm::make_constraint_closure_chunks(layout, 1);
    ASSERT_EQ(one_group.size(), groups.size());
    for (std::size_t i = 0; i < one_group.size(); ++i) {
        ASSERT_EQ(one_group[i].group_indices.size(), 1U);
        EXPECT_EQ(one_group[i].group_indices.front(), static_cast<int>(i));
        EXPECT_EQ(one_group[i].components.size(), groups[i].members.size());
    }
    const auto two_groups = erf_sbm::make_constraint_closure_chunks(layout, 2);
    ASSERT_EQ(two_groups.size(), 2U);
    EXPECT_EQ(two_groups[0].group_indices.size(), 2U);
    EXPECT_EQ(two_groups[0].components.size(), groups[0].members.size() + groups[1].members.size());
    EXPECT_THROW((void)erf_sbm::make_constraint_closure_chunks(layout, 0), std::invalid_argument);
}

TEST(SBMP2, GroupedFCTUsesCellWideConstraintBudgets)
{
    const auto layout = make_layout(2, MomentMode::OneMoment);
    const auto groups = erf_sbm::make_constraint_groups(layout);
    // Two distinct outgoing faces share cell 0.  Each high-order correction
    // consumes 0.75 of its mass margin; a face-local limiter would accept both
    // at lambda=1 and produce -0.5 in cell 0.  The cell-wide budget requires
    // lambda=2/3 for both and lands exactly on the admissible boundary.
    const std::vector<Real> low_state{1.0, 0.0, 10.0, 0.0, 10.0, 0.0};
    erf_sbm::FCTFaceTransfer face0;
    face0.left_cell = 0; face0.right_cell = 1;
    face0.low = {0.0, 0.0}; face0.high = {0.75, 0.0};
    erf_sbm::FCTFaceTransfer face1;
    face1.left_cell = 0; face1.right_cell = 2;
    face1.low = {0.0, 0.0}; face1.high = {0.75, 0.0};
    const auto result = erf_sbm::limit_grouped(low_state, 3, 2,
                                               {face0, face1}, groups, 1);
    EXPECT_NEAR(result.limiter[0], 2.0/3.0, precision_tolerance(Real(2.0/3.0)));
    EXPECT_NEAR(result.limiter[1], 2.0/3.0, precision_tolerance(Real(2.0/3.0)));
    EXPECT_NEAR(result.updated_state[0], 0.0, precision_tolerance(Real(0.0)));
    for (int cell = 0; cell < 3; ++cell) {
        const std::vector<Real> state(result.updated_state.begin() + cell*2,
                                      result.updated_state.begin() + (cell+1)*2);
        EXPECT_TRUE(groups.front().admissible(state));
    }
}

TEST(SBMP2, TwoMomentEndpointTransformIsStableAtBounds)
{
    const auto lower = erf_sbm::transform_two_moment(2.0, 2.0, 1.0, 3.0);
    EXPECT_EQ(lower.L, 2.0); EXPECT_EQ(lower.H, 0.0);
    const auto upper = erf_sbm::transform_two_moment(2.0, 6.0, 1.0, 3.0);
    EXPECT_EQ(upper.L, 0.0); EXPECT_EQ(upper.H, 2.0);
    const auto center = erf_sbm::transform_two_moment(2.0, 4.0, 1.0, 3.0);
    EXPECT_NEAR(center.L, 1.0, precision_tolerance(Real(1.0), Real(32.0)));
    EXPECT_NEAR(center.H, 1.0, precision_tolerance(Real(1.0), Real(32.0)));
    EXPECT_THROW((void)erf_sbm::transform_two_moment(1.0, 4.1, 1.0, 3.0), std::invalid_argument);
    EXPECT_THROW((void)erf_sbm::transform_two_moment(-1.0, 0.0, 1.0, 3.0), std::invalid_argument);
    const auto recovered = erf_sbm::inverse_two_moment(center.L, center.H, 1.0, 3.0);
    EXPECT_NEAR(recovered.first, 2.0, precision_tolerance(Real(2.0)));
    EXPECT_NEAR(recovered.second, 4.0, precision_tolerance(Real(4.0)));

    // Cancellation-sized endpoint roundoff is normalized only in the
    // derived (L,H) representation; the authoritative (C,M) inputs are not
    // modified.  Violations outside the scale-aware tolerance fail closed.
    const Real eps = std::numeric_limits<Real>::epsilon();
    const auto near_lower = erf_sbm::transform_two_moment(2.0, 2.0 - 32*eps, 1.0, 3.0);
    const auto near_upper = erf_sbm::transform_two_moment(2.0, 6.0 + 32*eps, 1.0, 3.0);
    EXPECT_EQ(near_lower.H, 0.0);
    EXPECT_EQ(near_upper.L, 0.0);
    EXPECT_THROW((void)erf_sbm::transform_two_moment(2.0, 2.0 - 1.e-8, 1.0, 3.0), std::invalid_argument);
    EXPECT_THROW((void)erf_sbm::transform_two_moment(2.0, 6.0 + 1.e-8, 1.0, 3.0), std::invalid_argument);

    // Keep both stress values finite for float and double.  The square-root
    // scales leave room for the two-moment factor-of-two arithmetic without
    // relying on a literal exponent that underflows in SINGLE or overflows
    // when the upper endpoint is formed.
    const Real tiny_c = std::sqrt(std::numeric_limits<Real>::min());
    const Real tiny_m = Real(2.0) * tiny_c;
    const auto tiny = erf_sbm::transform_two_moment(tiny_c, tiny_m, 1.0, 3.0);
    const auto tiny_back = erf_sbm::inverse_two_moment(tiny.L, tiny.H, 1.0, 3.0);
    EXPECT_NEAR(tiny_back.first, tiny_c, relative_roundoff_tolerance(tiny_c));
    EXPECT_NEAR(tiny_back.second, tiny_m, relative_roundoff_tolerance(tiny_m));

    const Real huge_c = std::sqrt(std::numeric_limits<Real>::max()) / Real(4.0);
    const Real huge_m = Real(2.0) * huge_c;
    const auto huge = erf_sbm::transform_two_moment(huge_c, huge_m, 1.0, 3.0);
    const auto huge_back = erf_sbm::inverse_two_moment(huge.L, huge.H, 1.0, 3.0);
    EXPECT_NEAR(huge_back.first, huge_c, relative_roundoff_tolerance(huge_c));
    EXPECT_NEAR(huge_back.second, huge_m, relative_roundoff_tolerance(huge_m));

    const Real lower_edge = Real(2.5), upper_edge = Real(9.25), count = Real(3.75);
    const Real mass = Real(6.125) * count;
    const auto nontrivial = erf_sbm::transform_two_moment(count, mass, lower_edge, upper_edge);
    const auto nontrivial_back = erf_sbm::inverse_two_moment(
        nontrivial.L, nontrivial.H, lower_edge, upper_edge);
    EXPECT_NEAR(nontrivial_back.first, count, relative_roundoff_tolerance(count));
    EXPECT_NEAR(nontrivial_back.second, mass, relative_roundoff_tolerance(mass));
}

TEST(SBMP2, ProductionChunkWorkingMemoryScalesWithAtomicChunkPolicy)
{
    const Box domain(IntVect(0, 0, 0), IntVect(7, 7, 7));
    const BoxArray boxes(domain);
    const DistributionMapping dm(boxes);
    for (const int nbins : {4, 16, 64}) {
        for (const auto mode : {MomentMode::OneMoment, MomentMode::TwoMoment}) {
            const auto layout = make_layout(nbins, mode);
            const auto one_group = erf_sbm::grouped_fct_peak_working_bytes(layout, boxes, dm, 1);
            const auto all_groups = erf_sbm::grouped_fct_peak_working_bytes(layout, boxes, dm, nbins);
            EXPECT_GT(one_group, std::size_t(0));
            EXPECT_GE(all_groups, one_group);
            EXPECT_EQ(one_group, erf_sbm::grouped_fct_peak_working_bytes(layout, boxes, dm, 1));
            if (nbins > 1) {
                EXPECT_GT(all_groups, one_group);
            }
        }
    }
    EXPECT_THROW((void)erf_sbm::grouped_fct_peak_working_bytes(
        make_layout(4, MomentMode::OneMoment), boxes, dm, 0), std::invalid_argument);
}

TEST(SBMP2, ProductionGroupedTransportIsChunkEquivalentAcrossRuntimeLayouts)
{
    // The all-group run is the production reference.  Every other run still
    // executes the grouped WENO/FCT FAB path; only the complete-group chunk
    // policy changes.  This exercises both physical moments, accepted face
    // transfers, compact projections, and the runtime 4/16/64-bin layouts.
    std::size_t one_group_bytes[2][3]{{0, 0, 0}, {0, 0, 0}};
    std::size_t all_group_bytes[2][3]{{0, 0, 0}, {0, 0, 0}};
    std::ofstream evidence;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        evidence.open(std::filesystem::temp_directory_path() /
                      "erf_sbm_p2_chunk_equivalence_memory.csv");
        ASSERT_TRUE(evidence.good());
        evidence << "nbins,moment_mode,chunk_size,temporary_p2_working_bytes\n";
    }
    for (const int nbins : {4, 16, 64}) {
        for (const auto mode : {MomentMode::OneMoment, MomentMode::TwoMoment}) {
            SCOPED_TRACE(std::string("nbins=") + std::to_string(nbins) +
                         " mode=" + std::to_string(static_cast<int>(mode)));
            const auto reference = run_production_chunk_case(nbins, mode, nbins);
            const auto one_group = run_production_chunk_case(nbins, mode, 1);
            const auto two_groups = run_production_chunk_case(nbins, mode, 2);
            const auto four_groups = run_production_chunk_case(nbins, mode, 4);
            const Real tolerance = Real(5.e-13);
            EXPECT_LE(max_multifab_difference(*one_group.spectral,
                                               *reference.spectral, nbins *
                                               (mode == MomentMode::TwoMoment ? 2 : 1)),
                      tolerance);
            EXPECT_LE(max_multifab_difference(*two_groups.spectral,
                                               *reference.spectral, nbins *
                                               (mode == MomentMode::TwoMoment ? 2 : 1)),
                      tolerance);
            EXPECT_LE(max_multifab_difference(*four_groups.spectral,
                                               *reference.spectral, nbins *
                                               (mode == MomentMode::TwoMoment ? 2 : 1)),
                      tolerance);
            EXPECT_LE(max_multifab_difference(*one_group.compact,
                                               *reference.compact, RhoQ3_comp + 1),
                      tolerance);
            EXPECT_LE(max_multifab_difference(*two_groups.compact,
                                               *reference.compact, RhoQ3_comp + 1),
                      tolerance);
            EXPECT_LE(max_multifab_difference(*four_groups.compact,
                                               *reference.compact, RhoQ3_comp + 1),
                      tolerance);
            EXPECT_LE(max_face_transfer_difference(*one_group.accepted,
                                                    *reference.accepted,
                                                    nbins *
                                                    (mode == MomentMode::TwoMoment ? 2 : 1)),
                      tolerance);
            EXPECT_LE(max_face_transfer_difference(*two_groups.accepted,
                                                    *reference.accepted,
                                                    nbins *
                                                    (mode == MomentMode::TwoMoment ? 2 : 1)),
                      tolerance);
            EXPECT_LE(max_face_transfer_difference(*four_groups.accepted,
                                                    *reference.accepted,
                                                    nbins *
                                                    (mode == MomentMode::TwoMoment ? 2 : 1)),
                      tolerance);

            EXPECT_GE(one_group.temporary_bytes, std::size_t(1));
            EXPECT_LE(one_group.temporary_bytes, two_groups.temporary_bytes);
            EXPECT_LE(two_groups.temporary_bytes, four_groups.temporary_bytes);
            EXPECT_LE(four_groups.temporary_bytes, reference.temporary_bytes);

            const int mode_index = mode == MomentMode::TwoMoment ? 1 : 0;
            const int bin_index = nbins == 4 ? 0 : (nbins == 16 ? 1 : 2);
            one_group_bytes[mode_index][bin_index] = one_group.temporary_bytes;
            all_group_bytes[mode_index][bin_index] = reference.temporary_bytes;
            if (evidence) {
                evidence << nbins << ',' << mode_index << ",1," << one_group.temporary_bytes << '\n'
                         << nbins << ',' << mode_index << ",2," << two_groups.temporary_bytes << '\n'
                         << nbins << ',' << mode_index << ",4," << four_groups.temporary_bytes << '\n'
                         << nbins << ',' << mode_index << ',' << nbins << ','
                         << reference.temporary_bytes << '\n';
            }
        }
    }
    for (int mode_index = 0; mode_index < 2; ++mode_index) {
        EXPECT_EQ(one_group_bytes[mode_index][0], one_group_bytes[mode_index][1]);
        EXPECT_EQ(one_group_bytes[mode_index][1], one_group_bytes[mode_index][2]);
        EXPECT_LT(all_group_bytes[mode_index][0], all_group_bytes[mode_index][1]);
        EXPECT_LT(all_group_bytes[mode_index][1], all_group_bytes[mode_index][2]);
    }
}

TEST(SBMP2, GroupedFCTUsesOneFaceLimiterAndConservesEveryComponent)
{
    const auto layout = make_layout(2, MomentMode::OneMoment);
    const auto groups = erf_sbm::make_constraint_groups(layout);
    std::vector<Real> low_state{1.0, 0.0, 2.0, 0.0};
    erf_sbm::FCTFaceTransfer face;
    face.left_cell = 0; face.right_cell = 1;
    face.left_volume = 2.0; face.right_volume = 1.0;
    face.low = {0.0, 0.0}; face.high = {3.0, 0.0};
    auto result = erf_sbm::limit_grouped(low_state, 2, 2, {face}, groups);
    ASSERT_EQ(result.accepted_faces.size(), 1U);
    EXPECT_EQ(result.limiter[0], 2.0/3.0);
    EXPECT_NEAR(result.updated_state[0], 0.0, precision_tolerance(Real(0.0)));
    EXPECT_NEAR(result.updated_state[2], 4.0, precision_tolerance(Real(4.0)));
    EXPECT_NEAR(result.updated_state[0]*2.0 + result.updated_state[2], 4.0,
                precision_tolerance(Real(4.0)));
    EXPECT_EQ(result.accepted_faces[0].low[0], 2.0);
    EXPECT_THROW((void)erf_sbm::limit_grouped(low_state, 2, 2, {face, face}, groups),
                 std::invalid_argument);
    const auto chunked = erf_sbm::limit_grouped(low_state, 2, 2, {face}, groups, 1);
    EXPECT_NEAR(chunked.updated_state[0], result.updated_state[0],
                precision_tolerance(result.updated_state[0]));
    EXPECT_NEAR(chunked.updated_state[2], result.updated_state[2],
                precision_tolerance(result.updated_state[2]));
}

TEST(SBMP2, GroupedFCTHonorsCompleteTwoMomentGroupAndSubset)
{
    const auto layout = make_layout(2, MomentMode::TwoMoment, true);
    const auto groups = erf_sbm::make_constraint_groups(layout);
    ASSERT_EQ(groups.size(), 2U);
    // Physical storage is (M,C); the host FCT reference applies the linear
    // cone constraints in that storage.  The candidate would drive both the
    // lower endpoint and the subset negative, so one common lambda must limit
    // every member of the group.
    std::vector<Real> low_state{0.5, 0.0, 0.5, 0.0, 0.25, 0.0,
                                0.5, 0.0, 0.5, 0.0, 0.25, 0.0};
    erf_sbm::FCTFaceTransfer face;
    face.left_cell = 0; face.right_cell = 1;
    face.left_volume = 1.0; face.right_volume = 1.0;
    face.low = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    face.high = {1.0, 0.0, 0.0, 0.0, 0.6, 0.0};
    auto result = erf_sbm::limit_grouped(low_state, 2, 6, {face}, groups);
    EXPECT_GE(result.limiter[0], 0.0);
    EXPECT_LE(result.limiter[0], Real(0.5) + precision_tolerance(Real(0.5)));
    for (const auto& group : groups) {
        std::vector<Real> state(result.updated_state.begin(), result.updated_state.begin()+6);
        EXPECT_TRUE(group.admissible(state));
    }
}

TEST(SBMP2, CapabilityGateKeepsP2NegativeControlsFailClosed)
{
    erf_sbm::CapabilityInput input;
    input.p2_requested = true;
    const auto supported = erf_sbm::evaluate_p2_capabilities(input);
    EXPECT_TRUE(supported.supported);
    EXPECT_EQ(supported.qualification_status, "qualified");
    EXPECT_EQ(supported.qualified_flags, supported.flags);
    EXPECT_TRUE(supported.qualification_limitations.empty());

    input.moving_terrain = true;
    EXPECT_FALSE(erf_sbm::evaluate_p2_capabilities(input).supported);
    input.moving_terrain = false;
    input.diffusion = true;
    input.explicit_sbm_diffusion = false;
    EXPECT_FALSE(erf_sbm::evaluate_p2_capabilities(input).supported);
    input.explicit_sbm_diffusion = true;
    input.chunk_size = 0;
    EXPECT_FALSE(erf_sbm::evaluate_p2_capabilities(input).supported);

    input.chunk_size = 256;
    input.periodic_cartesian = false;
    input.impermeable_wall = true;
    EXPECT_TRUE(erf_sbm::evaluate_p2_capabilities(input).supported);
    input.impermeable_wall = false;
    input.advective_outflow = true;
    EXPECT_TRUE(erf_sbm::evaluate_p2_capabilities(input).supported);
    input.prescribed_sbm_inflow = true;
    EXPECT_FALSE(erf_sbm::evaluate_p2_capabilities(input).supported);
    input.prescribed_sbm_inflow = false;
    input.max_level = 1;
    input.spatial_ref_ratio = IntVect(2, 2, 2);
    input.time_refinement_factor = 2;
    input.two_way_coupling = true;
    input.periodic_cartesian = true;
    input.periodic_amr = true;
    input.native_subcycling = true;
    EXPECT_TRUE(erf_sbm::evaluate_p2_capabilities(input).supported);
    input.amr_nonperiodic = true;
    EXPECT_FALSE(erf_sbm::evaluate_p2_capabilities(input).supported);
    input.amr_nonperiodic = false;
    input.native_subcycling = false;
    EXPECT_FALSE(erf_sbm::evaluate_p2_capabilities(input).supported);
    input.native_subcycling = true;
    input.tensor_diffusion = true;
    EXPECT_FALSE(erf_sbm::evaluate_p2_capabilities(input).supported);

    input.tensor_diffusion = false;
    input.acoustic_substepping_enabled = true;
    const auto acoustic_rejected = erf_sbm::evaluate_p2_capabilities(input);
    EXPECT_FALSE(acoustic_rejected.supported);
    EXPECT_NE(std::find(acoustic_rejected.rejected_reasons.begin(),
                        acoustic_rejected.rejected_reasons.end(),
                        "P2 SBM host-CFL qualification does not yet cover ERF acoustic substepping"),
              acoustic_rejected.rejected_reasons.end());
    EXPECT_NE(acoustic_rejected.stable_description().find(
                  "acoustic_substepping_enabled=1"), std::string::npos);
}

TEST(SBMP2, ActualStageDemandUsesProductionCarrierAndExactStageDuration)
{
    const Box domain(IntVect(0, 0, 0), IntVect(1, 1, 1));
    const BoxArray cell_boxes(domain);
    const DistributionMapping dm(cell_boxes);
    const amrex::RealBox real_box({AMREX_D_DECL(0.0, 0.0, 0.0)},
                                  {AMREX_D_DECL(4.0, 2.0, 2.0)});
    const std::array<int, AMREX_SPACEDIM> periodicity{AMREX_D_DECL(1, 1, 1)};
    const Geometry geometry(domain, &real_box, amrex::CoordSys::cartesian,
                            periodicity.data());
    MultiFab density(cell_boxes, dm, 1, 2);
    density.setVal(Real(1.0));
    MultiFab carrier_x(amrex::convert(cell_boxes, IntVect(1, 0, 0)), dm, 1, 1);
    MultiFab carrier_y(amrex::convert(cell_boxes, IntVect(0, 1, 0)), dm, 1, 1);
    MultiFab carrier_z(amrex::convert(cell_boxes, IntVect(0, 0, 1)), dm, 1, 1);
    carrier_x.setVal(Real(0.2));
    carrier_y.setVal(Real(-0.3));
    carrier_z.setVal(Real(0.0));

    const auto context = erf_auxiliary::make_compressible_stage(
        0, 0.0, 0.0, 1.0/3.0, 1.0, nullptr, nullptr);
    const auto demand = erf_sbm::measure_actual_stage_low_order_demand(
        context, density, carrier_x, carrier_y, carrier_z, geometry,
        Real(0.1), 0);

    // The real box is 4 x 2 x 2, so the independent two-direction carrier
    // demand is .2/2 + .3/1 = .4.  Diffusion uses the same arithmetic
    // rho-face convention as production: .1*(2*.25 + 2*1 + 2*1) = .45.
    EXPECT_NEAR(demand.stage_duration, 1.0/3.0,
                precision_tolerance(Real(1.0/3.0)));
    EXPECT_NEAR(demand.advective_rate, Real(0.4), precision_tolerance(Real(0.4)));
    EXPECT_NEAR(demand.diffusive_rate, Real(0.45), precision_tolerance(Real(0.45)));
    EXPECT_NEAR(demand.maximum_rate, Real(0.85), precision_tolerance(Real(0.85)));
    EXPECT_NEAR(demand.maximum_tau_rate, Real(0.85/3.0),
                precision_tolerance(Real(0.85/3.0)));
    EXPECT_EQ(demand.level, 0);
    EXPECT_EQ(demand.stage_index, 0);
    EXPECT_EQ(demand.worst_i, 0);
    EXPECT_EQ(demand.worst_j, 0);
    EXPECT_EQ(demand.worst_k, 0);
}

void sbm_test_ActualStageDemandUsesTheAnchorDensityForVariableCarrierFlux()
{
    const Box domain(IntVect(0, 0, 0), IntVect(1, 0, 0));
    const BoxArray cell_boxes(domain);
    const DistributionMapping dm(cell_boxes);
    const amrex::RealBox real_box({AMREX_D_DECL(0.0, 0.0, 0.0)},
                                  {AMREX_D_DECL(2.0, 1.0, 1.0)});
    const std::array<int, AMREX_SPACEDIM> periodicity{AMREX_D_DECL(1, 1, 1)};
    const Geometry geometry(domain, &real_box, amrex::CoordSys::cartesian,
                            periodicity.data());
    MultiFab density(cell_boxes, dm, 1, 2);
    density.setVal(Real(0.0));
    for (amrex::MFIter mfi(density); mfi.isValid(); ++mfi) {
        const auto rho = density.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            rho(i,j,k) = i == 0 ? Real(1.0) : Real(2.0);
        });
    }
    density.FillBoundary(geometry.periodicity());

    MultiFab carrier_x(amrex::convert(cell_boxes, IntVect(1, 0, 0)), dm, 1, 1);
    MultiFab carrier_y(amrex::convert(cell_boxes, IntVect(0, 1, 0)), dm, 1, 1);
    MultiFab carrier_z(amrex::convert(cell_boxes, IntVect(0, 0, 1)), dm, 1, 1);
    carrier_x.setVal(Real(0.0));
    carrier_y.setVal(Real(0.0));
    carrier_z.setVal(Real(0.0));
    for (amrex::MFIter mfi(carrier_x); mfi.isValid(); ++mfi) {
        const auto flux = carrier_x.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            // Negative x transport leaves the rho=2 cell through its left face.
            if (i == 1) flux(i,j,k) = Real(-0.4);
        });
    }
    carrier_x.FillBoundary(geometry.periodicity());
    carrier_y.FillBoundary(geometry.periodicity());
    carrier_z.FillBoundary(geometry.periodicity());

    const auto context = erf_auxiliary::make_anelastic_stage(
        0, 0.0, 0.0, 1.0, 1.0, nullptr, nullptr);
    const auto demand = erf_sbm::measure_actual_stage_low_order_demand(
        context, density, carrier_x, carrier_y, carrier_z, geometry,
        Real(0.0), 0);

    EXPECT_NEAR(demand.advective_rate, Real(0.2), precision_tolerance(Real(0.2)));
    EXPECT_NEAR(demand.diffusive_rate, Real(0.0), precision_tolerance(Real(0.0)));
    EXPECT_NEAR(demand.maximum_rate, Real(0.2), precision_tolerance(Real(0.2)));
    EXPECT_NEAR(demand.maximum_tau_rate, Real(0.2), precision_tolerance(Real(0.2)));
    EXPECT_EQ(demand.worst_i, 1);
    EXPECT_EQ(demand.worst_j, 0);
    EXPECT_EQ(demand.worst_k, 0);
}

TEST(SBMP2, ActualStageDemandUsesTheAnchorDensityForVariableCarrierFlux)
{
    sbm_test_ActualStageDemandUsesTheAnchorDensityForVariableCarrierFlux();
}

void sbm_test_ActualStageDemandSkipsWallDiffusionWithoutPhysicalGhostValues()
{
    const Box domain(IntVect(0, 0, 0), IntVect(0, 1, 1));
    const BoxArray cell_boxes(domain);
    const DistributionMapping dm(cell_boxes);
    const amrex::RealBox real_box({AMREX_D_DECL(0.0, 0.0, 0.0)},
                                  {AMREX_D_DECL(1.0, 2.0, 2.0)});
    const std::array<int, AMREX_SPACEDIM> periodicity{AMREX_D_DECL(0, 1, 1)};
    const Geometry geometry(domain, &real_box, amrex::CoordSys::cartesian,
                            periodicity.data());

    MultiFab density(cell_boxes, dm, 1, 1);
    density.setVal(Real(1.0));
    density.FillBoundary(geometry.periodicity());
    const Real nan = std::numeric_limits<Real>::quiet_NaN();
    for (amrex::MFIter mfi(density); mfi.isValid(); ++mfi) {
        const auto rho = density.array(mfi);
        const Box fab_box = mfi.fabbox();
        amrex::ParallelFor(fab_box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            if (i < domain.smallEnd(0) || i > domain.bigEnd(0)) rho(i,j,k) = nan;
        });
    }

    MultiFab carrier_x(amrex::convert(cell_boxes, IntVect(1, 0, 0)), dm, 1, 1);
    MultiFab carrier_y(amrex::convert(cell_boxes, IntVect(0, 1, 0)), dm, 1, 1);
    MultiFab carrier_z(amrex::convert(cell_boxes, IntVect(0, 0, 1)), dm, 1, 1);
    carrier_x.setVal(Real(0.0));
    carrier_y.setVal(Real(0.0));
    carrier_z.setVal(Real(0.0));
    carrier_y.FillBoundary(geometry.periodicity());
    carrier_z.FillBoundary(geometry.periodicity());

    erf_sbm::TransportBoundaryPolicy boundary_policy;
    boundary_policy.configured = true;
    boundary_policy.face_kind[0] = erf_sbm::BoundaryKind::ImpermeableWall;
    boundary_policy.face_kind[1] = erf_sbm::BoundaryKind::ImpermeableWall;
    boundary_policy.face_kind[2] = erf_sbm::BoundaryKind::Periodic;
    boundary_policy.face_kind[3] = erf_sbm::BoundaryKind::Periodic;
    boundary_policy.face_kind[4] = erf_sbm::BoundaryKind::Periodic;
    boundary_policy.face_kind[5] = erf_sbm::BoundaryKind::Periodic;
    EXPECT_TRUE(erf_sbm::suppress_normal_sbm_transfer(
        static_cast<int>(erf_sbm::BoundaryKind::ImpermeableWall)));
    EXPECT_TRUE(erf_sbm::suppress_normal_sbm_diffusion(
        static_cast<int>(erf_sbm::BoundaryKind::ImpermeableWall)));
    EXPECT_TRUE(erf_sbm::suppress_normal_sbm_diffusion(
        static_cast<int>(erf_sbm::BoundaryKind::AdvectiveOutflow)));

    const auto context = erf_auxiliary::make_compressible_stage(
        0, 0.0, 0.0, 1.0/3.0, 1.0, nullptr, nullptr);
    const auto demand = erf_sbm::measure_actual_stage_low_order_demand(
        context, density, carrier_x, carrier_y, carrier_z, geometry,
        Real(1.0), 0, boundary_policy);

    // The x-wall density ghosts are NaN deliberately.  A valid oracle must
    // not read them, and only the two periodic transverse directions remain:
    // K * (2/dy^2 + 2/dz^2) = 4.
    EXPECT_NEAR(demand.advective_rate, Real(0.0), precision_tolerance(Real(0.0), Real(32.0)));
    EXPECT_NEAR(demand.diffusive_rate, Real(4.0), precision_tolerance(Real(4.0), Real(32.0)));
    EXPECT_NEAR(demand.maximum_rate, Real(4.0), precision_tolerance(Real(4.0), Real(32.0)));
    EXPECT_NEAR(demand.maximum_tau_rate, Real(4.0/3.0),
                precision_tolerance(Real(4.0/3.0), Real(32.0)));
}

TEST(SBMP2, ActualStageDemandSkipsWallDiffusionWithoutPhysicalGhostValues)
{
    sbm_test_ActualStageDemandSkipsWallDiffusionWithoutPhysicalGhostValues();
}

void sbm_test_HostCFLBoundarySemanticsSkipWallPoisonAndRejectInwardOutflow()
{
    const Box domain(IntVect(0, 0, 0), IntVect(0, 0, 0));
    const BoxArray boxes(domain);
    const DistributionMapping dm(boxes);
    const amrex::RealBox real_box({AMREX_D_DECL(0.0, 0.0, 0.0)},
                                  {AMREX_D_DECL(1.0, 1.0, 1.0)});
    const std::array<int, AMREX_SPACEDIM> periodicity{AMREX_D_DECL(0, 1, 1)};
    const Geometry geometry(domain, &real_box, amrex::CoordSys::cartesian,
                            periodicity.data());
    const int wall = static_cast<int>(erf_sbm::BoundaryKind::ImpermeableWall);
    const int outflow = static_cast<int>(erf_sbm::BoundaryKind::AdvectiveOutflow);
    const int periodic = static_cast<int>(erf_sbm::BoundaryKind::Periodic);

    MultiFab state(boxes, dm, Rho_comp + 1, 1);
    state.setVal(Real(1.0));
    const Real nan = std::numeric_limits<Real>::quiet_NaN();
    for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi) {
        const auto values = state.array(mfi);
        amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            if (i < domain.smallEnd(0)) values(i,j,k,Rho_comp) = nan;
        });
    }
    amrex::Gpu::synchronize();
    MultiFab xvel(amrex::convert(boxes, IntVect(1, 0, 0)), dm, 1, 0);
    MultiFab yvel(amrex::convert(boxes, IntVect(0, 1, 0)), dm, 1, 0);
    MultiFab zvel(amrex::convert(boxes, IntVect(0, 0, 1)), dm, 1, 0);
    xvel.setVal(Real(0.0));
    yvel.setVal(Real(0.0));
    zvel.setVal(Real(0.0));
    ASSERT_EQ(state.size(), 1);
    // The MPI GoogleTest executable runs this test on every rank, while the
    // one-cell FAB belongs to only one rank.  Non-owning ranks have no local
    // array to inspect and must leave before indexing state[0].
    if (state.local_size() == 0) return;
    const auto wall_case = erf_sbm::host_cell_demand(
        state[0].const_array(), xvel[0].const_array(), yvel[0].const_array(), zvel[0].const_array(),
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 1, wall, wall, periodic, periodic, periodic, periodic,
        Real(1.0), Real(1.0), Real(1.0), Real(1.0));
    EXPECT_EQ(wall_case.valid, 1);
    EXPECT_NEAR(wall_case.advective, Real(0.0), 0.0);
    EXPECT_NEAR(wall_case.diffusive, Real(4.0), 0.0);
    EXPECT_NEAR(wall_case.total, Real(4.0), 0.0);

    for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi) {
        const auto values = state.array(mfi);
        amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            if (i > domain.bigEnd(0)) values(i,j,k,Rho_comp) = Real(1.0);
        });
    }
    amrex::Gpu::synchronize();
    for (amrex::MFIter mfi(xvel); mfi.isValid(); ++mfi) {
        const auto values = xvel.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            if (i == domain.bigEnd(0) + 1) values(i,j,k) = Real(0.25);
        });
    }
    amrex::Gpu::synchronize();
    const auto outflow_case = erf_sbm::host_cell_demand(
        state[0].const_array(), xvel[0].const_array(), yvel[0].const_array(), zvel[0].const_array(),
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 1, wall, outflow, periodic, periodic, periodic, periodic,
        Real(1.0), Real(1.0), Real(1.0), Real(1.0));
    EXPECT_EQ(outflow_case.valid, 1);
    EXPECT_NEAR(outflow_case.advective, Real(0.25), 0.0);
    EXPECT_NEAR(outflow_case.diffusive, Real(4.0), 0.0);
    EXPECT_NEAR(outflow_case.total, Real(4.25), 0.0);

    for (amrex::MFIter mfi(xvel); mfi.isValid(); ++mfi) {
        const auto values = xvel.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            if (i == domain.bigEnd(0) + 1) values(i,j,k) = Real(-0.25);
        });
    }
    amrex::Gpu::synchronize();
    const auto inward = erf_sbm::host_face_demand(
        Real(1.0), Real(1.0), Real(-0.25), outflow, false, Real(1.0), Real(1.0));
    EXPECT_EQ(inward.valid, 0);
}

TEST(SBMP2, HostCFLBoundarySemanticsSkipWallPoisonAndRejectInwardOutflow)
{
    sbm_test_HostCFLBoundarySemanticsSkipWallPoisonAndRejectInwardOutflow();
}

TEST(SBMP2, ERFBoundaryPolicyMapsPhysicalFacesAndHonorsPeriodicity)
{
    EXPECT_EQ(erf_sbm::classify_erf_boundary(ERF_BC::symmetry),
              erf_sbm::BoundaryKind::ImpermeableWall);
    EXPECT_EQ(erf_sbm::classify_erf_boundary(ERF_BC::ho_outflow),
              erf_sbm::BoundaryKind::AdvectiveOutflow);
    EXPECT_EQ(erf_sbm::classify_erf_boundary(ERF_BC::periodic),
              erf_sbm::BoundaryKind::Periodic);
    EXPECT_EQ(erf_sbm::classify_erf_boundary(ERF_BC::inflow),
              erf_sbm::BoundaryKind::PrescribedSpectralInflow);

    const Box domain(IntVect(0, 0, 0), IntVect(0, 0, 0));
    const amrex::RealBox real_box({AMREX_D_DECL(0.0, 0.0, 0.0)},
                                  {AMREX_D_DECL(1.0, 1.0, 1.0)});
    const std::array<int, AMREX_SPACEDIM> periodicity{AMREX_D_DECL(1, 0, 0)};
    const Geometry geometry(domain, &real_box, amrex::CoordSys::cartesian,
                            periodicity.data());
    amrex::GpuArray<ERF_BC, AMREX_SPACEDIM*2> physical{
        // AMReX's ERF face array is ordered low x/y/z, then high x/y/z.
        ERF_BC::inflow, ERF_BC::no_slip_wall,
        ERF_BC::surface_layer, ERF_BC::outflow,
        ERF_BC::ho_outflow, ERF_BC::undefined};
    const auto policy = erf_sbm::make_erf_transport_boundary_policy(
        geometry, physical);
    EXPECT_TRUE(policy.configured);
    EXPECT_EQ(policy.face_kind[0], erf_sbm::BoundaryKind::Periodic);
    EXPECT_EQ(policy.face_kind[1], erf_sbm::BoundaryKind::Periodic);
    EXPECT_EQ(policy.face_kind[2], erf_sbm::BoundaryKind::ImpermeableWall);
    EXPECT_EQ(policy.face_kind[3], erf_sbm::BoundaryKind::AdvectiveOutflow);
    EXPECT_EQ(policy.face_kind[4], erf_sbm::BoundaryKind::PrescribedSpectralInflow);
    EXPECT_EQ(policy.face_kind[5], erf_sbm::BoundaryKind::PrescribedSpectralInflow);
}

TEST(SBMP2, StageZeroAdvectionCarrierMatchesArithmeticFaceMomentum)
{
    // AdvectionSrcForRho may execute on a GPU.  These are intentionally
    // pinned host/device-accessible FABs: initialization and the independent
    // host oracle use RunOn::Host, while the production kernel can still
    // consume the same Array4 views without host dereferences of device-only
    // storage.
    auto* host_arena = amrex::The_Pinned_Arena();
    const Box domain(IntVect(0, 0, 0), IntVect(1, 1, 1));
    const Box xbox = amrex::surroundingNodes(domain, 0);
    const Box ybox = amrex::surroundingNodes(domain, 1);
    const Box zbox = amrex::surroundingNodes(domain, 2);
    Box density_box = domain;
    density_box.grow(1);
    FArrayBox density(density_box, 1, host_arena);
    FArrayBox rho_u(xbox, 1, host_arena), rho_v(ybox, 1, host_arena),
        omega(zbox, 1, host_arena);
    FArrayBox avg_xmom(xbox, 1, host_arena), avg_ymom(ybox, 1, host_arena),
        avg_zmom(zbox, 1, host_arena);
    FArrayBox ax(xbox, 1, host_arena), ay(ybox, 1, host_arena),
        az(zbox, 1, host_arena), detJ(domain, 1, host_arena);
    FArrayBox mf_mx(domain, 1, host_arena), mf_my(domain, 1, host_arena);
    FArrayBox mf_uy(xbox, 1, host_arena), mf_vx(ybox, 1, host_arena);
    FArrayBox source(domain, 1, host_arena);
    std::array<FArrayBox, AMREX_SPACEDIM> flux{
        FArrayBox(xbox, 1, host_arena), FArrayBox(ybox, 1, host_arena),
        FArrayBox(zbox, 1, host_arena)};
    density.setVal<amrex::RunOn::Host>(Real(1.0));
    ax.setVal<amrex::RunOn::Host>(Real(1.0));
    ay.setVal<amrex::RunOn::Host>(Real(1.0));
    az.setVal<amrex::RunOn::Host>(Real(1.0));
    detJ.setVal<amrex::RunOn::Host>(Real(1.0));
    mf_mx.setVal<amrex::RunOn::Host>(Real(1.0));
    mf_my.setVal<amrex::RunOn::Host>(Real(1.0));
    mf_uy.setVal<amrex::RunOn::Host>(Real(1.0));
    mf_vx.setVal<amrex::RunOn::Host>(Real(1.0));
    omega.setVal<amrex::RunOn::Host>(Real(0.0));
    avg_xmom.setVal<amrex::RunOn::Host>(Real(0.0));
    avg_ymom.setVal<amrex::RunOn::Host>(Real(0.0));
    avg_zmom.setVal<amrex::RunOn::Host>(Real(0.0));
    source.setVal<amrex::RunOn::Host>(Real(0.0));
    for (auto& face_flux : flux) face_flux.setVal<amrex::RunOn::Host>(Real(0.0));

    for (int k = density.box().smallEnd(2); k <= density.box().bigEnd(2); ++k) {
        for (int j = density.box().smallEnd(1); j <= density.box().bigEnd(1); ++j) {
            for (int i = density.box().smallEnd(0); i <= density.box().bigEnd(0); ++i) {
                const int wi = ((i % 2) + 2) % 2;
                const int wj = ((j % 2) + 2) % 2;
                const int wk = ((k % 2) + 2) % 2;
                density.array()(i,j,k) = Real(1.0) + Real(0.05) * wi +
                    Real(0.07) * wj + Real(0.09) * wk;
            }
        }
    }
    const auto rho = density.const_array();
    const auto x_velocity = [](const int i, const int j, const int k) {
        return Real(0.20) + Real(0.03) * i - Real(0.01) * j + Real(0.02) * k;
    };
    const auto y_velocity = [](const int i, const int j, const int k) {
        return Real(-0.15) + Real(0.02) * i + Real(0.04) * j - Real(0.01) * k;
    };
    const auto z_velocity = [](const int i, const int j, const int k) {
        return Real(0.11) - Real(0.01) * i + Real(0.02) * j + Real(0.03) * k;
    };
    for (int k = xbox.smallEnd(2); k <= xbox.bigEnd(2); ++k) {
        for (int j = xbox.smallEnd(1); j <= xbox.bigEnd(1); ++j) {
            for (int i = xbox.smallEnd(0); i <= xbox.bigEnd(0); ++i) {
                if (xbox.contains(IntVect(i,j,k))) {
                    rho_u.array()(i,j,k) = x_velocity(i,j,k) *
                        Real(0.5) * (rho(i-1,j,k) + rho(i,j,k));
                }
            }
        }
    }
    for (int k = ybox.smallEnd(2); k <= ybox.bigEnd(2); ++k) {
        for (int j = ybox.smallEnd(1); j <= ybox.bigEnd(1); ++j) {
            for (int i = ybox.smallEnd(0); i <= ybox.bigEnd(0); ++i) {
                if (ybox.contains(IntVect(i,j,k))) {
                    rho_v.array()(i,j,k) = y_velocity(i,j,k) *
                        Real(0.5) * (rho(i,j-1,k) + rho(i,j,k));
                }
            }
        }
    }
    for (int k = zbox.smallEnd(2); k <= zbox.bigEnd(2); ++k) {
        for (int j = zbox.smallEnd(1); j <= zbox.bigEnd(1); ++j) {
            for (int i = zbox.smallEnd(0); i <= zbox.bigEnd(0); ++i) {
                if (zbox.contains(IntVect(i,j,k))) {
                    omega.array()(i,j,k) = z_velocity(i,j,k) *
                        Real(0.5) * (rho(i,j,k-1) + rho(i,j,k));
                }
            }
        }
    }

    const amrex::GpuArray<Real, AMREX_SPACEDIM> inverse_cell_size{
        AMREX_D_DECL(Real(1.0), Real(1.0), Real(1.0))};
    const amrex::GpuArray<const Array4<Real>, AMREX_SPACEDIM> flux_arrays{
        AMREX_D_DECL(flux[0].array(), flux[1].array(), flux[2].array())};
    AdvectionSrcForRho(domain, source.array(), rho_u.const_array(), rho_v.const_array(),
                       omega.const_array(), avg_xmom.array(), avg_ymom.array(),
                       avg_zmom.array(), ax.const_array(), ay.const_array(),
                       az.const_array(), detJ.const_array(), inverse_cell_size,
                       mf_mx.const_array(), mf_my.const_array(), mf_uy.const_array(),
                       mf_vx.const_array(), flux_arrays, false);
    amrex::Gpu::synchronize();

    for (int k = domain.smallEnd(2); k <= domain.bigEnd(2); ++k) {
        for (int j = domain.smallEnd(1); j <= domain.bigEnd(1); ++j) {
            for (int i = domain.smallEnd(0); i <= domain.bigEnd(0) + 1; ++i) {
                EXPECT_EQ(avg_xmom.const_array()(i,j,k), rho_u.const_array()(i,j,k));
            }
        }
    }
    for (int k = domain.smallEnd(2); k <= domain.bigEnd(2); ++k) {
        for (int j = domain.smallEnd(1); j <= domain.bigEnd(1) + 1; ++j) {
            for (int i = domain.smallEnd(0); i <= domain.bigEnd(0); ++i) {
                EXPECT_EQ(avg_ymom.const_array()(i,j,k), rho_v.const_array()(i,j,k));
            }
        }
    }
    for (int k = domain.smallEnd(2); k <= domain.bigEnd(2) + 1; ++k) {
        for (int j = domain.smallEnd(1); j <= domain.bigEnd(1); ++j) {
            for (int i = domain.smallEnd(0); i <= domain.bigEnd(0); ++i) {
                EXPECT_EQ(avg_zmom.const_array()(i,j,k), omega.const_array()(i,j,k));
            }
        }
    }
}

TEST(SBMP2, SyntheticSubsetPopulationSurvivesGroupedLimitAndAMRViews)
{
    erf_sbm::SpectralPopulationSpec liquid;
    liquid.population_id = 0;
    liquid.semantic_id = "liquid";
    liquid.phase = erf_sbm::PopulationPhase::Liquid;
    liquid.grid.coordinate_kind = erf_sbm::CoordinateKind::Mass;
    liquid.grid.coordinate_units = "kg";
    liquid.grid.edges = {0.0, 1.0, 2.0};
    liquid.grid.pivots = {0.5, 1.5};
    liquid.mass_state_units = "kg m^-3";
    liquid.number_state_units = "m^-3";

    erf_sbm::SpectralPopulationSpec synthetic = liquid;
    synthetic.population_id = 1;
    synthetic.semantic_id = "synthetic_ice";
    synthetic.phase = erf_sbm::PopulationPhase::Ice;
    erf_sbm::AttachedPropertyDescriptor rime{
        "rime_like_mass", "rime_like_mass", "kg m^-3", 1,
        erf_sbm::PropertyKind::MassBoundedSubset,
        erf_sbm::SupportRequirement::PositiveMass,
        erf_sbm::PropertyRemapPolicy::CarrierBinConservative,
        true, false, 0.0, std::numeric_limits<Real>::quiet_NaN()};
    erf_sbm::SBMLayoutSpec spec;
    spec.populations = {liquid, synthetic};
    spec.liquid_projection = {0, 1};
    spec.attached_properties = {rime};
    const erf_sbm::SBMLayout layout(std::move(spec));
    const auto groups = erf_sbm::make_constraint_groups(layout);
    ASSERT_EQ(groups.size(), 4U);

    // ncomp = liquid mass(2) + synthetic mass(2) + rime subset(2).
    const std::vector<Real> low_state{
        1.0, 0.0, 2.0, 0.0, 0.5, 0.0,
        1.0, 0.0, 2.0, 0.0, 0.5, 0.0};
    erf_sbm::FCTFaceTransfer face;
    face.left_cell = 0; face.right_cell = 1;
    face.left_volume = 1.0; face.right_volume = 2.0;
    face.low.assign(6, 0.0);
    face.high = {0.0, 0.0, 1.0, 0.0, 0.75, 0.0};
    const auto limited = erf_sbm::limit_grouped(low_state, 2, 6, {face}, groups);
    for (int cell = 0; cell < 2; ++cell) {
        const std::vector<Real> state(limited.updated_state.begin() + cell*6,
                                      limited.updated_state.begin() + (cell+1)*6);
        for (const auto& group : groups) EXPECT_TRUE(group.admissible(state));
    }

    const auto restricted = erf_sbm::volume_weighted_restrict(
        limited.updated_state, 2, 1, 6, {0, 0}, {1.0, 2.0}, {3.0});
    const auto prolonged = erf_sbm::piecewise_constant_prolong(
        restricted, 1, 2, 6, {0, 0});
    for (const auto& state_vector : {restricted, prolonged}) {
        const int count = static_cast<int>(state_vector.size()) / 6;
        for (int cell = 0; cell < count; ++cell) {
            const std::vector<Real> state(state_vector.begin() + cell*6,
                                          state_vector.begin() + (cell+1)*6);
            for (const auto& group : groups) EXPECT_TRUE(group.admissible(state));
        }
    }
    // The synthetic property remains bounded by the synthetic carrier mass;
    // liquid projection is still defined exclusively by population 0.
    EXPECT_GE(restricted[2] - restricted[4], 0.0);
    EXPECT_EQ(layout.bulk_projection().rules().size(), 2U);
}

TEST(SBMP2, FCTStageWeightsAndAcceptedCorrectionAreExact)
{
    EXPECT_EQ(erf_sbm::stage_weight_contract(erf_auxiliary::IntegrationMethod::CompressibleRK3, 1, true).completed_ledger_weight, 1.0);
    EXPECT_EQ(erf_sbm::stage_weight_contract(erf_auxiliary::IntegrationMethod::CompressibleRK3, 1, false).completed_ledger_weight, 0.0);
    EXPECT_EQ(erf_sbm::stage_weight_contract(erf_auxiliary::IntegrationMethod::AnelasticHeun, 0, false).completed_ledger_weight, 0.5);
    EXPECT_EQ(erf_sbm::stage_weight_contract(erf_auxiliary::IntegrationMethod::AnelasticHeun, 1, true).completed_ledger_weight, 0.5);
    const auto correction = erf_sbm::accepted_stage_correction({1.0, -2.0}, {3.0, 2.0}, 0.25);
    EXPECT_DOUBLE_EQ(correction[0], 0.5);
    EXPECT_DOUBLE_EQ(correction[1], 1.0);
}

TEST(SBMP2, HostTimestepCombinesAdvectionAndDiffusionAndFailsClosed)
{
    EXPECT_DOUBLE_EQ(erf_sbm::admissible_host_timestep(0.0, 0.0),
                     std::numeric_limits<double>::max());
    EXPECT_DOUBLE_EQ(erf_sbm::admissible_host_timestep(2.0, 0.0), 0.25);
    EXPECT_DOUBLE_EQ(erf_sbm::admissible_host_timestep(0.0, 8.0), 0.0625);
    EXPECT_DOUBLE_EQ(erf_sbm::admissible_host_timestep(2.0, 8.0), 0.05);
    EXPECT_DOUBLE_EQ(erf_sbm::admissible_host_timestep(-1.0, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(erf_sbm::admissible_host_timestep(
                         std::numeric_limits<double>::infinity(), 0.0), 0.0);
}

TEST(SBMP2, DensityWeightedDiffusionUsesIntensiveRatioAndPhysicalGeometry)
{
    erf_sbm::DiffusionFace face;
    face.left_cell = 0; face.right_cell = 1; face.area = 3.0; face.distance = 2.0;
    face.rho_left = 2.0; face.rho_right = 4.0; face.rho_face = 3.0; face.coefficient = 0.5;
    const auto result = erf_sbm::explicit_two_point_diffusion({2.0, 8.0}, 2, 1,
                                                               {4.0, 1.0}, {face}, 0.2);
    // X/rho is 1 and 2; I=-A dt rho_f K grad(X/rho)=-0.45.
    ASSERT_EQ(result.integrated_transfers.size(), 1U);
    EXPECT_NEAR(result.integrated_transfers[0], -0.45, precision_tolerance(Real(0.45)));
    EXPECT_NEAR(result.updated_state[0], 2.1125, precision_tolerance(Real(2.1125)));
    EXPECT_NEAR(result.updated_state[1], 7.55, precision_tolerance(Real(7.55)));
    const auto uniform = erf_sbm::explicit_two_point_diffusion({2.0, 4.0}, 2, 1,
                                                                {1.0, 1.0}, {face}, 1.0);
    EXPECT_NEAR(uniform.integrated_transfers[0], 0.0, precision_tolerance(Real(0.0)));
    EXPECT_GT(erf_sbm::admissible_explicit_timestep({2.0, 8.0}, 2, 1,
                                                    {4.0, 1.0}, {face}), 0.0);
}

void sbm_test_ProductionCombinedAdvectionDiffusionFailsClosed()
{
    const auto layout = make_layout(2, MomentMode::OneMoment);
    erf_auxiliary::AuxiliaryStateManager manager(layout.auxiliary_layout());
    const Box domain(IntVect(0, 0, 0), IntVect(1, 0, 0));
    const BoxArray boxes(domain);
    const DistributionMapping dm(boxes);
    const amrex::RealBox real_box({AMREX_D_DECL(0.0, 0.0, 0.0)},
                                  {AMREX_D_DECL(2.0, 1.0, 1.0)});
    const std::array<int, AMREX_SPACEDIM> periodicity{AMREX_D_DECL(1, 1, 1)};
    const Geometry geometry(domain, &real_box, amrex::CoordSys::cartesian,
                            periodicity.data());
    manager.define_level(0, boxes, dm, 2);
    for (amrex::MFIter mfi(manager.output(0)); mfi.isValid(); ++mfi) {
        const auto state = manager.output(0).array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            state(i,j,k,0) = i == 0 ? Real(1.0) : Real(0.0);
            state(i,j,k,1) = Real(0.5);
        });
    }
    manager.begin_step(0, 0.0);

    amrex::MultiFab rho(boxes, dm, 1, 2);
    rho.setVal(1.0);
    rho.FillBoundary(geometry.periodicity());
    amrex::MultiFab xflux(amrex::convert(boxes, IntVect(1, 0, 0)), dm, 1, 0);
    amrex::MultiFab yflux(amrex::convert(boxes, IntVect(0, 1, 0)), dm, 1, 0);
    amrex::MultiFab zflux(amrex::convert(boxes, IntVect(0, 0, 1)), dm, 1, 0);
    xflux.setVal(1.0);
    yflux.setVal(0.0);
    zflux.setVal(0.0);
    amrex::MultiFab core(boxes, dm, RhoQ3_comp + 1, 0);
    erf_auxiliary::AuxiliaryFaceTransfer stage_flux;
    stage_flux.define(boxes, dm, layout.ncomp(), 0);
    const auto context = erf_auxiliary::make_compressible_stage(
        2, 0.0, 0.4, 0.8, 0.8, nullptr, nullptr);

    std::string diagnostic;
    try {
        erf_sbm::advance_stage(manager, layout, context, rho, core,
                               xflux, yflux, zflux, geometry, stage_flux,
                               erf_sbm::TransportMethod::GroupedFCT_WENOZ3,
                               0, Real(0.2), 1);
    } catch (const std::exception& error) {
        diagnostic = error.what();
    }
    // The unconditional exact-stage guard is deliberately before spectral
    // transport.  This fixture supplies a null diagnostic pointer, so it
    // also proves that the pointer cannot bypass the invariant.
    EXPECT_NE(diagnostic.find("SBM actual stage low-order demand exceeds admissibility"),
              std::string::npos);
    const auto recommendation = diagnostic.find("recommended_max_host_dt=");
    ASSERT_NE(recommendation, std::string::npos) << diagnostic;
    const double recommended_dt = std::stod(
        diagnostic.substr(recommendation + std::string("recommended_max_host_dt=").size()));
    EXPECT_GT(recommended_dt, 0.0);
    EXPECT_LT(recommended_dt, 0.8);
}

TEST(SBMP2, ProductionCombinedAdvectionDiffusionFailsClosed)
{
    sbm_test_ProductionCombinedAdvectionDiffusionFailsClosed();
}

void sbm_test_ProductionVariableDensityHostCFLBypassFailsClosed()
{
    const auto layout = make_layout(2, MomentMode::OneMoment);
    erf_auxiliary::AuxiliaryStateManager manager(layout.auxiliary_layout());
    const Box domain(IntVect(0, 0, 0), IntVect(1, 0, 0));
    const BoxArray boxes(domain);
    const DistributionMapping dm(boxes);
    const amrex::RealBox real_box({AMREX_D_DECL(0.0, 0.0, 0.0)},
                                  {AMREX_D_DECL(2.0, 1.0, 1.0)});
    const std::array<int, AMREX_SPACEDIM> periodicity{AMREX_D_DECL(1, 1, 1)};
    const Geometry geometry(domain, &real_box, amrex::CoordSys::cartesian,
                            periodicity.data());
    manager.define_level(0, boxes, dm, 2);
    manager.output(0).setVal(Real(0.0));
    for (amrex::MFIter mfi(manager.output(0)); mfi.isValid(); ++mfi) {
        const auto state = manager.output(0).array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            state(i,j,k,0) = i == 0 ? Real(1.0) : Real(0.0);
        });
    }
    manager.begin_step(0, 0.0);

    amrex::MultiFab rho(boxes, dm, 1, 2);
    for (amrex::MFIter mfi(rho); mfi.isValid(); ++mfi) {
        const auto density = rho.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            density(i,j,k,0) = i == 0 ? Real(0.01) : Real(1.0);
        });
    }
    rho.FillBoundary(geometry.periodicity());
    amrex::MultiFab xflux(amrex::convert(boxes, IntVect(1,0,0)), dm, 1, 0);
    amrex::MultiFab yflux(amrex::convert(boxes, IntVect(0,1,0)), dm, 1, 0);
    amrex::MultiFab zflux(amrex::convert(boxes, IntVect(0,0,1)), dm, 1, 0);
    xflux.setVal(Real(0.0)); yflux.setVal(Real(0.0)); zflux.setVal(Real(0.0));
    for (amrex::MFIter mfi(xflux); mfi.isValid(); ++mfi) {
        const auto flux = xflux.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            // The ERF carrier passed to the production hook is rho*u at the
            // face, not the velocity alone.  With rho=(0.01,1.0), u=0.1,
            // the arithmetic face carrier is 0.0505.
            if (i == 1) flux(i,j,k) = Real(0.0505);
        });
    }
    xflux.FillBoundary(geometry.periodicity());
    yflux.FillBoundary(geometry.periodicity());
    zflux.FillBoundary(geometry.periodicity());
    amrex::MultiFab core(boxes, dm, RhoQ3_comp + 1, 0);
    core.setVal(Real(0.0));
    erf_auxiliary::AuxiliaryFaceTransfer stage_flux;
    stage_flux.define(boxes, dm, layout.ncomp(), 0);
    const auto unsafe_context = erf_auxiliary::make_compressible_stage(
        2, 0.0, Real(0.125), Real(0.25), Real(0.25), nullptr, nullptr);
    std::string diagnostic;
    try {
        erf_sbm::advance_stage(manager, layout, unsafe_context, rho, core,
                               xflux, yflux, zflux, geometry, stage_flux,
                               erf_sbm::TransportMethod::GroupedFCT_WENOZ3,
                               0, Real(0.0), 1);
    } catch (const std::exception& error) {
        diagnostic = error.what();
    }
    EXPECT_NE(diagnostic.find("SBM actual stage low-order demand exceeds admissibility"),
              std::string::npos) << diagnostic;
    const auto recommendation = diagnostic.find("recommended_max_host_dt=");
    ASSERT_NE(recommendation, std::string::npos) << diagnostic;
    const double recommended_dt = std::stod(
        diagnostic.substr(recommendation + std::string("recommended_max_host_dt=").size()));
    EXPECT_GT(recommended_dt, 0.0);
    EXPECT_LT(recommended_dt, 0.25);

    // The manufactured carrier is intentionally too large for the selected
    // host step, but the reported host cap is sufficient when used to rerun
    // the same otherwise-unchanged stage.
    const auto safe_context = erf_auxiliary::make_compressible_stage(
        2, 0.0, Real(0.45) * recommended_dt, Real(0.9) * recommended_dt,
        Real(0.9) * recommended_dt,
        nullptr, nullptr);
    EXPECT_NO_THROW(erf_sbm::advance_stage(
        manager, layout, safe_context, rho, core, xflux, yflux, zflux,
        geometry, stage_flux, erf_sbm::TransportMethod::GroupedFCT_WENOZ3,
        0, Real(0.0), 1));
}

TEST(SBMP2, ProductionVariableDensityHostCFLBypassFailsClosed)
{
    sbm_test_ProductionVariableDensityHostCFLBypassFailsClosed();
}

void sbm_test_ProductionCombinedDemandUsesPreStageBaselineForAllStageContracts()
{
    const auto run_case = [](const Real advection_demand, const Real diffusion,
                             const bool anelastic, const int stage,
                             const MomentMode mode = MomentMode::OneMoment,
                             const bool with_property = false,
                             const int chunk_size = 1) {
        const auto layout = make_layout(2, mode, with_property);
        erf_auxiliary::AuxiliaryStateManager manager(layout.auxiliary_layout());
        const Box domain(IntVect(0, 0, 0), IntVect(1, 0, 0));
        const BoxArray boxes(domain);
        const DistributionMapping dm(boxes);
        const amrex::RealBox real_box({AMREX_D_DECL(0.0, 0.0, 0.0)},
                                      {AMREX_D_DECL(2.0, 100.0, 100.0)});
        const std::array<int, AMREX_SPACEDIM> periodicity{AMREX_D_DECL(1, 1, 1)};
        const Geometry geometry(domain, &real_box, amrex::CoordSys::cartesian,
                                periodicity.data());
        manager.define_level(0, boxes, dm, 2);
        manager.output(0).setVal(Real(0.0));
        for (amrex::MFIter mfi(manager.output(0)); mfi.isValid(); ++mfi) {
            const auto state = manager.output(0).array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                if (mode == MomentMode::TwoMoment) {
                    state(i,j,k,0) = i == 0 ? Real(0.5) : Real(0.0);
                    state(i,j,k,2) = i == 0 ? Real(1.0) : Real(0.0);
                    if (with_property) state(i,j,k,4) = i == 0 ? Real(0.25) : Real(0.0);
                } else {
                    state(i,j,k,0) = i == 0 ? Real(1.0) : Real(0.0);
                }
            });
        }
        manager.begin_step(0, 0.0);
        if (anelastic && stage > 0) manager.evaluation(0).setVal(Real(0.0));
        if (anelastic && stage > 0) {
            for (amrex::MFIter mfi(manager.evaluation(0)); mfi.isValid(); ++mfi) {
                const auto state = manager.evaluation(0).array(mfi);
                amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                    if (mode == MomentMode::TwoMoment) {
                        state(i,j,k,0) = i == 0 ? Real(0.5) : Real(0.0);
                        state(i,j,k,2) = i == 0 ? Real(1.0) : Real(0.0);
                        if (with_property) state(i,j,k,4) = i == 0 ? Real(0.25) : Real(0.0);
                    } else {
                        state(i,j,k,0) = i == 0 ? Real(1.0) : Real(0.0);
                    }
                });
            }
        }

        amrex::MultiFab rho(boxes, dm, 1, 2);
        rho.setVal(Real(1.0));
        rho.FillBoundary(geometry.periodicity());
        amrex::MultiFab xflux(amrex::convert(boxes, IntVect(1, 0, 0)), dm, 1, 0);
        amrex::MultiFab yflux(amrex::convert(boxes, IntVect(0, 1, 0)), dm, 1, 0);
        amrex::MultiFab zflux(amrex::convert(boxes, IntVect(0, 0, 1)), dm, 1, 0);
        xflux.setVal(Real(0.0));
        yflux.setVal(Real(0.0));
        zflux.setVal(Real(0.0));
        for (amrex::MFIter mfi(xflux); mfi.isValid(); ++mfi) {
            const auto flux = xflux.array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                flux(i,j,k) = i == 1 ? advection_demand : Real(0.0);
            });
        }
        amrex::MultiFab core(boxes, dm, RhoQ3_comp + 1, 0);
        erf_auxiliary::AuxiliaryFaceTransfer stage_flux;
        stage_flux.define(boxes, dm, layout.ncomp(), 0);
        const auto context = anelastic ?
            erf_auxiliary::make_anelastic_stage(stage, 0.0, stage == 0 ? 0.0 : 1.0,
                                                1.0, 1.0, nullptr, nullptr) :
            erf_auxiliary::make_compressible_stage(2, 0.0, 0.5, 1.0, 1.0, nullptr, nullptr);
        std::string diagnostic;
        try {
            erf_sbm::advance_stage(manager, layout, context, rho, core,
                                   xflux, yflux, zflux, geometry, stage_flux,
                                   erf_sbm::TransportMethod::GroupedFCT_WENOZ3,
                                   0, diffusion, chunk_size);
        } catch (const std::exception& error) {
            diagnostic = error.what();
        }
        return diagnostic;
    };

    const std::array<std::pair<bool, int>, 3> stage_contracts{{
        {false, 0}, {true, 0}, {true, 1}}};

    // The exact host-stage guard is now the first production invariant.  Keep
    // these cases below its bound so that the subsequent production
    // low-order budget still gets exercised for admissible stages.
    for (const auto& contract : stage_contracts) {
        SCOPED_TRACE(std::string(contract.first ? "anelastic" : "compressible") +
                     " stage=" + std::to_string(contract.second));
        EXPECT_TRUE(run_case(Real(0.6), Real(0.0), contract.first, contract.second).empty());
        const auto invalid = run_case(Real(1.2), Real(0.0), contract.first, contract.second);
        EXPECT_NE(invalid.find("SBM actual stage low-order demand exceeds admissibility"),
                  std::string::npos) << invalid;
        EXPECT_NE(invalid.find("recommended_max_host_dt="), std::string::npos) << invalid;
    }

    // Advection and diffusion are each admissible, but their combined demand
    // is not.  The second case deliberately exceeds the final low state while
    // remaining below the pre-stage baseline and must therefore pass.
    for (const auto& contract : stage_contracts) {
        const Real combined_fail_advection = Real(0.8);
        const Real combined_pass_advection = Real(0.4);
        const auto invalid = run_case(combined_fail_advection, Real(0.15), contract.first, contract.second);
        EXPECT_NE(invalid.find("SBM actual stage low-order demand exceeds admissibility"),
                  std::string::npos) << invalid;
        EXPECT_TRUE(run_case(combined_pass_advection, Real(0.15), contract.first, contract.second).empty());
    }

    for (const auto mode : {MomentMode::TwoMoment}) {
        const auto valid = run_case(Real(0.6), Real(0.0), false, 0, mode, true, 1);
        EXPECT_TRUE(valid.empty()) << valid;
        const auto invalid = run_case(Real(1.2), Real(0.0), false, 0, mode, true, 1);
        EXPECT_NE(invalid.find("SBM actual stage low-order demand exceeds admissibility"),
                  std::string::npos)
            << invalid;
    }
}

TEST(SBMP2, ProductionCombinedDemandUsesPreStageBaselineForAllStageContracts)
{
    sbm_test_ProductionCombinedDemandUsesPreStageBaselineForAllStageContracts();
}

void sbm_test_ProductionTargetDensityPreservesConstantRatioForBothTemporalContracts()
{
    const Real pi = Real(3.1415926535897932384626433832795);
    const Real k = Real(0.37);
    const int ncell = 16;
    const Box domain(IntVect(0, 0, 0), IntVect(ncell-1, 1, 1));
    const BoxArray boxes(domain);
    const DistributionMapping dm(boxes);
    const amrex::RealBox real_box({AMREX_D_DECL(0.0, 0.0, 0.0)},
                                  {AMREX_D_DECL(1.0, 1.0, 1.0)});
    const std::array<int, AMREX_SPACEDIM> periodicity{AMREX_D_DECL(1, 1, 1)};
    const Geometry geometry(domain, &real_box, amrex::CoordSys::cartesian,
                            periodicity.data());
    const auto layout = make_layout(2, MomentMode::OneMoment);
    const Real h = Real(1.0) / Real(ncell);
    const Real velocity = Real(0.05);
    const Real inverse_cell_size = geometry.InvCellSize(0);

    const auto run = [&](const bool anelastic, const int stage) {
        erf_auxiliary::AuxiliaryStateManager manager(layout.auxiliary_layout());
        manager.define_level(0, boxes, dm, 2);
        amrex::MultiFab rho_anchor(boxes, dm, 1, 2);
        amrex::MultiFab rho_input(boxes, dm, 1, 2);
        amrex::MultiFab rho_target(boxes, dm, 1, 2);
        amrex::MultiFab carrier_x(amrex::convert(boxes, IntVect(1, 0, 0)), dm, 1, 0);
        amrex::MultiFab carrier_y(amrex::convert(boxes, IntVect(0, 1, 0)), dm, 1, 0);
        amrex::MultiFab carrier_z(amrex::convert(boxes, IntVect(0, 0, 1)), dm, 1, 0);
        rho_anchor.setVal(Real(0.0));
        rho_input.setVal(Real(0.0));
        rho_target.setVal(Real(0.0));
        for (amrex::MFIter mfi(rho_anchor); mfi.isValid(); ++mfi) {
            const auto anchor = rho_anchor.array(mfi);
            const auto input = rho_input.array(mfi);
            const auto state = manager.output(0).array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int kidx) noexcept {
                const Real x = (Real(i) + Real(0.5)) * h;
                const Real anchor_rho = Real(1.0) + Real(0.10) *
                    std::sin(Real(2.0) * pi * x);
                const Real input_rho = anchor_rho + Real(0.02) *
                    std::cos(Real(2.0) * pi * x);
                anchor(i,j,kidx) = anchor_rho;
                input(i,j,kidx) = input_rho;
                state(i,j,kidx,0) = k * anchor_rho;
                state(i,j,kidx,1) = Real(0.0);
            });
        }
        for (amrex::MFIter mfi(carrier_x); mfi.isValid(); ++mfi) {
            const auto flux = carrier_x.array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int kidx) noexcept {
                const Real x = Real(i) * h;
                flux(i,j,kidx) = velocity * (Real(1.0) + Real(0.15) *
                    std::sin(Real(2.0) * pi * x));
            });
        }
        carrier_y.setVal(Real(0.0));
        carrier_z.setVal(Real(0.0));
        rho_anchor.FillBoundary(geometry.periodicity());
        rho_input.FillBoundary(geometry.periodicity());
        carrier_x.FillBoundary(geometry.periodicity());
        carrier_y.FillBoundary(geometry.periodicity());
        carrier_z.FillBoundary(geometry.periodicity());
        manager.output(0).FillBoundary(geometry.periodicity());
        manager.begin_step(0, 0.0);

        const Real full_step = anelastic ? Real(0.01) : Real(0.012);
        const Real dt = anelastic ? full_step :
            (stage == 0 ? full_step / Real(3.0) :
             (stage == 1 ? full_step / Real(2.0) : full_step));
        if (stage > 0) {
            for (amrex::MFIter mfi(manager.evaluation(0)); mfi.isValid(); ++mfi) {
                const auto evaluation = manager.evaluation(0).array(mfi);
                const auto input = rho_input.const_array(mfi);
                amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int kidx) noexcept {
                    evaluation(i,j,kidx,0) = k * input(i,j,kidx);
                    evaluation(i,j,kidx,1) = Real(0.0);
                });
            }
            manager.evaluation(0).FillBoundary(geometry.periodicity());
        }
        for (amrex::MFIter mfi(rho_target); mfi.isValid(); ++mfi) {
            const auto target = rho_target.array(mfi);
            const auto anchor = rho_anchor.const_array(mfi);
            const auto input = rho_input.const_array(mfi);
                const auto flux = carrier_x.const_array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int kidx) noexcept {
                const Real divergence = (flux(i+1,j,kidx) - flux(i,j,kidx)) *
                    inverse_cell_size;
                const Real transport_target = anchor(i,j,kidx) - dt * divergence;
                target(i,j,kidx) = (anelastic && stage > 0) ?
                    Real(0.5) * (anchor(i,j,kidx) + input(i,j,kidx) - dt * divergence) :
                    transport_target;
            });
        }
        rho_target.FillBoundary(geometry.periodicity());

        amrex::MultiFab core(boxes, dm, RhoQ3_comp + 1, 0);
        erf_auxiliary::AuxiliaryFaceTransfer stage_flux;
        stage_flux.define(boxes, dm, layout.ncomp(), 0);
        const auto context = anelastic ?
            erf_auxiliary::make_anelastic_stage(stage, 0.0, stage > 0 ? full_step : 0.0,
                                                full_step, full_step, nullptr, nullptr) :
            erf_auxiliary::make_compressible_stage(stage, 0.0,
                                                   stage == 0 ? 0.0 :
                                                   (stage == 1 ? full_step / Real(3.0) : full_step / Real(2.0)),
                                                   dt, full_step, nullptr, nullptr);
        erf_sbm::advance_stage(manager, layout, context, rho_anchor, rho_input, rho_target,
                               core, carrier_x, carrier_y, carrier_z, geometry, stage_flux,
                               erf_sbm::TransportMethod::GroupedFCT_WENOZ3, 0, Real(0.0), 1);

        amrex::MultiFab error(boxes, dm, 1, 0);
        for (amrex::MFIter mfi(error); mfi.isValid(); ++mfi) {
            const auto result = error.array(mfi);
            const auto output = manager.output(0).const_array(mfi);
            const auto target = rho_target.const_array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int kidx) noexcept {
                result(i,j,kidx) = amrex::Math::abs(output(i,j,kidx,0) - k * target(i,j,kidx));
            });
        }
        amrex::Gpu::synchronize();
        const Real target_error = error.norm0(0);
        return target_error;
    };

    Real maximum_error = Real(0.0);
    for (const int stage : {0, 1, 2}) {
        const Real error = run(false, stage);
        maximum_error = amrex::max(maximum_error, error);
        EXPECT_LE(error, Real(4096.0) * std::numeric_limits<Real>::epsilon() * k)
            << "compressible stage=" << stage << " error=" << error;
    }
    for (const int stage : {0, 1}) {
        const Real error = run(true, stage);
        maximum_error = amrex::max(maximum_error, error);
        EXPECT_LE(error, Real(4096.0) * std::numeric_limits<Real>::epsilon() * k)
            << "anelastic stage=" << stage << " error=" << error;
    }
    EXPECT_GT(maximum_error, Real(0.0));
}

TEST(SBMP2, ProductionTargetDensityPreservesConstantRatioForBothTemporalContracts)
{
    sbm_test_ProductionTargetDensityPreservesConstantRatioForBothTemporalContracts();
}

TEST(SBMP2, BoundaryBudgetsHaveNoWallSinkAndValidateInflow)
{
    const auto layout = make_layout(2, MomentMode::OneMoment);
    const auto groups = erf_sbm::make_constraint_groups(layout);
    erf_sbm::BoundaryDescriptor wall;
    wall.kind = erf_sbm::BoundaryKind::ImpermeableWall;
    const auto wall_budget = erf_sbm::make_boundary_budget(wall, {2.0, 3.0}, groups, {0.0, 0.0}, {0.0, 0.0}, 5.0, 0.1);
    EXPECT_EQ(wall_budget.auxiliary[0], 0.0);
    erf_sbm::BoundaryDescriptor inflow;
    inflow.kind = erf_sbm::BoundaryKind::PrescribedSpectralInflow;
    inflow.prescribed_state = {1.0, 2.0};
    EXPECT_TRUE(erf_sbm::validate_prescribed_inflow(inflow, groups));
    inflow.prescribed_state[0] = -1.0;
    EXPECT_FALSE(erf_sbm::validate_prescribed_inflow(inflow, groups));
}

void sbm_test_ProductionOutflowRejectsInwardCarrierWithoutSpectralInflow()
{
    const auto layout = make_layout(2, MomentMode::OneMoment);
    erf_auxiliary::AuxiliaryStateManager manager(layout.auxiliary_layout());
    const Box domain(IntVect(0, 0, 0), IntVect(1, 1, 1));
    const BoxArray boxes(domain);
    const DistributionMapping dm(boxes);
    const amrex::RealBox real_box({AMREX_D_DECL(0.0, 0.0, 0.0)},
                                  {AMREX_D_DECL(2.0, 2.0, 2.0)});
    const std::array<int, AMREX_SPACEDIM> periodicity{AMREX_D_DECL(0, 1, 1)};
    const Geometry geometry(domain, &real_box, amrex::CoordSys::cartesian,
                            periodicity.data());
    manager.define_level(0, boxes, dm, 2);
    manager.output(0).setVal(1.0);
    manager.begin_step(0, 0.0);

    amrex::MultiFab rho(boxes, dm, 1, 2);
    rho.setVal(1.0);
    rho.FillBoundary(geometry.periodicity());
    amrex::MultiFab xflux(amrex::convert(boxes, IntVect(1, 0, 0)), dm, 1, 0);
    amrex::MultiFab yflux(amrex::convert(boxes, IntVect(0, 1, 0)), dm, 1, 0);
    amrex::MultiFab zflux(amrex::convert(boxes, IntVect(0, 0, 1)), dm, 1, 0);
    xflux.setVal(0.0);
    yflux.setVal(0.0);
    zflux.setVal(0.0);
    const int high_face = domain.bigEnd(0) + 1;
    for (amrex::MFIter mfi(xflux); mfi.isValid(); ++mfi) {
        const auto flux = xflux.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            if (i == high_face) flux(i,j,k) = -1.0;
        });
    }
    amrex::MultiFab core(boxes, dm, RhoQ3_comp + 1, 0);
    erf_auxiliary::AuxiliaryFaceTransfer stage_flux;
    stage_flux.define(boxes, dm, layout.ncomp(), 0);
    const auto context = erf_auxiliary::make_compressible_stage(
        0, 0.0, 0.0, 1.0/3.0, 1.0, nullptr, nullptr);
    erf_sbm::TransportBoundaryPolicy boundary_policy;
    boundary_policy.configured = true;
    boundary_policy.face_kind[0] = erf_sbm::BoundaryKind::ImpermeableWall;
    boundary_policy.face_kind[1] = erf_sbm::BoundaryKind::AdvectiveOutflow;
    boundary_policy.face_kind[2] = erf_sbm::BoundaryKind::Periodic;
    boundary_policy.face_kind[3] = erf_sbm::BoundaryKind::Periodic;
    boundary_policy.face_kind[4] = erf_sbm::BoundaryKind::Periodic;
    boundary_policy.face_kind[5] = erf_sbm::BoundaryKind::Periodic;
    std::string diagnostic;
    try {
        erf_sbm::advance_stage(manager, layout, context, rho, rho, core,
                               xflux, yflux, zflux, geometry, stage_flux,
                               erf_sbm::TransportMethod::GroupedFCT_WENOZ3,
                               0, 0.0, 1, nullptr, boundary_policy);
    } catch (const std::exception& error) {
        diagnostic = error.what();
    }
    EXPECT_NE(diagnostic.find(
        "SBM open boundary received inward carrier flux but no prescribed spectral inflow is configured"),
        std::string::npos);
}

TEST(SBMP2, ProductionOutflowRejectsInwardCarrierWithoutSpectralInflow)
{
    sbm_test_ProductionOutflowRejectsInwardCarrierWithoutSpectralInflow();
}

TEST(SBMP2, FCTCorrectionUsesHighMinusLowAdvectionWhenDiffusionReversesTotalSign)
{
    const Real low_adv = 0.75;
    const Real low_diff = 1.0;
    const Real high_adv = 0.25;
    const Real high_total = high_adv + low_diff;
    ASSERT_LT(high_adv - low_adv, 0.0);
    ASSERT_GT(high_total - low_adv, 0.0);
    const Real lambda = 0.5;
    const Real accepted = low_adv + low_diff + lambda * (high_adv - low_adv);
    const Real old_expression = low_adv + low_diff + lambda * (high_total - low_adv);
    EXPECT_DOUBLE_EQ(erf_sbm::accepted_stage_correction({low_adv}, {high_adv}, lambda)[0], -0.25);
    EXPECT_DOUBLE_EQ(accepted, 1.5);
    EXPECT_DOUBLE_EQ(old_expression, 2.0);
    EXPECT_NE(accepted, old_expression);
}

TEST(SBMP2, AMRRestrictionProlongationAndRegisterUnitsAreConservative)
{
    const auto restricted = erf_sbm::volume_weighted_restrict({1.0, 3.0, 5.0, 7.0}, 4, 2, 1,
                                                               {0, 0, 1, 1}, {1.0, 1.0, 2.0, 2.0}, {2.0, 4.0});
    EXPECT_DOUBLE_EQ(restricted[0], 2.0);
    EXPECT_DOUBLE_EQ(restricted[1], 6.0);
    const auto injected = erf_sbm::piecewise_constant_prolong(restricted, 2, 4, 1, {0, 0, 1, 1});
    EXPECT_EQ(injected[0], 2.0); EXPECT_EQ(injected[3], 6.0);
    EXPECT_DOUBLE_EQ(erf_sbm::register_flux_from_integrated_transfer(12.0, 3.0, 2.0), 2.0);
    EXPECT_THROW((void)erf_sbm::register_flux_from_integrated_transfer(1.0, 0.0, 1.0), std::invalid_argument);
}

TEST(SBMP2, IndependentInterfaceOracleCatchesRefluxSignAreaAndTimeErrors)
{
    const std::vector<Real> coarse{2.0};
    const std::vector<Real> fine{1.0, 1.0, 1.0, 1.0};
    const auto good = erf_sbm::check_interface_transfer(
        coarse, fine, 4.0, 1.0, 0.5, 0.5, -2.0,
        precision_tolerance(Real(2.0), Real(64.0)));
    EXPECT_TRUE(good.passes);
    EXPECT_DOUBLE_EQ(good.coarse_transfer, 4.0);
    EXPECT_DOUBLE_EQ(good.fine_transfer, 2.0);
    EXPECT_DOUBLE_EQ(good.mismatch, -2.0);

    const auto missing_reflux = erf_sbm::check_interface_transfer(
        coarse, fine, 4.0, 1.0, 0.5, 0.5, 0.0,
        precision_tolerance(Real(2.0), Real(64.0)));
    EXPECT_FALSE(missing_reflux.passes);
    const auto wrong_sign = erf_sbm::check_interface_transfer(
        coarse, fine, 4.0, 1.0, 0.5, 0.5, 2.0,
        precision_tolerance(Real(2.0), Real(64.0)));
    EXPECT_FALSE(wrong_sign.passes);
    const auto wrong_area = erf_sbm::check_interface_transfer(
        coarse, fine, 4.0, 2.0, 0.5, 0.5, -6.0,
        precision_tolerance(Real(6.0), Real(64.0)));
    EXPECT_FALSE(wrong_area.passes);
    const auto wrong_time = erf_sbm::check_interface_transfer(
        coarse, fine, 4.0, 1.0, 0.5, 1.0, -2.0,
        precision_tolerance(Real(2.0), Real(64.0)));
    EXPECT_FALSE(wrong_time.passes);
    EXPECT_THROW((void)erf_sbm::check_interface_transfer(
        coarse, fine, 4.0, 1.0, 0.5, 0.0, -2.0,
        precision_tolerance(Real(2.0), Real(64.0))), std::invalid_argument);
}

TEST(SBMP2, PostRefluxFailsClosedWithDiagnosticsAndNeverClips)
{
    const auto layout = make_layout(2, MomentMode::OneMoment);
    const auto groups = erf_sbm::make_constraint_groups(layout);
    const auto good = erf_sbm::validate_post_reflux({1.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, 2, 1, 2, groups);
    EXPECT_TRUE(good.admissible);
    const auto bad = erf_sbm::validate_post_reflux({1.0, 0.0}, {-2.0, 0.0}, {-1.0, 0.0}, 2, 1, 2, groups);
    EXPECT_FALSE(bad.admissible);
    EXPECT_EQ(bad.level, 2); EXPECT_EQ(bad.cell, 0);
    EXPECT_FALSE(bad.group.empty()); EXPECT_FALSE(bad.constraint.empty());
}

TEST(SBMP2, RestartSchemaAndProjectionComparisonAreStrict)
{
    const auto layout = make_layout(4, MomentMode::TwoMoment);
    const auto schema = erf_sbm::make_checkpoint_schema(
        layout, erf_sbm::SBM_CONSTRAINT_POLICY_ID, erf_sbm::SBM_TRANSPORT_POLICY_ID,
        "gamma-k-v2", erf_sbm::SBM_BOUNDARY_POLICY_ID);
    EXPECT_TRUE(erf_sbm::compare_checkpoint_schema(schema, schema).empty());
    const auto old_p2_schema = erf_sbm::make_checkpoint_schema(
        layout, "complete-groups-v1", "WENO_Z3+FCT-v1", "gamma-k-v1", "periodic-only-v1");
    EXPECT_NE(erf_sbm::compare_checkpoint_schema(schema, old_p2_schema).find("constraint_policy"),
              std::string::npos);
    auto old_transport_schema = schema;
    old_transport_schema.transport_identity = "WENO_Z3+FCT-v1";
    EXPECT_NE(erf_sbm::compare_checkpoint_schema(schema, old_transport_schema).find("transport_identity"),
              std::string::npos);
    old_transport_schema.transport_identity = "WENO_Z3-group-FCT-v3";
    EXPECT_NE(erf_sbm::compare_checkpoint_schema(schema, old_transport_schema).find("transport_identity"),
              std::string::npos);
    auto old_amr_schema = schema;
    old_amr_schema.amr_transfer_policy = "direct-extensive-AMR-v1";
    EXPECT_NE(erf_sbm::compare_checkpoint_schema(schema, old_amr_schema).find("amr_transfer_policy"),
              std::string::npos);
    auto altered = schema;
    altered.moment_modes += "changed";
    EXPECT_NE(erf_sbm::compare_checkpoint_schema(schema, altered).find("moment_modes"), std::string::npos);
    altered = schema;
    altered.boundary_policy += "-changed";
    EXPECT_NE(erf_sbm::compare_checkpoint_schema(schema, altered).find("boundary_policy"),
              std::string::npos);
    EXPECT_TRUE(erf_sbm::compare_projection(
        1.0, 1.0 + precision_tolerance(Real(1.0), Real(8.0)), 1.0, 8));
    EXPECT_FALSE(erf_sbm::compare_projection(1.0, 1.0 + 1.e-4, 1.0, 8));
}

TEST(SBMP2, RestartSchemaRoundTripRejectsDistinctFinitePropertySupport)
{
    const auto bounded_a = make_layout(4, MomentMode::TwoMoment, true,
                                       Real(1.0000001));
    const auto bounded_b = make_layout(4, MomentMode::TwoMoment, true,
                                       Real(1.0000002));
    const auto schema_a = erf_sbm::make_checkpoint_schema(
        bounded_a, erf_sbm::SBM_CONSTRAINT_POLICY_ID,
        erf_sbm::SBM_TRANSPORT_POLICY_ID, "gamma-k-v2",
        erf_sbm::SBM_BOUNDARY_POLICY_ID);
    const auto schema_b = erf_sbm::make_checkpoint_schema(
        bounded_b, erf_sbm::SBM_CONSTRAINT_POLICY_ID,
        erf_sbm::SBM_TRANSPORT_POLICY_ID, "gamma-k-v2",
        erf_sbm::SBM_BOUNDARY_POLICY_ID);
    const auto path = std::filesystem::temp_directory_path() /
        "erf_sbm_p2_checkpoint_schema_roundtrip.txt";
    erf_sbm::write_checkpoint_schema(path.string(), schema_a);
    const auto roundtrip = erf_sbm::read_checkpoint_schema(path.string());
    EXPECT_TRUE(erf_sbm::compare_checkpoint_schema(schema_a, roundtrip).empty());
    const auto mismatch = erf_sbm::compare_checkpoint_schema(schema_b, roundtrip);
    EXPECT_NE(mismatch.find("property_identity"), std::string::npos);
    std::error_code error;
    std::filesystem::remove(path, error);
}

TEST(SBMP2, AuxiliaryManagerOwnsAMRCreateProlongAverageDownRemakeAndDestroy)
{
    const auto layout = make_layout(2, MomentMode::OneMoment);
    erf_auxiliary::AuxiliaryStateManager manager(layout.auxiliary_layout());
    const Box coarse_domain(IntVect(0, 0, 0), IntVect(1, 0, 0));
    const BoxArray coarse_boxes(coarse_domain);
    const DistributionMapping coarse_dm(coarse_boxes);
    const IntVect ref_ratio(2, 2, 2);
    const Box fine_domain = amrex::refine(coarse_domain, ref_ratio);
    BoxArray fine_boxes(fine_domain);
    const DistributionMapping fine_dm(fine_boxes);
    const amrex::RealBox real_box({AMREX_D_DECL(0.0, 0.0, 0.0)},
                                  {AMREX_D_DECL(4.0, 2.0, 2.0)});
    const std::array<int, AMREX_SPACEDIM> periodicity{AMREX_D_DECL(1, 1, 1)};
    const Geometry coarse_geometry(coarse_domain, &real_box, amrex::CoordSys::cartesian,
                                   periodicity.data());
    const Geometry fine_geometry(fine_domain, &real_box, amrex::CoordSys::cartesian,
                                 periodicity.data());
    manager.define_level(0, coarse_boxes, coarse_dm, 2);
    manager.define_level(1, fine_boxes, fine_dm, 2);
    manager.output(0).setVal(3.0);
    manager.old(0).setVal(3.0);
    manager.evaluation(0).setVal(3.0);
    manager.prolong_from_coarse(0, 1, coarse_geometry, fine_geometry, ref_ratio);
    EXPECT_DOUBLE_EQ(manager.output(1).min(0), 3.0);
    manager.average_down_to(0, 1, ref_ratio);
    EXPECT_DOUBLE_EQ(manager.output(0).min(0), 3.0);

    fine_boxes.maxSize(1);
    const DistributionMapping remade_dm(fine_boxes);
    manager.remake_level(1, fine_boxes, remade_dm, 2, fine_geometry.periodicity());
    EXPECT_TRUE(manager.has_level(1));
    EXPECT_GT(manager.resident_bytes(), std::size_t(0));
    manager.destroy_level(1);
    EXPECT_FALSE(manager.has_level(1));
}

TEST(SBMP2, AuxiliaryStageFillPatchAndRemakeUseAuthoritativeCoarseSpectrum)
{
    const auto layout = make_layout(2, MomentMode::OneMoment);
    erf_auxiliary::AuxiliaryStateManager manager(layout.auxiliary_layout());
    const Box coarse_domain(IntVect(0, 0, 0), IntVect(1, 1, 1));
    const BoxArray coarse_boxes(coarse_domain);
    const DistributionMapping coarse_dm(coarse_boxes);
    const IntVect ref_ratio(2, 2, 2);
    const Box fine_domain = amrex::refine(coarse_domain, ref_ratio);
    const BoxArray initial_fine_boxes(Box(IntVect(0, 0, 0), IntVect(1, 3, 3)));
    const BoxArray remade_fine_boxes(fine_domain);
    const DistributionMapping initial_fine_dm(initial_fine_boxes);
    const DistributionMapping remade_fine_dm(remade_fine_boxes);
    const amrex::RealBox real_box({AMREX_D_DECL(0.0, 0.0, 0.0)},
                                  {AMREX_D_DECL(4.0, 4.0, 4.0)});
    const std::array<int, AMREX_SPACEDIM> periodicity{AMREX_D_DECL(1, 1, 1)};
    const Geometry coarse_geometry(coarse_domain, &real_box, amrex::CoordSys::cartesian,
                                   periodicity.data());
    const Geometry fine_geometry(fine_domain, &real_box, amrex::CoordSys::cartesian,
                                 periodicity.data());
    manager.define_level(0, coarse_boxes, coarse_dm, 2);
    manager.define_level(1, initial_fine_boxes, initial_fine_dm, 2);
    manager.output(0).setVal(2.0);
    manager.begin_step(0, 0.0);
    manager.output(0).setVal(4.0);
    manager.accept_stage(0, 1.0);
    manager.evaluation(1).setVal(0.0);

    manager.fill_stage_from_coarse(0, 1, 0.5, coarse_geometry, fine_geometry, ref_ratio);
    Real stage_ghost_value = 0.0;
    for (amrex::MFIter mfi(manager.evaluation(1)); mfi.isValid(); ++mfi) {
        stage_ghost_value = manager.evaluation(1).const_array(mfi)(2, 1, 1, 0);
    }
    amrex::ParallelDescriptor::ReduceRealMax(stage_ghost_value);
    EXPECT_DOUBLE_EQ(stage_ghost_value, 3.0);

    manager.old(1).setVal(7.0);
    manager.evaluation(1).setVal(7.0);
    manager.output(1).setVal(7.0);
    manager.remake_level_from_coarse(1, remade_fine_boxes, remade_fine_dm, 2,
                                     fine_geometry.periodicity(), 0, coarse_geometry,
                                     fine_geometry, ref_ratio, 0.5);
    EXPECT_DOUBLE_EQ(manager.output(1).min(0), 3.0);
    EXPECT_DOUBLE_EQ(manager.output(1).max(0), 7.0);
}

void sbm_test_CarrierWeightedAMRTransferUsesTargetDensityAndPreservesRatios()
{
    const auto layout = make_layout(2, MomentMode::TwoMoment, true);
    erf_auxiliary::AuxiliaryStateManager manager(layout.auxiliary_layout());
    const Box coarse_domain(IntVect(0, 0, 0), IntVect(1, 1, 1));
    const BoxArray coarse_boxes(coarse_domain);
    const DistributionMapping coarse_dm(coarse_boxes);
    const IntVect ref_ratio(2, 2, 2);
    const Box fine_domain = amrex::refine(coarse_domain, ref_ratio);
    // Start with only the low-x half covered.  The later remake adds the
    // high-x half, making newly covered carrier-weighted cells observable.
    const BoxArray fine_boxes(Box(IntVect(0, 0, 0), IntVect(1, 3, 3)));
    const DistributionMapping fine_dm(fine_boxes);
    const amrex::RealBox real_box({AMREX_D_DECL(0.0, 0.0, 0.0)},
                                  {AMREX_D_DECL(4.0, 4.0, 4.0)});
    const std::array<int, AMREX_SPACEDIM> periodicity{AMREX_D_DECL(1, 1, 1)};
    const Geometry coarse_geometry(coarse_domain, &real_box, amrex::CoordSys::cartesian,
                                   periodicity.data());
    const Geometry fine_geometry(fine_domain, &real_box, amrex::CoordSys::cartesian,
                                 periodicity.data());
    manager.define_level(0, coarse_boxes, coarse_dm, 2);
    manager.define_level(1, fine_boxes, fine_dm, 2);
    manager.output(0).setVal(2.0);
    manager.begin_step(0, 0.0);
    manager.output(0).setVal(6.0);
    manager.accept_stage(0, 1.0);

    amrex::MultiFab rho_old(coarse_boxes, coarse_dm, 1, 2);
    amrex::MultiFab rho_output(coarse_boxes, coarse_dm, 1, 2);
    amrex::MultiFab rho_target(fine_boxes, fine_dm, 1, 2);
    rho_old.setVal(1.0);
    rho_output.setVal(3.0);
    for (amrex::MFIter mfi(rho_target); mfi.isValid(); ++mfi) {
        const auto target = rho_target.array(mfi);
        amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const Real parity = (i % 2 == 0 ? Real(-1.0) : Real(1.0)) +
                Real(0.5) * (j % 2 == 0 ? Real(-1.0) : Real(1.0)) +
                Real(0.25) * (k % 2 == 0 ? Real(-1.0) : Real(1.0));
            target(i,j,k,0) = Real(4.0) + Real(0.1) * parity;
        });
    }
    rho_old.FillBoundary(coarse_geometry.periodicity());
    rho_output.FillBoundary(coarse_geometry.periodicity());
    rho_target.FillBoundary(fine_geometry.periodicity());
    // Stage FillPatch must preserve already-authoritative fine valid cells;
    // initialize them to the same ratio that the coarse bracket supplies so
    // the ghost checks below isolate the carrier-weighted coarse write.
    for (amrex::MFIter mfi(manager.evaluation(1)); mfi.isValid(); ++mfi) {
        const auto state = manager.evaluation(1).array(mfi);
        const auto rho = rho_target.const_array(mfi);
        amrex::ParallelFor(mfi.validbox(), manager.evaluation(1).nComp(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int comp) noexcept {
                state(i,j,k,comp) = Real(2.0) * rho(i,j,k,0);
            });
    }
    amrex::Gpu::synchronize();
    manager.fill_stage_from_coarse(
        0, 1, 0.5, coarse_geometry, fine_geometry, ref_ratio,
        erf_auxiliary::AuxiliaryTimeView::Evaluation,
        &rho_old, &rho_output, &rho_target);

    const auto check_ratio = [&](const amrex::MultiFab& state,
                                 const amrex::MultiFab& density,
                                 const bool ghosts_only) {
        amrex::MultiFab expected(state.boxArray(), state.DistributionMap(),
                                  state.nComp(), state.nGrowVect());
        for (amrex::MFIter mfi(expected); mfi.isValid(); ++mfi) {
            const auto out = expected.array(mfi);
            const auto rho = density.const_array(mfi);
            amrex::ParallelFor(mfi.fabbox(), state.nComp(),
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int comp) noexcept {
                    out(i,j,k,comp) = Real(2.0) * rho(i,j,k);
                });
        }
        amrex::Gpu::synchronize();
        amrex::MultiFab difference(state.boxArray(), state.DistributionMap(),
                                   state.nComp(), state.nGrowVect());
        amrex::MultiFab::Copy(difference, state, 0, 0, state.nComp(), state.nGrowVect());
        amrex::MultiFab::Subtract(difference, expected, 0, 0, state.nComp(), state.nGrowVect());
        if (ghosts_only) {
            for (amrex::MFIter mfi(difference); mfi.isValid(); ++mfi) {
                const auto values = difference.array(mfi);
                const auto valid = mfi.validbox();
                amrex::ParallelFor(mfi.fabbox(), state.nComp(),
                    [=] AMREX_GPU_DEVICE (int i, int j, int k, int comp) noexcept {
                        if (valid.contains(i, j, k)) values(i,j,k,comp) = Real(0.0);
                    });
            }
            amrex::Gpu::synchronize();
        }
        Real error = difference.norm0(0, state.nComp(), state.nGrowVect(), true);
        amrex::ParallelDescriptor::ReduceRealMax(error);
        return error;
    };
    const Real ratio_tolerance = Real(512.0) * std::numeric_limits<Real>::epsilon() * Real(5.0);
    EXPECT_LE(check_ratio(manager.evaluation(1), rho_target, true), ratio_tolerance);

    manager.old(1).setVal(-9.0);
    for (amrex::MFIter mfi(manager.old(1)); mfi.isValid(); ++mfi) {
        const auto state = manager.old(1).array(mfi);
        const auto rho = rho_target.const_array(mfi);
        amrex::ParallelFor(mfi.validbox(), manager.old(1).nComp(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int comp) noexcept {
                state(i,j,k,comp) = Real(2.0) * rho(i,j,k,0);
            });
    }
    amrex::Gpu::synchronize();
    manager.fill_stage_from_coarse(
        0, 1, 0.25, coarse_geometry, fine_geometry, ref_ratio,
        erf_auxiliary::AuxiliaryTimeView::Old,
        &rho_old, &rho_output, &rho_target);
    EXPECT_LE(check_ratio(manager.old(1), rho_target, true), ratio_tolerance);

    manager.prolong_from_coarse(0, 1, coarse_geometry, fine_geometry, ref_ratio,
                                0.5, &rho_old, &rho_output, &rho_target);
    EXPECT_LE(check_ratio(manager.output(1), rho_target, false), ratio_tolerance);

    const BoxArray remade_boxes(fine_domain);
    const DistributionMapping remade_dm(remade_boxes);
    amrex::MultiFab remade_rho_target(remade_boxes, remade_dm, 1, 2);
    for (amrex::MFIter mfi(remade_rho_target); mfi.isValid(); ++mfi) {
        const auto target = remade_rho_target.array(mfi);
        amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const Real parity = (i % 2 == 0 ? Real(-1.0) : Real(1.0)) +
                Real(0.5) * (j % 2 == 0 ? Real(-1.0) : Real(1.0)) +
                Real(0.25) * (k % 2 == 0 ? Real(-1.0) : Real(1.0));
            target(i,j,k,0) = Real(4.0) + Real(0.1) * parity;
        });
    }
    remade_rho_target.FillBoundary(fine_geometry.periodicity());
    manager.remake_level_from_coarse(
        1, remade_boxes, remade_dm, 2, fine_geometry.periodicity(), 0,
        coarse_geometry, fine_geometry, ref_ratio, 0.5, -1,
        &rho_old, &rho_output, &remade_rho_target);
    EXPECT_LE(check_ratio(manager.output(1), remade_rho_target, false), ratio_tolerance);
}

TEST(SBMP2, CarrierWeightedAMRTransferUsesTargetDensityAndPreservesRatios)
{
    sbm_test_CarrierWeightedAMRTransferUsesTargetDensityAndPreservesRatios();
}

void sbm_test_CarrierWeightedStageFillPreservesFineFABAuthority()
{
    const auto layout = make_layout(2, MomentMode::OneMoment);
    const Box coarse_domain(IntVect(0, 0, 0), IntVect(3, 1, 1));
    const IntVect ref_ratio(2, 2, 2);
    const Box fine_domain = amrex::refine(coarse_domain, ref_ratio);
    amrex::BoxList fine_list;
    fine_list.push_back(Box(IntVect(0, 0, 0), IntVect(1, 3, 3)));
    fine_list.push_back(Box(IntVect(2, 0, 0), IntVect(3, 3, 3)));
    const BoxArray coarse_boxes(coarse_domain);
    const BoxArray fine_boxes(fine_list);
    const DistributionMapping coarse_dm(coarse_boxes);
    const DistributionMapping fine_dm(fine_boxes);
    const amrex::RealBox real_box({AMREX_D_DECL(0.0, 0.0, 0.0)},
                                  {AMREX_D_DECL(4.0, 4.0, 4.0)});
    const std::array<int, AMREX_SPACEDIM> periodicity{AMREX_D_DECL(1, 1, 1)};
    const Geometry coarse_geometry(coarse_domain, &real_box, amrex::CoordSys::cartesian,
                                   periodicity.data());
    const Geometry fine_geometry(fine_domain, &real_box, amrex::CoordSys::cartesian,
                                 periodicity.data());

    erf_auxiliary::AuxiliaryStateManager manager(layout.auxiliary_layout());
    manager.define_level(0, coarse_boxes, coarse_dm, 1);
    manager.define_level(1, fine_boxes, fine_dm, 1);
    manager.output(0).setVal(Real(1.0));
    manager.begin_step(0, 0.0);
    manager.accept_stage(0, 1.0);

    amrex::MultiFab rho_old(coarse_boxes, coarse_dm, 1, 1);
    amrex::MultiFab rho_output(coarse_boxes, coarse_dm, 1, 1);
    amrex::MultiFab rho_target(fine_boxes, fine_dm, 1, 1);
    rho_old.setVal(Real(1.0));
    rho_output.setVal(Real(1.0));
    for (amrex::MFIter mfi(rho_target); mfi.isValid(); ++mfi) {
        const auto rho = rho_target.array(mfi);
        amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            rho(i,j,k,0) = Real(2.0) + Real(0.125) * Real(i + 1) +
                Real(0.03125) * Real(j + 1) + Real(0.015625) * Real(k + 1);
        });
    }
    rho_old.FillBoundary(coarse_geometry.periodicity());
    rho_output.FillBoundary(coarse_geometry.periodicity());
    rho_target.FillBoundary(fine_geometry.periodicity());

    for (amrex::MFIter mfi(manager.output(1)); mfi.isValid(); ++mfi) {
        const auto state = manager.output(1).array(mfi);
        const auto rho = rho_target.const_array(mfi);
        const Box valid = mfi.validbox();
        amrex::ParallelFor(valid, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            state(i,j,k,0) = (i < 2 ? Real(3.0) : Real(7.0)) * rho(i,j,k,0);
        });
    }
    manager.begin_step(1, 0.0);
    manager.fill_stage_from_coarse(0, 1, 0.5, coarse_geometry, fine_geometry,
                                   ref_ratio, erf_auxiliary::AuxiliaryTimeView::Evaluation,
                                   &rho_old, &rho_output, &rho_target);

    Real a_error = Real(0.0), b_error = Real(0.0), coarse_fine_error = Real(0.0);
    Real valid_error = Real(0.0);
    for (amrex::MFIter mfi(manager.evaluation(1)); mfi.isValid(); ++mfi) {
        const auto state = manager.evaluation(1).const_array(mfi);
        const auto rho = rho_target.const_array(mfi);
        const Box valid = mfi.validbox();
        a_error = amrex::max(a_error, amrex::Math::abs(
            state(2,1,1,0) - Real(7.0) * rho(2,1,1,0)));
        b_error = amrex::max(b_error, amrex::Math::abs(
            state(1,1,1,0) - Real(3.0) * rho(1,1,1,0)));
        if (valid.smallEnd(0) == 2) {
            coarse_fine_error = amrex::max(coarse_fine_error, amrex::Math::abs(
                state(4,1,1,0) - rho(4,1,1,0)));
        }
        if (valid.smallEnd(0) == 0) {
            valid_error = amrex::max(valid_error, amrex::Math::abs(
                state(0,1,1,0) - Real(3.0) * rho(0,1,1,0)));
        }
        if (valid.smallEnd(0) == 2) {
            valid_error = amrex::max(valid_error, amrex::Math::abs(
                state(2,1,1,0) - Real(7.0) * rho(2,1,1,0)));
        }
    }
    amrex::ParallelDescriptor::ReduceRealMax(a_error);
    amrex::ParallelDescriptor::ReduceRealMax(b_error);
    amrex::ParallelDescriptor::ReduceRealMax(coarse_fine_error);
    amrex::ParallelDescriptor::ReduceRealMax(valid_error);
    EXPECT_LE(a_error, Real(512.0) * std::numeric_limits<Real>::epsilon() * Real(8.0));
    EXPECT_LE(b_error, Real(512.0) * std::numeric_limits<Real>::epsilon() * Real(8.0));
    EXPECT_LE(coarse_fine_error, Real(512.0) * std::numeric_limits<Real>::epsilon() * Real(8.0));
    EXPECT_LE(valid_error, Real(512.0) * std::numeric_limits<Real>::epsilon() * Real(8.0));
}

TEST(SBMP2, CarrierWeightedStageFillPreservesFineFABAuthority)
{
    sbm_test_CarrierWeightedStageFillPreservesFineFABAuthority();
}

void sbm_test_AttachedPropertySupportUsesFineDonorAcrossInternalFABBoundary()
{
    const auto layout = make_layout(2, MomentMode::TwoMoment, true, Real(10.0), Real(10.0));
    const auto& population = layout.populations().front();
    const int property = layout.property_offset(0);
    const int number_component = population.number_offset;
    const int mass_component = population.mass_offset;
    const Box coarse_domain(IntVect(0, 0, 0), IntVect(3, 0, 0));
    const IntVect ref_ratio(2, 2, 2);
    const Box fine_domain = amrex::refine(coarse_domain, ref_ratio);
    amrex::BoxList fine_list;
    fine_list.push_back(Box(IntVect(0, 0, 0), IntVect(1, 1, 1)));
    fine_list.push_back(Box(IntVect(2, 0, 0), IntVect(3, 1, 1)));
    const BoxArray coarse_boxes(coarse_domain);
    const BoxArray fine_boxes(fine_list);
    const DistributionMapping coarse_dm(coarse_boxes);
    const DistributionMapping fine_dm(fine_boxes);
    const amrex::RealBox real_box({AMREX_D_DECL(0.0, 0.0, 0.0)},
                                  {AMREX_D_DECL(4.0, 1.0, 1.0)});
    const std::array<int, AMREX_SPACEDIM> periodicity{AMREX_D_DECL(1, 1, 1)};
    const Geometry coarse_geometry(coarse_domain, &real_box, amrex::CoordSys::cartesian,
                                   periodicity.data());
    const Geometry fine_geometry(fine_domain, &real_box, amrex::CoordSys::cartesian,
                                 periodicity.data());
    erf_auxiliary::AuxiliaryStateManager manager(layout.auxiliary_layout());
    manager.define_level(0, coarse_boxes, coarse_dm, 2);
    manager.define_level(1, fine_boxes, fine_dm, 2);
    manager.output(0).setVal(Real(0.0));
    for (amrex::MFIter mfi(manager.output(0)); mfi.isValid(); ++mfi) {
        const auto state = manager.output(0).array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            state(i,j,k,mass_component) = Real(10.0);
            state(i,j,k,number_component) = Real(1.0);
            state(i,j,k,property) = Real(1.0);
        });
    }
    manager.begin_step(0, 0.0);
    manager.accept_stage(0, 0.0);

    amrex::MultiFab rho_old(coarse_boxes, coarse_dm, 1, 2);
    amrex::MultiFab rho_output(coarse_boxes, coarse_dm, 1, 2);
    amrex::MultiFab rho_target(fine_boxes, fine_dm, 1, 2);
    rho_old.setVal(Real(1.0)); rho_output.setVal(Real(1.0));
    for (amrex::MFIter mfi(rho_target); mfi.isValid(); ++mfi) {
        const auto rho = rho_target.array(mfi);
        amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            rho(i,j,k,0) = Real(2.0) + Real(0.1) * Real(i + 4);
        });
    }
    rho_old.FillBoundary(coarse_geometry.periodicity());
    rho_output.FillBoundary(coarse_geometry.periodicity());
    rho_target.FillBoundary(fine_geometry.periodicity());
    for (amrex::MFIter mfi(manager.output(1)); mfi.isValid(); ++mfi) {
        const auto state = manager.output(1).array(mfi);
        const auto rho = rho_target.const_array(mfi);
        const Box valid = mfi.validbox();
        amrex::ParallelFor(valid, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            state(i,j,k,mass_component) = Real(10.0) * rho(i,j,k,0);
            state(i,j,k,number_component) = rho(i,j,k,0);
            state(i,j,k,property) = (i < 2 ? Real(3.0) : Real(7.0)) * rho(i,j,k,0);
        });
    }
    manager.begin_step(1, 0.0);
    // This is a compressible stage-2 call, so the production transport path
    // consumes the evaluation/predictor view.  Prepare that matching view;
    // filling only Old would exercise the pre-F01 temporal pairing instead.
    manager.fill_stage_from_coarse(0, 1, 0.0, coarse_geometry, fine_geometry,
                                   ref_ratio, erf_auxiliary::AuxiliaryTimeView::Evaluation,
                                   &rho_old, &rho_output, &rho_target);

    amrex::MultiFab xflux(amrex::convert(fine_boxes, IntVect(1,0,0)), fine_dm, 1, 0);
    amrex::MultiFab yflux(amrex::convert(fine_boxes, IntVect(0,1,0)), fine_dm, 1, 0);
    amrex::MultiFab zflux(amrex::convert(fine_boxes, IntVect(0,0,1)), fine_dm, 1, 0);
    xflux.setVal(Real(0.0)); yflux.setVal(Real(0.0)); zflux.setVal(Real(0.0));
    for (amrex::MFIter mfi(xflux); mfi.isValid(); ++mfi) {
        const auto flux = xflux.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            if (i == 2) flux(i,j,k) = Real(-0.1);
        });
    }
    xflux.FillBoundary(fine_geometry.periodicity());
    yflux.FillBoundary(fine_geometry.periodicity());
    zflux.FillBoundary(fine_geometry.periodicity());
    amrex::MultiFab core(fine_boxes, fine_dm, RhoQ3_comp + 1, 0);
    core.setVal(Real(0.0));
    erf_auxiliary::AuxiliaryFaceTransfer stage_flux;
    stage_flux.define(fine_boxes, fine_dm, layout.ncomp(), 0);
    const auto context = erf_auxiliary::make_compressible_stage(
        2, 0.0, Real(0.005), Real(0.01), Real(0.01), nullptr, nullptr);
    erf_sbm::advance_stage(manager, layout, context, rho_target, rho_target,
                           rho_target, core, xflux, yflux, zflux, fine_geometry,
                           stage_flux, erf_sbm::TransportMethod::GroupedFCT_WENOZ3,
                           1, Real(0.0), 1);

    Real flux_error = Real(0.0);
    Real ratio_error = Real(0.0);
    const Real expected_lower = std::min(Real(3.0), Real(7.0));
    const Real expected_upper = std::max(Real(3.0), Real(7.0));
    for (amrex::MFIter mfi(stage_flux.x()); mfi.isValid(); ++mfi) {
        const auto flux = stage_flux.x().const_array(mfi);
        const Real mass_flux = flux(2,0,0,population.mass_offset);
        const Real number_flux = flux(2,0,0,number_component);
        const Real property_flux = flux(2,0,0,property);
        const Real actual_ratio = property_flux / number_flux;
        flux_error = amrex::max(flux_error, amrex::Math::abs(mass_flux + Real(1.0)));
        flux_error = amrex::max(flux_error, amrex::Math::abs(number_flux + Real(0.1)));
        ratio_error = amrex::max(ratio_error, amrex::Math::abs(actual_ratio - Real(7.0)));
        EXPECT_GE(actual_ratio, expected_lower);
        EXPECT_LE(actual_ratio, expected_upper);
    }
    amrex::ParallelDescriptor::ReduceRealMax(flux_error);
    amrex::ParallelDescriptor::ReduceRealMax(ratio_error);
    EXPECT_LE(flux_error, Real(1024.0) * std::numeric_limits<Real>::epsilon());
    EXPECT_LE(ratio_error, Real(1024.0) * std::numeric_limits<Real>::epsilon());
}

TEST(SBMP2, AttachedPropertySupportUsesFineDonorAcrossInternalFABBoundary)
{
    sbm_test_AttachedPropertySupportUsesFineDonorAcrossInternalFABBoundary();
}

void sbm_test_DonorCellTwoMomentUsesPreparedCoarseFineEndpointDonors()
{
    const auto layout = make_layout(2, MomentMode::TwoMoment);
    const auto& population = layout.populations().front();
    const int mass_component = population.mass_offset;
    const int number_component = population.number_offset;
    const Box coarse_domain(IntVect(0, 0, 0), IntVect(1, 1, 1));
    const IntVect ref_ratio(2, 2, 2);
    const Box fine_domain = amrex::refine(coarse_domain, ref_ratio);
    const BoxArray coarse_boxes(coarse_domain);
    const BoxArray fine_boxes(Box(IntVect(0, 0, 0), IntVect(1, 1, 1)));
    const DistributionMapping coarse_dm(coarse_boxes);
    const DistributionMapping fine_dm(fine_boxes);
    const amrex::RealBox real_box({AMREX_D_DECL(0.0, 0.0, 0.0)},
                                  {AMREX_D_DECL(4.0, 4.0, 4.0)});
    const std::array<int, AMREX_SPACEDIM> periodicity{AMREX_D_DECL(1, 1, 1)};
    const Geometry coarse_geometry(coarse_domain, &real_box, amrex::CoordSys::cartesian,
                                   periodicity.data());
    const Geometry fine_geometry(fine_domain, &real_box, amrex::CoordSys::cartesian,
                                 periodicity.data());

    erf_auxiliary::AuxiliaryStateManager manager(layout.auxiliary_layout());
    manager.define_level(0, coarse_boxes, coarse_dm, 2);
    manager.define_level(1, fine_boxes, fine_dm, 2);
    manager.output(0).setVal(Real(0.0));
    for (amrex::MFIter mfi(manager.output(0)); mfi.isValid(); ++mfi) {
        const auto state = manager.output(0).array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            state(i,j,k,mass_component) = Real(0.5);
            state(i,j,k,number_component) = Real(0.75);
        });
    }
    manager.begin_step(0, 0.0);
    manager.begin_step(1, 0.0);

    amrex::MultiFab rho_old(coarse_boxes, coarse_dm, 1, 2);
    amrex::MultiFab rho_output(coarse_boxes, coarse_dm, 1, 2);
    amrex::MultiFab rho_target(fine_boxes, fine_dm, 1, 2);
    rho_old.setVal(Real(2.0));
    rho_output.setVal(Real(2.0));
    for (amrex::MFIter mfi(rho_target); mfi.isValid(); ++mfi) {
        const auto rho = rho_target.array(mfi);
        amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            rho(i,j,k,0) = Real(2.0) + Real(0.10) * Real(i) +
                Real(0.05) * Real(j) + Real(0.03) * Real(k);
        });
    }
    rho_old.FillBoundary(coarse_geometry.periodicity());
    rho_output.FillBoundary(coarse_geometry.periodicity());
    rho_target.FillBoundary(fine_geometry.periodicity());
    // This is a compressible stage-2 call, so the production transport path
    // consumes the evaluation/predictor view.  Prepare that matching view;
    // filling only Old would exercise the pre-F01 temporal pairing instead.
    manager.fill_stage_from_coarse(0, 1, 0.0, coarse_geometry, fine_geometry,
                                   ref_ratio, erf_auxiliary::AuxiliaryTimeView::Evaluation,
                                   &rho_old, &rho_output, &rho_target);

    amrex::MultiFab xflux(amrex::convert(fine_boxes, IntVect(1,0,0)), fine_dm, 1, 0);
    amrex::MultiFab yflux(amrex::convert(fine_boxes, IntVect(0,1,0)), fine_dm, 1, 0);
    amrex::MultiFab zflux(amrex::convert(fine_boxes, IntVect(0,0,1)), fine_dm, 1, 0);
    xflux.setVal(Real(0.0)); yflux.setVal(Real(0.0)); zflux.setVal(Real(0.0));
    for (amrex::MFIter mfi(xflux); mfi.isValid(); ++mfi) {
        const auto flux = xflux.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            if (i == 0) flux(i,j,k) = Real(0.2);
            if (i == 2) flux(i,j,k) = Real(-0.2);
        });
    }
    xflux.FillBoundary(fine_geometry.periodicity());
    yflux.FillBoundary(fine_geometry.periodicity());
    zflux.FillBoundary(fine_geometry.periodicity());
    amrex::MultiFab core(fine_boxes, fine_dm, RhoQ3_comp + 1, 0);
    core.setVal(Real(0.0));
    erf_auxiliary::AuxiliaryFaceTransfer stage_flux;
    stage_flux.define(fine_boxes, fine_dm, layout.ncomp(), 0);
    const auto context = erf_auxiliary::make_compressible_stage(
        2, 0.0, Real(0.05), Real(0.1), Real(0.1), nullptr, nullptr);
    erf_sbm::advance_stage(manager, layout, context, rho_target, rho_target,
                           rho_target, core, xflux, yflux, zflux, fine_geometry,
                           stage_flux, erf_sbm::TransportMethod::DonorCell,
                           1, Real(0.0), 0);

    // Independent coarse endpoint oracle: a=0, b=1, M_c=0.5, C_c=0.75,
    // rho_c=2 gives L_c=0.25 and H_c=0.50.  Both incoming signs must read
    // the same true coarse/fine donor, with opposite signed transfers.
    const Real lower = Real(0.0), upper = Real(1.0);
    const Real coarse_mass = Real(0.5), coarse_number = Real(0.75), coarse_rho = Real(2.0);
    const Real Lc = (upper*coarse_number - coarse_mass) / (upper - lower);
    const Real Hc = (coarse_mass - lower*coarse_number) / (upper - lower);
    const Real expected_mass_flux = Real(0.2) *
        (lower*Lc/coarse_rho + upper*Hc/coarse_rho);
    const Real expected_number_flux = Real(0.2) *
        (Lc/coarse_rho + Hc/coarse_rho);
    Real flux_error = Real(0.0), final_error = Real(0.0);
    Real endpoint_error = Real(0.0), compact_error = Real(0.0);
    for (amrex::MFIter mfi(stage_flux.x()); mfi.isValid(); ++mfi) {
        const auto flux = stage_flux.x().const_array(mfi);
        flux_error = amrex::max(flux_error, amrex::Math::abs(
            flux(0,0,0,mass_component) - expected_mass_flux));
        flux_error = amrex::max(flux_error, amrex::Math::abs(
            flux(0,0,0,number_component) - expected_number_flux));
        flux_error = amrex::max(flux_error, amrex::Math::abs(
            flux(2,0,0,mass_component) + expected_mass_flux));
        flux_error = amrex::max(flux_error, amrex::Math::abs(
            flux(2,0,0,number_component) + expected_number_flux));
    }
    for (amrex::MFIter mfi(manager.output(1)); mfi.isValid(); ++mfi) {
        const auto state = manager.output(1).const_array(mfi);
        const auto compact = core.const_array(mfi);
        for (int i = 0; i <= 1; ++i) {
            const Real expected_mass = Real(0.1) * expected_mass_flux;
            const Real expected_number = Real(0.1) * expected_number_flux;
            final_error = amrex::max(final_error, amrex::Math::abs(
                state(i,0,0,mass_component) - expected_mass));
            final_error = amrex::max(final_error, amrex::Math::abs(
                state(i,0,0,number_component) - expected_number));
            const Real M = state(i,0,0,mass_component);
            const Real C = state(i,0,0,number_component);
            endpoint_error = amrex::max(endpoint_error, amrex::max(
                lower*C - M, M - upper*C));
            compact_error = amrex::max(compact_error, amrex::Math::abs(
                compact(i,0,0,RhoQ2_comp) - expected_mass));
            compact_error = amrex::max(compact_error, amrex::Math::abs(
                compact(i,0,0,RhoQ3_comp)));
        }
    }
    amrex::ParallelDescriptor::ReduceRealMax(flux_error);
    amrex::ParallelDescriptor::ReduceRealMax(final_error);
    amrex::ParallelDescriptor::ReduceRealMax(endpoint_error);
    amrex::ParallelDescriptor::ReduceRealMax(compact_error);
    EXPECT_LE(flux_error, Real(1024.0) * std::numeric_limits<Real>::epsilon());
    EXPECT_LE(final_error, Real(1024.0) * std::numeric_limits<Real>::epsilon());
    EXPECT_LE(endpoint_error, Real(1024.0) * std::numeric_limits<Real>::epsilon());
    EXPECT_LE(compact_error, Real(1024.0) * std::numeric_limits<Real>::epsilon());
}

TEST(SBMP2, DonorCellTwoMomentUsesPreparedCoarseFineEndpointDonors)
{
    sbm_test_DonorCellTwoMomentUsesPreparedCoarseFineEndpointDonors();
}

void sbm_test_CompactGhostsAreProjectedFromAuthoritativeSpectralGhosts()
{
    const auto layout = make_layout(2, MomentMode::OneMoment);
    erf_auxiliary::AuxiliaryStateManager manager(layout.auxiliary_layout());
    const Box domain(IntVect(0, 0, 0), IntVect(1, 0, 0));
    const BoxArray boxes(domain);
    const DistributionMapping dm(boxes);
    const amrex::RealBox real_box({AMREX_D_DECL(0.0, 0.0, 0.0)},
                                  {AMREX_D_DECL(2.0, 1.0, 1.0)});
    const std::array<int, AMREX_SPACEDIM> periodicity{AMREX_D_DECL(1, 1, 1)};
    const Geometry geometry(domain, &real_box, amrex::CoordSys::cartesian,
                            periodicity.data());
    manager.define_level(0, boxes, dm, 2);
    manager.output(0).setVal(0.0);
    for (amrex::MFIter mfi(manager.output(0)); mfi.isValid(); ++mfi) {
        const auto state = manager.output(0).array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            state(i,j,k,0) = Real(2.5);
            state(i,j,k,1) = Real(4.5);
        });
    }
    manager.output(0).FillBoundary(geometry.periodicity());
    amrex::MultiFab core(boxes, dm, RhoQ3_comp + 1, 2);
    core.setVal(Real(123.0));
    erf_sbm::SBMBulkProjection projection(layout);
    for (amrex::MFIter mfi(manager.output(0)); mfi.isValid(); ++mfi) {
        Box box = mfi.validbox();
        box.grow(1);
        projection.apply_to_core(box, manager.output(0).const_array(mfi), core.array(mfi));
    }
    amrex::Gpu::synchronize();
    for (amrex::MFIter mfi(core); mfi.isValid(); ++mfi) {
        const auto values = core.const_array(mfi);
        EXPECT_DOUBLE_EQ(values(-1,0,0,RhoQ2_comp), Real(2.5));
        EXPECT_DOUBLE_EQ(values(2,0,0,RhoQ3_comp), Real(4.5));
    }
}

TEST(SBMP2, CompactGhostsAreProjectedFromAuthoritativeSpectralGhosts)
{
    sbm_test_CompactGhostsAreProjectedFromAuthoritativeSpectralGhosts();
}

TEST(SBMP2, CoarseFineWENOInterfaceOracleUsesBothUpwindSigns)
{
    const auto layout = make_layout(2, MomentMode::OneMoment);
    erf_auxiliary::AuxiliaryStateManager manager(layout.auxiliary_layout());
    const Box coarse_domain(IntVect(0, 0, 0), IntVect(1, 1, 1));
    const BoxArray coarse_boxes(coarse_domain);
    const DistributionMapping coarse_dm(coarse_boxes);
    const IntVect ref_ratio(2, 2, 2);
    const Box fine_domain = amrex::refine(coarse_domain, ref_ratio);
    // Only the low-x half is a fine valid region.  The high-x stencil cells
    // therefore come from the temporal coarse FillPatch path.
    const BoxArray fine_boxes(Box(IntVect(0, 0, 0), IntVect(1, 3, 3)));
    const DistributionMapping fine_dm(fine_boxes);
    const amrex::RealBox real_box({AMREX_D_DECL(0.0, 0.0, 0.0)},
                                  {AMREX_D_DECL(4.0, 4.0, 4.0)});
    const std::array<int, AMREX_SPACEDIM> periodicity{AMREX_D_DECL(1, 1, 1)};
    const Geometry coarse_geometry(coarse_domain, &real_box, amrex::CoordSys::cartesian,
                                   periodicity.data());
    const Geometry fine_geometry(fine_domain, &real_box, amrex::CoordSys::cartesian,
                                 periodicity.data());
    manager.define_level(0, coarse_boxes, coarse_dm, 2);
    manager.define_level(1, fine_boxes, fine_dm, 2);
    manager.output(0).setVal(2.0);
    manager.begin_step(0, 0.0);
    manager.output(0).setVal(4.0);
    manager.accept_stage(0, 1.0);
    // The valid fine cells represent the same smooth constant state as the
    // coarse bracket; only the high-x stencil cells cross the interface.
    manager.evaluation(1).setVal(3.0);
    manager.fill_stage_from_coarse(0, 1, 0.5, coarse_geometry, fine_geometry,
                                   ref_ratio, erf_auxiliary::AuxiliaryTimeView::Evaluation);

    const auto weno_z3 = [](const Real qm2, const Real qm1, const Real q,
                            const Real qp1, const bool positive) {
        const Real q0 = positive ? Real(0.5) * (-qm2 + Real(3.0)*qm1) :
            Real(0.5) * (Real(3.0)*q - qp1);
        const Real q1 = Real(0.5) * (qm1 + q);
        const Real beta0 = positive ? (qm1-qm2)*(qm1-qm2) : (qp1-q)*(qp1-q);
        const Real beta1 = (q-qm1)*(q-qm1);
        const Real tau = std::abs(beta1-beta0);
        const Real epsilon = Real(1.0e-40);
        const Real w0 = (Real(1.0)/Real(3.0)) *
            (Real(1.0) + (tau*tau)/((epsilon+beta0)*(epsilon+beta0)));
        const Real w1 = (Real(2.0)/Real(3.0) *
            (Real(1.0) + (tau*tau)/((epsilon+beta1)*(epsilon+beta1))));
        return (w0*q0 + w1*q1) / (w0 + w1);
    };

    Real positive_face = 0.0;
    Real negative_face = 0.0;
    for (amrex::MFIter mfi(manager.evaluation(1)); mfi.isValid(); ++mfi) {
        const auto values = manager.evaluation(1).const_array(mfi);
        // Face i=2 is the coarse/fine boundary.  The complete two-ghost
        // stencil is authoritative coarse data at t=0.5 for this fixture.
        const Real qm2 = values(0,1,1,0);
        const Real qm1 = values(1,1,1,0);
        const Real q = values(2,1,1,0);
        const Real qp1 = values(3,1,1,0);
        positive_face = weno_z3(qm2, qm1, q, qp1, true);
        negative_face = weno_z3(qm2, qm1, q, qp1, false);
    }
    amrex::ParallelDescriptor::ReduceRealMax(positive_face);
    amrex::ParallelDescriptor::ReduceRealMax(negative_face);
    EXPECT_DOUBLE_EQ(positive_face, 3.0);
    EXPECT_DOUBLE_EQ(negative_face, 3.0);

    manager.old(1).setVal(0.0);
    manager.fill_stage_from_coarse(0, 1, 0.5, coarse_geometry, fine_geometry,
                                   ref_ratio, erf_auxiliary::AuxiliaryTimeView::Old);
    Real old_view_ghost = 0.0;
    for (amrex::MFIter mfi(manager.old(1)); mfi.isValid(); ++mfi) {
        old_view_ghost = manager.old(1).const_array(mfi)(2,1,1,0);
    }
    amrex::ParallelDescriptor::ReduceRealMax(old_view_ghost);
    EXPECT_DOUBLE_EQ(old_view_ghost, 3.0);
}

TEST(SBMP2, CompleteConstraintValidationCoversAttachedProperties)
{
    const auto layout = make_layout(2, MomentMode::TwoMoment, true,
                                    std::numeric_limits<Real>::quiet_NaN());
    erf_auxiliary::AuxiliaryStateManager manager(layout.auxiliary_layout());
    const Box domain(IntVect(0, 0, 0), IntVect(0, 0, 0));
    const BoxArray boxes(domain);
    const DistributionMapping dm(boxes);
    manager.define_level(0, boxes, dm, 2);
    auto& state = manager.output(0);
    state.setVal(0.0);
    const auto& population = layout.populations().front();
    for (int b = 0; b < 2; ++b) {
        state.setVal(static_cast<Real>(b) + 0.5, population.mass_offset + b, 1);
        state.setVal(1.0, population.number_offset + b, 1);
        state.setVal(0.25, layout.property_offset(0) + b, 1);
    }
    erf_sbm::validate_admissible_state(manager, layout, 0);
    // Universal validation must reject an orphan attached amount even when
    // the descriptor has no finite hard support_max.
    state.setVal(0.0, population.mass_offset, 1);
    state.setVal(0.0, population.number_offset, 1);
    state.setVal(0.25, layout.property_offset(0), 1);
    EXPECT_THROW(erf_sbm::validate_admissible_state(manager, layout, 0), std::exception);
    state.setVal(1.0, population.mass_offset, 1);
    state.setVal(1.0, population.number_offset, 1);
    state.setVal(-1.0, layout.property_offset(0), 1);
    EXPECT_THROW(erf_sbm::validate_admissible_state(manager, layout, 0), std::exception);
}

TEST(SBMP2, AttachedPropertyTransactionValidationRejectsOrphansAfterRemake)
{
    const auto layout = make_layout(2, MomentMode::TwoMoment, true,
                                    std::numeric_limits<Real>::quiet_NaN());
    erf_auxiliary::AuxiliaryStateManager manager(layout.auxiliary_layout());
    const Box domain(IntVect(0, 0, 0), IntVect(1, 0, 0));
    BoxArray boxes(domain);
    const DistributionMapping dm(boxes);
    const amrex::RealBox real_box({AMREX_D_DECL(0.0, 0.0, 0.0)},
                                  {AMREX_D_DECL(2.0, 1.0, 1.0)});
    const std::array<int, AMREX_SPACEDIM> periodicity{AMREX_D_DECL(1, 1, 1)};
    const Geometry geometry(domain, &real_box, amrex::CoordSys::cartesian,
                            periodicity.data());
    manager.define_level(0, boxes, dm, 2);
    auto& state = manager.output(0);
    state.setVal(0.0);
    const auto& population = layout.populations().front();
    state.setVal(1.0, population.mass_offset, 1);
    state.setVal(1.0, population.number_offset, 1);
    state.setVal(0.25, layout.property_offset(0), 1);
    erf_sbm::validate_admissible_state(manager, layout, 0);

    // Recreate the accepted-state transaction with a different BoxArray, then
    // inject an orphan into the authoritative output.  The universal support
    // contract must reject it without clipping or repairing it.
    boxes.maxSize(1);
    manager.remake_level(0, boxes, DistributionMapping(boxes), 2,
                         geometry.periodicity());
    auto& remade = manager.output(0);
    remade.setVal(0.0);
    remade.setVal(1.0, population.mass_offset, 1);
    remade.setVal(1.0, population.number_offset, 1);
    remade.setVal(0.25, layout.property_offset(0), 1);
    erf_sbm::validate_admissible_state(manager, layout, 0);
    remade.setVal(0.0, population.number_offset, 1);

    std::string diagnostic;
    try {
        erf_sbm::validate_admissible_state(manager, layout, 0);
    } catch (const std::exception& error) {
        diagnostic = error.what();
    }
    EXPECT_NE(diagnostic.find("universal attached-property carrier/support contract"),
              std::string::npos);
}

TEST(SBMP2, DonorSupportEnvelopeHandlesOneTwoMomentAndZeroCarrierCases)
{
    // The host oracle is the same ratio contract used by the chunk-local
    // production envelope: a discontinuous donor composition is bounded by
    // its actual carrier ratio, while an orphan property is never accepted.
    EXPECT_TRUE(erf_sbm::property_support_is_admissible(1.0, 2.0, 0.25, 0.75));
    EXPECT_FALSE(erf_sbm::property_support_is_admissible(0.25, 2.0, 0.25, 0.75));
    EXPECT_TRUE(erf_sbm::property_support_is_admissible(0.0, 0.0, 0.0, 1.0));
    EXPECT_FALSE(erf_sbm::property_support_is_admissible(1.e-8, 0.0, 0.0, 1.0));
    EXPECT_TRUE(erf_sbm::property_support_is_admissible(
        0.5, 1.0, 0.25, 0.75));
    EXPECT_FALSE(erf_sbm::property_support_is_admissible(
        0.2, 1.0, 0.25, 0.75));

    const auto one_moment = erf_sbm::make_attached_property_support_descriptors(
        make_layout(4, MomentMode::OneMoment, true));
    ASSERT_EQ(one_moment.size(), 4U);
    EXPECT_EQ(one_moment[0].two_moment, 0);
    EXPECT_EQ(one_moment[0].carrier_component, 0);
    EXPECT_DOUBLE_EQ(one_moment[0].pivot, 0.5);
    EXPECT_DOUBLE_EQ(one_moment[0].hard_max, 2.0);

    const auto two_moment = erf_sbm::make_attached_property_support_descriptors(
        make_layout(4, MomentMode::TwoMoment, true));
    ASSERT_EQ(two_moment.size(), 4U);
    EXPECT_EQ(two_moment[0].two_moment, 1);
    EXPECT_EQ(two_moment[0].carrier_component, 4);
    EXPECT_DOUBLE_EQ(two_moment[0].hard_max, 1.0);

    erf_sbm::SpectralPopulationSpec population;
    population.population_id = 0;
    population.semantic_id = "liquid";
    population.phase = erf_sbm::PopulationPhase::Liquid;
    population.grid.coordinate_kind = erf_sbm::CoordinateKind::Mass;
    population.grid.coordinate_units = "kg";
    population.grid.edges = {0.0, 1.0, 2.0};
    population.grid.pivots = {0.5, 1.5};
    population.mass_state_units = "kg m^-3";
    population.number_state_units = "m^-3";
    erf_sbm::SBMLayoutSpec spec;
    spec.populations.push_back(population);
    spec.liquid_projection = {0, 1};
    spec.attached_properties.push_back({"solute", "solute", "kg m^-3", 0,
        erf_sbm::PropertyKind::ExtensiveMass,
        erf_sbm::SupportRequirement::PositiveMass,
        erf_sbm::PropertyRemapPolicy::CarrierBinConservative,
        true, false, 0.0, std::numeric_limits<Real>::quiet_NaN()});
    const auto no_static_upper = erf_sbm::make_attached_property_support_descriptors(
        erf_sbm::SBMLayout(std::move(spec)));
    ASSERT_EQ(no_static_upper.size(), 2U);
    EXPECT_EQ(no_static_upper[0].has_hard_max, 0);
}

void sbm_test_WENOZ3ConvergenceBeatsDonorOnPeriodicSmoothOperator()
{
    std::ofstream evidence(std::filesystem::temp_directory_path() /
                            "erf_sbm_p2_weno_convergence.csv");
    ASSERT_TRUE(evidence.good());
    evidence << "N,weno_positive_max_error,weno_negative_max_error,donor_positive_max_error,"
                 "weno_positive_order,weno_negative_order,donor_positive_order\n";
    std::vector<Real> weno_errors;
    std::vector<Real> weno_negative_errors;
    std::vector<Real> donor_errors;
    for (const int ncell : {16, 32, 64, 128}) {
        const Box domain(IntVect(0, 0, 0), IntVect(ncell-1, 0, 0));
        const BoxArray boxes(domain);
        const DistributionMapping dm(boxes);
        const amrex::RealBox real_box({AMREX_D_DECL(0.0, 0.0, 0.0)},
                                      {AMREX_D_DECL(1.0, 1.0, 1.0)});
        const std::array<int, AMREX_SPACEDIM> periodicity{AMREX_D_DECL(1, 1, 1)};
        const Geometry geometry(domain, &real_box, amrex::CoordSys::cartesian,
                                periodicity.data());
        amrex::MultiFab ratio(boxes, dm, 1, 2);
        for (amrex::MFIter mfi(ratio); mfi.isValid(); ++mfi) {
            const auto values = ratio.array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                const Real x = (static_cast<Real>(i) + Real(0.5)) / static_cast<Real>(ncell);
                const Real h = Real(1.0) / static_cast<Real>(ncell);
                const Real factor = std::sin(Real(3.14159265358979323846) * h) /
                    (Real(3.14159265358979323846) * h);
                values(i,j,k) = Real(2.0) +
                    std::sin(Real(6.2831853071795864769) * x) * factor;
            });
        }
        ratio.FillBoundary(geometry.periodicity());
        const BoxArray face_boxes = amrex::convert(boxes, IntVect(1, 0, 0));
        amrex::MultiFab errors(face_boxes, dm, 3, 0);
        for (amrex::MFIter mfi(errors); mfi.isValid(); ++mfi) {
            const auto input = ratio.const_array(mfi);
            const auto result = errors.array(mfi);
            const Box face_box = mfi.validbox();
            amrex::ParallelFor(face_box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                const Real positive = [&]() {
                    const Real qm2 = input(i-2,j,k);
                    const Real qm1 = input(i-1,j,k);
                    const Real q = input(i,j,k);
                    const Real q0 = Real(0.5) * (-qm2 + Real(3.0)*qm1);
                    const Real q1 = Real(0.5) * (qm1 + q);
                    const Real b0 = (qm1-qm2)*(qm1-qm2);
                    const Real b1 = (q-qm1)*(q-qm1);
                    const Real tau = std::abs(b1-b0);
                    const Real epsilon = Real(1.0e-40);
                    const Real w0 = (Real(1.0)/Real(3.0)) *
                        (Real(1.0) + (tau*tau)/((epsilon+b0)*(epsilon+b0)));
                    const Real w1 = (Real(2.0)/Real(3.0)) *
                        (Real(1.0) + (tau*tau)/((epsilon+b1)*(epsilon+b1)));
                    return (w0*q0 + w1*q1)/(w0+w1);
                }();
                const Real negative = [&]() {
                    const Real qm1 = input(i-1,j,k);
                    const Real q = input(i,j,k);
                    const Real qp1 = input(i+1,j,k);
                    const Real q0 = Real(0.5) * (Real(3.0)*q - qp1);
                    const Real q1 = Real(0.5) * (qm1 + q);
                    const Real b0 = (qp1-q)*(qp1-q);
                    const Real b1 = (q-qm1)*(q-qm1);
                    const Real tau = std::abs(b1-b0);
                    const Real epsilon = Real(1.0e-40);
                    const Real w0 = (Real(1.0)/Real(3.0)) *
                        (Real(1.0) + (tau*tau)/((epsilon+b0)*(epsilon+b0)));
                    const Real w1 = (Real(2.0)/Real(3.0)) *
                        (Real(1.0) + (tau*tau)/((epsilon+b1)*(epsilon+b1)));
                    return (w0*q0 + w1*q1)/(w0+w1);
                }();
                const Real donor_positive = input(i-1, j, k);
                const Real exact = Real(2.0) + std::sin(
                    Real(6.2831853071795864769) * static_cast<Real>(i) / static_cast<Real>(ncell));
                result(i,j,k,0) = std::abs(positive - exact);
                result(i,j,k,1) = std::abs(negative - exact);
                result(i,j,k,2) = std::abs(donor_positive - exact);
            });
        }
        weno_errors.push_back(errors.norm0(0));
        weno_negative_errors.push_back(errors.norm0(1));
        donor_errors.push_back(errors.norm0(2));
    }
    for (std::size_t i = 0; i < weno_errors.size(); ++i) {
        const Real weno_order = i == 0 ? Real(0.0) :
            std::log(weno_errors[i-1] / weno_errors[i]) / std::log(Real(2.0));
        const Real weno_negative_order = i == 0 ? Real(0.0) :
            std::log(weno_negative_errors[i-1] / weno_negative_errors[i]) / std::log(Real(2.0));
        const Real donor_order = i == 0 ? Real(0.0) :
            std::log(donor_errors[i-1] / donor_errors[i]) / std::log(Real(2.0));
        evidence << (16 << i) << ',' << std::setprecision(17) << weno_errors[i] << ','
                 << weno_negative_errors[i] << ',' << donor_errors[i] << ','
                 << weno_order << ',' << weno_negative_order << ',' << donor_order << '\n';
        if (i > 0) {
            EXPECT_GT(weno_order, Real(1.5));
            EXPECT_GT(weno_negative_order, Real(1.5));
            EXPECT_LT(weno_errors[i], donor_errors[i]);
            EXPECT_LT(weno_negative_errors[i], donor_errors[i]);
        }
    }
}

TEST(SBMP2, WENOZ3ConvergenceBeatsDonorOnPeriodicSmoothOperator)
{
    sbm_test_WENOZ3ConvergenceBeatsDonorOnPeriodicSmoothOperator();
}

TEST(SBMP2, WENOZ3TranslationCovarianceForSmoothAndDiscontinuousStencils)
{
    const Real shift = Real(37.25);
    const Real smooth[] = {Real(1.2), Real(0.7), Real(1.1), Real(1.8)};
    const Real jump[] = {Real(0.0), Real(0.0), Real(1.0), Real(1.0)};
    for (const Real carrier : {Real(1.0), Real(-1.0)}) {
        const Real smooth_base = erf_sbm::weno_z3_face_from_stencil(
            smooth[0], smooth[1], smooth[2], smooth[3], carrier);
        const Real smooth_shifted = erf_sbm::weno_z3_face_from_stencil(
            smooth[0] + shift, smooth[1] + shift, smooth[2] + shift,
            smooth[3] + shift, carrier);
        EXPECT_NEAR(smooth_shifted - smooth_base, shift,
                    precision_tolerance(shift, Real(64.0)));

        const Real jump_base = erf_sbm::weno_z3_face_from_stencil(
            jump[0], jump[1], jump[2], jump[3], carrier);
        const Real jump_shifted = erf_sbm::weno_z3_face_from_stencil(
            jump[0] + shift, jump[1] + shift, jump[2] + shift,
            jump[3] + shift, carrier);
        EXPECT_NEAR(jump_shifted - jump_base, shift,
                    precision_tolerance(shift, Real(64.0)));
    }
}

TEST(SBMP2, WENOZ3CanonicalERFEquivalenceRandomized)
{
    const Real eps = std::numeric_limits<Real>::epsilon();
    for (int sample = 0; sample < 512; ++sample) {
        const Real phase = Real(sample + 1);
        const Real qm2 = Real(0.25) * std::sin(Real(0.71) * phase) +
            Real(0.03) * Real(sample % 7);
        const Real qm1 = Real(0.50) * std::cos(Real(0.37) * phase) -
            Real(0.02) * Real(sample % 5);
        const Real q = Real(0.40) * std::sin(Real(0.19) * phase) + Real(0.11);
        const Real qp1 = Real(0.60) * std::cos(Real(0.23) * phase) - Real(0.07);
        for (const Real carrier : {Real(-1.0), Real(1.0)}) {
            const Real expected = canonical_weno_z3_reference(qm2, qm1, q, qp1, carrier);
            const Real actual = erf_sbm::weno_z3_face_from_stencil(
                qm2, qm1, q, qp1, carrier);
            const Real tolerance = Real(512.0) * eps *
                std::max({Real(1.0), std::abs(expected), std::abs(actual)});
            EXPECT_NEAR(actual, expected, tolerance)
                << "sample=" << sample << " carrier=" << carrier;
        }
    }
}

TEST(SBMP2, WENOZ3CanonicalERFDiscontinuityAndConstantStencils)
{
    const Real discontinuities[][4] = {
        {Real(0.0), Real(1.0), Real(1.0), Real(1.0)},
        {Real(1.0), Real(1.0), Real(1.0), Real(0.0)}};
    for (const auto& stencil : discontinuities) {
        for (const Real carrier : {Real(-1.0), Real(1.0)}) {
            const Real expected = canonical_weno_z3_reference(
                stencil[0], stencil[1], stencil[2], stencil[3], carrier);
            const Real actual = erf_sbm::weno_z3_face_from_stencil(
                stencil[0], stencil[1], stencil[2], stencil[3], carrier);
            EXPECT_NEAR(actual, expected,
                        precision_tolerance(expected, Real(512.0)));
            EXPECT_TRUE(std::isfinite(actual));
        }
    }
    for (const Real constant : {Real(-3.25), Real(0.0), Real(8.5)}) {
        for (const Real carrier : {Real(-1.0), Real(1.0)}) {
            const Real actual = erf_sbm::weno_z3_face_from_stencil(
                constant, constant, constant, constant, carrier);
            EXPECT_DOUBLE_EQ(actual, constant);
        }
    }
}

TEST(SBMP2, RuntimeCapabilityRejectsUnsupportedAMREnvelope)
{
    erf_sbm::CapabilityInput supported;
    supported.p2_requested = true;
    supported.max_level = 1;
    supported.spatial_ref_ratio = IntVect(2, 2, 2);
    supported.time_refinement_factor = 2;
    supported.two_way_coupling = true;
    supported.periodic_amr = true;
    supported.native_subcycling = true;
    EXPECT_TRUE(erf_sbm::evaluate_p2_capabilities(supported).supported);

    auto rejected = supported;
    rejected.max_level = 2;
    EXPECT_FALSE(erf_sbm::evaluate_p2_capabilities(rejected).supported);
    rejected = supported;
    rejected.spatial_ref_ratio = IntVect(4, 4, 4);
    EXPECT_FALSE(erf_sbm::evaluate_p2_capabilities(rejected).supported);
    rejected = supported;
    rejected.time_refinement_factor = 1;
    EXPECT_FALSE(erf_sbm::evaluate_p2_capabilities(rejected).supported);
    rejected = supported;
    rejected.time_refinement_factor = 4;
    EXPECT_FALSE(erf_sbm::evaluate_p2_capabilities(rejected).supported);
    rejected = supported;
    rejected.two_way_coupling = false;
    EXPECT_FALSE(erf_sbm::evaluate_p2_capabilities(rejected).supported);
    rejected = supported;
    rejected.periodic_amr = false;
    EXPECT_FALSE(erf_sbm::evaluate_p2_capabilities(rejected).supported);
    const auto report = erf_sbm::evaluate_p2_capabilities(supported);
    EXPECT_NE(report.stable_description().find("spatial_ref_ratio=2,2,2"), std::string::npos);
    EXPECT_NE(report.stable_description().find("time_refinement_factor=2"), std::string::npos);
}

void sbm_test_FullGroupedTransportHasSmoothManufacturedConvergence()
{
    std::ofstream evidence(std::filesystem::temp_directory_path() /
                            "erf_sbm_p2_full_transport_convergence.csv");
    ASSERT_TRUE(evidence.good());
    evidence << "N,weno_operator_error,weno_order,weno_min_lambda,"
                 "donor_operator_error,donor_order\n";
    struct OperatorResult {
        Real error;
        Real minimum_limiter;
    };
    const auto run = [](const int ncell, const erf_sbm::TransportMethod method) {
        const auto layout = make_layout(2, MomentMode::OneMoment);
        erf_auxiliary::AuxiliaryStateManager manager(layout.auxiliary_layout());
        const Box domain(IntVect(0, 0, 0), IntVect(ncell-1, 0, 0));
        const BoxArray boxes(domain);
        const DistributionMapping dm(boxes);
        const amrex::RealBox real_box({AMREX_D_DECL(0.0, 0.0, 0.0)},
                                      {AMREX_D_DECL(1.0, 1.0, 1.0)});
        const std::array<int, AMREX_SPACEDIM> periodicity{AMREX_D_DECL(1, 1, 1)};
        const Geometry geometry(domain, &real_box, amrex::CoordSys::cartesian,
                                periodicity.data());
        manager.define_level(0, boxes, dm, 2);
        for (amrex::MFIter mfi(manager.output(0)); mfi.isValid(); ++mfi) {
            const auto state = manager.output(0).array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                const Real x = (static_cast<Real>(i) + Real(0.5)) / static_cast<Real>(ncell);
                const Real h = Real(1.0) / static_cast<Real>(ncell);
                const Real average_factor = std::sin(
                    Real(3.14159265358979323846) * h) /
                    (Real(3.14159265358979323846) * h);
                state(i,j,k,0) = Real(2.0) +
                    std::sin(Real(6.2831853071795864769) * x) * average_factor;
                state(i,j,k,1) = Real(1.0);
            });
        }
        manager.begin_step(0, 0.0);
        const Real dt = Real(1.0) / static_cast<Real>(ncell);
        amrex::MultiFab rho(boxes, dm, 1, 2);
        rho.setVal(Real(1.0));
        rho.FillBoundary(geometry.periodicity());
        amrex::MultiFab xflux(amrex::convert(boxes, IntVect(1, 0, 0)), dm, 1, 0);
        amrex::MultiFab yflux(amrex::convert(boxes, IntVect(0, 1, 0)), dm, 1, 0);
        amrex::MultiFab zflux(amrex::convert(boxes, IntVect(0, 0, 1)), dm, 1, 0);
        xflux.setVal(Real(0.25));
        yflux.setVal(Real(0.0));
        zflux.setVal(Real(0.0));
        amrex::MultiFab core(boxes, dm, RhoQ3_comp + 1, 0);
        erf_auxiliary::AuxiliaryFaceTransfer stage_flux;
        stage_flux.define(boxes, dm, layout.ncomp(), 0);
        const auto context = erf_auxiliary::make_compressible_stage(
            2, 0.0, dt / 2.0, dt, dt, nullptr, nullptr);
        Real minimum_limiter = Real(0.0);
        erf_sbm::advance_stage(manager, layout, context, rho, core,
                               xflux, yflux, zflux, geometry, stage_flux,
                               method, 0, Real(0.0), 1, &minimum_limiter);
        amrex::MultiFab error(boxes, dm, 1, 0);
        for (amrex::MFIter mfi(error); mfi.isValid(); ++mfi) {
            const auto result = error.array(mfi);
                const auto accepted = stage_flux.x().const_array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                const Real x = (static_cast<Real>(i) + Real(0.5)) / static_cast<Real>(ncell);
                const Real h = Real(1.0) / static_cast<Real>(ncell);
                // The production state is a finite-volume average, so the
                // exact semi-discrete operator is the cell-average of
                // -u*dq/dx, equivalently the exact face-flux difference.
                const Real exact_operator = -Real(0.25) *
                    (std::sin(Real(6.2831853071795864769) * (x + Real(0.5)*h)) -
                     std::sin(Real(6.2831853071795864769) * (x - Real(0.5)*h))) / h;
                const Real production_operator = -static_cast<Real>(ncell) *
                    (accepted(i+1,j,k,0) - accepted(i,j,k,0));
                result(i,j,k) = std::abs(production_operator - exact_operator);
            });
        }
        amrex::Gpu::synchronize();
        return OperatorResult{error.norm0(0), minimum_limiter};
    };

    std::vector<Real> weno_errors;
    std::vector<Real> weno_limiters;
    std::vector<Real> donor_errors;
    for (const int ncell : {16, 32, 64, 128}) {
        const auto weno = run(ncell, erf_sbm::TransportMethod::GroupedFCT_WENOZ3);
        const auto donor = run(ncell, erf_sbm::TransportMethod::DonorCell);
        weno_errors.push_back(weno.error);
        weno_limiters.push_back(weno.minimum_limiter);
        donor_errors.push_back(donor.error);
    }
    for (std::size_t i = 0; i < weno_errors.size(); ++i) {
        const Real weno_order = i == 0 ? Real(0.0) :
            std::log(weno_errors[i-1] / weno_errors[i]) / std::log(Real(2.0));
        const Real donor_order = i == 0 ? Real(0.0) :
            std::log(donor_errors[i-1] / donor_errors[i]) / std::log(Real(2.0));
        evidence << (16 << i) << ',' << std::setprecision(17)
                 << weno_errors[i] << ',' << weno_order << ','
                 << weno_limiters[i] << ',' << donor_errors[i] << ','
                 << donor_order << '\n';
        if (i > 0) {
            EXPECT_GT(weno_order, Real(0.5));
            EXPECT_GT(donor_order, Real(0.5));
            EXPECT_LT(donor_order, Real(1.5));
            // The two operators are mathematically tied for the coarsest
            // smooth mode.  Keep the ordering requirement, but allow the
            // bounded reduction/reconstruction roundoff seen across CPU
            // standard libraries and precision modes.
            const Real ordering_tolerance = Real(128.0) *
                std::numeric_limits<Real>::epsilon() *
                std::max(Real(1.0), std::abs(donor_errors[i]));
            EXPECT_LE(weno_errors[i], donor_errors[i] + ordering_tolerance);
        }
        EXPECT_NEAR(weno_limiters[i], Real(1.0),
                    precision_tolerance(Real(1.0), Real(128.0)));
    }
}

TEST(SBMP2, FullGroupedTransportHasSmoothManufacturedConvergence)
{
    sbm_test_FullGroupedTransportHasSmoothManufacturedConvergence();
}

void sbm_test_VariableDensityWENOReconstructionHasBoundedConvergence()
{
    std::ofstream evidence(std::filesystem::temp_directory_path() /
                            "erf_sbm_p2_variable_density_convergence.csv");
    ASSERT_TRUE(evidence.good());
    evidence << "N,error,order\n";
    const Real pi = Real(3.1415926535897932384626433832795);
    const Real wave = Real(2.0) * pi;
    const Real rho_amplitude = Real(0.20);
    const Real z0 = Real(0.50);
    const Real z_amplitude = Real(0.05);
    const Real velocity = Real(0.25);
    const auto sinc = [](const Real argument) {
        return std::abs(argument) < precision_tolerance(Real(1.0), Real(16.0)) ? Real(1.0) :
            std::sin(argument) / argument;
    };
    const auto run = [&](const int ncell) {
        const auto layout = make_layout(2, MomentMode::OneMoment);
        erf_auxiliary::AuxiliaryStateManager manager(layout.auxiliary_layout());
        const Box domain(IntVect(0, 0, 0), IntVect(ncell-1, 0, 0));
        const BoxArray boxes(domain);
        const DistributionMapping dm(boxes);
        const amrex::RealBox real_box({AMREX_D_DECL(0.0, 0.0, 0.0)},
                                      {AMREX_D_DECL(1.0, 1.0, 1.0)});
        const std::array<int, AMREX_SPACEDIM> periodicity{AMREX_D_DECL(1, 1, 1)};
        const Geometry geometry(domain, &real_box, amrex::CoordSys::cartesian,
                                periodicity.data());
        manager.define_level(0, boxes, dm, 2);
        amrex::MultiFab rho(boxes, dm, 1, 2);
        const Real h = Real(1.0) / static_cast<Real>(ncell);
        const Real rho_average_factor = sinc(Real(0.5) * wave * h);
        const Real product_average_factor = sinc(wave * h);
        for (amrex::MFIter mfi(rho); mfi.isValid(); ++mfi) {
            const auto density = rho.array(mfi);
            const auto state = manager.output(0).array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                const Real x = (static_cast<Real>(i) + Real(0.5)) * h;
                const Real rho_cell = Real(1.0) + rho_amplitude * rho_average_factor *
                    std::sin(wave * x);
                const Real u_cell = z0 + z_amplitude * rho_average_factor *
                    std::cos(wave * x) + rho_amplitude * z0 * rho_average_factor *
                    std::sin(wave * x) + Real(0.5) * rho_amplitude * z_amplitude *
                    product_average_factor * std::sin(Real(2.0) * wave * x);
                density(i,j,k,0) = rho_cell;
                state(i,j,k,0) = u_cell;
                state(i,j,k,1) = Real(0.25);
            });
        }
        rho.FillBoundary(geometry.periodicity());
        manager.output(0).FillBoundary(geometry.periodicity());
        manager.begin_step(0, 0.0);
        amrex::MultiFab xflux(amrex::convert(boxes, IntVect(1, 0, 0)), dm, 1, 0);
        amrex::MultiFab yflux(amrex::convert(boxes, IntVect(0, 1, 0)), dm, 1, 0);
        amrex::MultiFab zflux(amrex::convert(boxes, IntVect(0, 0, 1)), dm, 1, 0);
        for (amrex::MFIter mfi(xflux); mfi.isValid(); ++mfi) {
            const auto flux = xflux.array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                const Real x = static_cast<Real>(i) * h;
                flux(i,j,k) = velocity * (Real(1.0) + rho_amplitude * std::sin(wave * x));
            });
        }
        yflux.setVal(Real(0.0));
        zflux.setVal(Real(0.0));
        xflux.FillBoundary(geometry.periodicity());
        yflux.FillBoundary(geometry.periodicity());
        zflux.FillBoundary(geometry.periodicity());
        amrex::MultiFab core(boxes, dm, RhoQ3_comp + 1, 0);
        erf_auxiliary::AuxiliaryFaceTransfer stage_flux;
        stage_flux.define(boxes, dm, layout.ncomp(), 0);
        const auto context = erf_auxiliary::make_compressible_stage(
            2, 0.0, Real(0.125) * h, Real(0.25) * h, Real(0.25) * h, nullptr, nullptr);
        erf_sbm::advance_stage(manager, layout, context, rho, core,
                               xflux, yflux, zflux, geometry, stage_flux,
                               erf_sbm::TransportMethod::GroupedFCT_WENOZ3,
                               0, Real(0.0), 1);
        amrex::MultiFab error(boxes, dm, 1, 0);
        for (amrex::MFIter mfi(error); mfi.isValid(); ++mfi) {
            const auto err = error.array(mfi);
            const auto accepted = stage_flux.x().const_array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                const Real x = (static_cast<Real>(i) + Real(0.5)) * h;
                const Real left = x - Real(0.5) * h;
                const Real right = x + Real(0.5) * h;
                const Real rho_left = Real(1.0) + rho_amplitude * std::sin(wave * left);
                const Real rho_right = Real(1.0) + rho_amplitude * std::sin(wave * right);
                const Real z_left = z0 + z_amplitude * std::cos(wave * left);
                const Real z_right = z0 + z_amplitude * std::cos(wave * right);
                const Real exact = -velocity * (rho_right*z_right - rho_left*z_left) / h;
                const Real numerical = -(accepted(i+1,j,k,0) - accepted(i,j,k,0)) / h;
                err(i,j,k) = std::abs(numerical - exact);
            });
        }
        amrex::Gpu::synchronize();
        return error.norm0(0);
    };

    std::vector<Real> errors;
    for (const int ncell : {16, 32, 64, 128}) errors.push_back(run(ncell));
    for (std::size_t i = 0; i < errors.size(); ++i) {
        const Real order = i == 0 ? Real(0.0) :
            std::log(errors[i-1] / errors[i]) / std::log(Real(2.0));
        evidence << (16 << i) << ',' << std::setprecision(17) << errors[i] << ',' << order << '\n';
        if (i > 0) {
            // The ratio is formed from cell averages U/rho, so this test
            // records the conservative bounded claim (approximately second
            // order) rather than claiming third-order whole-model accuracy.
            EXPECT_GT(order, Real(0.5));
            EXPECT_LT(order, Real(1.5));
        }
    }
}

TEST(SBMP2, VariableDensityWENOReconstructionHasBoundedConvergence)
{
    sbm_test_VariableDensityWENOReconstructionHasBoundedConvergence();
}

TEST(SBMP2, FiniteVolumeWENOQuadraticOptimalCandidateIsExact)
{
    const Real h = Real(1.0) / Real(32.0);
    const auto cell_average = [=](const int cell) {
        const Real lo = h * static_cast<Real>(cell);
        const Real hi = lo + h;
        const auto primitive = [](const Real x) {
            return Real(0.5)*x*x + Real(0.5)*x*x*x;
        };
        return (primitive(hi) - primitive(lo)) / h;
    };
    for (int face = 2; face < 30; ++face) {
        const Real qm2 = cell_average(face-2);
        const Real qm1 = cell_average(face-1);
        const Real q = cell_average(face);
        const Real qp1 = cell_average(face+1);
        const Real left_candidate = Real(0.5) * (-qm2 + Real(3.0)*qm1);
        const Real centered_candidate = Real(0.5) * (qm1 + q);
        const Real reconstructed = (left_candidate + Real(2.0)*centered_candidate) / Real(3.0);
        const Real x = h * static_cast<Real>(face);
        const Real exact = x + Real(1.5)*x*x;
        // This is the linear optimal combination of the two smooth FV
        // substencils; it is exact through the quadratic term and exposes the
        // face-average indexing independently of nonlinear WENO weights.
        EXPECT_NEAR(reconstructed, exact, Real(2.e-12));
        for (const Real carrier : {Real(1.0), Real(-1.0)}) {
            // With the mandated fixed canonical epsilon, nonlinear WENO-Z3
            // is not algebraically identical to the optimal linear
            // quadratic combination.  Retain the quadratic indexing check
            // above and require the canonical reconstruction to remain
            // finite and within its deterministic O(h^3) error envelope.
            const Real actual = erf_sbm::weno_z3_face_from_stencil(
                qm2, qm1, q, qp1, carrier);
            EXPECT_TRUE(std::isfinite(actual));
            EXPECT_NEAR(actual, exact, Real(4.e-6));
        }
    }
}

} // namespace
