#include <gtest/gtest.h>

#include <AMReX_BoxArray.H>
#include <AMReX_BaseFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_Math.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_YAFluxRegister.H>

#include "ERF_SBMAMRTransfer.H"
#include "ERF_SBMAMRFluxRegister.H"
#include "ERF_SBMRestart.H"

#include <cmath>
#include <limits>
#include <string>
#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

namespace {

using amrex::Box;
using amrex::BoxArray;
using amrex::DistributionMapping;
using amrex::Geometry;
using amrex::IntVect;
using amrex::MFIter;
using amrex::MultiFab;
using amrex::Real;

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real ya_oracle_coarse_flux (int dir, int i, int j, int k, int n) noexcept
{
    return Real(1.0) + Real(3.0) * dir + Real(0.1) * i + Real(0.2) * j +
           Real(0.3) * k + Real(0.01) * n;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real ya_oracle_fine_flux (int dir, int i, int j, int k, int n) noexcept
{
    return Real(2.0) + Real(5.0) * dir + Real(0.01) * i + Real(0.02) * j +
           Real(0.03) * k + Real(0.001) * n;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real accepted_chunk_rate (int dir, int i, int j, int k, int global_component,
                          int method, int stage, int substep) noexcept
{
    const Real low = Real(0.4) + Real(0.03) * dir + Real(0.001) * i +
                     Real(0.002) * j + Real(0.003) * k +
                     Real(0.004) * global_component;
    const Real high = low + Real(0.8) + Real(0.01) * stage +
                      Real(0.02) * substep;
    const Real lambda = Real(0.35) + Real(0.01) * (global_component % 4);
    return low + lambda * (high - low);
}

erf_auxiliary::MappedFaceFluxRate make_rate_chunk (
    const BoxArray& cell_ba, const DistributionMapping& mapping,
    const int first_global_component, const int ncomp,
    const int method, const int stage, const int substep)
{
    erf_auxiliary::MappedFaceFluxRate rate;
    rate.define(cell_ba, mapping, ncomp, 0);
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        auto& field = rate.dir(dir);
        for (MFIter mfi(field, false); mfi.isValid(); ++mfi) {
            const Box box = mfi.validbox();
            const auto flux = field.array(mfi);
            ParallelFor(box, ncomp,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int local) noexcept {
                flux(i,j,k,local) = accepted_chunk_rate(
                    dir, i, j, k, first_global_component + local,
                    method, stage, substep);
            });
        }
    }
    amrex::Gpu::streamSynchronize();
    return rate;
}

void accumulate_reference_integral (
    erf_auxiliary::IntegratedMappedFaceFlux& integral,
    const erf_auxiliary::MappedFaceFluxRate& rate,
    const int first_global_component, const double weight)
{
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        for (int local = 0; local < rate.nComp(); ++local) {
            MultiFab::Saxpy(integral.dir(dir), static_cast<Real>(weight),
                            rate.dir(dir), local,
                            first_global_component + local, 1, 0);
        }
    }
}

void add_integral_to_reference_register (
    amrex::YAFluxRegister& flux_register,
    const erf_auxiliary::IntegratedMappedFaceFlux& integral,
    const BoxArray& cell_ba, const DistributionMapping& mapping,
    const amrex::Geometry& geometry, const bool coarse_side)
{
    MultiFab cell_layout(cell_ba, mapping, 1, 0,
                         amrex::MFInfo().SetAlloc(false));
    const auto dx = geometry.CellSizeArray();
    for (MFIter mfi(cell_layout, amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
        std::array<amrex::FArrayBox const*, AMREX_SPACEDIM> faces{};
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            faces[static_cast<std::size_t>(dir)] = &integral.dir(dir)[mfi];
        }
        if (coarse_side) {
            flux_register.CrseAdd(mfi, faces, dx.data(), Real(1.0),
                                  amrex::RunOn::Host);
        } else {
            flux_register.FineAdd(mfi, faces, dx.data(), Real(1.0),
                                  amrex::RunOn::Host);
        }
    }
}

struct ComponentOffsets {
    int one_mass;
    int one_property;
    int two_mass;
    int two_number;
    int two_property;
};

erf_sbm::SBMLayout make_transfer_layout ()
{
    erf_sbm::SBMLayoutSpec spec;
    erf_sbm::SpectralPopulationSpec one;
    one.population_id = 0;
    one.semantic_id = "one_moment";
    one.phase = erf_sbm::PopulationPhase::Liquid;
    one.moment_mode = erf_sbm::MomentMode::OneMoment;
    one.grid.coordinate_kind = erf_sbm::CoordinateKind::Mass;
    one.grid.coordinate_units = "kg";
    one.grid.edges = {Real(0.0), Real(1.0), Real(2.0)};
    one.grid.pivots = {Real(0.5), Real(1.5)};
    spec.populations.push_back(one);

    erf_sbm::SpectralPopulationSpec two;
    two.population_id = 1;
    two.semantic_id = "two_moment";
    two.phase = erf_sbm::PopulationPhase::Aerosol;
    two.moment_mode = erf_sbm::MomentMode::TwoMoment;
    two.grid.coordinate_kind = erf_sbm::CoordinateKind::Mass;
    two.grid.coordinate_units = "kg";
    two.grid.edges = {Real(0.0), Real(1.0), Real(2.0)};
    two.grid.pivots = {Real(0.5), Real(1.5)};
    spec.populations.push_back(two);
    spec.liquid_projection = {0, 1};

    for (const auto& property_info : std::vector<std::pair<std::string, int>>{
             {"one_moment_extensive", 0}, {"two_moment_carried", 1}}) {
        erf_sbm::AttachedPropertyDescriptor property;
        property.name = property_info.first;
        property.semantic_id = property_info.first + ".semantic";
        property.units = "kg m^-3";
        property.carrier_population = property_info.second;
        property.kind = property_info.second == 0
                            ? erf_sbm::PropertyKind::ExtensiveMass
                            : erf_sbm::PropertyKind::NumberCarried;
        property.support = erf_sbm::SupportRequirement::None;
        property.remap_policy = erf_sbm::PropertyRemapPolicy::CarrierBinConservative;
        property.transported = true;
        property.support_min = Real(0.0);
        spec.attached_properties.push_back(property);
    }
    return erf_sbm::SBMLayout(std::move(spec));
}

ComponentOffsets offsets_for (const erf_sbm::SBMLayout& layout)
{
    return {layout.populations()[0].mass_offset,
            layout.property_offset(0),
            layout.populations()[1].mass_offset,
            layout.populations()[1].number_offset,
            layout.property_offset(1)};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real amplitude (const int i, const int j, const int k) noexcept
{
    return Real(1.0) + Real(0.025) * i + Real(0.015) * j + Real(0.01) * k;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real base_component (const int component, const int i, const int j, const int k,
                     const ComponentOffsets offsets,
                     const bool empty_upper_bin) noexcept
{
    const Real a = amplitude(i, j, k);
    if (component >= offsets.one_mass && component < offsets.one_mass + 2) {
        return Real(0.2) * a * static_cast<Real>(component - offsets.one_mass + 1);
    }
    if (component >= offsets.one_property && component < offsets.one_property + 2) {
        return Real(0.02) * a * static_cast<Real>(component - offsets.one_property + 1);
    }
    if (component >= offsets.two_mass && component < offsets.two_mass + 2) {
        const int bin = component - offsets.two_mass;
        if (empty_upper_bin && bin == 1 && i == 1 && j == 1 && k == 1) return Real(0.0);
        const Real number = Real(0.3) * a * static_cast<Real>(bin + 1);
        const Real mean = bin == 0 ? Real(0.35) : Real(1.35);
        return number * mean;
    }
    if (component >= offsets.two_number && component < offsets.two_number + 2) {
        const int bin = component - offsets.two_number;
        if (empty_upper_bin && bin == 1 && i == 1 && j == 1 && k == 1) return Real(0.0);
        return Real(0.3) * a * static_cast<Real>(bin + 1);
    }
    if (component >= offsets.two_property && component < offsets.two_property + 2) {
        const int bin = component - offsets.two_property;
        if (empty_upper_bin && bin == 1 && i == 1 && j == 1 && k == 1) return Real(0.0);
        return Real(0.15) * a * static_cast<Real>(bin + 1);
    }
    return Real(0.0);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real coarse_density (const int i, const int j, const int k) noexcept
{
    return Real(1.0) + Real(0.01) * i + Real(0.02) * j + Real(0.03) * k;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real coarse_measure (const int i, const int j, const int k) noexcept
{
    return Real(0.7) + Real(0.02) * i + Real(0.01) * j + Real(0.015) * k;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real restriction_fine_density (const int i, const int j, const int k) noexcept
{
    return Real(0.8) + Real(0.004) * i + Real(0.006) * j + Real(0.003) * k;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real restriction_fine_measure (const int i, const int j, const int k) noexcept
{
    return Real(0.45) + Real(0.012) * i + Real(0.009) * j + Real(0.007) * k;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real prolong_fine_measure (const int i, const int j, const int k) noexcept
{
    constexpr Real delta = Real(0.2);
    const int ci = i / 2;
    const int cj = j / 2;
    const int ck = k / 2;
    const int parity = (i % 2 + j % 2 + k % 2) % 2;
    const Real sign = parity == 0 ? Real(-1.0) : Real(1.0);
    return coarse_measure(ci, cj, ck) * (Real(1.0) + delta * sign);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real prolong_fine_density (const int i, const int j, const int k) noexcept
{
    constexpr Real delta = Real(0.2);
    const int ci = i / 2;
    const int cj = j / 2;
    const int ck = k / 2;
    const int parity = (i % 2 + j % 2 + k % 2) % 2;
    const Real sign = parity == 0 ? Real(-1.0) : Real(1.0);
    return coarse_density(ci, cj, ck) * (Real(1.0) - delta * sign) /
           (Real(1.0) - delta * delta);
}

void fill_carriers (MultiFab& density, MultiFab& measure, const bool coarse,
                    const bool prolongation_fine = false)
{
    for (amrex::MFIter mfi(density, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box box = mfi.tilebox();
        const auto rho = density.array(mfi);
        const auto omega = measure.array(mfi);
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            if (coarse) {
                rho(i, j, k, 0) = coarse_density(i, j, k);
                omega(i, j, k, 0) = coarse_measure(i, j, k);
            } else if (prolongation_fine) {
                rho(i, j, k, 0) = prolong_fine_density(i, j, k);
                omega(i, j, k, 0) = prolong_fine_measure(i, j, k);
            } else {
                rho(i, j, k, 0) = restriction_fine_density(i, j, k);
                omega(i, j, k, 0) = restriction_fine_measure(i, j, k);
            }
        });
    }
}

void fill_spectrum (MultiFab& spectrum, const MultiFab& density,
                    const erf_sbm::SBMLayout& layout,
                    const bool empty_upper_bin = false)
{
    const ComponentOffsets offsets = offsets_for(layout);
    const int ncomp = layout.ncomp();
    for (amrex::MFIter mfi(spectrum, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box box = mfi.tilebox();
        const auto rho = density.const_array(mfi);
        const auto state = spectrum.array(mfi);
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const Real carrier = rho(i, j, k, 0);
            for (int component = 0; component < ncomp; ++component) {
                state(i, j, k, component) = carrier * base_component(
                    component, i, j, k, offsets, empty_upper_bin);
            }
        });
    }
}

erf_sbm::SBMAMRStateView timed_view (const MultiFab& spectrum,
                                    const MultiFab& density,
                                    const MultiFab& measure,
                                    const double time)
{
    return {&spectrum, time, &density, 0, time, &measure, 0, time};
}

void expect_same_values (const MultiFab& actual, const MultiFab& expected)
{
    ASSERT_EQ(actual.nComp(), expected.nComp());
    MultiFab difference(actual.boxArray(), actual.DistributionMap(),
                        actual.nComp(), 0);
    MultiFab::Copy(difference, actual, 0, 0, actual.nComp(), 0);
    MultiFab::Subtract(difference, expected, 0, 0, actual.nComp(), 0);
    for (int component = 0; component < actual.nComp(); ++component) {
        EXPECT_EQ(difference.norm0(component), Real(0.0))
            << "component=" << component;
    }
}

void expect_roundoff_equivalent (const MultiFab& actual,
                                 const MultiFab& expected)
{
    ASSERT_EQ(actual.nComp(), expected.nComp());
    MultiFab difference(actual.boxArray(), actual.DistributionMap(),
                        actual.nComp(), 0);
    MultiFab::Copy(difference, actual, 0, 0, actual.nComp(), 0);
    MultiFab::Subtract(difference, expected, 0, 0, actual.nComp(), 0);
    constexpr Real operation_reordering_bound =
        Real(64.0) * std::numeric_limits<Real>::epsilon();
    for (int component = 0; component < actual.nComp(); ++component) {
        const Real scale = std::max(
            Real(1.0), std::max(actual.norm0(component), expected.norm0(component)));
        EXPECT_LE(difference.norm0(component), operation_reordering_bound * scale)
            << "component=" << component;
    }
}

Geometry make_geometry (const Box& domain)
{
    const amrex::RealBox physical({0.0, 0.0, 0.0}, {1.0, 1.0, 1.0});
    const int periodic[AMREX_SPACEDIM] = {1, 1, 1};
    return Geometry(domain, &physical, amrex::CoordSys::cartesian, periodic);
}

Geometry
make_geometry_with_nonperiodic_direction (const Box& domain,
                                          const int direction)
{
    const amrex::RealBox physical({0.0, 0.0, 0.0}, {1.0, 1.0, 1.0});
    int periodic[AMREX_SPACEDIM];
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        periodic[dir] = dir == direction ? 0 : 1;
    }
    return Geometry(domain, &physical, amrex::CoordSys::cartesian, periodic);
}

Box coarse_domain ()
{
    return Box(IntVect(0, 0, 0), IntVect(3, 3, 3));
}

Box fine_coverage ()
{
    return Box(IntVect(2, 2, 2), IntVect(5, 5, 5));
}

void run_restriction_averages_mapped_amount_and_preserves_uncovered_state ()
{
    const auto layout = make_transfer_layout();
    const BoxArray coarse_ba(coarse_domain());
    const DistributionMapping coarse_dm(coarse_ba);
    const BoxArray fine_ba(fine_coverage());
    const DistributionMapping fine_dm(fine_ba);
    MultiFab coarse_rho(coarse_ba, coarse_dm, 1, 0);
    MultiFab coarse_omega(coarse_ba, coarse_dm, 1, 0);
    MultiFab fine_rho(fine_ba, fine_dm, 1, 0);
    MultiFab fine_omega(fine_ba, fine_dm, 1, 0);
    fill_carriers(coarse_rho, coarse_omega, true);
    fill_carriers(fine_rho, fine_omega, false);

    MultiFab coarse_state(coarse_ba, coarse_dm, layout.ncomp(), 0);
    MultiFab fine_state(fine_ba, fine_dm, layout.ncomp(), 0);
    fill_spectrum(coarse_state, coarse_rho, layout);
    fill_spectrum(fine_state, fine_rho, layout);
    MultiFab coarse_before(coarse_ba, coarse_dm, layout.ncomp(), 0);
    MultiFab::Copy(coarse_before, coarse_state, 0, 0, layout.ncomp(), 0);
    MultiFab fine_before(fine_ba, fine_dm, layout.ncomp(), 0);
    MultiFab::Copy(fine_before, fine_state, 0, 0, layout.ncomp(), 0);
    MultiFab candidate(coarse_ba, coarse_dm, layout.ncomp(), 0);
    candidate.setVal(Real(99.0));

    constexpr int ratio_value = 2;
    const IntVect ratio(ratio_value);
    auto fine_view = timed_view(fine_state, fine_rho, fine_omega, 0.5);
    auto coarse_view = timed_view(coarse_state, coarse_rho, coarse_omega, 0.5);
    std::string diagnostic;

    ASSERT_TRUE(erf_sbm::authoritative_state_admissible(coarse_state, layout, 0,
                                                        &diagnostic))
        << diagnostic;
    MultiFab uncovered_roundtrip_change(coarse_ba, coarse_dm, 1, 0);
    uncovered_roundtrip_change.setVal(Real(0.0));
    for (amrex::MFIter mfi(coarse_state, amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
        const Box box = mfi.tilebox();
        const auto state = coarse_state.const_array(mfi);
        const auto omega = coarse_omega.const_array(mfi);
        const auto changed = uncovered_roundtrip_change.array(mfi);
        const int ncomp = layout.ncomp();
        amrex::ParallelFor(
            box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                const bool covered =
                    i >= 1 && i <= 2 && j >= 1 && j <= 2 && k >= 1 && k <= 2;
                int found_change = 0;
                if (!covered) {
                    const Real measure = omega(i, j, k, 0);
                    for (int component = 0; component < ncomp; ++component) {
                        const Real value = state(i, j, k, component);
                        if ((measure * value) / measure != value)
                            found_change = 1;
                    }
                }
                changed(i, j, k, 0) = static_cast<Real>(found_change);
            });
    }
    ASSERT_EQ(uncovered_roundtrip_change.max(0), Real(1.0))
        << "fixture must distinguish the old uncovered H round trip";

    ASSERT_TRUE(erf_sbm::RestrictMappedSpectrum(
        layout, fine_view, coarse_view, ratio, 0, candidate, diagnostic)) << diagnostic;
    ASSERT_TRUE(erf_sbm::authoritative_state_admissible(
        candidate, layout, 0, &diagnostic)) << diagnostic;

    MultiFab private_candidate(coarse_ba, coarse_dm, layout.ncomp(), 0);
    private_candidate.setVal(Real(99.0));
    ASSERT_TRUE(erf_sbm::RestrictMappedSpectrumPrivate(
        layout, fine_view, coarse_view, ratio, 0, private_candidate,
        diagnostic)) << diagnostic;
    MultiFab private_difference(coarse_ba, coarse_dm, layout.ncomp(), 0);
    MultiFab::Copy(private_difference, private_candidate, 0, 0,
                   layout.ncomp(), 0);
    MultiFab::Subtract(private_difference, candidate, 0, 0,
                       layout.ncomp(), 0);
    for (int component = 0; component < layout.ncomp(); ++component) {
        EXPECT_LE(private_difference.norm0(component),
                  Real(64.0) * std::numeric_limits<Real>::epsilon())
            << "private synchronization candidate differs from the independent "
            << "transactional restriction oracle at component " << component;
    }

    // Independently evaluate every child amount from its analytic fixture
    // values.  The expected candidate includes the preexisting uncovered U.
    MultiFab expected(coarse_ba, coarse_dm, layout.ncomp(), 0);
    MultiFab arithmetic_u(coarse_ba, coarse_dm, layout.ncomp(), 0);
    const ComponentOffsets offsets = offsets_for(layout);
    const int ncomp = layout.ncomp();
    for (amrex::MFIter mfi(expected, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box box = mfi.tilebox();
        const auto expected_array = expected.array(mfi);
        const auto arithmetic_array = arithmetic_u.array(mfi);
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const Real omega_c = coarse_measure(i, j, k);
            const Real rho_c = coarse_density(i, j, k);
            const bool covered = i >= 1 && i <= 2 && j >= 1 && j <= 2 && k >= 1 && k <= 2;
            for (int component = 0; component < ncomp; ++component) {
                if (!covered) {
                    expected_array(i, j, k, component) = rho_c *
                        base_component(component, i, j, k, offsets, false);
                    arithmetic_array(i, j, k, component) =
                        expected_array(i, j, k, component);
                    continue;
                }
                Real mapped_sum = Real(0.0);
                Real u_sum = Real(0.0);
                for (int oz = 0; oz < ratio_value; ++oz) {
                    for (int oy = 0; oy < ratio_value; ++oy) {
                        for (int ox = 0; ox < ratio_value; ++ox) {
                            const int fi = ratio_value * i + ox;
                            const int fj = ratio_value * j + oy;
                            const int fk = ratio_value * k + oz;
                            const Real rho_f = restriction_fine_density(fi, fj, fk);
                            const Real u = rho_f * base_component(
                                component, fi, fj, fk, offsets, false);
                            const Real omega_f = restriction_fine_measure(fi, fj, fk);
                            mapped_sum += omega_f * u;
                            u_sum += u;
                        }
                    }
                }
                constexpr Real child_count = Real(8.0);
                expected_array(i, j, k, component) =
                    (mapped_sum / child_count) / omega_c;
                arithmetic_array(i, j, k, component) = u_sum / child_count;
            }
        });
    }
    MultiFab difference(coarse_ba, coarse_dm, layout.ncomp(), 0);
    MultiFab::Copy(difference, candidate, 0, 0, ncomp, 0);
    MultiFab::Subtract(difference, expected, 0, 0, ncomp, 0);
    for (int component = 0; component < ncomp; ++component) {
        EXPECT_LE(difference.norm0(component),
                  Real(64.0) * std::numeric_limits<Real>::epsilon())
            << "component=" << component;
    }

    MultiFab uncovered_error(coarse_ba, coarse_dm, ncomp, 0);
    MultiFab coarse_source_error(coarse_ba, coarse_dm, ncomp, 0);
    for (amrex::MFIter mfi(candidate, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi) {
        const Box box = mfi.tilebox();
        const auto actual = candidate.const_array(mfi);
        const auto original = coarse_before.const_array(mfi);
        const auto source = coarse_state.const_array(mfi);
        const auto uncovered = uncovered_error.array(mfi);
        const auto source_error = coarse_source_error.array(mfi);
        amrex::ParallelFor(
            box, ncomp,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int component) noexcept {
                const bool covered =
                    i >= 1 && i <= 2 && j >= 1 && j <= 2 && k >= 1 && k <= 2;
                uncovered(i, j, k, component) =
                    covered ? Real(0.0)
                            : amrex::Math::abs(actual(i, j, k, component) -
                                               original(i, j, k, component));
                source_error(i, j, k, component) = amrex::Math::abs(
                    source(i, j, k, component) - original(i, j, k, component));
            });
    }
    for (int component = 0; component < ncomp; ++component) {
        EXPECT_EQ(uncovered_error.norm0(component), Real(0.0))
            << "uncovered component=" << component;
        EXPECT_EQ(coarse_source_error.norm0(component), Real(0.0))
            << "coarse source component=" << component;
    }

    MultiFab arithmetic_difference(coarse_ba, coarse_dm, layout.ncomp(), 0);
    MultiFab::Copy(arithmetic_difference, candidate, 0, 0, ncomp, 0);
    MultiFab::Subtract(arithmetic_difference, arithmetic_u, 0, 0, ncomp, 0);
    for (int component = 0; component < ncomp; ++component) {
        EXPECT_GT(arithmetic_difference.norm0(component),
                  Real(1.0e-3) * candidate.norm0(component))
            << "component=" << component;
    }

    MultiFab::Subtract(fine_state, fine_before, 0, 0, ncomp, 0);
    for (int component = 0; component < ncomp; ++component) {
        EXPECT_EQ(fine_state.norm0(component), Real(0.0))
            << "component=" << component;
    }

    // A noncanonical fine source fails before any candidate can be committed.
    MultiFab invalid_fine(fine_ba, fine_dm, ncomp, 0);
    MultiFab::Copy(invalid_fine, fine_before, 0, 0, ncomp, 0);
    invalid_fine.setVal(Real(-1.0), layout.populations()[1].mass_offset, 1, 0);
    candidate.setVal(Real(123.0));
    auto invalid_view = timed_view(invalid_fine, fine_rho, fine_omega, 0.5);
    EXPECT_FALSE(erf_sbm::RestrictMappedSpectrum(
        layout, invalid_view, coarse_view, ratio, 0, candidate, diagnostic));
    for (int component = 0; component < ncomp; ++component) {
        EXPECT_DOUBLE_EQ(candidate.min(component), Real(123.0))
            << "component=" << component;
        EXPECT_DOUBLE_EQ(candidate.max(component), Real(123.0))
            << "component=" << component;
    }
}

TEST(SBMAMRTransfer, RestrictionAveragesMappedAmountAndPreservesUncoveredState)
{
    run_restriction_averages_mapped_amount_and_preserves_uncovered_state();
}

void
run_restriction_late_quotient_failure_is_atomic ()
{
    const auto layout = make_transfer_layout();
    const Box c_domain = coarse_domain();
    Box f_domain = c_domain;
    f_domain.refine(IntVect(2));
    const BoxArray coarse_ba(c_domain);
    const BoxArray fine_ba(f_domain);
    const DistributionMapping coarse_dm(coarse_ba);
    const DistributionMapping fine_dm(fine_ba);

    MultiFab coarse_rho(coarse_ba, coarse_dm, 1, 0);
    MultiFab coarse_omega(coarse_ba, coarse_dm, 1, 0);
    MultiFab fine_rho(fine_ba, fine_dm, 1, 0);
    MultiFab fine_omega(fine_ba, fine_dm, 1, 0);
    fill_carriers(coarse_rho, coarse_omega, true);
    fill_carriers(fine_rho, fine_omega, false);
    coarse_omega.setVal(std::numeric_limits<Real>::min());

    MultiFab coarse_state(coarse_ba, coarse_dm, layout.ncomp(), 0);
    MultiFab fine_state(fine_ba, fine_dm, layout.ncomp(), 0);
    fill_spectrum(coarse_state, coarse_rho, layout);
    fill_spectrum(fine_state, fine_rho, layout);
    fine_state.mult(Real(1.0e30), 0, layout.ncomp(), 0);
    std::string diagnostic;
    ASSERT_TRUE(erf_sbm::authoritative_state_admissible(fine_state, layout, 1,
                                                        &diagnostic))
        << diagnostic;

    MultiFab candidate(coarse_ba, coarse_dm, layout.ncomp(), 0);
    candidate.setVal(Real(41.0));
    MultiFab candidate_before(coarse_ba, coarse_dm, layout.ncomp(), 0);
    MultiFab coarse_before(coarse_ba, coarse_dm, layout.ncomp(), 0);
    MultiFab fine_before(fine_ba, fine_dm, layout.ncomp(), 0);
    MultiFab coarse_rho_before(coarse_ba, coarse_dm, 1, 0);
    MultiFab coarse_omega_before(coarse_ba, coarse_dm, 1, 0);
    MultiFab fine_rho_before(fine_ba, fine_dm, 1, 0);
    MultiFab fine_omega_before(fine_ba, fine_dm, 1, 0);
    MultiFab::Copy(candidate_before, candidate, 0, 0, layout.ncomp(), 0);
    MultiFab::Copy(coarse_before, coarse_state, 0, 0, layout.ncomp(), 0);
    MultiFab::Copy(fine_before, fine_state, 0, 0, layout.ncomp(), 0);
    MultiFab::Copy(coarse_rho_before, coarse_rho, 0, 0, 1, 0);
    MultiFab::Copy(coarse_omega_before, coarse_omega, 0, 0, 1, 0);
    MultiFab::Copy(fine_rho_before, fine_rho, 0, 0, 1, 0);
    MultiFab::Copy(fine_omega_before, fine_omega, 0, 0, 1, 0);

    const auto fine_view = timed_view(fine_state, fine_rho, fine_omega, 0.5);
    const auto coarse_view =
        timed_view(coarse_state, coarse_rho, coarse_omega, 0.5);
    EXPECT_FALSE(erf_sbm::RestrictMappedSpectrum(
        layout, fine_view, coarse_view, IntVect(2), 0, candidate, diagnostic));
    EXPECT_NE(diagnostic.find("recovering coarse spectrum"), std::string::npos)
        << diagnostic;
    expect_same_values(candidate, candidate_before);
    expect_same_values(coarse_state, coarse_before);
    expect_same_values(fine_state, fine_before);
    expect_same_values(coarse_rho, coarse_rho_before);
    expect_same_values(coarse_omega, coarse_omega_before);
    expect_same_values(fine_rho, fine_rho_before);
    expect_same_values(fine_omega, fine_omega_before);
}

TEST(SBMAMRTransfer, RestrictionLateQuotientFailureIsAtomic)
{
    run_restriction_late_quotient_failure_is_atomic();
}

void run_restriction_rejects_positive_mapped_average_underflow ()
{
    const auto layout = make_transfer_layout();
    const BoxArray coarse_ba(coarse_domain());
    const DistributionMapping coarse_dm(coarse_ba);
    const Box fine_box(IntVect(0, 0, 0), IntVect(1, 1, 1));
    const BoxArray fine_ba(fine_box);
    const DistributionMapping fine_dm(fine_ba);
    MultiFab coarse_rho(coarse_ba, coarse_dm, 1, 0);
    MultiFab coarse_omega(coarse_ba, coarse_dm, 1, 0);
    MultiFab fine_rho(fine_ba, fine_dm, 1, 0);
    MultiFab fine_omega(fine_ba, fine_dm, 1, 0);
    coarse_rho.setVal(Real(1.0));
    coarse_omega.setVal(Real(1.0));
    fine_rho.setVal(Real(1.0));
    fine_omega.setVal(Real(1.0));

    MultiFab coarse_state(coarse_ba, coarse_dm, layout.ncomp(), 0);
    MultiFab fine_state(fine_ba, fine_dm, layout.ncomp(), 0);
    MultiFab candidate(coarse_ba, coarse_dm, layout.ncomp(), 0);
    coarse_state.setVal(Real(0.0));
    fine_state.setVal(Real(0.0));
    candidate.setVal(Real(29.0));
    const int component = layout.populations()[0].mass_offset;
    const Real smallest = std::numeric_limits<Real>::denorm_min();
    ASSERT_GT(smallest, Real(0.0));
    volatile Real runtime_smallest = smallest;
    const Real averaged = static_cast<Real>(runtime_smallest) / Real(8.0);
    ASSERT_EQ(averaged, Real(0.0))
        << "the selected precision must round this one-child average to zero";

    const auto set_one_child = [&fine_state, component] (const Real amount) {
        fine_state.setVal(Real(0.0));
        for (amrex::MFIter mfi(fine_state, amrex::TilingIfNotGPU());
             mfi.isValid(); ++mfi) {
            const Box valid_box = mfi.validbox();
            const IntVect first_cell = valid_box.smallEnd();
            const Box cell(first_cell, first_cell);
            const auto state = fine_state.array(mfi);
            amrex::ParallelFor(
                cell, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    state(i, j, k, component) = amount;
                });
        }
    };

    set_one_child(smallest);
    std::string diagnostic;
    ASSERT_TRUE(fine_state.is_finite(0, layout.ncomp(), 0));
    ASSERT_TRUE(erf_sbm::authoritative_state_admissible(fine_state, layout, 1,
                                                        &diagnostic))
        << diagnostic;
    MultiFab fine_before(fine_ba, fine_dm, layout.ncomp(), 0);
    MultiFab coarse_before(coarse_ba, coarse_dm, layout.ncomp(), 0);
    MultiFab::Copy(fine_before, fine_state, 0, 0, layout.ncomp(), 0);
    MultiFab::Copy(coarse_before, coarse_state, 0, 0, layout.ncomp(), 0);

    const auto fine_view = timed_view(fine_state, fine_rho, fine_omega, 0.0);
    const auto coarse_view =
        timed_view(coarse_state, coarse_rho, coarse_omega, 0.0);
    EXPECT_FALSE(erf_sbm::RestrictMappedSpectrum(
        layout, fine_view, coarse_view, IntVect(2), 0, candidate, diagnostic));
    EXPECT_NE(diagnostic.find("underflow"), std::string::npos) << diagnostic;

    MultiFab fine_source_error(fine_ba, fine_dm, layout.ncomp(), 0);
    MultiFab coarse_source_error(coarse_ba, coarse_dm, layout.ncomp(), 0);
    MultiFab::Copy(fine_source_error, fine_state, 0, 0, layout.ncomp(), 0);
    MultiFab::Subtract(fine_source_error, fine_before, 0, 0, layout.ncomp(), 0);
    MultiFab::Copy(coarse_source_error, coarse_state, 0, 0, layout.ncomp(), 0);
    MultiFab::Subtract(coarse_source_error, coarse_before, 0, 0, layout.ncomp(),
                       0);
    for (int comp = 0; comp < layout.ncomp(); ++comp) {
        EXPECT_EQ(fine_source_error.norm0(comp), Real(0.0))
            << "fine component=" << comp;
        EXPECT_EQ(coarse_source_error.norm0(comp), Real(0.0))
            << "coarse component=" << comp;
    }

    // Eight subnormal units average to one representable subnormal unit.
    const Real representable_amount = Real(8.0) * smallest;
    ASSERT_GT(representable_amount, smallest);
    set_one_child(representable_amount);
    ASSERT_TRUE(erf_sbm::authoritative_state_admissible(fine_state, layout, 1,
                                                        &diagnostic))
        << diagnostic;
    EXPECT_TRUE(erf_sbm::RestrictMappedSpectrum(
        layout, fine_view, coarse_view, IntVect(2), 0, candidate, diagnostic))
        << diagnostic;
    MultiFab expected(coarse_ba, coarse_dm, layout.ncomp(), 0);
    expected.setVal(Real(0.0));
    for (amrex::MFIter mfi(expected, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi) {
        const IntVect coarse_cell(0, 0, 0);
        const Box cell(coarse_cell, coarse_cell);
        const auto state = expected.array(mfi);
        amrex::ParallelFor(cell,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                               state(i, j, k, component) = smallest;
                           });
    }
    MultiFab positive_control_error(coarse_ba, coarse_dm, layout.ncomp(), 0);
    MultiFab::Copy(positive_control_error, candidate, 0, 0, layout.ncomp(), 0);
    MultiFab::Subtract(positive_control_error, expected, 0, 0, layout.ncomp(),
                       0);
    for (int comp = 0; comp < layout.ncomp(); ++comp) {
        EXPECT_EQ(positive_control_error.norm0(comp), Real(0.0))
            << "positive-control component=" << comp;
    }
}

TEST(SBMAMRTransfer, RestrictionRejectsPositiveMappedAverageUnderflow)
{
    run_restriction_rejects_positive_mapped_average_underflow();
}

void run_prolongation_uses_same_time_piecewise_constant_carrier_ratio ()
{
    const auto layout = make_transfer_layout();
    const Box c_domain = coarse_domain();
    Box f_domain = c_domain;
    f_domain.refine(IntVect(2));
    const Geometry cgeom = make_geometry(c_domain);
    const Geometry fgeom = make_geometry(f_domain);
    const BoxArray coarse_ba(c_domain);
    const DistributionMapping coarse_dm(coarse_ba);
    const BoxArray fine_ba(fine_coverage());
    const DistributionMapping fine_dm(fine_ba);

    MultiFab coarse_rho(coarse_ba, coarse_dm, 1, 0);
    MultiFab coarse_omega(coarse_ba, coarse_dm, 1, 0);
    MultiFab fine_rho(fine_ba, fine_dm, 1, 0);
    MultiFab fine_omega(fine_ba, fine_dm, 1, 0);
    fill_carriers(coarse_rho, coarse_omega, true);
    fill_carriers(fine_rho, fine_omega, false, true);
    MultiFab coarse_state(coarse_ba, coarse_dm, layout.ncomp(), 0);
    MultiFab fine_storage(fine_ba, fine_dm, layout.ncomp(), 0);
    MultiFab candidate(fine_ba, fine_dm, layout.ncomp(), 0);
    fill_spectrum(coarse_state, coarse_rho, layout, true);
    fine_storage.setVal(Real(0.0));
    candidate.setVal(Real(77.0));

    auto coarse_view = timed_view(coarse_state, coarse_rho, coarse_omega, 0.5);
    auto fine_view = timed_view(fine_storage, fine_rho, fine_omega, 0.5);
    std::string diagnostic;
    ASSERT_TRUE(erf_sbm::ProlongCarrierRelativeSpectrum(
        layout, coarse_view, fine_view, cgeom, fgeom, IntVect(2), 1,
        candidate, diagnostic)) << diagnostic;
    ASSERT_TRUE(erf_sbm::authoritative_state_admissible(
        candidate, layout, 1, &diagnostic)) << diagnostic;

    MultiFab expected(fine_ba, fine_dm, layout.ncomp(), 0);
    const ComponentOffsets offsets = offsets_for(layout);
    const int ncomp = layout.ncomp();
    for (amrex::MFIter mfi(expected, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box box = mfi.tilebox();
        const auto rho = fine_rho.const_array(mfi);
        const auto result = expected.array(mfi);
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const int ci = i / 2;
            const int cj = j / 2;
            const int ck = k / 2;
            const Real carrier = rho(i, j, k, 0);
            for (int component = 0; component < ncomp; ++component) {
                result(i, j, k, component) = carrier * base_component(
                    component, ci, cj, ck, offsets, true);
            }
        });
    }
    MultiFab difference(fine_ba, fine_dm, ncomp, 0);
    MultiFab::Copy(difference, candidate, 0, 0, ncomp, 0);
    MultiFab::Subtract(difference, expected, 0, 0, ncomp, 0);
    for (int component = 0; component < ncomp; ++component) {
        EXPECT_LE(difference.norm0(component),
                  Real(64.0) * std::numeric_limits<Real>::epsilon())
            << "component=" << component;
    }

    // Every child carries the parent dry-air-relative state, including the
    // 2M mass/number means and attached material ratios.
    const auto& two = layout.populations()[1];
    MultiFab ratio_error(fine_ba, fine_dm, 2, 0);
    for (amrex::MFIter mfi(ratio_error, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box box = mfi.tilebox();
        const auto state = candidate.const_array(mfi);
        const auto errors = ratio_error.array(mfi);
        const int mass0 = two.mass_offset;
        const int mass1 = two.mass_offset + 1;
        const int number0 = two.number_offset;
        const int number1 = two.number_offset + 1;
        const int property0 = layout.property_offset(1);
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const int ci = i / 2;
            const int cj = j / 2;
            const int ck = k / 2;
            const bool empty_upper = ci == 1 && cj == 1 && ck == 1;
            const Real n0 = state(i, j, k, number0);
            const Real n1 = state(i, j, k, number1);
            errors(i, j, k, 0) = n0 > Real(0.0)
                ? amrex::Math::abs(state(i, j, k, mass0) / n0 - Real(0.35))
                : Real(0.0);
            errors(i, j, k, 1) = empty_upper
                ? amrex::Math::abs(n1) + amrex::Math::abs(state(i, j, k, mass1)) +
                      amrex::Math::abs(state(i, j, k, property0 + 1))
                : (n1 > Real(0.0)
                       ? amrex::Math::abs(state(i, j, k, mass1) / n1 - Real(1.35)) +
                             amrex::Math::abs(state(i, j, k, property0 + 1) / n1 - Real(0.5))
                       : Real(0.0));
        });
    }
    EXPECT_LE(ratio_error.norm0(0),
              Real(64.0) * std::numeric_limits<Real>::epsilon());
    EXPECT_LE(ratio_error.norm0(1),
              Real(64.0) * std::numeric_limits<Real>::epsilon());

    // Compare mapped inventories over the refined patch.  Fine cell volumes
    // are 1/8 of coarse computational volumes at this refinement ratio.
    MultiFab coarse_inventory(coarse_ba, coarse_dm, ncomp, 0);
    MultiFab fine_inventory(fine_ba, fine_dm, ncomp, 0);
    coarse_inventory.setVal(Real(0.0));
    for (amrex::MFIter mfi(coarse_inventory, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box box = mfi.tilebox();
        const auto u = coarse_state.const_array(mfi);
        const auto omega = coarse_omega.const_array(mfi);
        const auto h = coarse_inventory.array(mfi);
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const bool covered = i >= 1 && i <= 2 && j >= 1 && j <= 2 && k >= 1 && k <= 2;
            for (int component = 0; component < ncomp; ++component) {
                h(i, j, k, component) = covered
                    ? omega(i, j, k, 0) * u(i, j, k, component)
                    : Real(0.0);
            }
        });
    }
    for (amrex::MFIter mfi(fine_inventory, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box box = mfi.tilebox();
        const auto u = candidate.const_array(mfi);
        const auto omega = fine_omega.const_array(mfi);
        const auto h = fine_inventory.array(mfi);
        amrex::ParallelFor(box, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int component) noexcept {
            h(i, j, k, component) = omega(i, j, k, 0) * u(i, j, k, component);
        });
    }
    for (int component = 0; component < ncomp; ++component) {
        const Real coarse_sum = coarse_inventory.sum(component);
        const Real fine_sum = fine_inventory.sum(component) / Real(8.0);
        EXPECT_NEAR(coarse_sum, fine_sum,
                    Real(128.0) * std::numeric_limits<Real>::epsilon() *
                        std::max(Real(1.0), amrex::Math::abs(coarse_sum)));
    }

    // A tuple with a wrong-time density is rejected before candidate writes.
    candidate.setVal(Real(77.0));
    auto stale_fine_view = fine_view;
    stale_fine_view.density_time = 0.75;
    EXPECT_FALSE(erf_sbm::ProlongCarrierRelativeSpectrum(
        layout, coarse_view, stale_fine_view, cgeom, fgeom, IntVect(2), 1,
        candidate, diagnostic));
    for (int component = 0; component < ncomp; ++component) {
        EXPECT_DOUBLE_EQ(candidate.min(component), Real(77.0))
            << "component=" << component;
        EXPECT_DOUBLE_EQ(candidate.max(component), Real(77.0))
            << "component=" << component;
    }
    EXPECT_NE(diagnostic.find("times do not match"), std::string::npos);

    candidate.setVal(Real(77.0));
    EXPECT_FALSE(erf_sbm::ProlongCarrierRelativeSpectrum(
        layout, coarse_view, fine_view, cgeom, fgeom, IntVect(2), 0,
        candidate, diagnostic));
    EXPECT_NE(diagnostic.find("fine level"), std::string::npos);
    for (int component = 0; component < ncomp; ++component) {
        EXPECT_DOUBLE_EQ(candidate.min(component), Real(77.0))
            << "component=" << component;
        EXPECT_DOUBLE_EQ(candidate.max(component), Real(77.0))
            << "component=" << component;
    }

    auto incomplete_view = fine_view;
    incomplete_view.dry_air_density = nullptr;
    EXPECT_FALSE(erf_sbm::ProlongCarrierRelativeSpectrum(
        layout, coarse_view, incomplete_view, cgeom, fgeom, IntVect(2), 1,
        candidate, diagnostic));
    EXPECT_NE(diagnostic.find("incomplete"), std::string::npos);
    for (int component = 0; component < ncomp; ++component) {
        EXPECT_DOUBLE_EQ(candidate.min(component), Real(77.0))
            << "component=" << component;
        EXPECT_DOUBLE_EQ(candidate.max(component), Real(77.0))
            << "component=" << component;
    }

    // Exact same-time views also work at a later semantic time after the
    // carrier and conservative spectrum are both changed consistently.
    coarse_rho.mult(Real(2.0), 0, 1, 0);
    fine_rho.mult(Real(2.0), 0, 1, 0);
    coarse_state.mult(Real(2.0), 0, ncomp, 0);
    coarse_view = timed_view(coarse_state, coarse_rho, coarse_omega, 1.5);
    fine_view = timed_view(fine_storage, fine_rho, fine_omega, 1.5);
    ASSERT_TRUE(erf_sbm::ProlongCarrierRelativeSpectrum(
        layout, coarse_view, fine_view, cgeom, fgeom, IntVect(2), 1,
        candidate, diagnostic)) << diagnostic;
    candidate.mult(Real(0.5), 0, ncomp, 0);
    MultiFab::Subtract(candidate, expected, 0, 0, ncomp, 0);
    for (int component = 0; component < ncomp; ++component) {
        EXPECT_LE(candidate.norm0(component),
                  Real(64.0) * std::numeric_limits<Real>::epsilon())
            << "component=" << component;
    }
}

TEST(SBMAMRTransfer, ProlongationUsesSameTimePiecewiseConstantCarrierRatio)
{
    run_prolongation_uses_same_time_piecewise_constant_carrier_ratio();
}

TEST(SBMAMRTransfer, ProlongationRejectsQuotientAndProductUnderflow)
{
    const auto layout = make_transfer_layout();
    const Box c_domain = coarse_domain();
    Box f_domain = c_domain;
    f_domain.refine(IntVect(2));
    const Geometry cgeom = make_geometry(c_domain);
    const Geometry fgeom = make_geometry(f_domain);
    const BoxArray coarse_ba(c_domain);
    const DistributionMapping coarse_dm(coarse_ba);
    const BoxArray fine_ba(fine_coverage());
    const DistributionMapping fine_dm(fine_ba);
    MultiFab coarse_rho(coarse_ba, coarse_dm, 1, 0);
    MultiFab coarse_omega(coarse_ba, coarse_dm, 1, 0);
    MultiFab fine_rho(fine_ba, fine_dm, 1, 0);
    MultiFab fine_omega(fine_ba, fine_dm, 1, 0);
    fill_carriers(coarse_rho, coarse_omega, true);
    fill_carriers(fine_rho, fine_omega, false, true);
    MultiFab coarse_state(coarse_ba, coarse_dm, layout.ncomp(), 0);
    MultiFab fine_storage(fine_ba, fine_dm, layout.ncomp(), 0);
    MultiFab candidate(fine_ba, fine_dm, layout.ncomp(), 0);
    fill_spectrum(coarse_state, coarse_rho, layout);
    fine_storage.setVal(Real(0.0));
    candidate.setVal(Real(33.0));
    const Real tiny = std::numeric_limits<Real>::denorm_min();
    const Real tiny_scale = Real(1000.0) * std::numeric_limits<Real>::min();
    ASSERT_GT(tiny, Real(0.0));

    coarse_state.mult(tiny_scale, 0, layout.ncomp(), 0);
    coarse_rho.setVal(std::numeric_limits<Real>::max());
    auto coarse_view = timed_view(coarse_state, coarse_rho, coarse_omega, 0.5);
    auto fine_view = timed_view(fine_storage, fine_rho, fine_omega, 0.5);
    std::string diagnostic;
    EXPECT_FALSE(erf_sbm::ProlongCarrierRelativeSpectrum(
        layout, coarse_view, fine_view, cgeom, fgeom, IntVect(2), 1,
        candidate, diagnostic));
    EXPECT_NE(diagnostic.find("underflowed"), std::string::npos);

    coarse_rho.setVal(Real(1.0));
    fill_spectrum(coarse_state, coarse_rho, layout);
    coarse_state.mult(tiny_scale, 0, layout.ncomp(), 0);
    fine_rho.setVal(tiny);
    fine_omega.setVal(Real(1.0));
    candidate.setVal(Real(34.0));
    MultiFab candidate_before(fine_ba, fine_dm, layout.ncomp(), 0);
    MultiFab coarse_before(coarse_ba, coarse_dm, layout.ncomp(), 0);
    MultiFab fine_rho_before(fine_ba, fine_dm, 1, 0);
    MultiFab fine_omega_before(fine_ba, fine_dm, 1, 0);
    MultiFab coarse_state_before(coarse_ba, coarse_dm, layout.ncomp(), 0);
    MultiFab::Copy(candidate_before, candidate, 0, 0, layout.ncomp(), 0);
    MultiFab::Copy(coarse_before, coarse_state, 0, 0, layout.ncomp(), 0);
    MultiFab::Copy(fine_rho_before, fine_rho, 0, 0, 1, 0);
    MultiFab::Copy(fine_omega_before, fine_omega, 0, 0, 1, 0);
    MultiFab::Copy(coarse_state_before, coarse_state, 0, 0, layout.ncomp(), 0);
    coarse_view = timed_view(coarse_state, coarse_rho, coarse_omega, 0.5);
    fine_view = timed_view(fine_storage, fine_rho, fine_omega, 0.5);
    EXPECT_FALSE(erf_sbm::ProlongCarrierRelativeSpectrum(
        layout, coarse_view, fine_view, cgeom, fgeom, IntVect(2), 1,
        candidate, diagnostic));
    EXPECT_NE(diagnostic.find("underflowed"), std::string::npos);
    expect_same_values(candidate, candidate_before);
    expect_same_values(coarse_state, coarse_state_before);
    expect_same_values(fine_rho, fine_rho_before);
    expect_same_values(fine_omega, fine_omega_before);

    // A finite but large fine carrier makes rho_f*z overflow only during the
    // last reconstruction pass, after the old implementation had written the
    // public destination.
    coarse_rho.setVal(Real(1.0));
    fill_spectrum(coarse_state, coarse_rho, layout);
    coarse_state.mult(Real(1.0e30), 0, layout.ncomp(), 0);
    fine_rho.setVal(std::numeric_limits<Real>::max());
    fine_omega.setVal(Real(1.0));
    candidate.setVal(Real(35.0));
    MultiFab::Copy(candidate_before, candidate, 0, 0, layout.ncomp(), 0);
    MultiFab::Copy(coarse_state_before, coarse_state, 0, 0, layout.ncomp(), 0);
    MultiFab::Copy(fine_rho_before, fine_rho, 0, 0, 1, 0);
    MultiFab::Copy(fine_omega_before, fine_omega, 0, 0, 1, 0);
    coarse_view = timed_view(coarse_state, coarse_rho, coarse_omega, 0.5);
    fine_view = timed_view(fine_storage, fine_rho, fine_omega, 0.5);
    EXPECT_FALSE(erf_sbm::ProlongCarrierRelativeSpectrum(
        layout, coarse_view, fine_view, cgeom, fgeom, IntVect(2), 1, candidate,
        diagnostic));
    EXPECT_NE(diagnostic.find("overflowed or underflowed"), std::string::npos)
        << diagnostic;
    expect_same_values(candidate, candidate_before);
    expect_same_values(coarse_state, coarse_state_before);
    expect_same_values(fine_rho, fine_rho_before);
    expect_same_values(fine_omega, fine_omega_before);
}

TEST(SBMAMRTransfer, RestrictionRejectsNoncoarsenableFineBoxesBeforeAverage)
{
    const auto layout = make_transfer_layout();
    const Box coarse_domain(IntVect(0), IntVect(2));
    const Box fine_domain(IntVect(0), IntVect(7));
    const BoxArray coarse_ba(coarse_domain);
    const BoxArray fine_ba(fine_domain);
    const DistributionMapping coarse_dm(coarse_ba);
    const DistributionMapping fine_dm(fine_ba);
    const IntVect ratio(3);
    ASSERT_FALSE(fine_ba.coarsenable(ratio));

    MultiFab coarse_rho(coarse_ba, coarse_dm, 1, 0);
    MultiFab coarse_omega(coarse_ba, coarse_dm, 1, 0);
    MultiFab fine_rho(fine_ba, fine_dm, 1, 0);
    MultiFab fine_omega(fine_ba, fine_dm, 1, 0);
    fill_carriers(coarse_rho, coarse_omega, true);
    fill_carriers(fine_rho, fine_omega, false);
    MultiFab coarse_state(coarse_ba, coarse_dm, layout.ncomp(), 0);
    MultiFab fine_state(fine_ba, fine_dm, layout.ncomp(), 0);
    fill_spectrum(coarse_state, coarse_rho, layout);
    fill_spectrum(fine_state, fine_rho, layout);
    MultiFab coarse_before(coarse_ba, coarse_dm, layout.ncomp(), 0);
    MultiFab fine_before(fine_ba, fine_dm, layout.ncomp(), 0);
    MultiFab::Copy(coarse_before, coarse_state, 0, 0, layout.ncomp(), 0);
    MultiFab::Copy(fine_before, fine_state, 0, 0, layout.ncomp(), 0);

    MultiFab candidate(coarse_ba, coarse_dm, layout.ncomp(), 0);
    MultiFab candidate_before(coarse_ba, coarse_dm, layout.ncomp(), 0);
    candidate.setVal(Real(-23.0));
    MultiFab::Copy(candidate_before, candidate, 0, 0, layout.ncomp(), 0);
    const auto fine_view = timed_view(fine_state, fine_rho, fine_omega, 0.3);
    const auto coarse_view =
        timed_view(coarse_state, coarse_rho, coarse_omega, 0.3);
    std::string diagnostic;
    EXPECT_FALSE(erf_sbm::RestrictMappedSpectrum(
        layout, fine_view, coarse_view, ratio, 0, candidate, diagnostic));
    EXPECT_NE(diagnostic.find("coarsenable"), std::string::npos) << diagnostic;
    expect_same_values(candidate, candidate_before);
    expect_same_values(coarse_state, coarse_before);
    expect_same_values(fine_state, fine_before);
}

void
run_prolongation_diagnostic_precedence ()
{
    const auto layout = make_transfer_layout();
    const Box c_domain = coarse_domain();
    Box f_domain = c_domain;
    f_domain.refine(IntVect(2));
    const BoxArray coarse_ba(c_domain);
    const BoxArray fine_ba(f_domain);
    const DistributionMapping coarse_dm(coarse_ba);
    const DistributionMapping fine_dm(fine_ba);
    const Geometry cgeom = make_geometry(c_domain);
    const Geometry nonperiodic_fgeom =
        make_geometry_with_nonperiodic_direction(f_domain, 0);
    Box bad_f_domain = f_domain;
    IntVect bad_hi = bad_f_domain.bigEnd();
    bad_hi[0] -= 1;
    bad_f_domain = Box(bad_f_domain.smallEnd(), bad_hi);
    const Geometry bad_fgeom = make_geometry(bad_f_domain);

    MultiFab coarse_rho(coarse_ba, coarse_dm, 1, 0);
    MultiFab coarse_omega(coarse_ba, coarse_dm, 1, 0);
    MultiFab fine_rho(fine_ba, fine_dm, 1, 0);
    MultiFab fine_omega(fine_ba, fine_dm, 1, 0);
    fill_carriers(coarse_rho, coarse_omega, true);
    fill_carriers(fine_rho, fine_omega, false, true);
    MultiFab coarse_state(coarse_ba, coarse_dm, layout.ncomp(), 0);
    MultiFab fine_target_state(fine_ba, fine_dm, layout.ncomp(), 0);
    fill_spectrum(coarse_state, coarse_rho, layout, true);
    fine_target_state.setVal(Real(0.0));
    const auto coarse_view =
        timed_view(coarse_state, coarse_rho, coarse_omega, 0.5);
    const auto fine_view =
        timed_view(fine_target_state, fine_rho, fine_omega, 0.5);
    MultiFab coarse_before(coarse_ba, coarse_dm, layout.ncomp(), 0);
    MultiFab fine_before(fine_ba, fine_dm, layout.ncomp(), 0);
    MultiFab coarse_rho_before(coarse_ba, coarse_dm, 1, 0);
    MultiFab coarse_omega_before(coarse_ba, coarse_dm, 1, 0);
    MultiFab fine_rho_before(fine_ba, fine_dm, 1, 0);
    MultiFab fine_omega_before(fine_ba, fine_dm, 1, 0);
    MultiFab::Copy(coarse_before, coarse_state, 0, 0, layout.ncomp(), 0);
    MultiFab::Copy(fine_before, fine_target_state, 0, 0, layout.ncomp(), 0);
    MultiFab::Copy(coarse_rho_before, coarse_rho, 0, 0, 1, 0);
    MultiFab::Copy(coarse_omega_before, coarse_omega, 0, 0, 1, 0);
    MultiFab::Copy(fine_rho_before, fine_rho, 0, 0, 1, 0);
    MultiFab::Copy(fine_omega_before, fine_omega, 0, 0, 1, 0);

    std::string diagnostic;
    // The earlier alias/layout defect must survive a later geometry failure.
    EXPECT_FALSE(erf_sbm::ProlongCarrierRelativeSpectrum(
        layout, coarse_view, fine_view, cgeom, nonperiodic_fgeom, IntVect(2), 1,
        fine_target_state, diagnostic));
    EXPECT_NE(diagnostic.find("separate from both input tuples"),
              std::string::npos)
        << diagnostic;
    expect_same_values(fine_target_state, fine_before);

    MultiFab candidate(fine_ba, fine_dm, layout.ncomp(), 0);
    candidate.setVal(Real(83.0));
    MultiFab candidate_before(fine_ba, fine_dm, layout.ncomp(), 0);
    MultiFab::Copy(candidate_before, candidate, 0, 0, layout.ncomp(), 0);
    EXPECT_FALSE(erf_sbm::ProlongCarrierRelativeSpectrum(
        layout, coarse_view, fine_view, cgeom, nonperiodic_fgeom, IntVect(2), 1,
        candidate, diagnostic));
    EXPECT_NE(diagnostic.find("periodic geometry"), std::string::npos)
        << diagnostic;
    expect_same_values(candidate, candidate_before);

    candidate.setVal(Real(84.0));
    MultiFab::Copy(candidate_before, candidate, 0, 0, layout.ncomp(), 0);
    EXPECT_FALSE(erf_sbm::ProlongCarrierRelativeSpectrum(
        layout, coarse_view, fine_view, cgeom, bad_fgeom, IntVect(2), 1,
        candidate, diagnostic));
    EXPECT_NE(diagnostic.find("does not map the coarse domain"),
              std::string::npos)
        << diagnostic;
    expect_same_values(candidate, candidate_before);
    expect_same_values(coarse_state, coarse_before);
    expect_same_values(fine_target_state, fine_before);
    expect_same_values(coarse_rho, coarse_rho_before);
    expect_same_values(coarse_omega, coarse_omega_before);
    expect_same_values(fine_rho, fine_rho_before);
    expect_same_values(fine_omega, fine_omega_before);
}

TEST(SBMAMRTransfer, ProlongationPreservesDiagnosticPrecedence)
{
    run_prolongation_diagnostic_precedence();
}

TEST(SBMAMRYA, SignedMappedCorrectionMatchesIndependentFaceAmounts)
{
    ASSERT_EQ(AMREX_SPACEDIM, 3);
    const auto layout = make_transfer_layout();
    const int ncomp = layout.ncomp();
    const IntVect ratio(2);
    const Box coarse_domain(IntVect(0), IntVect(7));
    const Box fine_domain(IntVect(0), IntVect(15));
    const Box covered_fine_box = amrex::refine(Box(IntVect(2), IntVect(5)), ratio);
    const BoxArray coarse_ba(coarse_domain);
    const BoxArray fine_ba(covered_fine_box);
    const DistributionMapping coarse_dm(coarse_ba);
    const DistributionMapping fine_dm(fine_ba);
    const Geometry coarse_geom = make_geometry(coarse_domain);
    const Geometry fine_geom = make_geometry(fine_domain);

    MultiFab coarse_cells(coarse_ba, coarse_dm, ncomp, 0);
    MultiFab fine_cells(fine_ba, fine_dm, ncomp, 0);
    for (int n = 0; n < ncomp; ++n) {
        coarse_cells.setVal(Real(0.25) * (n + 1), n, 1, 0);
    }

    std::array<MultiFab, AMREX_SPACEDIM> coarse_flux;
    std::array<MultiFab, AMREX_SPACEDIM> fine_flux;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        BoxArray coarse_faces = coarse_ba;
        BoxArray fine_faces = fine_ba;
        coarse_faces.surroundingNodes(dir);
        fine_faces.surroundingNodes(dir);
        coarse_flux[dir].define(coarse_faces, coarse_dm, ncomp, 0);
        fine_flux[dir].define(fine_faces, fine_dm, ncomp, 0);
        for (MFIter mfi(coarse_flux[dir], false); mfi.isValid(); ++mfi) {
            const Box box = mfi.validbox();
            const auto flux = coarse_flux[dir].array(mfi);
            ParallelFor(box, ncomp,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept {
                flux(i,j,k,n) = ya_oracle_coarse_flux(dir, i, j, k, n);
            });
        }
        for (MFIter mfi(fine_flux[dir], false); mfi.isValid(); ++mfi) {
            const Box box = mfi.validbox();
            const auto flux = fine_flux[dir].array(mfi);
            ParallelFor(box, ncomp,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept {
                flux(i,j,k,n) = ya_oracle_fine_flux(dir, i, j, k, n);
            });
        }
    }
    amrex::Gpu::streamSynchronize();

    amrex::YAFluxRegister register_for_sbm(
        fine_ba, coarse_ba, fine_dm, coarse_dm, fine_geom, coarse_geom,
        ratio, 1, ncomp);
    register_for_sbm.reset();
    const auto coarse_dx = coarse_geom.CellSizeArray();
    const auto fine_dx = fine_geom.CellSizeArray();
    for (MFIter mfi(coarse_cells, false); mfi.isValid(); ++mfi) {
        std::array<amrex::FArrayBox const*, AMREX_SPACEDIM> fluxes{
            &coarse_flux[0][mfi], &coarse_flux[1][mfi], &coarse_flux[2][mfi]};
        register_for_sbm.CrseAdd(mfi, fluxes, coarse_dx.data(), Real(1.0),
                                 amrex::RunOn::Host);
    }
    for (MFIter mfi(fine_cells, false); mfi.isValid(); ++mfi) {
        std::array<amrex::FArrayBox const*, AMREX_SPACEDIM> fluxes{
            &fine_flux[0][mfi], &fine_flux[1][mfi], &fine_flux[2][mfi]};
        register_for_sbm.FineAdd(mfi, fluxes, fine_dx.data(), Real(1.0),
                                 amrex::RunOn::Host);
    }
    register_for_sbm.Reflux(coarse_cells, 0, 0, ncomp);
    amrex::Gpu::streamSynchronize();

    const auto coarse_dx_product =
        coarse_dx[0] * coarse_dx[1] * coarse_dx[2];
    const int tangential[3][2] = {{1, 2}, {0, 2}, {0, 1}};
    for (MFIter mfi(coarse_cells, false); mfi.isValid(); ++mfi) {
        const Box box = mfi.validbox();
        const auto state = coarse_cells.const_array(mfi);
        for (int k = box.smallEnd(2); k <= box.bigEnd(2); ++k) {
            for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
                for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
                    const int c[3] = {i, j, k};
                    const bool covered = i >= 2 && i <= 5 &&
                                         j >= 2 && j <= 5 &&
                                         k >= 2 && k <= 5;
                    const Real omega = Real(1.5) + Real(0.01) * i +
                                       Real(0.02) * j + Real(0.03) * k;
                    for (int n = 0; n < ncomp; ++n) {
                        const Real initial = Real(0.25) * (n + 1);
                        Real expected_delta_h = Real(0.0);
                        Real expected_delta_q = Real(0.0);
                        if (!covered) {
                            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                                for (int side : {-1, 1}) {
                                    const int neighbor = c[dir] + side;
                                    const bool neighbor_covered = neighbor >= 2 &&
                                        neighbor <= 5 &&
                                        c[(dir + 1) % 3] >= 2 &&
                                        c[(dir + 1) % 3] <= 5 &&
                                        c[(dir + 2) % 3] >= 2 &&
                                        c[(dir + 2) % 3] <= 5;
                                    if (!neighbor_covered) continue;

                                    const int face = c[dir] + (side > 0 ? 1 : 0);
                                    int coarse_face[3] = {i, j, k};
                                    coarse_face[dir] = face;
                                    const Real coarse_integral =
                                        ya_oracle_coarse_flux(
                                            dir, coarse_face[0], coarse_face[1],
                                            coarse_face[2], n);

                                    int fine_face[3] = {2*i, 2*j, 2*k};
                                    fine_face[dir] = 2 * face;
                                    Real fine_sum = Real(0.0);
                                    for (int a = 0; a < 2; ++a) {
                                        for (int b = 0; b < 2; ++b) {
                                            int subface[3] = {fine_face[0],
                                                              fine_face[1],
                                                              fine_face[2]};
                                            subface[tangential[dir][0]] += a;
                                            subface[tangential[dir][1]] += b;
                                            fine_sum += ya_oracle_fine_flux(
                                                dir, subface[0], subface[1],
                                                subface[2], n);
                                        }
                                    }
                                    const Real sigma = side > 0
                                        ? Real(1.0) : Real(-1.0);
                                    expected_delta_h += sigma / coarse_dx[dir] *
                                        (coarse_integral - fine_sum / Real(4.0));

                                    const Real coarse_area =
                                        coarse_dx_product / coarse_dx[dir];
                                    const Real fine_cell_volume =
                                        coarse_dx_product / Real(8.0);
                                    const Real fine_area =
                                        fine_cell_volume / fine_dx[dir];
                                    expected_delta_q += sigma *
                                        (coarse_area * coarse_integral -
                                         fine_area * fine_sum);
                                }
                            }
                        }
                        const Real expected = initial + expected_delta_h;
                        EXPECT_NEAR(state(i,j,k,n), expected, Real(2.0e-11))
                            << "cell=" << i << ',' << j << ',' << k
                            << " component=" << n;
                        const Real delta_u = (state(i,j,k,n) - initial) / omega;
                        const Real mapped_amount = coarse_dx_product * omega * delta_u;
                        EXPECT_NEAR(mapped_amount,
                                     expected_delta_q,
                                     Real(2.0e-11));
                        if (covered) {
                            EXPECT_EQ(state(i,j,k,n), initial)
                                << "covered coarse parent changed at "
                                << i << ',' << j << ',' << k
                                << " component=" << n;
                        }
                    }
                }
            }
        }
    }
}

TEST(SBMAMRYA, DirectAcceptedStageRatesMatchCompletedIntegralReference)
{
    const auto layout = make_transfer_layout();
    const int ncomp = layout.ncomp();
    const IntVect ratio(2);
    const Box coarse_domain(IntVect(0), IntVect(7));
    const Box fine_domain(IntVect(0), IntVect(15));
    const BoxArray coarse_ba(coarse_domain);
    const BoxArray fine_ba(amrex::refine(Box(IntVect(2), IntVect(5)), ratio));
    const DistributionMapping coarse_dm(coarse_ba);
    const DistributionMapping fine_dm(fine_ba);
    const Geometry coarse_geom = make_geometry(coarse_domain);
    const Geometry fine_geom = make_geometry(fine_domain);
    const double coarse_begin = 0.0;
    const double coarse_end = 0.3;
    constexpr int fine_substeps = 2;
    const double fine_dt = (coarse_end - coarse_begin) / fine_substeps;

    for (const auto method : {erf_auxiliary::HostIntegrator::CompressibleRK3,
                              erf_auxiliary::HostIntegrator::AnelasticHeun}) {
        const int method_id = method ==
            erf_auxiliary::HostIntegrator::CompressibleRK3 ? 0 : 1;
        const int stages = method_id == 0 ? 3 : 2;
        const auto stage_weight = [=] (int stage, double dt) {
            if (method_id == 0) return stage == 2 ? dt : 0.0;
            return 0.5 * dt;
        };

        erf_sbm::SBMAMRFluxRegister direct_register;
        direct_register.define(fine_ba, coarse_ba, fine_dm, coarse_dm,
                               fine_geom, coarse_geom, ratio, 1, ncomp);
        std::string diagnostic;
        ASSERT_TRUE(direct_register.begin_interval(
            coarse_begin, coarse_end, fine_substeps, diagnostic)) << diagnostic;

        erf_auxiliary::IntegratedMappedFaceFlux coarse_integral;
        coarse_integral.define(coarse_ba, coarse_dm, ncomp, 0);
        coarse_integral.setVal(Real(0.0));
        erf_auxiliary::IntegratedMappedFaceFlux fine_integral;
        fine_integral.define(fine_ba, fine_dm, ncomp, 0);
        fine_integral.setVal(Real(0.0));

        auto advance_reference_step = [&] (
            const bool coarse_side, const double step_begin,
            const double step_end, const double dt, const int substep) {
            const auto& ba = coarse_side ? coarse_ba : fine_ba;
            const auto& dm = coarse_side ? coarse_dm : fine_dm;
            auto& integral = coarse_side ? coarse_integral : fine_integral;
            for (int stage = 0; stage < stages; ++stage) {
                EXPECT_TRUE(direct_register.begin_stage(
                    coarse_side, method, stage, step_begin, diagnostic))
                    << diagnostic;
                const double weight = stage_weight(stage, dt);
                // Reverse chunk order on alternating stages/substeps.  The
                // global component identity must be independent of traversal.
                for (int chunk = 1; chunk >= 0; --chunk) {
                    const int first = chunk * 5;
                    auto accepted = make_rate_chunk(
                        ba, dm, first, 5, method_id, stage, substep);
                    const std::vector<int> component_map{
                        first, first + 1, first + 2, first + 3, first + 4};
                    ASSERT_TRUE(direct_register.add_stage_rates(
                        coarse_side, accepted, component_map, weight,
                        diagnostic)) << diagnostic;
                    accumulate_reference_integral(
                        integral, accepted, first, weight);
                }
                EXPECT_TRUE(direct_register.finish_stage(
                    coarse_side, diagnostic)) << diagnostic;
            }
            EXPECT_TRUE(direct_register.accept_level_step(
                coarse_side, step_begin, step_end, diagnostic)) << diagnostic;
        };

        ASSERT_TRUE(direct_register.begin_level_step(
            true, coarse_begin, coarse_end, diagnostic)) << diagnostic;
        advance_reference_step(true, coarse_begin, coarse_end,
                               coarse_end - coarse_begin, 0);

        MultiFab direct_state(coarse_ba, coarse_dm, ncomp, 0);
        MultiFab reference_state(coarse_ba, coarse_dm, ncomp, 0);
        MultiFab before_reflux(coarse_ba, coarse_dm, ncomp, 0);
        for (int n = 0; n < ncomp; ++n) {
            direct_state.setVal(Real(0.25) * (n + 1), n, 1, 0);
            reference_state.setVal(Real(0.25) * (n + 1), n, 1, 0);
        }

        for (int substep = 0; substep < fine_substeps; ++substep) {
            const double step_begin = coarse_begin + substep * fine_dt;
            const double step_end = step_begin + fine_dt;
            ASSERT_TRUE(direct_register.begin_level_step(
                false, step_begin, step_end, diagnostic)) << diagnostic;
            advance_reference_step(false, step_begin, step_end,
                                   fine_dt, substep + 1);
            if (substep == 0) {
                EXPECT_FALSE(direct_register.ready_to_reflux());
                MultiFab::Copy(before_reflux, direct_state, 0, 0, ncomp, 0);
                EXPECT_FALSE(direct_register.reflux(direct_state, diagnostic));
                expect_same_values(direct_state, before_reflux);
            }
        }
        ASSERT_TRUE(direct_register.ready_to_reflux());

        amrex::YAFluxRegister reference_register(
            fine_ba, coarse_ba, fine_dm, coarse_dm, fine_geom, coarse_geom,
            ratio, 1, ncomp);
        reference_register.reset();
        add_integral_to_reference_register(
            reference_register, coarse_integral, coarse_ba, coarse_dm,
            coarse_geom, true);
        add_integral_to_reference_register(
            reference_register, fine_integral, fine_ba, fine_dm,
            fine_geom, false);
        reference_register.Reflux(reference_state, 0, 0, ncomp);
        amrex::Gpu::streamSynchronize();

        ASSERT_TRUE(direct_register.reflux(direct_state, diagnostic))
            << diagnostic;
        expect_roundoff_equivalent(direct_state, reference_state);
        EXPECT_TRUE(direct_register.has_outstanding_amount());
        ASSERT_TRUE(direct_register.accept_reflux(diagnostic)) << diagnostic;
        EXPECT_FALSE(direct_register.has_outstanding_amount());
        MultiFab consumed_snapshot(coarse_ba, coarse_dm, ncomp, 0);
        MultiFab::Copy(consumed_snapshot, direct_state, 0, 0, ncomp, 0);
        EXPECT_FALSE(direct_register.reflux(direct_state, diagnostic));
        expect_same_values(direct_state, consumed_snapshot);
    }
}

TEST(SBMAMRYA, ReportsDirectAndReferenceAllocationCensus)
{
    const IntVect ratio(2);
    const Box coarse_domain(IntVect(0), IntVect(31));
    const Box fine_domain(IntVect(0), IntVect(63));
    BoxArray coarse_ba(coarse_domain);
    coarse_ba.maxSize(16);
    BoxArray fine_ba(amrex::refine(Box(IntVect(8), IntVect(23)), ratio));
    fine_ba.maxSize(16);
    const DistributionMapping coarse_dm(coarse_ba);
    const DistributionMapping fine_dm(fine_ba);
    const Geometry coarse_geom = make_geometry(coarse_domain);
    const Geometry fine_geom = make_geometry(fine_domain);
    constexpr int chunk_components = 16;

    auto payload_bytes = [] (const MultiFab& field) {
        std::uint64_t bytes = 0;
        for (amrex::MFIter mfi(field); mfi.isValid(); ++mfi) {
            const auto& fab = field[mfi];
            bytes += static_cast<std::uint64_t>(fab.box().numPts()) *
                     static_cast<std::uint64_t>(fab.nComp()) * sizeof(Real);
        }
        return bytes;
    };
    auto valid_payload_bytes = [] (const MultiFab& field) {
        std::uint64_t bytes = 0;
        for (amrex::MFIter mfi(field); mfi.isValid(); ++mfi) {
            bytes += static_cast<std::uint64_t>(mfi.validbox().numPts()) *
                     static_cast<std::uint64_t>(field.nComp()) * sizeof(Real);
        }
        return bytes;
    };

    for (const int ncomp : {32, 128, 256}) {
        std::uint64_t local_coarse_cells = 0;
        std::uint64_t local_fine_cells = 0;
        for (int box = 0; box < coarse_ba.size(); ++box) {
            if (coarse_dm[box] == amrex::ParallelDescriptor::MyProc()) {
                local_coarse_cells +=
                    static_cast<std::uint64_t>(coarse_ba[box].numPts());
            }
        }
        for (int box = 0; box < fine_ba.size(); ++box) {
            if (fine_dm[box] == amrex::ParallelDescriptor::MyProc()) {
                local_fine_cells +=
                    static_cast<std::uint64_t>(fine_ba[box].numPts());
            }
        }
        const std::uint64_t dense_coarse_bytes = local_coarse_cells *
            static_cast<std::uint64_t>(ncomp) * sizeof(Real);
        const auto before = amrex::TotalBytesAllocatedInFabs();
        std::uint64_t direct_allocated = 0;
        std::uint64_t reference_extra = 0;
        {
            erf_sbm::SBMAMRFluxRegister owner;
            owner.define(fine_ba, coarse_ba, fine_dm, coarse_dm,
                         fine_geom, coarse_geom, ratio, 1, ncomp);
            erf_auxiliary::MappedFaceFluxRate low, high, accepted, lambda,
                projected;
            low.define(fine_ba, fine_dm, chunk_components, 0);
            high.define(fine_ba, fine_dm, chunk_components, 0);
            accepted.define(fine_ba, fine_dm, chunk_components, 0);
            lambda.define(fine_ba, fine_dm, 1, 0);
            projected.define(fine_ba, fine_dm, 2, 0);
            low.setVal(Real(0.0));
            high.setVal(Real(0.0));
            accepted.setVal(Real(0.0));
            lambda.setVal(Real(0.0));
            projected.setVal(Real(0.0));
            erf_auxiliary::IntegratedMappedFaceFlux projected_ledger;
            projected_ledger.define(fine_ba, fine_dm, 2, 0);
            projected_ledger.setVal(Real(0.0));
            erf_auxiliary::MappedFaceFluxRate coarse_low, coarse_high,
                coarse_accepted, coarse_lambda, coarse_projected;
            coarse_low.define(coarse_ba, coarse_dm, chunk_components, 0);
            coarse_high.define(coarse_ba, coarse_dm, chunk_components, 0);
            coarse_accepted.define(coarse_ba, coarse_dm, chunk_components, 0);
            coarse_lambda.define(coarse_ba, coarse_dm, 1, 0);
            coarse_projected.define(coarse_ba, coarse_dm, 2, 0);
            coarse_low.setVal(Real(0.0));
            coarse_high.setVal(Real(0.0));
            coarse_accepted.setVal(Real(0.0));
            coarse_lambda.setVal(Real(0.0));
            coarse_projected.setVal(Real(0.0));
            erf_auxiliary::IntegratedMappedFaceFlux coarse_projected_ledger;
            coarse_projected_ledger.define(coarse_ba, coarse_dm, 2, 0);
            coarse_projected_ledger.setVal(Real(0.0));
            amrex::Gpu::streamSynchronize();
            direct_allocated = static_cast<std::uint64_t>(
                amrex::TotalBytesAllocatedInFabs() - before);

            // Count the existing M3 transport storage on both active levels:
            // anchor/target, four full-component WENO input views with their
            // two-cell ghost regions, and the old/new authoritative views.
            const auto persistent_before = amrex::TotalBytesAllocatedInFabs();
            std::vector<std::unique_ptr<MultiFab>> persistent_fields;
            auto add_persistent_field = [&] (const BoxArray& ba,
                                             const DistributionMapping& dm,
                                             const int components,
                                             const int ghosts) {
                auto field = std::make_unique<MultiFab>(
                    ba, dm, components, ghosts);
                field->setVal(Real(0.0));
                persistent_fields.push_back(std::move(field));
            };
            for (const auto& level : std::array{
                     std::pair<const BoxArray*, const DistributionMapping*>{
                         &coarse_ba, &coarse_dm},
                     std::pair<const BoxArray*, const DistributionMapping*>{
                         &fine_ba, &fine_dm}}) {
                for (int state_view = 0; state_view < 2; ++state_view) {
                    add_persistent_field(*level.first, *level.second,
                                         ncomp, 0);
                }
                for (int transport_view = 0; transport_view < 2;
                     ++transport_view) {
                    add_persistent_field(*level.first, *level.second,
                                         ncomp, 0);
                }
                for (int ghost_view = 0; ghost_view < 4; ++ghost_view) {
                    add_persistent_field(*level.first, *level.second,
                                         ncomp, 2);
                }
                add_persistent_field(*level.first, *level.second,
                                     chunk_components, 0); // low_trial_h
                add_persistent_field(*level.first, *level.second,
                                     1, 1); // minimum cell_ratios storage
                for (int scalar_view = 0; scalar_view < 3; ++scalar_view) {
                    add_persistent_field(*level.first, *level.second, 1, 0);
                }
            }
            amrex::Gpu::streamSynchronize();
            const auto state_and_transport_bytes = static_cast<std::uint64_t>(
                amrex::TotalBytesAllocatedInFabs() - persistent_before);
            const auto persistent_payload = [&] () {
                std::uint64_t bytes = 0;
                for (const auto& field : persistent_fields) {
                    bytes += payload_bytes(*field);
                }
                return bytes;
            }();
            std::uint64_t ghost_region_bytes = 0;
            for (std::size_t field = 0; field < persistent_fields.size(); ++field) {
                if (field % 13 >= 4 && field % 13 < 8) {
                    ghost_region_bytes += payload_bytes(*persistent_fields[field]) -
                                          valid_payload_bytes(*persistent_fields[field]);
                }
            }
            EXPECT_EQ(state_and_transport_bytes, persistent_payload);

            // Peak synchronization scratch: the one full coarse candidate,
            // one fine mapped-H field, and the scalar support/coverage masks.
            const auto sync_before = amrex::TotalBytesAllocatedInFabs();
            MultiFab coarse_candidate(coarse_ba, coarse_dm, ncomp, 0);
            MultiFab fine_mapped(fine_ba, fine_dm, ncomp, 0);
            MultiFab fine_support(fine_ba, fine_dm, 1, 0);
            MultiFab fine_coverage(fine_ba, fine_dm, 1, 0);
            MultiFab coarse_support(coarse_ba, coarse_dm, 1, 0);
            MultiFab coarse_coverage(coarse_ba, coarse_dm, 1, 0);
            MultiFab lost_support(coarse_ba, coarse_dm, 1, 0);
            MultiFab fine_invalid(fine_ba, fine_dm, 1, 0);
            MultiFab coarse_invalid(coarse_ba, coarse_dm, 1, 0);
            coarse_candidate.setVal(Real(0.0));
            fine_mapped.setVal(Real(0.0));
            fine_support.setVal(Real(0.0));
            fine_coverage.setVal(Real(0.0));
            coarse_support.setVal(Real(0.0));
            coarse_coverage.setVal(Real(0.0));
            lost_support.setVal(Real(0.0));
            fine_invalid.setVal(Real(0.0));
            coarse_invalid.setVal(Real(0.0));
            amrex::Gpu::streamSynchronize();
            const auto sync_scratch_bytes = static_cast<std::uint64_t>(
                amrex::TotalBytesAllocatedInFabs() - sync_before);

            {
                const auto ledger_before = amrex::TotalBytesAllocatedInFabs();
                erf_auxiliary::IntegratedMappedFaceFlux reference_ledger;
                reference_ledger.define(fine_ba, fine_dm, ncomp, 0);
                reference_ledger.setVal(Real(0.0));
                erf_auxiliary::IntegratedMappedFaceFlux coarse_reference_ledger;
                coarse_reference_ledger.define(coarse_ba, coarse_dm, ncomp, 0);
                coarse_reference_ledger.setVal(Real(0.0));
                amrex::Gpu::streamSynchronize();
                reference_extra = static_cast<std::uint64_t>(
                    amrex::TotalBytesAllocatedInFabs() - ledger_before);
                EXPECT_EQ(payload_bytes(reference_ledger.dir(0)) +
                              payload_bytes(reference_ledger.dir(1)) +
                              payload_bytes(reference_ledger.dir(2)) +
                              payload_bytes(coarse_reference_ledger.dir(0)) +
                              payload_bytes(coarse_reference_ledger.dir(1)) +
                              payload_bytes(coarse_reference_ledger.dir(2)),
                          reference_extra);
            }

            const auto direct_peak_bytes = direct_allocated +
                state_and_transport_bytes + sync_scratch_bytes;
            for (int rank = 0;
                 rank < amrex::ParallelDescriptor::NProcs(); ++rank) {
                if (amrex::ParallelDescriptor::MyProc() == rank) {
                    std::cout << "SBM_ALLOC_CENSUS ranks="
                              << amrex::ParallelDescriptor::NProcs()
                              << " ncomp=" << ncomp
                              << " rank=" << amrex::ParallelDescriptor::MyProc()
                              << " coarse_cells_local="
                              << local_coarse_cells
                              << " fine_cells_local=" << local_fine_cells
                              << " YA_dense_coarse_bytes=" << dense_coarse_bytes
                              << " YA_plus_face_chunks_bytes=" << direct_allocated
                              << " old_new_transport_and_ghost_bytes="
                              << state_and_transport_bytes
                              << " ghost_region_bytes=" << ghost_region_bytes
                              << " sync_candidate_peak_bytes=" << sync_scratch_bytes
                              << " direct_sync_peak_total_bytes=" << direct_peak_bytes
                              << " completed_integral_ledger_bytes=" << reference_extra
                              << " rate_chunk_components=" << chunk_components
                              << " cell_ratio_components_assumed=1_minimum"
                              << std::endl;
                }
                amrex::ParallelDescriptor::Barrier();
            }
        }
    }
}

} // namespace
