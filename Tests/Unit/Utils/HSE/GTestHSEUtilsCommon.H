#ifndef GTEST_HSEUTILS_COMMON_H_
#define GTEST_HSEUTILS_COMMON_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <AMReX_BaseFab.H>
#include <AMReX_Box.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_Math.H>

#include <ERF_Constants.H>
#include <ERF_EOS.H>
#include <ERF_HSEUtils.H>

namespace hse_test {

using amrex::Real;

// HSEUtils test inventory and branch map:
// - Newton_Raphson_hse: dry-density closure in pressure with moist total mass.
// - init_isentropic_hse / init_isentropic_hse_terrain: first-cell half-step and
//   interior trapezoidal HSE residuals.
// - compute_saturation_pressure: default and empirical water-saturation units.
// - compute_theta: lower power-law branch and upper exponential branch.
// - compute_rho / compute_F / compute_p_k: T_from_theta branches plus first-cell
//   and interior residual branches.
// Tolerance policy:
// - machine_precision_tol: floating-point roundoff and a small accumulation margin.
// - fd_truncation_tol: forward/central finite-difference truncation plus roundoff.
// - independent_root_tol: bisection bracket tolerance plus machine precision.
// - host_device_tol: backend math-library variation plus float/double behavior.

// MUST MATCH Source/ERF_Constants.H: public model constants used by the
// independent reference equations.
constexpr Real kRd = Real(287.0);
constexpr Real kRv = Real(461.505);
constexpr Real kCpD = Real(1004.5);
constexpr Real kCpL = Real(4200.0);
constexpr Real kLv = Real(2.5e6);
constexpr Real kP0 = Real(1.0e5);
constexpr Real kGamma = Real(1.4);
constexpr Real kConstGrav = Real(9.81);
constexpr Real kEpsVapor = kRd / kRv;
constexpr Real kChi = kRv / kRd;

constexpr Real kRdOcp = kRd / kCpD;
constexpr Real kInvGamma = Real(1.0) / kGamma;
constexpr Real kZone1QV = Real(0.014);
constexpr int kKernelMetricsPerCase = 6;

inline Real abs_max (const Real a, const Real b)
{
    return std::max(std::abs(a), std::abs(b));
}

inline Real scale_from_values (const std::initializer_list<Real>& values)
{
    Real scale = Real(1.0);
    for (const Real value : values) {
        scale = std::max(scale, std::abs(value));
    }
    return scale;
}

inline Real machine_precision_tol (const Real scale)
{
    // Provenance: allow about 64 ulps on the resolved scale for scalar algebra
    // and small accumulation across a few EOS/HSE operations.
    return Real(64.0) * std::numeric_limits<Real>::epsilon() * std::max(Real(1.0), std::abs(scale));
}

inline Real fd_truncation_tol (const Real step, const Real scale)
{
    // Provenance: finite-difference truncation is O(step); combine a 4*step
    // branch-local truncation allowance with machine precision on the value scale.
    return Real(4.0) * std::abs(step) + machine_precision_tol(scale);
}

inline Real independent_root_tol (const Real scale)
{
    // Provenance: bisection terminates with a narrow bracket; allow the final
    // bracket width plus roundoff on the resolved pressure/density scale.
    return Real(8.0) * machine_precision_tol(scale);
}

inline Real residual_tol (const Real scale)
{
    // Provenance: HSE/EOS residuals combine a few floating-point operations on
    // comparable terms; allow a modest roundoff multiple on that scale.
    return Real(96.0) * std::numeric_limits<Real>::epsilon() * std::max(Real(1.0), std::abs(scale));
}

inline Real host_device_tol (const Real scale)
{
    // Provenance: host/device comparisons can differ by backend math-library
    // details; allow a wider ulp budget than scalar-only checks.
#ifdef AMREX_USE_FLOAT
    constexpr Real factor = Real(512.0);
#else
    constexpr Real factor = Real(192.0);
#endif
    return factor * std::numeric_limits<Real>::epsilon() * std::max(Real(1.0), std::abs(scale));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real fd_step_temperature (const Real temperature)
{
    return std::sqrt(std::numeric_limits<Real>::epsilon()) *
           amrex::max(amrex::Math::abs(temperature), Real(300.0));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real fd_step_pressure (const Real pressure)
{
    return std::sqrt(std::numeric_limits<Real>::epsilon()) *
           amrex::max(amrex::Math::abs(pressure), kP0);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real fd_step_density (const Real density)
{
    return std::sqrt(std::numeric_limits<Real>::epsilon()) *
           amrex::max(amrex::Math::abs(density), Real(1.0));
}

// Independent reference helpers: EOS closures, algebraic moisture relations,
// analytic profile branches, discrete HSE residuals, and bracketed root solves.
// These helpers avoid calling HSEutils so the scalar tests constrain the
// intended contracts without reusing the production Newton implementation.

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real pressure_from_density_theta_qv (const Real rho, const Real theta, const Real qv)
{
    return kP0 * std::pow(kRd * rho * theta * (Real(1.0) + kChi * qv) / kP0, kGamma);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real dry_density_from_theta_pressure_qv (const Real theta, const Real pressure, const Real qv)
{
    return std::pow(kP0, kRdOcp) * std::pow(pressure, kInvGamma) /
           (kRd * theta * (Real(1.0) + kChi * qv));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real dpdrho_const_theta_qv (const Real rho, const Real theta, const Real qv)
{
    return kGamma * pressure_from_density_theta_qv(rho, theta, qv) / rho;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real temperature_from_theta_pressure (const Real theta, const Real pressure)
{
    return theta * std::pow(pressure / kP0, kRdOcp);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real theta_from_temperature_pressure (const Real temperature, const Real pressure)
{
    return temperature * std::pow(kP0 / pressure, kRdOcp);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real vapor_pressure_from_qv (const Real pressure, const Real qv)
{
    return qv * pressure / (kEpsVapor + qv);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real rh_from_qv (const Real qv, const Real pressure, const Real psat)
{
    return qv * pressure / (psat * (kEpsVapor + qv));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real qv_from_rh (const Real pressure, const Real psat, const Real rh)
{
    const Real vapor_pressure = rh * psat;
    return kEpsVapor * vapor_pressure / (pressure - vapor_pressure);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real theta_profile (const Real scaled_height, const Real theta_0,
                    const Real theta_tr, const Real z_tr, const Real t_tr)
{
    if (scaled_height <= Real(1.0)) {
        return theta_0 + (theta_tr - theta_0) * std::pow(scaled_height, Real(1.25));
    }

    return theta_tr * std::exp((kConstGrav * z_tr / (kCpD * t_tr)) * (scaled_height - Real(1.0)));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real theta_profile_lower_inverse (const Real theta, const Real theta_0, const Real theta_tr)
{
    return std::pow((theta - theta_0) / (theta_tr - theta_0), Real(0.8));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real theta_profile_upper_inverse (const Real theta, const Real theta_tr,
                                  const Real z_tr, const Real t_tr)
{
    return Real(1.0) + (kCpD * t_tr / (kConstGrav * z_tr)) * std::log(theta / theta_tr);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real theta_profile_lower_derivative (const Real scaled_height,
                                     const Real theta_0, const Real theta_tr)
{
    return Real(1.25) * (theta_tr - theta_0) * std::pow(scaled_height, Real(0.25));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real theta_profile_upper_derivative (const Real scaled_height,
                                     const Real theta_0, const Real theta_tr,
                                     const Real z_tr, const Real t_tr)
{
    const Real alpha = kConstGrav * z_tr / (kCpD * t_tr);
    return alpha * theta_profile(scaled_height, theta_0, theta_tr, z_tr, t_tr);
}

// Production-contract helpers: these intentionally follow the current HSEutils
// humidity and saturation behavior so host/device sweeps and profile-based
// roots cover the same branches as production. The zone-1 RH/p_v/q_v contract
// is preserved here pending a separate physics decision.

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real relative_humidity_profile (const Real pressure, const Real temperature,
                                const bool use_empirical, const int which_zone,
                                const Real scaled_height)
{
    if (which_zone > 0) {
        if (which_zone == 1) {
            const Real p_sat = HSEutils::compute_saturation_pressure(temperature, use_empirical);
            const Real qsat = kEpsVapor * p_sat / (pressure - p_sat);
            return kZone1QV / qsat;
        }
        if (which_zone == 2) {
            return Real(1.0) - Real(0.75) * std::pow(scaled_height, Real(1.25));
        }
        return Real(0.25);
    }

    return Real(1.0);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real vapor_pressure_profile (const Real pressure, const Real temperature,
                             const bool use_empirical, const int which_zone,
                             const Real scaled_height)
{
    const Real p_sat = HSEutils::compute_saturation_pressure(temperature, use_empirical);
    const Real rh = relative_humidity_profile(pressure, temperature, use_empirical, which_zone, scaled_height);
    return compute_vapor_pressure(p_sat, rh);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real qv_profile (const Real pressure, const Real temperature,
                 const bool use_empirical, const int which_zone,
                 const Real scaled_height)
{
    if (which_zone == 1) {
        return kZone1QV;
    }

    const Real p_sat = HSEutils::compute_saturation_pressure(temperature, use_empirical);
    const Real rh = relative_humidity_profile(pressure, temperature, use_empirical, which_zone, scaled_height);
    return qv_from_rh(pressure, p_sat, rh);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real equivalent_potential_temperature_from_state (const Real temperature, const Real pressure,
                                                  const Real qv, const Real qt)
{
    const Real fac = kCpD + kCpL * qt;
    const Real vapor_pressure = vapor_pressure_from_qv(pressure, qv);
    return temperature * std::pow((pressure - vapor_pressure) / kP0, -kRd / fac) *
           std::exp(kLv * qv / (fac * temperature));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real equivalent_potential_temperature_from_profile (const Real temperature, const Real pressure,
                                                    const Real qt, const bool use_empirical,
                                                    const int which_zone, const Real scaled_height)
{
    const Real qv = qv_profile(pressure, temperature, use_empirical, which_zone, scaled_height);
    const Real fac = kCpD + kCpL * qt;
    const Real vapor_pressure = vapor_pressure_profile(pressure, temperature, use_empirical, which_zone,
                                                       scaled_height);
    return temperature * std::pow((pressure - vapor_pressure) / kP0, -kRd / fac) *
           std::exp(kLv * qv / (fac * temperature));
}

template <typename Function>
inline Real bisection_root (Function&& function,
                            Real lower, Real upper,
                            const Real value_tol,
                            const int max_iter = 200)
{
    Real f_lower = function(lower);
    Real f_upper = function(upper);

    AMREX_ALWAYS_ASSERT(f_lower == Real(0.0) || f_upper == Real(0.0) || f_lower * f_upper < Real(0.0));

    for (int iter = 0; iter < max_iter; ++iter) {
        const Real mid = Real(0.5) * (lower + upper);
        const Real f_mid = function(mid);

        if (std::abs(f_mid) <= value_tol || std::abs(upper - lower) <= value_tol) {
            return mid;
        }

        if (f_lower == Real(0.0)) {
            return lower;
        }
        if (f_upper == Real(0.0)) {
            return upper;
        }

        if (f_lower * f_mid <= Real(0.0)) {
            upper = mid;
            f_upper = f_mid;
        } else {
            lower = mid;
            f_lower = f_mid;
        }
    }

    return Real(0.5) * (lower + upper);
}

inline Real compute_temperature_reference (const Real pressure, const Real qt,
                                           const Real eq_pot_temp, const bool use_empirical,
                                           const int which_zone, const Real scaled_height,
                                           const Real lower = Real(180.0),
                                           const Real upper = Real(380.0))
{
    auto residual = [=] (const Real temperature) {
        return eq_pot_temp - equivalent_potential_temperature_from_profile(temperature, pressure, qt,
                                                                           use_empirical, which_zone,
                                                                           scaled_height);
    };

    return bisection_root(residual, lower, upper, machine_precision_tol(eq_pot_temp));
}

struct RhoState {
    Real theta;
    Real rho;
    Real qv;
    Real temperature;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RhoState compute_rho_reference (const Real pressure, const Real q_t,
                                const Real eq_pot_temp, const bool use_empirical,
                                const int which_zone, const Real scaled_height,
                                const bool t_from_theta,
                                const Real theta_0, const Real theta_tr,
                                const Real z_tr, const Real t_tr)
{
    RhoState state{};

    if (t_from_theta) {
        state.theta = theta_profile(scaled_height, theta_0, theta_tr, z_tr, t_tr);
        state.temperature = temperature_from_theta_pressure(state.theta, pressure);
    } else {
        state.temperature = compute_temperature_reference(pressure, q_t, eq_pot_temp,
                                                          use_empirical, which_zone, scaled_height);
        state.theta = theta_from_temperature_pressure(state.temperature, pressure);
    }

    state.qv = qv_profile(pressure, state.temperature, use_empirical, which_zone, scaled_height);
    const Real dry_density = pressure / (kRd * state.temperature * (Real(1.0) + kChi * state.qv));
    state.rho = dry_density * (Real(1.0) + (t_from_theta ? state.qv : q_t));

    return state;
}

inline Real compute_f_reference (const Real pressure, const Real pressure_minus_1,
                                 const Real dz, const Real rho_minus_1,
                                 const Real q_t, const Real eq_pot_temp,
                                 const bool use_empirical, const int which_zone,
                                 const Real scaled_height, const bool t_from_theta,
                                 const Real theta_0, const Real theta_tr,
                                 const Real z_tr, const Real t_tr)
{
    const RhoState state = compute_rho_reference(pressure, q_t, eq_pot_temp, use_empirical, which_zone,
                                                 scaled_height, t_from_theta, theta_0, theta_tr, z_tr, t_tr);
    if (rho_minus_1 == Real(0.0)) {
        return pressure - pressure_minus_1 + Real(0.5) * state.rho * kConstGrav * dz;
    }

    return pressure - pressure_minus_1 + Real(0.5) * (state.rho + rho_minus_1) * kConstGrav * dz;
}

inline Real compute_pk_reference (const Real pressure_guess, const Real pressure_minus_1,
                                  const Real dz, const Real rho_minus_1,
                                  const Real q_t, const Real eq_pot_temp,
                                  const bool use_empirical, const int which_zone,
                                  const Real scaled_height, const bool t_from_theta,
                                  const Real theta_0, const Real theta_tr,
                                  const Real z_tr, const Real t_tr)
{
    const Real pressure_scale = scale_from_values({pressure_guess, pressure_minus_1, kP0});
    const Real lower = std::max(Real(1.0e3), pressure_guess - Real(8.0e4));
    const Real upper = pressure_guess + Real(2.0e4);

    auto residual = [=] (const Real pressure) {
        return compute_f_reference(pressure, pressure_minus_1, dz, rho_minus_1,
                                   q_t, eq_pot_temp, use_empirical, which_zone,
                                   scaled_height, t_from_theta, theta_0, theta_tr, z_tr, t_tr);
    };

    return bisection_root(residual, lower, upper, independent_root_tol(pressure_scale));
}

template <typename Function>
inline Real central_difference (Function&& function, const Real x, const Real step)
{
    return (function(x + step) - function(x - step)) / (Real(2.0) * step);
}

inline std::string label (const char* base, const Real a, const Real b = Real(-1.0))
{
    std::string text = std::string(base) + " a=" + std::to_string(a);
    if (b >= Real(0.0)) {
        text += " b=" + std::to_string(b);
    }
    return text;
}

struct SaturationReference {
    Real temperature;
    Real expected_pa;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real magnus_water_saturation_pressure (const Real temperature)
{
    const Real temperature_c = temperature - Real(273.15);
    return Real(610.94) * std::exp((Real(17.625) * temperature_c) /
                                   (temperature_c + Real(243.04)));
}

inline std::array<SaturationReference, 3> saturation_reference_cases ()
{
    // Independent meteorological reference values for saturation vapor pressure
    // over water in Pa near 250 K, 273.15 K, and 300 K.
    return {{
        {Real(250.0), Real(95.5)},
        {Real(273.15), Real(611.2)},
        {Real(300.0), Real(3534.0)}
    }};
}

inline std::array<SaturationReference, 2> saturation_formula_reference_cases ()
{
    return {{
        {Real(273.15), magnus_water_saturation_pressure(Real(273.15))},
        {Real(300.0), magnus_water_saturation_pressure(Real(300.0))}
    }};
}

inline Real saturation_formula_tol (const Real expected_pa)
{
    // Provenance: allow 2% relative disagreement against an independent
    // Magnus-style fit plus machine precision on the pressure scale.
    return Real(0.02) * expected_pa + machine_precision_tol(expected_pa);
}

inline amrex::Gpu::HostVector<Real> make_host_vector (const std::vector<Real>& values)
{
    amrex::Gpu::HostVector<Real> host(values.size());
    std::copy(values.begin(), values.end(), host.begin());
    return host;
}

template <typename HostVectorT>
void copy_host_to_device (const HostVectorT& host, amrex::Gpu::DeviceVector<Real>& device)
{
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, host.begin(), host.end(), device.begin());
}

struct KernelCase {
    Real pressure;
    Real pressure_minus_1;
    Real dz;
    Real rho_minus_1;
    Real q_t;
    Real eq_pot_temp;
    Real scaled_height;
    Real theta_0;
    Real theta_tr;
    Real z_tr;
    Real t_tr;
    int which_zone;
    int use_empirical;
    int t_from_theta;
};

inline std::vector<KernelCase> make_kernel_cases ()
{
    const std::array<int, 4> zones = {{-1, 1, 2, 3}};
    const std::array<Real, 3> scaled_heights = {{Real(0.5), Real(1.0), Real(1.5)}};
    const std::array<int, 2> empirical = {{0, 1}};
    const std::array<int, 2> theta_modes = {{0, 1}};
    std::vector<KernelCase> cases;
    cases.reserve(zones.size() * scaled_heights.size() * empirical.size() * theta_modes.size());

    int case_index = 0;
    for (const int zone : zones) {
        for (const Real scaled_height : scaled_heights) {
            for (const int use_empirical : empirical) {
                for (const int t_from_theta : theta_modes) {
                    const Real pressure = Real(7.5e4) + Real(1500.0) * case_index;
                    const Real q_t = (zone == 1) ? kZone1QV : Real(0.002) + Real(2.0e-4) * (case_index % 5);
                    const Real theta_0 = Real(298.0) + Real(0.4) * (case_index % 3);
                    const Real theta_tr = Real(312.0) + Real(0.5) * (case_index % 4);
                    const Real z_tr = Real(12000.0);
                    const Real t_tr = Real(295.0);
                    const Real temperature = t_from_theta != 0
                        ? temperature_from_theta_pressure(theta_profile(scaled_height, theta_0, theta_tr, z_tr, t_tr), pressure)
                        : Real(286.0) + Real(0.35) * case_index;
                    const Real qv = (zone == 1)
                        ? kZone1QV
                        : qv_profile(pressure, temperature, use_empirical != 0, zone, scaled_height);
                    const Real eq_pot_temp = equivalent_potential_temperature_from_state(temperature, pressure, qv, q_t);
                    const Real rho_minus_1 = Real(0.95) + Real(0.01) * (case_index % 4);

                    cases.push_back({
                        pressure,
                        pressure + Real(400.0),
                        Real(800.0) + Real(50.0) * (case_index % 3),
                        rho_minus_1,
                        q_t,
                        eq_pot_temp,
                        scaled_height,
                        theta_0,
                        theta_tr,
                        z_tr,
                        t_tr,
                        zone,
                        use_empirical,
                        t_from_theta
                    });
                    ++case_index;
                }
            }
        }
    }

    return cases;
}

} // namespace hse_test

#endif