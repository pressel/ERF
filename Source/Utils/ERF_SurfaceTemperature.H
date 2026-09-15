#ifndef ERF_SURFACE_TEMPERATURE_H_
#define ERF_SURFACE_TEMPERATURE_H_

#include <cmath>

#include <ERF_Constants.H>
#include <ERF_EOS.H>

/**
 * Small, GPU-safe helpers for the surface-temperature contract.
 *
 * SurfaceLayer stores dry potential temperature.  External surface
 * temperature data and consumers such as radiation and saturation humidity
 * use absolute temperature.  These helpers keep the EOS conversion and its
 * validity check in one place without imposing a representation on callers.
 */
namespace erf_surface_temperature {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool valid_pressure (const amrex::Real pressure)
{
    return std::isfinite(pressure) && pressure > amrex::Real(0.0);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool valid_temperature (const amrex::Real temperature)
{
    return std::isfinite(temperature) && temperature > amrex::Real(0.0);
}

/**
 * Convert an absolute surface temperature to dry potential temperature.
 *
 * @return false when either input is invalid; `theta` is not modified then.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool absolute_to_theta (const amrex::Real temperature,
                        const amrex::Real pressure,
                        amrex::Real& theta)
{
    if (!valid_temperature(temperature) || !valid_pressure(pressure)) {
        return false;
    }

    const amrex::Real converted = getThgivenTandP(temperature, pressure, RdoCp);
    if (!valid_temperature(converted)) {
        return false;
    }

    theta = converted;
    return true;
}

/**
 * Convert dry potential temperature to absolute temperature.
 *
 * @return false when either input is invalid; `temperature` is not modified
 *         then.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool theta_to_absolute (const amrex::Real theta,
                        const amrex::Real pressure,
                        amrex::Real& temperature)
{
    if (!valid_temperature(theta) || !valid_pressure(pressure)) {
        return false;
    }

    const amrex::Real converted = getTgivenPandTh(pressure, theta, RdoCp);
    if (!valid_temperature(converted)) {
        return false;
    }

    temperature = converted;
    return true;
}

/**
 * Diagnose pressure at the physical surface from the lowest cell state.
 *
 * `getPgivenRTh` gives the pressure at the cell center.  The hydrostatic
 * correction uses the same lowest-cell state and terrain-relative height as
 * SurfaceLayer's saturation-humidity path.  The result is in Pa.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real pressure_at_surface (const amrex::Real rho,
                                 const amrex::Real rho_theta,
                                 const amrex::Real qv,
                                 const amrex::Real delta_z)
{
    if (!(rho > amrex::Real(0.0)) || !std::isfinite(rho) ||
        !(rho_theta > amrex::Real(0.0)) || !std::isfinite(rho_theta) ||
        !(qv >= amrex::Real(0.0)) || !std::isfinite(qv) ||
        !(delta_z >= amrex::Real(0.0)) || !std::isfinite(delta_z)) {
        return amrex::Real(0.0);
    }

    const amrex::Real pressure = getPgivenRTh(rho_theta, qv) +
                                 rho * CONST_GRAV * delta_z;
    return valid_pressure(pressure) ? pressure : amrex::Real(0.0);
}

} // namespace erf_surface_temperature

#endif
