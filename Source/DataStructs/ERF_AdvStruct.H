#ifndef ERF_ADV_STRUCT_H_
#define ERF_ADV_STRUCT_H_

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Gpu.H>
#include <AMReX_Geometry.H>

#include <ERF_NumericalConstants.H>
#include <ERF_IndexDefines.H>

/**
 * Container holding the advection-related choices.
 */
struct AdvChoice {
  public:
    /**
     * @brief Read advection options from the input parameter database.
     *
     * Parse into fresh local values and replace the stored configuration only
     * after every supplied value has passed validation. This makes repeated
     * initialization deterministic and leaves the object unchanged on error.
     * @param pp_prefix ParmParse prefix for the ERF input namespace.
     */
    void init_params (const std::string& pp_prefix)
    {
        amrex::ParmParse pp(pp_prefix);

        bool new_efficient_advection = false;
        pp.query("use_efficient_advection", new_efficient_advection);

        const auto parse_scheme = [&](const std::string& key, AdvType default_type,
                                      bool scalar_family) -> AdvType {
            std::string value;
            if (pp.query(key, value) == 0) {
                return default_type;
            }
            if (value.empty()) {
                amrex::Abort(full_key(pp_prefix, key) +
                             " is explicitly empty; provide a supported scheme name");
            }

            const AdvType parsed = adv_type_convert_string_to_advtype(value);
            const bool common = parsed == AdvType::Centered_2nd || parsed == AdvType::Upwind_3rd ||
                parsed == AdvType::Centered_4th || parsed == AdvType::Upwind_5th ||
                parsed == AdvType::Centered_6th || parsed == AdvType::Weno_3 ||
                parsed == AdvType::Weno_3Z || parsed == AdvType::Weno_5 ||
                parsed == AdvType::Weno_5Z || parsed == AdvType::Weno_7 ||
                parsed == AdvType::Weno_7Z;
            const bool scalar_only = parsed == AdvType::Upwind_3rd_SL || parsed == AdvType::Weno_3MZQ;
            if (!(common || (scalar_family && scalar_only))) {
                amrex::Abort(full_key(pp_prefix, key) + " has unsupported value '" + value + "'");
            }
            return parsed;
        };

        const AdvType new_dycore_h = parse_scheme("dycore_horiz_adv_type", AdvType::Upwind_3rd, false);
        const AdvType new_dycore_v = parse_scheme("dycore_vert_adv_type", AdvType::Upwind_3rd, false);
        const AdvType new_dry_h = parse_scheme("dryscal_horiz_adv_type", AdvType::Upwind_3rd, true);
        const AdvType new_dry_v = parse_scheme("dryscal_vert_adv_type", AdvType::Upwind_3rd, true);
        const AdvType new_moist_h = parse_scheme("moistscal_horiz_adv_type", AdvType::Weno_3, true);
        const AdvType new_moist_v = parse_scheme("moistscal_vert_adv_type", AdvType::Weno_3, true);

        const auto parse_fraction = [&](const std::string& key, AdvType scheme) -> amrex::Real {
            amrex::Real value = one;
            const bool supplied = pp.query(key, value) > 0;
            const bool consumes_fraction = scheme == AdvType::Upwind_3rd || scheme == AdvType::Upwind_5th;
            if (!(std::isfinite(static_cast<double>(value)) && value >= zero && value <= one)) {
                std::ostringstream message;
                message << full_key(pp_prefix, key) << " must be finite and in [0,1]; got " << value;
                amrex::Abort(message.str());
            }
            if (supplied && !consumes_fraction) {
                std::ostringstream message;
                message << full_key(pp_prefix, key) << " = " << value
                        << " is unused by the selected advection scheme "
                        << adv_type_convert_int_to_string(scheme);
                amrex::Abort(message.str());
            }
            return value;
        };

        const amrex::Real new_dycore_h_frac = parse_fraction("dycore_horiz_upw_frac", new_dycore_h);
        const amrex::Real new_dycore_v_frac = parse_fraction("dycore_vert_upw_frac", new_dycore_v);
        const amrex::Real new_dry_h_frac = parse_fraction("dryscal_horiz_upw_frac", new_dry_h);
        const amrex::Real new_dry_v_frac = parse_fraction("dryscal_vert_upw_frac", new_dry_v);
        const amrex::Real new_moist_h_frac = parse_fraction("moistscal_horiz_upw_frac", new_moist_h);
        const amrex::Real new_moist_v_frac = parse_fraction("moistscal_vert_upw_frac", new_moist_v);

        amrex::Vector<amrex::IntVect> new_zero_xflux, new_zero_yflux, new_zero_zflux;
        pp.queryarr("zero_xflux_faces", new_zero_xflux);
        pp.queryarr("zero_yflux_faces", new_zero_yflux);
        pp.queryarr("zero_zflux_faces", new_zero_zflux);
        normalize_face_list(new_zero_xflux);
        normalize_face_list(new_zero_yflux);
        normalize_face_list(new_zero_zflux);

        use_efficient_advection = new_efficient_advection;
        dycore_horiz_adv_type = new_dycore_h;
        dycore_vert_adv_type = new_dycore_v;
        dryscal_horiz_adv_type = new_dry_h;
        dryscal_vert_adv_type = new_dry_v;
        moistscal_horiz_adv_type = new_moist_h;
        moistscal_vert_adv_type = new_moist_v;
        dycore_horiz_upw_frac = new_dycore_h_frac;
        dycore_vert_upw_frac = new_dycore_v_frac;
        dryscal_horiz_upw_frac = new_dry_h_frac;
        dryscal_vert_upw_frac = new_dry_v_frac;
        moistscal_horiz_upw_frac = new_moist_h_frac;
        moistscal_vert_upw_frac = new_moist_v_frac;
        zero_xflux = std::move(new_zero_xflux);
        zero_yflux = std::move(new_zero_yflux);
        zero_zflux = std::move(new_zero_zflux);
        have_zero_flux_faces = !zero_xflux.empty() || !zero_yflux.empty() || !zero_zflux.empty();

        if (use_efficient_advection) {
            amrex::Print() << "Using efficient advection scheme" << std::endl;
        }
    }

    /**
     * @brief Print the effective stored advection configuration.
     * @param pp_prefix Kept for source compatibility; display uses stored state.
     */
    void display (const std::string& pp_prefix) const
    {
        amrex::ignore_unused(pp_prefix);
        amrex::Print() << "Advection Choices: " << std::endl;
        amrex::Print() << "    use_efficient_advection : " << std::boolalpha << use_efficient_advection
                       << std::noboolalpha << std::endl;
        const auto print_choice = [](const char* key, AdvType scheme, amrex::Real fraction) {
            amrex::Print() << "    " << key << " : " << adv_type_convert_int_to_string(scheme);
            if (scheme == AdvType::Upwind_3rd || scheme == AdvType::Upwind_5th) {
                amrex::Print() << " (upwind fraction = " << fraction << ')';
            }
            amrex::Print() << std::endl;
        };
        print_choice("dycore_horiz_adv_type", dycore_horiz_adv_type, dycore_horiz_upw_frac);
        print_choice("dycore_vert_adv_type", dycore_vert_adv_type, dycore_vert_upw_frac);
        print_choice("dryscal_horiz_adv_type", dryscal_horiz_adv_type, dryscal_horiz_upw_frac);
        print_choice("dryscal_vert_adv_type", dryscal_vert_adv_type, dryscal_vert_upw_frac);
        print_choice("moistscal_horiz_adv_type", moistscal_horiz_adv_type, moistscal_horiz_upw_frac);
        print_choice("moistscal_vert_adv_type", moistscal_vert_adv_type, moistscal_vert_upw_frac);
    }

    /** Convert an advection enum to its canonical input string. */
    static std::string adv_type_convert_int_to_string (AdvType adv_int)
    {
        if (adv_int == AdvType::Centered_2nd) return "Centered_2nd";
        if (adv_int == AdvType::Upwind_3rd) return "Upwind_3rd";
        if (adv_int == AdvType::Upwind_3rd_SL) return "Upwind_3rd_SL";
        if (adv_int == AdvType::Centered_4th) return "Centered_4th";
        if (adv_int == AdvType::Upwind_5th) return "Upwind_5th";
        if (adv_int == AdvType::Centered_6th) return "Centered_6th";
        if (adv_int == AdvType::Weno_3) return "WENO3";
        if (adv_int == AdvType::Weno_3Z) return "WENOZ3";
        if (adv_int == AdvType::Weno_5) return "WENO5";
        if (adv_int == AdvType::Weno_5Z) return "WENOZ5";
        if (adv_int == AdvType::Weno_3MZQ) return "WENOMZQ3";
        if (adv_int == AdvType::Weno_7) return "WENO7";
        if (adv_int == AdvType::Weno_7Z) return "WENOZ7";
        return "Unknown";
    }

    /** Convert a user-facing advection name to its enum; unknown names stay Unknown. */
    static AdvType adv_type_convert_string_to_advtype (const std::string& adv_string)
    {
        if (adv_string == "Centered_2nd") return AdvType::Centered_2nd;
        if (adv_string == "Upwind_3rd" || adv_string == "Blended_3rd4th") return AdvType::Upwind_3rd;
        if (adv_string == "Upwind_3rd_SL") return AdvType::Upwind_3rd_SL;
        if (adv_string == "Centered_4th") return AdvType::Centered_4th;
        if (adv_string == "Upwind_5th" || adv_string == "Blended_5th6th") return AdvType::Upwind_5th;
        if (adv_string == "Centered_6th") return AdvType::Centered_6th;
        if (adv_string == "WENO3") return AdvType::Weno_3;
        if (adv_string == "WENOZ3") return AdvType::Weno_3Z;
        if (adv_string == "WENO5") return AdvType::Weno_5;
        if (adv_string == "WENOZ5") return AdvType::Weno_5Z;
        if (adv_string == "WENOMZQ3") return AdvType::Weno_3MZQ;
        if (adv_string == "WENO7") return AdvType::Weno_7;
        if (adv_string == "WENOZ7") return AdvType::Weno_7Z;
        return AdvType::Unknown;
    }

    /** Validate scalar WENO pairing and actual EB cut-cell capabilities. */
    void validate_active_configuration (bool use_embedded_boundary, bool moisture_active) const
    {
        // Dycore choices also drive the scalar thermodynamic-state flux path.
        validate_scalar_weno_pair("dycore_horiz_adv_type", dycore_horiz_adv_type,
                                  "dycore_vert_adv_type", dycore_vert_adv_type);
        validate_scalar_weno_pair("dryscal_horiz_adv_type", dryscal_horiz_adv_type,
                                  "dryscal_vert_adv_type", dryscal_vert_adv_type);
        if (moisture_active) {
            validate_scalar_weno_pair("moistscal_horiz_adv_type", moistscal_horiz_adv_type,
                                      "moistscal_vert_adv_type", moistscal_vert_adv_type);
        }
        if (!use_embedded_boundary) return;

        validate_eb_momentum_choice("dycore_horiz_adv_type", dycore_horiz_adv_type);
        validate_eb_momentum_choice("dycore_vert_adv_type", dycore_vert_adv_type);
        validate_eb_scalar_choice("dryscal_horiz_adv_type", dryscal_horiz_adv_type);
        validate_eb_scalar_choice("dryscal_vert_adv_type", dryscal_vert_adv_type);
        if (moisture_active) {
            validate_eb_scalar_choice("moistscal_horiz_adv_type", moistscal_horiz_adv_type);
            validate_eb_scalar_choice("moistscal_vert_adv_type", moistscal_vert_adv_type);
        }
    }

    static constexpr bool is_weno (AdvType scheme) noexcept
    {
        return scheme == AdvType::Weno_3 || scheme == AdvType::Weno_3Z ||
               scheme == AdvType::Weno_3MZQ || scheme == AdvType::Weno_5 ||
               scheme == AdvType::Weno_5Z || scheme == AdvType::Weno_7 ||
               scheme == AdvType::Weno_7Z;
    }

    /** Largest ghost reach required by a production face reconstructor. */
    static int required_stencil_radius (AdvType scheme)
    {
        switch (scheme) {
        case AdvType::Centered_2nd: return 1;
        case AdvType::Upwind_3rd:
        case AdvType::Upwind_3rd_SL:
        case AdvType::Centered_4th:
        case AdvType::Weno_3:
        case AdvType::Weno_3Z:
        case AdvType::Weno_3MZQ: return 2;
        case AdvType::Upwind_5th:
        case AdvType::Centered_6th:
        case AdvType::Weno_5:
        case AdvType::Weno_5Z: return 3;
        case AdvType::Weno_7:
        case AdvType::Weno_7Z: return 4;
        default:
            amrex::Abort("Unsupported advection enum while computing stencil halo: " +
                         adv_type_convert_int_to_string(scheme));
            return 0;
        }
    }

    /** Compute the common allocation halo while preserving production minimums. */
    static int required_ghost_cells (const AdvChoice& choice,
                                     bool numerical_diffusion,
                                     bool embedded_boundary,
                                     bool moisture_active)
    {
        int ngrow = numerical_diffusion ? 3 : 2;
        const auto add_scheme = [&ngrow](AdvType scheme) {
            ngrow = amrex::max(ngrow, required_stencil_radius(scheme));
        };

        add_scheme(choice.dycore_horiz_adv_type);
        add_scheme(choice.dycore_vert_adv_type);
        add_scheme(choice.dryscal_horiz_adv_type);
        add_scheme(choice.dryscal_vert_adv_type);
        if (moisture_active) {
            add_scheme(choice.moistscal_horiz_adv_type);
            add_scheme(choice.moistscal_vert_adv_type);
        }
        if (embedded_boundary) {
            ngrow = amrex::max(ngrow, 4);
        }
        return ngrow;
    }

    static bool is_eb_supported_scheme (AdvType scheme) noexcept
    {
        return scheme == AdvType::Centered_2nd || scheme == AdvType::Upwind_3rd ||
               scheme == AdvType::Centered_4th || scheme == AdvType::Upwind_5th ||
               scheme == AdvType::Centered_6th;
    }

    static std::string full_key (const std::string& prefix, const std::string& key)
    {
        return prefix.empty() ? key : prefix + "." + key;
    }

    static void normalize_face_list (amrex::Vector<amrex::IntVect>& faces)
    {
        std::sort(faces.begin(), faces.end(), [](const amrex::IntVect& lhs, const amrex::IntVect& rhs) {
            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                if (lhs[dir] != rhs[dir]) return lhs[dir] < rhs[dir];
            }
            return false;
        });
        faces.erase(std::unique(faces.begin(), faces.end()), faces.end());
    }

    static void validate_scalar_weno_pair (const char* h_key, AdvType h_scheme,
                                           const char* v_key, AdvType v_scheme)
    {
        if ((is_weno(h_scheme) || is_weno(v_scheme)) && h_scheme != v_scheme) {
            amrex::Abort(std::string("erf.") + h_key + " = " + adv_type_convert_int_to_string(h_scheme) +
                         " and erf." + v_key + " = " + adv_type_convert_int_to_string(v_scheme) +
                         " must select the same WENO scheme for scalar advection");
        }
    }

    static void validate_eb_momentum_choice (const char* key, AdvType scheme)
    {
        if (!is_eb_supported_scheme(scheme)) {
            amrex::Abort(std::string("erf.") + key + " = " + adv_type_convert_int_to_string(scheme) +
                         " is not supported for EB momentum advection; supported choices are "
                         "Centered_2nd, Upwind_3rd, Centered_4th, Upwind_5th, and Centered_6th");
        }
    }

    static void validate_eb_scalar_choice (const char* key, AdvType scheme)
    {
        if (!is_eb_supported_scheme(scheme)) {
            amrex::Abort(std::string("erf.") + key + " = " + adv_type_convert_int_to_string(scheme) +
                         " is not supported for EB cut-cell scalar advection; supported choices are "
                         "Centered_2nd, Upwind_3rd, Centered_4th, Upwind_5th, and Centered_6th");
        }
    }

    // Effective advection configuration.
    bool use_efficient_advection = false;
    AdvType dycore_horiz_adv_type    = AdvType::Upwind_3rd;
    AdvType dycore_vert_adv_type     = AdvType::Upwind_3rd;
    AdvType dryscal_horiz_adv_type   = AdvType::Upwind_3rd;
    AdvType dryscal_vert_adv_type    = AdvType::Upwind_3rd;
    AdvType moistscal_horiz_adv_type = AdvType::Weno_3;
    AdvType moistscal_vert_adv_type  = AdvType::Weno_3;

    // Fraction of the high-order upwind correction; default 1 is pure upwind.
    amrex::Real dycore_horiz_upw_frac    = one;
    amrex::Real dycore_vert_upw_frac     = one;
    amrex::Real dryscal_horiz_upw_frac   = one;
    amrex::Real dryscal_vert_upw_frac    = one;
    amrex::Real moistscal_horiz_upw_frac = one;
    amrex::Real moistscal_vert_upw_frac  = one;

    // Thin immersed-body faces; duplicates are normalized during parsing.
    amrex::Vector<amrex::IntVect> zero_xflux;
    amrex::Vector<amrex::IntVect> zero_yflux;
    amrex::Vector<amrex::IntVect> zero_zflux;
    bool have_zero_flux_faces = false;
};

#endif
