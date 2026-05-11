#ifndef ERF_SHOC_TYPES_H_
#define ERF_SHOC_TYPES_H_

#include "ERF_DataStruct.H"

#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Vector.H>

#include <algorithm>
#include <cctype>
#include <string>

enum class ShocTransportMode
{
    Tendencies,
    HostDiffusion
};

inline const char*
shoc_transport_mode_name (ShocTransportMode mode)
{
    switch (mode) {
    case ShocTransportMode::Tendencies:   return "tendencies";
    case ShocTransportMode::HostDiffusion:return "host_diffusion";
    }
    return "unknown";
}

inline bool
shoc_uses_internal_transport (ShocTransportMode mode)
{
    return mode == ShocTransportMode::Tendencies;
}

inline bool
shoc_uses_host_diffusion (ShocTransportMode mode)
{
    return mode == ShocTransportMode::HostDiffusion;
}

struct ShocRuntimeOptions
{
    amrex::Real lambda_low = 0.001;
    amrex::Real lambda_high = 0.04;
    amrex::Real lambda_slope = 2.65;
    amrex::Real lambda_thresh = 0.02;

    amrex::Real thl2tune = 1.0;
    amrex::Real qw2tune = 1.0;
    amrex::Real qwthl2tune = 1.0;
    amrex::Real w2tune = 1.0;

    amrex::Real length_fac = 0.5;
    amrex::Real c_diag_3rd_mom = 7.0;
    amrex::Real coeff_kh = 0.1;
    amrex::Real coeff_km = 0.1;
    amrex::Real top_taper_depth = 0.0;
    amrex::Real top_taper_min_factor = 0.0;

    bool shoc_1p5tke = false;
    bool extra_shoc_diags = false;
    bool apply_tms = false;
    bool check_flux_state = false;
    bool column_conservation_check = false;
    bool debug_summary = false;
    bool allow_tendency_microphysics_overlap = false;
    bool signed_tke_production = false;
    ShocTransportMode transport_mode = ShocTransportMode::Tendencies;
};

struct ShocColumnLayout
{
    int nx = 0;
    int ny = 0;
    int ncell = 0;
    int nlev = 0;
    int imin = 0;
    int jmin = 0;
    int kmin = 0;
    int kmax = -1;
};

struct ShocColumnData
{
    ShocColumnLayout layout;

    amrex::FArrayBox zt;
    amrex::FArrayBox zi;
    amrex::FArrayBox dz;
    amrex::FArrayBox p_mid;
    amrex::FArrayBox p_int;
    amrex::FArrayBox rho;
    amrex::FArrayBox theta;
    amrex::FArrayBox exner;
    amrex::FArrayBox theta_v;
    amrex::FArrayBox thetal;
    amrex::FArrayBox qv;
    amrex::FArrayBox qc;
    amrex::FArrayBox qi;
    amrex::FArrayBox qw;
    amrex::FArrayBox tabs;
    amrex::FArrayBox tke;
    amrex::FArrayBox u;
    amrex::FArrayBox v;
    amrex::FArrayBox w;
    amrex::FArrayBox host_dse;

    amrex::FArrayBox pblh;
    amrex::FArrayBox obklen;
    amrex::FArrayBox ustar;
    amrex::FArrayBox shoc_mix;
    amrex::FArrayBox brunt;
    amrex::FArrayBox isotropy;
    amrex::FArrayBox tk;
    amrex::FArrayBox tkh;
    amrex::FArrayBox shear_prod;
    amrex::FArrayBox buoy_prod;
    amrex::FArrayBox diss_tke;
    amrex::FArrayBox wthv_sec;
    amrex::FArrayBox shoc_cldfrac;
    amrex::FArrayBox shoc_ql;
    amrex::FArrayBox shoc_ql2;
    amrex::FArrayBox wqls_sec;
    amrex::FArrayBox shoc_cond;
    amrex::FArrayBox shoc_evap;

    amrex::FArrayBox thl_sec;
    amrex::FArrayBox qw_sec;
    amrex::FArrayBox qwthl_sec;
    amrex::FArrayBox wthl_sec;
    amrex::FArrayBox wqw_sec;
    amrex::FArrayBox uw_sec;
    amrex::FArrayBox vw_sec;
    amrex::FArrayBox wtke_sec;
    amrex::FArrayBox w3;
    amrex::FArrayBox w_sec;

    amrex::FArrayBox thetal_base;
    amrex::FArrayBox theta_base;
    amrex::FArrayBox qv_base;
    amrex::FArrayBox qc_base;
    amrex::FArrayBox qi_base;
    amrex::FArrayBox u_base;
    amrex::FArrayBox v_base;
    amrex::FArrayBox tke_base_state;

    amrex::FArrayBox theta_tend;
    amrex::FArrayBox qv_tend;
    amrex::FArrayBox qc_tend;
    amrex::FArrayBox qi_tend;
    amrex::FArrayBox u_tend;
    amrex::FArrayBox v_tend;
    amrex::FArrayBox tke_tend;
    amrex::FArrayBox surf_sens_flux;
    amrex::FArrayBox surf_lat_flux;
    amrex::FArrayBox surf_tau_u;
    amrex::FArrayBox surf_tau_v;
};

inline void
read_shoc_runtime_options (ShocRuntimeOptions& opts)
{
    amrex::ParmParse pp("erf.shoc");

    pp.query("lambda_low", opts.lambda_low);
    pp.query("lambda_high", opts.lambda_high);
    pp.query("lambda_slope", opts.lambda_slope);
    pp.query("lambda_thresh", opts.lambda_thresh);
    pp.query("thl2tune", opts.thl2tune);
    pp.query("qw2tune", opts.qw2tune);
    pp.query("qwthl2tune", opts.qwthl2tune);
    pp.query("w2tune", opts.w2tune);
    pp.query("length_fac", opts.length_fac);
    pp.query("c_diag_3rd_mom", opts.c_diag_3rd_mom);
    pp.query("coeff_kh", opts.coeff_kh);
    pp.query("coeff_km", opts.coeff_km);
    pp.query("top_taper_depth", opts.top_taper_depth);
    pp.query("top_taper_min_factor", opts.top_taper_min_factor);
    pp.query("shoc_1p5tke", opts.shoc_1p5tke);
    pp.query("extra_shoc_diags", opts.extra_shoc_diags);
    pp.query("apply_tms", opts.apply_tms);
    pp.query("check_flux_state", opts.check_flux_state);
    pp.query("column_conservation_check", opts.column_conservation_check);
    pp.query("debug_summary", opts.debug_summary);
    pp.query("allow_tendency_microphysics_overlap", opts.allow_tendency_microphysics_overlap);
    pp.query("signed_tke_production", opts.signed_tke_production);

    std::string transport_mode = shoc_transport_mode_name(opts.transport_mode);
    if (pp.query("transport_mode", transport_mode)) {
        std::transform(transport_mode.begin(), transport_mode.end(), transport_mode.begin(),
                       [] (unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (transport_mode == "tendencies") {
            opts.transport_mode = ShocTransportMode::Tendencies;
        } else if (transport_mode == "host_diffusion") {
            opts.transport_mode = ShocTransportMode::HostDiffusion;
        } else {
            amrex::Abort("erf.shoc.transport_mode must be 'tendencies' or 'host_diffusion'");
        }
    }
}

inline void
validate_shoc_runtime_options (const ShocRuntimeOptions& opts)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(opts.lambda_low > 0.0, "erf.shoc.lambda_low must be positive");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(opts.lambda_high >= opts.lambda_low,
                                     "erf.shoc.lambda_high must be >= lambda_low");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(opts.length_fac > 0.0, "erf.shoc.length_fac must be positive");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(opts.coeff_kh >= 0.0, "erf.shoc.coeff_kh must be non-negative");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(opts.coeff_km >= 0.0, "erf.shoc.coeff_km must be non-negative");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(opts.top_taper_depth >= 0.0,
                                     "erf.shoc.top_taper_depth must be non-negative");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(opts.top_taper_min_factor >= 0.0 &&
                                     opts.top_taper_min_factor <= 1.0,
                                     "erf.shoc.top_taper_min_factor must be in [0, 1]");
}

AMREX_FORCE_INLINE
bool
shoc_valid_comp (int comp, int ncomp)
{
    return comp >= 0 && comp < ncomp;
}

#endif
