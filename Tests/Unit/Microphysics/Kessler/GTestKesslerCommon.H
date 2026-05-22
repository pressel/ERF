#ifndef GTEST_KESSLER_COMMON_H_
#define GTEST_KESSLER_COMMON_H_

#include <cstdint>
#include <cstring>
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>
#include <string>

#include <AMReX_Arena.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Gpu.H>
#include <AMReX_Geometry.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>

#include <ERF_Constants.H>
#include <ERF_DataStruct.H>
#include <ERF_EOS.H>
#include <ERF_IndexDefines.H>
#include <ERF_Kessler.H>
#include <ERF_MicrophysicsUtils.H>

#ifndef ERF_ENABLE_KESSLER_BFB_REFERENCE
#error "Kessler BFB tests require ERF_ENABLE_KESSLER_BFB_REFERENCE"
#endif

struct KesslerBFBTestAccessor {
    static void AdvanceLegacy (Kessler& kessler, const SolverChoice& solver_choice)
    {
        kessler.AdvanceKesslerLegacyForBFB(solver_choice);
    }

    static void AdvanceRefactored (Kessler& kessler, const SolverChoice& solver_choice)
    {
        kessler.AdvanceKesslerRefactored(solver_choice);
    }

    static amrex::MultiFab& MicroVar (Kessler& kessler, const int var_idx)
    {
        return *kessler.mic_fab_vars[var_idx];
    }

    static const amrex::MultiFab& MicroVar (const Kessler& kessler, const int var_idx)
    {
        return *kessler.mic_fab_vars[var_idx];
    }
};

namespace kessler_bfb_test {

#ifdef AMREX_USE_FLOAT
using BitPattern = std::uint32_t;
#else
using BitPattern = std::uint64_t;
#endif

constexpr amrex::Real kRdOcp = R_d / Cp_d;

struct CellState {
    amrex::Real tabs;
    amrex::Real pres_mbar;
    amrex::Real qv;
    amrex::Real qc;
    amrex::Real qp;
    amrex::Real rain_accum;
};

enum class FixtureFlavor {
    BranchQuilt,
    ForceQcCap,
    ForceQpCap
};

struct FixtureOptions {
    MoistureType moisture_type;
    FixtureFlavor flavor;
    bool use_shoc;
    bool with_detj;
    bool inject_clip_cases;
    int real_width;
    int nx;
    int ny;
    int nz;
    amrex::Real dt;
    amrex::Real dzmin;
};

struct BitMismatch {
    int fab_index;
    int i;
    int j;
    int k;
    int comp;
    BitPattern expected_bits;
    BitPattern actual_bits;
    amrex::Real expected_value;
    amrex::Real actual_value;
};

inline void sync ()
{
#ifdef AMREX_USE_GPU
    amrex::Gpu::streamSynchronize();
#endif
}

template <typename F>
inline void run_and_sync (F&& f)
{
    std::forward<F>(f)();
    sync();
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
amrex::Real qsat (const amrex::Real tabs, const amrex::Real pres_mbar)
{
    amrex::Real qsat_local;
    erf_qsatw(tabs, pres_mbar, qsat_local);
    return qsat_local;
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
CellState make_rain_cell_state (const FixtureFlavor flavor,
                                const int selector,
                                const bool boundary_cell)
{
    const amrex::Real tabs = boundary_cell ? amrex::Real(289.5)
                                           : amrex::Real(285.0 + selector);
    const amrex::Real pres_mbar = boundary_cell ? amrex::Real(905.0)
                                                : amrex::Real(860.0 + 7.5 * selector);
    const amrex::Real qsat_local = qsat(tabs, pres_mbar);

    if (flavor == FixtureFlavor::ForceQcCap) {
        return CellState{tabs,
                         pres_mbar,
                         amrex::Real(0.95) * qsat_local,
                         amrex::Real(1.0e-4),
                         amrex::Real(2.0e-1),
                         amrex::Real(0.0)};
    }

    if (flavor == FixtureFlavor::ForceQpCap) {
        return CellState{tabs,
                         pres_mbar,
                         amrex::Real(0.01) * qsat_local,
                         amrex::Real(0.0),
                         amrex::Real(1.0e-8),
                         amrex::Real(0.0)};
    }

    CellState state{tabs, pres_mbar,
                    amrex::Real(0.9) * qsat_local,
                    amrex::Real(5.0e-4),
                    amrex::Real(2.0e-3),
                    amrex::Real(0.1)};

    if (boundary_cell) {
        state.qv = amrex::Real(0.95) * qsat_local;
        state.qc = amrex::Real(4.0e-4);
        state.qp = amrex::Real(1.0e-3);
        state.rain_accum = amrex::Real(0.25);
        return state;
    }

    switch (selector % 8) {
    case 0:
        state.qv = qsat_local + amrex::Real(8.0e-4);
        state.qc = amrex::Real(4.0e-4);
        state.qp = amrex::Real(1.0e-3);
        state.rain_accum = amrex::Real(0.05);
        break;
    case 1:
        state.qv = amrex::Real(0.6) * qsat_local;
        state.qc = amrex::Real(1.8e-3);
        state.qp = amrex::Real(0.0);
        state.rain_accum = amrex::Real(0.06);
        break;
    case 2:
        state.qv = amrex::Real(0.1) * qsat_local;
        state.qc = amrex::Real(0.0);
        state.qp = amrex::Real(5.0e-2);
        state.rain_accum = amrex::Real(0.07);
        break;
    case 3:
        state.qv = qsat_local;
        state.qc = amrex::Real(0.5) * qcw0;
        state.qp = amrex::Real(1.0e-3);
        state.rain_accum = amrex::Real(0.08);
        break;
    case 4:
        state.qv = qsat_local;
        state.qc = amrex::Real(3.0) * qcw0;
        state.qp = amrex::Real(4.0e-3);
        state.rain_accum = amrex::Real(0.09);
        break;
    case 5:
        state.qv = amrex::Real(0.9) * qsat_local;
        state.qc = amrex::Real(8.0e-4);
        state.qp = amrex::Real(2.0e-3);
        state.rain_accum = amrex::Real(0.10);
        break;
    case 6:
        state.qv = qsat_local;
        state.qc = amrex::Real(0.0);
        state.qp = amrex::Real(1.0e-20);
        state.rain_accum = amrex::Real(0.11);
        break;
    default:
        state.qv = amrex::Real(0.7) * qsat_local;
        state.qc = amrex::Real(1.2e-3);
        state.qp = amrex::Real(7.0e-2);
        state.rain_accum = amrex::Real(0.12);
        break;
    }

    return state;
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
CellState make_norain_cell_state (const int selector, const bool boundary_cell)
{
    const amrex::Real tabs = boundary_cell ? amrex::Real(288.0)
                                           : amrex::Real(286.0 + selector);
    const amrex::Real pres_mbar = boundary_cell ? amrex::Real(900.0)
                                                : amrex::Real(875.0 + 5.0 * selector);
    const amrex::Real qsat_local = qsat(tabs, pres_mbar);

    CellState state{tabs, pres_mbar,
                    amrex::Real(0.8) * qsat_local,
                    amrex::Real(3.0e-4),
                    amrex::Real(2.0e-3),
                    amrex::Real(0.2)};

    if (boundary_cell) {
        state.qv = amrex::Real(0.9) * qsat_local;
        state.qc = amrex::Real(2.0e-4);
        state.qp = amrex::Real(1.0e-3);
        state.rain_accum = amrex::Real(0.30);
        return state;
    }

    switch (selector % 4) {
    case 0:
        state.qv = qsat_local + amrex::Real(6.0e-4);
        state.qc = amrex::Real(3.0e-4);
        state.qp = amrex::Real(2.0e-3);
        state.rain_accum = amrex::Real(0.13);
        break;
    case 1:
        state.qv = amrex::Real(0.5) * qsat_local;
        state.qc = amrex::Real(1.2e-3);
        state.qp = amrex::Real(4.0e-3);
        state.rain_accum = amrex::Real(0.14);
        break;
    case 2:
        state.qv = qsat_local - amrex::Real(1.0e-6);
        state.qc = amrex::Real(1.0e-5);
        state.qp = amrex::Real(0.0);
        state.rain_accum = amrex::Real(0.15);
        break;
    default:
        state.qv = amrex::Real(0.7) * qsat_local;
        state.qc = amrex::Real(0.0);
        state.qp = amrex::Real(1.0e-3);
        state.rain_accum = amrex::Real(0.16);
        break;
    }

    return state;
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
CellState make_cell_state (const FixtureOptions& options,
                           const int i, const int j, const int k)
{
    const bool boundary_cell = (i == 0 || i == options.nx - 1 ||
                                j == 0 || j == options.ny - 1);
    const int selector = (i + 2 * j + 3 * k) % 8;
    return (options.moisture_type == MoistureType::Kessler)
    ? make_rain_cell_state(options.flavor, selector, boundary_cell)
        : make_norain_cell_state(selector, boundary_cell);
}

inline amrex::Geometry make_geometry (const int nx, const int ny, const int nz)
{
    amrex::Box domain(amrex::IntVect(AMREX_D_DECL(0, 0, 0)),
                      amrex::IntVect(AMREX_D_DECL(nx - 1, ny - 1, nz - 1)));
    amrex::RealBox real_box({AMREX_D_DECL(0.0, 0.0, 0.0)},
                            {AMREX_D_DECL(1.0, 1.0, 1.0)});
    amrex::Array<int, AMREX_SPACEDIM> periodicity{AMREX_D_DECL(1, 1, 1)};
    return amrex::Geometry(domain, &real_box, amrex::CoordSys::cartesian, periodicity.data());
}

inline SolverChoice make_solver_choice (const FixtureOptions& options)
{
    SolverChoice solver_choice;
    solver_choice.c_p = Cp_d;
    solver_choice.rdOcp = kRdOcp;
    solver_choice.use_shoc = options.use_shoc;
    solver_choice.moisture_type = options.moisture_type;
    solver_choice.ave_plane = 2;
    return solver_choice;
}

inline void fill_base_conserved_state_portable (amrex::MultiFab& cons,
                                                const FixtureOptions& options)
{
    for (amrex::MFIter mfi(cons, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto arr = cons.array(mfi);

        run_and_sync([=] {
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                const CellState state = make_cell_state(options, i, j, k);
                const amrex::Real pres_pa = amrex::Real(100.0) * state.pres_mbar;
                const amrex::Real rho = getRhogivenTandPress(state.tabs, pres_pa, state.qv);
                const amrex::Real theta = getThgivenTandP(state.tabs, pres_pa, kRdOcp);

                arr(i,j,k,Rho_comp) = rho;
                arr(i,j,k,RhoTheta_comp) = rho * theta;
                arr(i,j,k,RhoKE_comp) = amrex::Real(0.0);
                arr(i,j,k,RhoScalar_comp) = amrex::Real(0.0);
                arr(i,j,k,RhoQ1_comp) = rho * state.qv;
                arr(i,j,k,RhoQ2_comp) = rho * state.qc;
                arr(i,j,k,RhoQ3_comp) = rho * state.qp;
            });
        });
    }
}

inline void fill_detj_portable (amrex::MultiFab& detj,
                                const FixtureOptions& options)
{
    for (amrex::MFIter mfi(detj, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto arr = detj.array(mfi);

        run_and_sync([=] {
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                const amrex::Real selector = amrex::Real((i + 2 * j + 3 * k) % 5);
                arr(i,j,k) = amrex::Real(1.05) + amrex::Real(0.03) * selector;
            });
        });
    }
}

inline void initialize_micro_state_portable (Kessler& kessler,
                                             const FixtureOptions& options)
{
    auto& qv = KesslerBFBTestAccessor::MicroVar(kessler, MicVar_Kess::qv);
    auto& qc = KesslerBFBTestAccessor::MicroVar(kessler, MicVar_Kess::qcl);
    auto& qp = KesslerBFBTestAccessor::MicroVar(kessler, MicVar_Kess::qp);
    auto& qt = KesslerBFBTestAccessor::MicroVar(kessler, MicVar_Kess::qt);
    auto& rain_accum = KesslerBFBTestAccessor::MicroVar(kessler, MicVar_Kess::rain_accum);

    for (amrex::MFIter mfi(qv, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto qv_arr = qv.array(mfi);
        auto qc_arr = qc.array(mfi);
        auto qp_arr = qp.array(mfi);
        auto qt_arr = qt.array(mfi);
        auto rain_arr = rain_accum.array(mfi);

        run_and_sync([=] {
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                CellState state = make_cell_state(options, i, j, k);
                const int selector = (i + 2 * j + 3 * k) % 8;

                if (options.inject_clip_cases && options.moisture_type == MoistureType::Kessler &&
                    i > 0 && i < options.nx - 1 && j > 0 && j < options.ny - 1)
                {
                    if (selector == 5) {
                        state.qv = -amrex::Real(1.0e-6);
                        state.qc = -amrex::Real(2.0e-4);
                        state.qp = -amrex::Real(3.0e-4);
                    }
                }

                qv_arr(i,j,k) = state.qv;
                qc_arr(i,j,k) = state.qc;
                qp_arr(i,j,k) = state.qp;
                qt_arr(i,j,k) = state.qv + state.qc;
                rain_arr(i,j,k) = state.rain_accum;
            });
        });
    }
}

inline BitPattern real_bits (const amrex::Real value) noexcept
{
    BitPattern bits = 0;
    amrex::Gpu::memcpy(&bits, &value, sizeof(amrex::Real));
    return bits;
}

inline std::string format_bits (const BitPattern bits)
{
    std::ostringstream stream;
    stream << "0x"
           << std::hex << std::setfill('0')
           << std::setw(static_cast<int>(2 * sizeof(BitPattern)))
           << bits;
    return stream.str();
}

inline amrex::FArrayBox copy_fab_to_host (const amrex::FArrayBox& src)
{
    amrex::FArrayBox host(src.box(), src.nComp(), amrex::The_Pinned_Arena());
#ifdef AMREX_USE_GPU
    if (src.arena() != nullptr && (src.arena()->isManaged() || src.arena()->isDevice())) {
        amrex::Gpu::dtoh_memcpy_async(host.dataPtr(), src.dataPtr(), src.size() * sizeof(amrex::Real));
        amrex::Gpu::streamSynchronize();
        return host;
    }
#endif
    std::memcpy(host.dataPtr(), src.dataPtr(), src.size() * sizeof(amrex::Real));
    return host;
}

inline std::optional<BitMismatch>
find_first_multifab_mismatch (const amrex::MultiFab& expected,
                              const amrex::MultiFab& actual)
{
    AMREX_ALWAYS_ASSERT(expected.nComp() == actual.nComp());
    AMREX_ALWAYS_ASSERT(expected.boxArray() == actual.boxArray());

    int fab_index = 0;
    for (amrex::MFIter mfi(expected); mfi.isValid(); ++mfi, ++fab_index) {
        amrex::FArrayBox expected_host = copy_fab_to_host(expected[mfi]);
        amrex::FArrayBox actual_host = copy_fab_to_host(actual[mfi]);

        const amrex::Box& bx = expected_host.box();
        auto expected_arr = expected_host.const_array();
        auto actual_arr = actual_host.const_array();

        for (int comp = 0; comp < expected_host.nComp(); ++comp) {
            for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k) {
                for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
                    for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                        const amrex::Real expected_value = expected_arr(i,j,k,comp);
                        const amrex::Real actual_value = actual_arr(i,j,k,comp);
                        const BitPattern expected_bits = real_bits(expected_value);
                        const BitPattern actual_bits = real_bits(actual_value);

                        if (expected_bits != actual_bits) {
                            return BitMismatch{fab_index, i, j, k, comp,
                                               expected_bits, actual_bits,
                                               expected_value, actual_value};
                        }
                    }
                }
            }
        }
    }

    return std::nullopt;
}

inline const char* micro_var_name (const int mic_var)
{
    switch (mic_var) {
    case MicVar_Kess::rho: return "rho";
    case MicVar_Kess::theta: return "theta";
    case MicVar_Kess::tabs: return "tabs";
    case MicVar_Kess::pres: return "pres";
    case MicVar_Kess::qt: return "qt";
    case MicVar_Kess::qv: return "qv";
    case MicVar_Kess::qcl: return "qcl";
    case MicVar_Kess::qp: return "qp";
    case MicVar_Kess::rain_accum: return "rain_accum";
    default: return "unknown";
    }
}

inline bool condensation_qv_cap_is_reachable ()
{
    // Under the current warm-rain helper contract, erf_qsatw returns positive
    // saturation pressure and erf_qsat_from_esat returns positive qsat for the
    // pressures carried by these fixtures. With dtqsat >= 0, the unclipped
    // condensation tendency (qv - qsat)/(1 + fac) cannot exceed qv, so the
    // min(..., qv) arm is not reachable without changing that contract.
    return false;
}

inline bool has_cloud_to_rain_qc_cap (const Kessler& model,
                                      const amrex::Real dt)
{
    const amrex::MultiFab& qv_mf = KesslerBFBTestAccessor::MicroVar(model, MicVar_Kess::qv);
    const amrex::MultiFab& qc_mf = KesslerBFBTestAccessor::MicroVar(model, MicVar_Kess::qcl);
    const amrex::MultiFab& qp_mf = KesslerBFBTestAccessor::MicroVar(model, MicVar_Kess::qp);

    for (amrex::MFIter mfi(qc_mf); mfi.isValid(); ++mfi) {
        amrex::FArrayBox qv_host = copy_fab_to_host(qv_mf[mfi]);
        amrex::FArrayBox qc_host = copy_fab_to_host(qc_mf[mfi]);
        amrex::FArrayBox qp_host = copy_fab_to_host(qp_mf[mfi]);

        const amrex::Box& bx = qc_host.box();
        auto qc_arr = qc_host.const_array();
        auto qp_arr = qp_host.const_array();

        for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k) {
            for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
                for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                    const amrex::Real qc = std::max(amrex::Real(0), qc_arr(i,j,k));
                    const amrex::Real qp = std::max(amrex::Real(0), qp_arr(i,j,k));
                    if (qc <= amrex::Real(0)) {
                        continue;
                    }

                    amrex::Real auto_r = amrex::Real(0);
                    if (qc > qcw0) {
                        auto_r = alphaelq;
                    }

                    amrex::Real accrr = amrex::Real(2.2) * std::pow(qp, amrex::Real(0.875));
                    const amrex::Real raw_cloud_to_rain = dt * (accrr * qc + auto_r * (qc - qcw0));
                    if (raw_cloud_to_rain > qc) {
                        return true;
                    }
                }
            }
        }
    }

    return false;
}

inline bool has_rain_to_vapor_qp_cap (const Kessler& model,
                                      const amrex::Real dt)
{
    const amrex::MultiFab& tabs_mf = KesslerBFBTestAccessor::MicroVar(model, MicVar_Kess::tabs);
    const amrex::MultiFab& pres_mf = KesslerBFBTestAccessor::MicroVar(model, MicVar_Kess::pres);
    const amrex::MultiFab& qv_mf   = KesslerBFBTestAccessor::MicroVar(model, MicVar_Kess::qv);
    const amrex::MultiFab& qp_mf   = KesslerBFBTestAccessor::MicroVar(model, MicVar_Kess::qp);
    const amrex::MultiFab& rho_mf  = KesslerBFBTestAccessor::MicroVar(model, MicVar_Kess::rho);

    for (amrex::MFIter mfi(qp_mf); mfi.isValid(); ++mfi) {
        amrex::FArrayBox tabs_host = copy_fab_to_host(tabs_mf[mfi]);
        amrex::FArrayBox pres_host = copy_fab_to_host(pres_mf[mfi]);
        amrex::FArrayBox qv_host = copy_fab_to_host(qv_mf[mfi]);
        amrex::FArrayBox qp_host = copy_fab_to_host(qp_mf[mfi]);
        amrex::FArrayBox rho_host = copy_fab_to_host(rho_mf[mfi]);

        const amrex::Box& bx = qp_host.box();
        auto tabs_arr = tabs_host.const_array();
        auto pres_arr = pres_host.const_array();
        auto qv_arr = qv_host.const_array();
        auto qp_arr = qp_host.const_array();
        auto rho_arr = rho_host.const_array();

        for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k) {
            for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
                for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                    const amrex::Real qv = std::max(amrex::Real(0), qv_arr(i,j,k));
                    const amrex::Real qp = std::max(amrex::Real(0), qp_arr(i,j,k));
                    if (qp <= amrex::Real(0)) {
                        continue;
                    }

                    amrex::Real qsat_local;
                    erf_qsatw(tabs_arr(i,j,k), pres_arr(i,j,k), qsat_local);
                    if (!(qv < qsat_local)) {
                        continue;
                    }

                    const amrex::Real rho = rho_arr(i,j,k);
                    const amrex::Real coeff = amrex::Real(1.6)
                        + amrex::Real(124.9) * std::pow(amrex::Real(0.001) * rho * qp, amrex::Real(0.2046));
                    const amrex::Real raw_rain_to_vapor = amrex::Real(1) / (amrex::Real(0.001) * rho)
                        * (amrex::Real(1) - qv / qsat_local)
                        * coeff
                        * std::pow(amrex::Real(0.001) * rho * qp, amrex::Real(0.525))
                        / (amrex::Real(5.4e5) + amrex::Real(2.55e6) / (pres_arr(i,j,k) * qsat_local))
                        * dt;
                    if (raw_rain_to_vapor > qp) {
                        return true;
                    }
                }
            }
        }
    }

    return false;
}

inline amrex::Real max_bottom_rain_accum (const Kessler& model)
{
    const amrex::MultiFab& rain_mf = KesslerBFBTestAccessor::MicroVar(model, MicVar_Kess::rain_accum);
    amrex::Real max_value = -std::numeric_limits<amrex::Real>::max();

    for (amrex::MFIter mfi(rain_mf); mfi.isValid(); ++mfi) {
        amrex::FArrayBox rain_host = copy_fab_to_host(rain_mf[mfi]);
        const amrex::Box& bx = rain_host.box();
        auto rain_arr = rain_host.const_array();
        const int k = bx.smallEnd(2);

        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
            for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                max_value = std::max(max_value, rain_arr(i,j,k));
            }
        }
    }

    return max_value;
}

struct Harness {
    explicit Harness (const FixtureOptions& options_in)
        : options(options_in),
          geom(make_geometry(options.nx, options.ny, options.nz)),
          ba(geom.Domain()),
          dm(ba),
          cons(ba, dm, RhoQ3_comp + 1, 0),
          solver_choice(make_solver_choice(options))
    {
        cons.setVal(amrex::Real(0.0));
        fill_base_conserved_state_portable(cons, options);

        if (options.with_detj) {
            detj_cc = std::make_unique<amrex::MultiFab>(ba, dm, 1, 0);
            detj_cc->setVal(amrex::Real(1.0));
            fill_detj_portable(*detj_cc, options);
        }

        model.Define(solver_choice);
        model.Set_RealWidth(options.real_width);
        model.Set_dzmin(options.dzmin);
        model.Init(cons, cons.boxArray(), geom, options.dt, z_phys_nd, detj_cc);
        model.Copy_State_to_Micro(cons);
        initialize_micro_state_portable(model, options);
    }

    FixtureOptions options;
    amrex::Geometry geom;
    amrex::BoxArray ba;
    amrex::DistributionMapping dm;
    amrex::MultiFab cons;
    std::unique_ptr<amrex::MultiFab> z_phys_nd;
    std::unique_ptr<amrex::MultiFab> detj_cc;
    Kessler model;
    SolverChoice solver_choice;
};

} // namespace kessler_bfb_test

#endif