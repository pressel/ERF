#include "ERF_Morrison.H"

using namespace amrex;

namespace {

struct PlotVarEntry {
    const char* name;
    int index;
};

constexpr PlotVarEntry plot_entries[] = {
    {"micro_rho",    MicVar_Morr::rho},
    {"micro_theta",  MicVar_Morr::theta},
    {"micro_temp",   MicVar_Morr::tabs},
    {"micro_pres",   MicVar_Morr::pres},
    {"micro_qv",     MicVar_Morr::qv},
    {"micro_qc",     MicVar_Morr::qcl},
    {"micro_qi",     MicVar_Morr::qci},
    {"micro_qn",     MicVar_Morr::qn},
    {"micro_qt",     MicVar_Morr::qt},
    {"micro_qp",     MicVar_Morr::qp},
    {"micro_qrain",  MicVar_Morr::qpr},
    {"micro_qsnow",  MicVar_Morr::qps},
    {"micro_qgraup", MicVar_Morr::qpg},
    {"micro_nc",     MicVar_Morr::nc},
    {"micro_nr",     MicVar_Morr::nr},
    {"micro_ni",     MicVar_Morr::ni},
    {"micro_ns",     MicVar_Morr::ns},
    {"micro_ng",     MicVar_Morr::ng},
    {"micro_omega",  MicVar_Morr::omega}
};

} // namespace

void
Morrison::GetPlotVarNames (Vector<std::string>& a_vec) const
{
    a_vec.clear();
    a_vec.reserve(sizeof(plot_entries) / sizeof(plot_entries[0]));
    for (const auto& entry : plot_entries) {
        a_vec.emplace_back(entry.name);
    }
}

void
Morrison::GetPlotVar (const std::string& a_name,
                      MultiFab& a_mf) const
{
    const MultiFab* src = nullptr;

    for (const auto& entry : plot_entries) {
        if (a_name == entry.name) {
            src = mic_fab_vars[entry.index].get();
            break;
        }
    }

    if (src == nullptr) {
        amrex::Abort("Morrison::GetPlotVar: unknown plot variable " + a_name);
    }

    MultiFab::Copy(a_mf, *src, 0, 0, 1, 0);
}
