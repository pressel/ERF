if(NOT DEFINED ERF_SOURCE_FILE OR NOT EXISTS "${ERF_SOURCE_FILE}")
  message(FATAL_ERROR "ERF_SOURCE_FILE must name the checked-out Source/ERF.cpp")
endif()

file(READ "${ERF_SOURCE_FILE}" erf_source)
string(FIND "${erf_source}"
  "ERF::post_timestep (int nstep, double time, double dt_lev0)"
  post_timestep_begin)
if(post_timestep_begin LESS 0)
  message(FATAL_ERROR "Could not find ERF::post_timestep in Source/ERF.cpp")
endif()
string(SUBSTRING "${erf_source}" ${post_timestep_begin} -1 post_timestep)

string(FIND "${post_timestep}"
  "if (cloud_chamber_budget &&\n        solverChoice.moisture_type != MoistureType::SBM)"
  non_sbm_guard)
string(FIND "${post_timestep}" "cloud_chamber_budget->report(" non_sbm_report)
string(FIND "${post_timestep}"
  "if (solverChoice.coupling_type == CouplingType::TwoWay)" native_sync)
string(FIND "${post_timestep}"
  "if (cloud_chamber_budget &&\n        solverChoice.moisture_type == MoistureType::SBM)"
  sbm_guard)
if(non_sbm_report LESS 0)
  message(FATAL_ERROR "Could not find the non-SBM cloud-budget report")
endif()
math(EXPR second_report_offset "${non_sbm_report} + 1")
string(SUBSTRING "${post_timestep}" ${second_report_offset} -1 after_non_sbm_report)
string(FIND "${after_non_sbm_report}" "cloud_chamber_budget->report("
  sbm_report_relative)
if(sbm_report_relative GREATER_EQUAL 0)
  math(EXPR sbm_report "${second_report_offset} + ${sbm_report_relative}")
else()
  set(sbm_report -1)
endif()

if(non_sbm_guard LESS 0 OR native_sync LESS 0 OR sbm_guard LESS 0 OR
   sbm_report LESS 0)
  message(FATAL_ERROR "Cloud-budget mode guards or native sync are missing")
endif()
if(NOT (non_sbm_guard LESS non_sbm_report AND
        non_sbm_report LESS native_sync AND
        native_sync LESS sbm_guard AND sbm_guard LESS sbm_report))
  message(FATAL_ERROR
    "Non-SBM budget reporting must precede native AMR sync; SBM reporting must follow its accepted sync")
endif()

message(STATUS "Cloud-budget report ordering preserves non-SBM pre-sync and SBM post-sync semantics")
