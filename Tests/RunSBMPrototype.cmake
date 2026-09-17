if(NOT DEFINED MPIEXEC OR NOT DEFINED MPIEXEC_NUMPROC_FLAG OR
   NOT DEFINED NRANKS OR NOT DEFINED TEST_EXE OR NOT DEFINED INPUT OR
   NOT DEFINED WORKING_DIRECTORY OR NOT DEFINED LOG OR
   NOT DEFINED DIAGNOSTIC OR NOT DEFINED CHECKER OR
   NOT DEFINED EXPECTED_COMPONENTS)
    message(FATAL_ERROR "RunSBMPrototype.cmake missing required argument")
endif()

set(_command ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${NRANKS})
if(DEFINED MPIEXEC_PREFLAGS AND NOT "${MPIEXEC_PREFLAGS}" STREQUAL "")
    separate_arguments(_mpi_preflags UNIX_COMMAND "${MPIEXEC_PREFLAGS}")
    list(APPEND _command ${_mpi_preflags})
endif()
list(APPEND _command ${TEST_EXE} ${INPUT})
if(DEFINED RUNTIME_OPTIONS AND NOT "${RUNTIME_OPTIONS}" STREQUAL "")
    separate_arguments(_runtime_options UNIX_COMMAND "${RUNTIME_OPTIONS}")
    list(APPEND _command ${_runtime_options})
endif()

execute_process(
    COMMAND ${_command}
    WORKING_DIRECTORY "${WORKING_DIRECTORY}"
    OUTPUT_FILE "${LOG}"
    ERROR_FILE "${LOG}"
    RESULT_VARIABLE simulation_result)
if(NOT simulation_result EQUAL 0)
    message(FATAL_ERROR "SBM P1 simulation failed: ${simulation_result}")
endif()

file(READ "${LOG}" simulation_log)
foreach(expected_text
        "SBM layout identity"
        "SBM P1 auxiliary state"
        "components=${EXPECTED_COMPONENTS}"
        "Coarse STEP 1 ends"
        "Coarse STEP 2 ends")
    string(FIND "${simulation_log}" "${expected_text}" expected_offset)
    if(expected_offset EQUAL -1)
        message(FATAL_ERROR "SBM P1 log is missing expected text: ${expected_text}")
    endif()
endforeach()

if(NOT DEFINED METHOD)
    message(FATAL_ERROR "SBM P1 runner requires METHOD for temporal-stage qualification")
endif()
if(METHOD STREQUAL "compressible")
    set(_expected_actual_stages 6)
    set(_expected_temporal_mode "compressible_rk3")
elseif(METHOD STREQUAL "anelastic")
    set(_expected_actual_stages 4)
    set(_expected_temporal_mode "anelastic_heun")
else()
    message(FATAL_ERROR "unknown SBM prototype temporal method: ${METHOD}")
endif()
string(REGEX MATCHALL
    "SBM actual stage low-order demand level=[0-9]+ stage=[0-9]+ temporal_mode=[A-Za-z0-9_]+ acoustic_substepping=disabled tau=[0-9eE+.-]+ actual_rate=[0-9eE+.-]+ actual_advective_rate=[0-9eE+.-]+ actual_diffusive_rate=[0-9eE+.-]+ tau_actual_rate=[0-9eE+.-]+"
    _actual_stage_lines "${simulation_log}")
list(LENGTH _actual_stage_lines _actual_stage_count)
if(NOT _actual_stage_count EQUAL _expected_actual_stages)
    message(FATAL_ERROR "${METHOD} SBM run reported ${_actual_stage_count} actual stages, expected ${_expected_actual_stages}")
endif()
set(_actual_max_tau_rate 0.0)
set(_actual_modes)
foreach(_line IN LISTS _actual_stage_lines)
    string(REGEX MATCH
        "stage=([0-9]+) temporal_mode=([A-Za-z0-9_]+).*tau_actual_rate=([0-9eE+.-]+)"
        _actual_match "${_line}")
    if(NOT _actual_match OR NOT CMAKE_MATCH_2 STREQUAL _expected_temporal_mode)
        message(FATAL_ERROR "${METHOD} SBM actual-stage temporal diagnostic is inconsistent: ${_line}")
    endif()
    if(CMAKE_MATCH_3 GREATER _actual_max_tau_rate)
        set(_actual_max_tau_rate "${CMAKE_MATCH_3}")
    endif()
    list(APPEND _actual_modes "${CMAKE_MATCH_2}")
endforeach()
if(_actual_max_tau_rate GREATER 1.0000000001)
    message(FATAL_ERROR "${METHOD} SBM selected timestep violates actual stage demand: ${_actual_max_tau_rate}")
endif()
list(REMOVE_DUPLICATES _actual_modes)
file(WRITE "${DIAGNOSTIC}.actual_stage"
    "temporal_mode=${_expected_temporal_mode}\n"
    "actual_stage_count=${_actual_stage_count}\n"
    "actual_stage_max_tau_rate=${_actual_max_tau_rate}\n"
    "acoustic_substepping=disabled\n"
    "mpi_ranks=${NRANKS}\n")

if(NOT EXISTS "${DIAGNOSTIC}")
    message(FATAL_ERROR "SBM P1 simulation did not produce numerical diagnostic: ${DIAGNOSTIC}")
endif()
execute_process(
    COMMAND "${CHECKER}" "${DIAGNOSTIC}"
    WORKING_DIRECTORY "${WORKING_DIRECTORY}"
    RESULT_VARIABLE checker_result)
if(NOT checker_result EQUAL 0)
    message(FATAL_ERROR "SBM P1 numerical qualification failed: ${checker_result}")
endif()
