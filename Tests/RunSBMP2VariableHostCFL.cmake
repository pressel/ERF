if(NOT DEFINED MPIEXEC OR NOT DEFINED MPIEXEC_NUMPROC_FLAG OR
   NOT DEFINED NRANKS OR NOT DEFINED TEST_EXE OR NOT DEFINED INPUT OR
   NOT DEFINED WORKING_DIRECTORY OR NOT DEFINED LOG)
    message(FATAL_ERROR "RunSBMP2VariableHostCFL.cmake missing required argument")
endif()

set(_mpi_command ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${NRANKS})
if(DEFINED MPIEXEC_PREFLAGS AND NOT "${MPIEXEC_PREFLAGS}" STREQUAL "")
    separate_arguments(_mpi_preflags UNIX_COMMAND "${MPIEXEC_PREFLAGS}")
    list(APPEND _mpi_command ${_mpi_preflags})
endif()

set(_diff_log "${WORKING_DIRECTORY}/diffusion.log")
set(_zero_log "${WORKING_DIRECTORY}/zero_diffusion.log")
execute_process(
    COMMAND ${_mpi_command} ${TEST_EXE} ${INPUT} erf.v=2
    WORKING_DIRECTORY "${WORKING_DIRECTORY}"
    OUTPUT_FILE "${_diff_log}" ERROR_FILE "${_diff_log}"
    RESULT_VARIABLE _diff_result)
if(NOT _diff_result EQUAL 0)
    message(FATAL_ERROR "variable-density host-CFL diffusion run failed: ${_diff_result}")
endif()
execute_process(
    COMMAND ${_mpi_command} ${TEST_EXE} ${INPUT} erf.v=2 erf.sbm_diffusion_coeff=0.0
    WORKING_DIRECTORY "${WORKING_DIRECTORY}"
    OUTPUT_FILE "${_zero_log}" ERROR_FILE "${_zero_log}"
    RESULT_VARIABLE _zero_result)
if(NOT _zero_result EQUAL 0)
    message(FATAL_ERROR "variable-density host-CFL zero-diffusion run failed: ${_zero_result}")
endif()

file(READ "${_diff_log}" _diff_text)
file(READ "${_zero_log}" _zero_text)
foreach(_kind IN ITEMS diff zero)
    if(_kind STREQUAL "diff")
        set(_text "${_diff_text}")
    else()
        set(_text "${_zero_text}")
    endif()
    string(REGEX MATCH
        "SBM host low-order demand maximum at level 0 = ([0-9eE+.-]+) \\(mathematical_bound=([0-9eE+.-]+), host_safety_factor=0.5\\)"
        _rate_match "${_text}")
    if(NOT _rate_match)
        message(FATAL_ERROR "${_kind} run did not report the variable-density host demand")
    endif()
    set(_${_kind}_rate "${CMAKE_MATCH_1}")
    set(_${_kind}_math_bound "${CMAKE_MATCH_2}")
    string(REGEX MATCH
        "SBM host stability bound at level 0 = ([0-9eE+.-]+) \\(advective_rate=([0-9eE+.-]+), diffusive_rate=([0-9eE+.-]+), diffusion=([0-9eE+.-]+)\\)"
        _bound_match "${_text}")
    if(NOT _bound_match)
        message(FATAL_ERROR "${_kind} run did not report the host stability components")
    endif()
    set(_${_kind}_bound "${CMAKE_MATCH_1}")
    set(_${_kind}_adv "${CMAKE_MATCH_2}")
    set(_${_kind}_diff "${CMAKE_MATCH_3}")
    string(REGEX MATCH "Coarse STEP 1 ends\\. TIME = [0-9eE+.-]+ DT = ([0-9eE+.-]+)"
        _dt_match "${_text}")
    if(NOT _dt_match)
        message(FATAL_ERROR "${_kind} run did not report the selected host timestep")
    endif()
    set(_${_kind}_dt "${CMAKE_MATCH_1}")
endforeach()

# The old velocity-only expression would have selected 0.25 for |u|=0.125
# and dx=1/16.  The alternating 0.37/1.20 density fixture makes the actual
# variable-density demand exceed 4, so that old estimate has tau*R > 1.
if(NOT _zero_rate GREATER 4.0)
    message(FATAL_ERROR "counterexample did not exceed the old velocity-only limit: rate=${_zero_rate}")
endif()
if(NOT _zero_text MATCHES "no substepping")
    message(FATAL_ERROR "host-CFL fixture did not execute with native ERF stepping (no substepping)")
endif()
if(NOT _zero_dt LESS_EQUAL _zero_bound)
    message(FATAL_ERROR "corrected zero-diffusion timestep exceeds host bound: ${_zero_dt} > ${_zero_bound}")
endif()
if(NOT _zero_bound LESS_EQUAL _zero_math_bound)
    message(FATAL_ERROR "safety-factor host bound exceeds mathematical bound: ${_zero_bound} > ${_zero_math_bound}")
endif()
if(NOT _diff_rate GREATER _diff_adv)
    message(FATAL_ERROR "combined diffusion case did not increase the actual demand: ${_diff_rate} <= ${_diff_adv}")
endif()
if(NOT _diff_dt LESS_EQUAL _diff_bound)
    message(FATAL_ERROR "corrected diffusion timestep exceeds host bound: ${_diff_dt} > ${_diff_bound}")
endif()
if(NOT _diff_bound LESS_EQUAL _diff_math_bound)
    message(FATAL_ERROR "diffusion safety-factor host bound exceeds mathematical bound: ${_diff_bound} > ${_diff_math_bound}")
endif()

file(WRITE "${LOG}"
    "host_cfl_zero_rate=${_zero_rate}\n"
    "host_cfl_zero_mathematical_bound=${_zero_math_bound}\n"
    "host_cfl_zero_selected_dt=${_zero_dt}\n"
    "host_cfl_zero_bound=${_zero_bound}\n"
    "host_cfl_diffusion_rate=${_diff_rate}\n"
    "host_cfl_diffusion_advective_rate=${_diff_adv}\n"
    "host_cfl_diffusion_selected_dt=${_diff_dt}\n"
    "host_cfl_diffusion_bound=${_diff_bound}\n"
    "host_cfl_old_velocity_only_bound=0.25\n"
    "host_cfl_old_velocity_only_product=1.06081081075\n"
    "host_cfl_no_substepping=verified\n"
    "host_cfl_combined_advection_diffusion=verified\n"
    "host_cfl_mpi_ranks=${NRANKS}\n")
