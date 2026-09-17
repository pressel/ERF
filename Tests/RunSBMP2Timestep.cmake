if(NOT DEFINED MPIEXEC OR NOT DEFINED MPIEXEC_NUMPROC_FLAG OR
   NOT DEFINED NRANKS OR NOT DEFINED TEST_EXE OR NOT DEFINED INPUT OR
   NOT DEFINED WORKING_DIRECTORY OR NOT DEFINED LOG)
    message(FATAL_ERROR "RunSBMP2Timestep.cmake missing required argument")
endif()

set(_mpi_command ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${NRANKS})
if(DEFINED MPIEXEC_PREFLAGS AND NOT "${MPIEXEC_PREFLAGS}" STREQUAL "")
    separate_arguments(_mpi_preflags UNIX_COMMAND "${MPIEXEC_PREFLAGS}")
    list(APPEND _mpi_command ${_mpi_preflags})
endif()

set(_diff_log "${WORKING_DIRECTORY}/diffusion.log")
set(_zero_log "${WORKING_DIRECTORY}/zero_diffusion.log")
set(_diff_command ${_mpi_command} ${TEST_EXE} ${INPUT} erf.v=2)
execute_process(
    COMMAND ${_diff_command}
    WORKING_DIRECTORY "${WORKING_DIRECTORY}"
    OUTPUT_FILE "${_diff_log}"
    ERROR_FILE "${_diff_log}"
    RESULT_VARIABLE _diff_result)
if(NOT _diff_result EQUAL 0)
    message(FATAL_ERROR "SBM host timestep diffusion run failed: ${_diff_result}")
endif()

set(_zero_command ${_mpi_command} ${TEST_EXE} ${INPUT} erf.v=2 erf.sbm_diffusion_coeff=0.0)
execute_process(
    COMMAND ${_zero_command}
    WORKING_DIRECTORY "${WORKING_DIRECTORY}"
    OUTPUT_FILE "${_zero_log}"
    ERROR_FILE "${_zero_log}"
    RESULT_VARIABLE _zero_result)
if(NOT _zero_result EQUAL 0)
    message(FATAL_ERROR "SBM host timestep zero-diffusion run failed: ${_zero_result}")
endif()

file(READ "${_diff_log}" _diff_text)
file(READ "${_zero_log}" _zero_text)
string(REGEX MATCH "SBM host stability bound at level 0 = ([0-9eE+.-]+)" _bound_match "${_diff_text}")
if(NOT _bound_match)
    message(FATAL_ERROR "diffusion run did not report the SBM host bound")
endif()
set(_bound "${CMAKE_MATCH_1}")
string(REGEX MATCH "Coarse STEP 1 ends\\. TIME = [0-9eE+.-]+ DT = ([0-9eE+.-]+)" _dt_match "${_diff_text}")
if(NOT _dt_match)
    message(FATAL_ERROR "diffusion run did not report the selected host timestep")
endif()
set(_diff_dt "${CMAKE_MATCH_1}")
string(REGEX MATCH "SBM host stability bound at level 0 = ([0-9eE+.-]+)" _zero_bound_match "${_zero_text}")
if(NOT _zero_bound_match)
    message(FATAL_ERROR "zero-diffusion run did not report the SBM host bound")
endif()
set(_zero_bound "${CMAKE_MATCH_1}")
string(REGEX MATCH "Coarse STEP 1 ends\\. TIME = [0-9eE+.-]+ DT = ([0-9eE+.-]+)" _zero_dt_match "${_zero_text}")
if(NOT _zero_dt_match)
    message(FATAL_ERROR "zero-diffusion run did not report the selected host timestep")
endif()
set(_zero_dt "${CMAKE_MATCH_1}")

if(NOT _diff_dt LESS 5.0e-3)
    message(FATAL_ERROR "SBM diffusion did not reduce the fixed host timestep: ${_diff_dt}")
endif()
if(_diff_dt GREATER _bound)
    message(FATAL_ERROR "selected diffusion timestep exceeds SBM bound: ${_diff_dt} > ${_bound}")
endif()
if(NOT _zero_dt EQUAL 5.0e-3)
    message(FATAL_ERROR "zero-diffusion SBM changed the fixed host timestep: ${_zero_dt}")
endif()
if(NOT _zero_bound GREATER _zero_dt)
    message(FATAL_ERROR "zero-diffusion host bound is not above the selected fixed timestep: ${_zero_bound}")
endif()

file(WRITE "${LOG}"
     "host_timestep_diffusion_bound=${_bound}\n"
     "host_timestep_diffusion_selected=${_diff_dt}\n"
     "host_timestep_zero_diffusion_bound=${_zero_bound}\n"
     "host_timestep_zero_diffusion_selected=${_zero_dt}\n"
     "host_timestep_combined_advection_diffusion=verified\n"
     "host_timestep_mpi_ranks=${NRANKS}\n")
