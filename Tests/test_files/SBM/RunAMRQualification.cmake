foreach(_required IN ITEMS ERF_EXECUTABLE INPUT_FILE TEST_ROOT MPIEXEC
                           MPIEXEC_NUMPROC_FLAG MPI_RANKS EXPECTED_COMPONENTS)
  if(NOT DEFINED ${_required})
    message(FATAL_ERROR "${_required} is required for the SBM AMR qualification")
  endif()
endforeach()

file(MAKE_DIRECTORY "${TEST_ROOT}")
set(_log_file "${TEST_ROOT}/qualification.log")
set(_command "${MPIEXEC}" "${MPIEXEC_NUMPROC_FLAG}" "${MPI_RANKS}"
             "${ERF_EXECUTABLE}" "${INPUT_FILE}")
if(ZERO_FLOW)
  list(APPEND _command
    "erf.sbm_amr_qualification_expect_zero_flow=true"
    "prob.U_0=0" "prob.V_0=0" "prob.W_0=0")
endif()
if(DEFINED MAX_STEPS)
  list(APPEND _command "max_step=${MAX_STEPS}")
endif()
execute_process(
  COMMAND ${_command}
  WORKING_DIRECTORY "${TEST_ROOT}"
  RESULT_VARIABLE _result
  OUTPUT_VARIABLE _stdout
  ERROR_VARIABLE _stderr)
set(_log "${_stdout}\n${_stderr}")
file(WRITE "${_log_file}" "${_log}")

if(NOT _result EQUAL 0)
  message(FATAL_ERROR
    "SBM AMR qualification exited with ${_result}; see ${_log_file}\n${_log}")
endif()

if(NOT _log MATCHES "MPI initialized with ${MPI_RANKS} MPI processes")
  message(FATAL_ERROR
    "MPI launcher did not initialize the requested ${MPI_RANKS} ranks; see ${_log_file}")
endif()

if(NOT DEFINED EXPECTED_STEPS)
  set(EXPECTED_STEPS 2)
endif()
string(REGEX MATCHALL "SBM_AMR_QUAL step=[^\n]*" _step_lines "${_log}")
list(LENGTH _step_lines _step_count)
if(NOT _step_count EQUAL EXPECTED_STEPS)
  message(FATAL_ERROR
    "Expected ${EXPECTED_STEPS} accepted coarse-step qualification records, got ${_step_count}; see ${_log_file}")
endif()
foreach(_line IN LISTS _step_lines)
  if(ZERO_FLOW)
    if(NOT _line MATCHES
       "fine_substeps=2.*reflux_calls=1.*reflux_syncs=1.*nonzero_reflux_linf=[0-9.eE+-]+.*limiter_active_groups=[0-9]+.*limiter_changed_rate=false.*zero_flow=true.*zero_flow_tolerance=[0-9.eE+-]+")
      message(FATAL_ERROR
        "Zero-flow qualification did not preserve zero flux and inactive limiting: ${_line}\nSee ${_log_file}")
    endif()
  else()
    if(NOT _line MATCHES
       "fine_substeps=2.*reflux_calls=1.*reflux_syncs=1.*nonzero_reflux_linf=[1-9][0-9.eE+-]*.*limiter_active_groups=[1-9][0-9]*.*limiter_changed_rate=true.*zero_flow=false")
      message(FATAL_ERROR
        "Qualification did not exercise two fine substeps, nonzero reflux, and active limiting: ${_line}\nSee ${_log_file}")
    endif()
  endif()
endforeach()

string(REGEX MATCHALL "SBM_AMR_QUAL_INVENTORY component=[^\n]*"
       _inventory_lines "${_log}")
math(EXPR _expected_inventory_count "${EXPECTED_STEPS} * ${EXPECTED_COMPONENTS}")
list(LENGTH _inventory_lines _inventory_count)
if(NOT _inventory_count EQUAL _expected_inventory_count)
  message(FATAL_ERROR
    "Expected ${_expected_inventory_count} per-component composite inventory records, got ${_inventory_count}; see ${_log_file}")
endif()

message(STATUS
  "SBM AMR qualification passed: ranks=${MPI_RANKS}, components=${EXPECTED_COMPONENTS}, coarse_steps=2; log=${_log_file}")
