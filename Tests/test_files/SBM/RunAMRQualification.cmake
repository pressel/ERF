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
if(DEFINED SBM_NBINS)
  math(EXPR _sbm_cloud_rain_split "${SBM_NBINS} / 2")
  list(APPEND _command "erf.sbm_nbins=${SBM_NBINS}"
       "erf.sbm_cloud_rain_split=${_sbm_cloud_rain_split}")
endif()
if(EXPECT_SPLIT)
  list(APPEND _command "erf.sbm_amr_qualification_expect_split=true")
endif()
if(EXPECT_EMPTY_FINE_RANK)
  list(APPEND _command "erf.sbm_amr_qualification_expect_empty_fine_rank=true")
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
if(NOT DEFINED EXPECTED_FINE_SUBSTEPS)
  set(EXPECTED_FINE_SUBSTEPS 3)
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
       "fine_substeps=${EXPECTED_FINE_SUBSTEPS}.*reflux_calls=[1-9][0-9]*.*reflux_syncs=[1-9][0-9]*.*nonzero_reflux_linf=[0-9.eE+-]+.*limiter_active_groups=[0-9]+.*limiter_changed_rate=(true|false).*zero_flow=true.*zero_flow_tolerance=[0-9.eE+-]+")
      message(FATAL_ERROR
        "Zero-flow qualification exceeded the accepted correction or inventory bound: ${_line}\nSee ${_log_file}")
    endif()
  else()
    if(NOT _line MATCHES
       "fine_substeps=${EXPECTED_FINE_SUBSTEPS}.*reflux_calls=1.*reflux_syncs=1.*nonzero_reflux_linf=[1-9][0-9.eE+-]*.*limiter_active_groups=[1-9][0-9]*.*limiter_changed_rate=true.*zero_flow=false")
      message(FATAL_ERROR
        "Qualification did not exercise two fine substeps, nonzero reflux, and active limiting: ${_line}\nSee ${_log_file}")
    endif()
  endif()
endforeach()

if(EXPECT_SPLIT AND NOT _log MATCHES
   "SBM_AMR_QUAL_SPLIT_LAYOUT coarse_ranks=[2-9][0-9]* fine_ranks=[2-9][0-9]* remote_parent=true remote_fine=true")
  message(FATAL_ERROR
    "Runtime did not prove split coarse/fine ownership and remote parent/fine donors; see ${_log_file}")
endif()
if(EXPECT_EMPTY_FINE_RANK AND NOT _log MATCHES
   "SBM_AMR_QUAL_EMPTY_FINE_RANK empty_ranks=[1-9][0-9]*")
  message(FATAL_ERROR
    "Runtime did not prove a rank with no fine cells; see ${_log_file}")
endif()

if(DEFINED SBM_NBINS)
  math(EXPR _expected_profile_chunks "(${EXPECTED_COMPONENTS} + 15) / 16")
  math(EXPR _expected_componentwise_copies
       "2 * ${EXPECTED_COMPONENTS} + 2 * ${_expected_profile_chunks}")
  string(REGEX MATCHALL "SBM_AMR_PROFILE_PARENT rank=[^\n]*"
         _parent_profile_lines "${_log}")
  math(EXPR _expected_parent_profiles
       "${MPI_RANKS} * ${EXPECTED_FINE_SUBSTEPS} * 9")
  list(LENGTH _parent_profile_lines _parent_profile_count)
  if(NOT _parent_profile_count EQUAL _expected_parent_profiles)
    message(FATAL_ERROR
      "Expected ${_expected_parent_profiles} measured parent-role records, got ${_parent_profile_count}; see ${_log_file}")
  endif()
  foreach(_line IN LISTS _parent_profile_lines)
    if(NOT _line MATCHES
       "ncomp=${EXPECTED_COMPONENTS}.*chunks=${_expected_profile_chunks} parent_fill_mode=componentwise parallel_copy_calls=${_expected_componentwise_copies} parallel_copy_wall_s=[0-9.eE+-]+ parent_fill_wall_s=[0-9.eE+-]+ coarse_stage_fab_bytes_local=[0-9]+ coarse_source_pack_fab_bytes_local=0 rank_live_fab_bytes=[0-9]+")
      message(FATAL_ERROR
        "Incomplete measured parent-fill record: ${_line}\nSee ${_log_file}")
    endif()
  endforeach()
  string(REGEX MATCHALL "SBM_AMR_PROFILE_FINE_STEP rank=[^\n]*"
         _fine_step_profile_lines "${_log}")
  math(EXPR _expected_fine_step_profiles
       "${MPI_RANKS} * ${EXPECTED_FINE_SUBSTEPS}")
  list(LENGTH _fine_step_profile_lines _fine_step_profile_count)
  if(NOT _fine_step_profile_count EQUAL _expected_fine_step_profiles)
    message(FATAL_ERROR
      "Expected ${_expected_fine_step_profiles} measured fine-step records, got ${_fine_step_profile_count}; see ${_log_file}")
  endif()
endif()

if(DEFINED REFERENCE_LOG)
  if(NOT DEFINED PYTHON_EXECUTABLE OR NOT EXISTS "${PYTHON_EXECUTABLE}")
    message(FATAL_ERROR "PYTHON_EXECUTABLE is required for decomposition parity")
  endif()
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}"
            "${CMAKE_CURRENT_LIST_DIR}/CompareAMRQualificationLogs.py"
            "${REFERENCE_LOG}" "${_log_file}" "${EXPECTED_COMPONENTS}"
    RESULT_VARIABLE _compare_result
    OUTPUT_VARIABLE _compare_stdout
    ERROR_VARIABLE _compare_stderr)
  if(NOT _compare_result EQUAL 0)
    message(FATAL_ERROR
      "Split decomposition comparison failed: ${_compare_stdout}${_compare_stderr}")
  endif()
endif()

string(REGEX MATCHALL "SBM_AMR_QUAL_INVENTORY component=[^\n]*"
       _inventory_lines "${_log}")
math(EXPR _expected_inventory_count "${EXPECTED_STEPS} * ${EXPECTED_COMPONENTS}")
list(LENGTH _inventory_lines _inventory_count)
if(NOT _inventory_count EQUAL _expected_inventory_count)
  message(FATAL_ERROR
    "Expected ${_expected_inventory_count} per-component composite inventory records, got ${_inventory_count}; see ${_log_file}")
endif()

message(STATUS
  "SBM AMR qualification passed: ranks=${MPI_RANKS}, components=${EXPECTED_COMPONENTS}, coarse_steps=${EXPECTED_STEPS}, fine_substeps=${EXPECTED_FINE_SUBSTEPS}; log=${_log_file}")
