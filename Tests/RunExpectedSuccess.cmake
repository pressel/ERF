include("${CMAKE_CURRENT_LIST_DIR}/ResolveExecutable.cmake")
if(NOT DEFINED ERF_EXECUTABLE)
  message(FATAL_ERROR "ERF_EXECUTABLE must name a built erf_exec")
endif()
erf_resolve_executable(_erf_executable "${ERF_EXECUTABLE}" CONFIG "${CONFIG}"
  CONTEXT "Startup-smoke test executable")
if(NOT EXISTS "${_erf_executable}")
  message(FATAL_ERROR "ERF_EXECUTABLE does not exist: ${_erf_executable}")
endif()
if(NOT DEFINED INPUT_FILE OR NOT EXISTS "${INPUT_FILE}")
  message(FATAL_ERROR "INPUT_FILE must name an ERF input deck")
endif()
if(NOT DEFINED TEST_ROOT OR NOT DEFINED TEST_NAME)
  message(FATAL_ERROR "TEST_ROOT and TEST_NAME are required")
endif()

set(_command "${_erf_executable}" "${INPUT_FILE}"
  "max_step=1" "erf.plot_int_1=-1" "erf.plot_int_2=-1"
  "erf.check_int=-1" "amrex.call_addr2line=0")
if(DEFINED RUNTIME_OPTIONS AND NOT "${RUNTIME_OPTIONS}" STREQUAL "")
  separate_arguments(_runtime_options NATIVE_COMMAND "${RUNTIME_OPTIONS}")
  list(APPEND _command ${_runtime_options})
endif()

if(NOT DEFINED RUN_TIMEOUT OR "${RUN_TIMEOUT}" STREQUAL "")
  set(RUN_TIMEOUT 600)
endif()
execute_process(
  COMMAND ${_command}
  WORKING_DIRECTORY "${TEST_ROOT}"
  RESULT_VARIABLE _result
  OUTPUT_VARIABLE _stdout
  ERROR_VARIABLE _stderr
  TIMEOUT ${RUN_TIMEOUT})

set(_output "${_stdout}\n${_stderr}")
set(_log_file "${TEST_ROOT}/${TEST_NAME}.log")
file(WRITE "${_log_file}" "${_output}")
if(NOT "${_result}" MATCHES "^-?[0-9]+$")
  message(FATAL_ERROR
    "ERF did not return a numeric process status (result '${_result}').\n"
    "stdout:\n${_stdout}\nstderr:\n${_stderr}")
endif()
if(NOT "${_result}" STREQUAL "0")
  message(FATAL_ERROR
    "ERF returned ${_result}; expected successful startup and one step.\n"
    "stdout:\n${_stdout}\nstderr:\n${_stderr}")
endif()
if(DEFINED EXPECTED_MESSAGE AND NOT "${EXPECTED_MESSAGE}" STREQUAL "")
  string(FIND "${_output}" "${EXPECTED_MESSAGE}" _diagnostic_index)
  if(_diagnostic_index EQUAL -1)
    message(FATAL_ERROR
      "ERF exited successfully without the expected diagnostic '${EXPECTED_MESSAGE}'.\n"
      "stdout:\n${_stdout}\nstderr:\n${_stderr}")
  endif()
endif()
message(STATUS "ERF completed startup smoke test ${TEST_NAME}")
