if(NOT DEFINED CHECKER OR NOT DEFINED WORKING_DIRECTORY OR
   NOT DEFINED METHOD OR NOT DEFINED MOMENT OR NOT DEFINED LOG)
    message(FATAL_ERROR "RunSBMP2AcousticSubsteppingRejection.cmake missing required argument")
endif()

execute_process(
    COMMAND "${CHECKER}" "${METHOD}" "${MOMENT}"
    WORKING_DIRECTORY "${WORKING_DIRECTORY}"
    OUTPUT_FILE "${LOG}"
    ERROR_FILE "${LOG}"
    RESULT_VARIABLE _result)
if(NOT _result EQUAL 0)
    message(FATAL_ERROR
        "deterministic P2 acoustic-substepping preflight failed for method=${METHOD} moment=${MOMENT}")
endif()

file(READ "${LOG}" _text)
set(_reason "P2 SBM host-CFL qualification does not yet cover ERF acoustic substepping")
string(FIND "${_text}" "${_reason}" _reason_offset)
if(_reason_offset LESS 0)
    message(FATAL_ERROR
        "acoustic-substepping rejection did not report the required capability reason")
endif()
if(NOT _text MATCHES "acoustic_substepping_enabled=1")
    message(FATAL_ERROR
        "acoustic-substepping rejection did not expose the enabled mode")
endif()
if(NOT _text MATCHES "startup_before_time_integration=verified")
    message(FATAL_ERROR
        "acoustic-substepping preflight did not establish startup-before-time-integration")
endif()

file(WRITE "${LOG}.evidence"
     "acoustic_substepping=enabled\n"
     "capability_rejection=verified\n"
     "startup_before_time_integration=verified\n"
     "method=${METHOD}\n"
     "moment_mode=${MOMENT}\n"
     "oracle=host_capability_preflight\n")
