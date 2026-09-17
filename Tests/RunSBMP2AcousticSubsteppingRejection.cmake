if(NOT DEFINED MPIEXEC OR NOT DEFINED MPIEXEC_NUMPROC_FLAG OR
   NOT DEFINED NRANKS OR NOT DEFINED TEST_EXE OR NOT DEFINED INPUT OR
   NOT DEFINED WORKING_DIRECTORY OR NOT DEFINED METHOD OR
   NOT DEFINED MOMENT OR NOT DEFINED LOG)
    message(FATAL_ERROR "RunSBMP2AcousticSubsteppingRejection.cmake missing required argument")
endif()

set(_mpi_command ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${NRANKS})
if(DEFINED MPIEXEC_PREFLAGS AND NOT "${MPIEXEC_PREFLAGS}" STREQUAL "")
    separate_arguments(_mpi_preflags UNIX_COMMAND "${MPIEXEC_PREFLAGS}")
    list(APPEND _mpi_command ${_mpi_preflags})
endif()

execute_process(
    COMMAND ${_mpi_command} ${TEST_EXE} ${INPUT}
            erf.v=0 erf.substepping_type=Implicit
            erf.sbm_transport_method=${METHOD} erf.sbm_moment_mode=${MOMENT}
    WORKING_DIRECTORY "${WORKING_DIRECTORY}"
    OUTPUT_FILE "${LOG}"
    ERROR_FILE "${LOG}"
    RESULT_VARIABLE _result)
if(_result EQUAL 0)
    message(FATAL_ERROR
        "P2 incorrectly admitted acoustic substepping for method=${METHOD} moment=${MOMENT}")
endif()

file(READ "${LOG}" _text)
set(_reason "P2 SBM host-CFL qualification does not yet cover ERF acoustic substepping")
string(FIND "${_text}" "${_reason}" _reason_offset)
if(_reason_offset LESS 0)
    message(FATAL_ERROR
        "acoustic-substepping rejection did not report the required capability reason")
endif()
if(_text MATCHES "Coarse STEP")
    message(FATAL_ERROR
        "acoustic-substepping rejection occurred after time integration began")
endif()
if(NOT _text MATCHES "acoustic_substepping_enabled=1")
    message(FATAL_ERROR
        "acoustic-substepping rejection did not expose the enabled mode")
endif()

file(WRITE "${LOG}.evidence"
     "acoustic_substepping=enabled\n"
     "capability_rejection=verified\n"
     "startup_before_time_integration=verified\n"
     "method=${METHOD}\n"
     "moment_mode=${MOMENT}\n"
     "mpi_ranks=${NRANKS}\n")
