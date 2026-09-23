if(NOT DEFINED MPIEXEC OR NOT DEFINED MPIEXEC_NUMPROC_FLAG OR
   NOT DEFINED TEST_EXE OR NOT DEFINED INPUT OR NOT DEFINED WORKING_DIRECTORY OR
   NOT DEFINED LOG)
    message(FATAL_ERROR "RunSBMP2OwnershipFault.cmake missing required argument")
endif()

set(_mpi_command ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1)
if(DEFINED MPIEXEC_PREFLAGS AND NOT "${MPIEXEC_PREFLAGS}" STREQUAL "")
    separate_arguments(_mpi_preflags UNIX_COMMAND "${MPIEXEC_PREFLAGS}")
    list(APPEND _mpi_command ${_mpi_preflags})
endif()

set(_summary "")
foreach(_fault IN ITEMS qc_advection qr_diffusion bulk_clip)
    set(_fault_log "${WORKING_DIRECTORY}/${_fault}.log")
    file(REMOVE "${_fault_log}")
    set(_command ${_mpi_command} ${TEST_EXE} ${INPUT}
        "erf.sbm_test_fault_injection=${_fault}")
    execute_process(
        COMMAND ${_command}
        WORKING_DIRECTORY "${WORKING_DIRECTORY}"
        OUTPUT_FILE "${_fault_log}"
        ERROR_FILE "${_fault_log}"
        RESULT_VARIABLE _result)
    if(_result EQUAL 0)
        message(FATAL_ERROR
            "SBM ownership fault ${_fault} unexpectedly completed successfully")
    endif()
    file(READ "${_fault_log}" _fault_text)
    string(FIND "${_fault_text}"
        "SBM native qc/qr ownership invariant failed before projection"
        _invariant_offset)
    if(_invariant_offset EQUAL -1)
        message(FATAL_ERROR
            "SBM ownership fault ${_fault} failed without the pre-projection invariant; see ${_fault_log}")
    endif()
    string(APPEND _summary
        "fault=${_fault} result=${_result} pre_projection_invariant=detected\n")
endforeach()

file(WRITE "${LOG}" "format=erf-sbm-p2-ownership-faults-v1\n${_summary}passed=1\n")
