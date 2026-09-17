if(NOT DEFINED MPIEXEC OR NOT DEFINED MPIEXEC_NUMPROC_FLAG OR
   NOT DEFINED NRANKS OR NOT DEFINED TEST_EXE OR NOT DEFINED INPUT OR
   NOT DEFINED WORKING_DIRECTORY OR NOT DEFINED LOG OR NOT DEFINED COMPOSITE)
    message(FATAL_ERROR "RunSBMP2AMR.cmake missing required argument")
endif()

set(_command ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${NRANKS})
if(DEFINED MPIEXEC_PREFLAGS AND NOT "${MPIEXEC_PREFLAGS}" STREQUAL "")
    separate_arguments(_mpi_preflags UNIX_COMMAND "${MPIEXEC_PREFLAGS}")
    list(APPEND _command ${_mpi_preflags})
endif()
list(APPEND _command ${TEST_EXE} ${INPUT}
     erf.sbm_composite_diagnostic_file=${COMPOSITE})

execute_process(
    COMMAND ${_command}
    WORKING_DIRECTORY "${WORKING_DIRECTORY}"
    OUTPUT_FILE "${LOG}"
    ERROR_FILE "${LOG}"
    RESULT_VARIABLE simulation_result)
if(NOT simulation_result EQUAL 0)
    message(FATAL_ERROR "SBM P2 AMR simulation failed: ${simulation_result}")
endif()

file(READ "${LOG}" simulation_log)
foreach(expected_text
        "SBM layout identity"
        "ncomp=8"
        "moment=1"
        "Coarse STEP 2 ends"
        "SBM dynamic qualification refinement")
    string(FIND "${simulation_log}" "${expected_text}" expected_offset)
    if(expected_offset EQUAL -1)
        message(FATAL_ERROR "SBM P2 AMR log is missing expected text: ${expected_text}")
    endif()
endforeach()

if(NOT EXISTS "${COMPOSITE}")
    message(FATAL_ERROR "SBM P2 AMR run did not create composite diagnostic ${COMPOSITE}")
endif()
file(READ "${COMPOSITE}" composite_text)
foreach(expected_text
        "format=erf-sbm-p2-composite-v1"
        "finest_level=1"
        "level_count=2"
        "moment_mode=2"
        "interface_oracle_passed=1"
        "interface_oracle=accepted-transfer-mismatch-vs-authoritative-reflux-correction"
        "passed=1")
    string(FIND "${composite_text}" "${expected_text}" expected_offset)
    if(expected_offset EQUAL -1)
        message(FATAL_ERROR "SBM P2 composite diagnostic is missing expected text: ${expected_text}")
    endif()
endforeach()
foreach(comp IN ITEMS 0 1 2 3 4 5 6 7)
    foreach(field IN ITEMS "interface_coarse_transfer_comp_${comp}"
                           "interface_fine_transfer_comp_${comp}"
                           "interface_mismatch_comp_${comp}"
                           "interface_reflux_correction_comp_${comp}"
                           "interface_oracle_error_comp_${comp}")
        string(FIND "${composite_text}" "${field}=" field_offset)
        if(field_offset EQUAL -1)
            message(FATAL_ERROR "SBM P2 composite diagnostic is missing ${field}")
        endif()
    endforeach()
endforeach()
foreach(comp IN ITEMS 0 1 2 3 4 5 6 7)
    string(REGEX MATCH "composite_error_comp_${comp}=([0-9eE+.-]+)" _error_match "${composite_text}")
    set(_error_value "${CMAKE_MATCH_1}")
    string(REGEX MATCH "composite_tolerance_comp_${comp}=([0-9eE+.-]+)" _tolerance_match "${composite_text}")
    set(_tolerance_value "${CMAKE_MATCH_1}")
    if(NOT _error_match OR NOT _tolerance_match)
        message(FATAL_ERROR "SBM P2 composite diagnostic is missing component ${comp} error/tolerance")
    endif()
    # The production diagnostic owns the physical, per-component tolerance.
    # Do not reintroduce a checker-side order-one or fixed 1e-10 floor.
    if(_error_value GREATER _tolerance_value)
        message(FATAL_ERROR "SBM P2 composite component ${comp} exceeds its physical tolerance: ${_error_value} > ${_tolerance_value}")
    endif()
endforeach()
