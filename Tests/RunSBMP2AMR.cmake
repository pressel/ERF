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
list(APPEND _command ${TEST_EXE} ${INPUT})
if(DEFINED RUNTIME_OPTIONS AND NOT "${RUNTIME_OPTIONS}" STREQUAL "")
    separate_arguments(_runtime_options UNIX_COMMAND "${RUNTIME_OPTIONS}")
    list(APPEND _command ${_runtime_options})
endif()
list(APPEND _command
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
set(_variable_density_case FALSE)
if(DEFINED RUNTIME_OPTIONS)
    string(FIND "${RUNTIME_OPTIONS}" "erf.sbm_manufactured_variable_density=true" _variable_density_offset)
    if(NOT _variable_density_offset EQUAL -1)
        set(_variable_density_case TRUE)
    endif()
endif()
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
if(_variable_density_case)
    foreach(expected_text IN ITEMS
            "variable_density_ratio_error="
            "variable_density_ratio_tolerance=")
        string(FIND "${composite_text}" "${expected_text}" _variable_density_field_offset)
        if(_variable_density_field_offset EQUAL -1)
            message(FATAL_ERROR "variable-density SBM P2 diagnostic is missing ${expected_text}")
        endif()
    endforeach()
    string(REGEX MATCH "variable_density_ratio_error=([0-9eE+.-]+)" _variable_density_error_match "${composite_text}")
    set(_variable_density_error "${CMAKE_MATCH_1}")
    string(REGEX MATCH "variable_density_ratio_tolerance=([0-9eE+.-]+)" _variable_density_tolerance_match "${composite_text}")
    set(_variable_density_tolerance "${CMAKE_MATCH_1}")
    if(NOT _variable_density_error_match OR NOT _variable_density_tolerance_match)
        message(FATAL_ERROR "variable-density SBM P2 diagnostic is missing ratio error/tolerance values")
    endif()
    if(_variable_density_error GREATER _variable_density_tolerance)
        message(FATAL_ERROR "variable-density SBM ratio error exceeds tolerance: ${_variable_density_error} > ${_variable_density_tolerance}")
    endif()
endif()
string(REGEX MATCH "post_reflux_validation_count=([1-9][0-9]*)" _validation_match "${composite_text}")
if(NOT _validation_match)
    message(FATAL_ERROR "SBM P2 composite diagnostic did not record a positive post-reflux validation count")
endif()
string(REGEX MATCH "post_reflux_material_rejection_count=0([^0-9]|$)" _material_match "${composite_text}")
if(NOT _material_match)
    message(FATAL_ERROR "SBM P2 composite diagnostic recorded a material post-reflux rejection")
endif()
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
