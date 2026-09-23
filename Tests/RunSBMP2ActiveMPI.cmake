if(NOT DEFINED MPIEXEC OR NOT DEFINED MPIEXEC_NUMPROC_FLAG OR
   NOT DEFINED TEST_EXE OR NOT DEFINED INPUT OR NOT DEFINED WORKING_DIRECTORY OR
   NOT DEFINED LOG OR NOT DEFINED COMPARE)
    message(FATAL_ERROR "RunSBMP2ActiveMPI.cmake missing required argument")
endif()

set(_root "${WORKING_DIRECTORY}/active_mpi")
file(REMOVE_RECURSE "${_root}")
file(MAKE_DIRECTORY "${_root}")

set(_mpi_command ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG})
if(DEFINED MPIEXEC_PREFLAGS AND NOT "${MPIEXEC_PREFLAGS}" STREQUAL "")
    separate_arguments(_mpi_preflags UNIX_COMMAND "${MPIEXEC_PREFLAGS}")
    list(APPEND _mpi_command ${_mpi_preflags})
endif()

set(_summary "")
foreach(_diffusion IN ITEMS 0.0 1.0e-4)
    string(REPLACE "." "p" _diff_tag "${_diffusion}")
    foreach(_case IN ITEMS A B C)
        if(_case STREQUAL "A")
            set(_nranks 1)
            set(_grid_size 16)
        elseif(_case STREQUAL "B")
            set(_nranks 1)
            set(_grid_size 8)
        else()
            set(_nranks 2)
            set(_grid_size 8)
        endif()
        set(_run_log "${_root}/diff_${_diff_tag}_${_case}.log")
        set(_diagnostic "${_root}/diff_${_diff_tag}_${_case}.composite")
        set(_command ${_mpi_command} ${_nranks} ${TEST_EXE} ${INPUT}
            erf.sbm_composite_diagnostic_file=${_diagnostic}
            erf.sbm_diffusion_coeff=${_diffusion}
            amr.max_grid_size=${_grid_size})
        execute_process(
            COMMAND ${_command}
            WORKING_DIRECTORY "${WORKING_DIRECTORY}"
            OUTPUT_FILE "${_run_log}"
            ERROR_FILE "${_run_log}"
            RESULT_VARIABLE _result)
        if(NOT _result EQUAL 0)
            message(FATAL_ERROR
                "active-limiter diffusion=${_diffusion} case=${_case} production run failed: ${_result}")
        endif()
        file(READ "${_diagnostic}" _diagnostic_text)
        foreach(_required IN ITEMS "format=erf-sbm-p2-composite-v1" "active_limiter=1" "passed=1")
            string(FIND "${_diagnostic_text}" "${_required}" _offset)
            if(_offset EQUAL -1)
                message(FATAL_ERROR
                    "diffusion=${_diffusion} case=${_case} diagnostic is missing ${_required}")
            endif()
        endforeach()
        string(REGEX MATCH "minimum_accepted_limiter=([0-9eE+.-]+)" _limiter_match "${_diagnostic_text}")
        if(NOT _limiter_match OR NOT CMAKE_MATCH_1 LESS 1.0)
            message(FATAL_ERROR
                "diffusion=${_diffusion} case=${_case} did not report a strict active limiter")
        endif()
        string(APPEND _summary
            "diffusion=${_diffusion} case=${_case} ranks=${_nranks} grid_size=${_grid_size} active_limiter=verified\n")
    endforeach()

    execute_process(
        COMMAND python3 "${COMPARE}"
            "${_root}/diff_${_diff_tag}_A.composite"
            "${_root}/diff_${_diff_tag}_B.composite"
        WORKING_DIRECTORY "${WORKING_DIRECTORY}"
        OUTPUT_FILE "${LOG}"
        ERROR_FILE "${LOG}"
        RESULT_VARIABLE _compare_ab)
    if(NOT _compare_ab EQUAL 0)
        message(FATAL_ERROR
            "active-limiter one-FAB/split-FAB diagnostics differ for diffusion=${_diffusion}: ${_compare_ab}")
    endif()
    execute_process(
        COMMAND python3 "${COMPARE}"
            "${_root}/diff_${_diff_tag}_B.composite"
            "${_root}/diff_${_diff_tag}_C.composite"
        WORKING_DIRECTORY "${WORKING_DIRECTORY}"
        OUTPUT_FILE "${LOG}"
        ERROR_FILE "${LOG}"
        RESULT_VARIABLE _compare_bc)
    if(NOT _compare_bc EQUAL 0)
        message(FATAL_ERROR
            "active-limiter split-FAB rank-ownership diagnostics differ for diffusion=${_diffusion}: ${_compare_bc}")
    endif()
endforeach()

file(WRITE "${LOG}" "format=erf-sbm-p2-active-limiter-layout-v2\n${_summary}passed=1\n")
