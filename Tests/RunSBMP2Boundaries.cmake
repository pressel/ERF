if(NOT DEFINED MPIEXEC OR NOT DEFINED MPIEXEC_NUMPROC_FLAG OR
   NOT DEFINED TEST_EXE OR NOT DEFINED WALL_INPUT OR NOT DEFINED OUTFLOW_INPUT OR
   NOT DEFINED WORKING_DIRECTORY OR NOT DEFINED LOG)
    message(FATAL_ERROR "RunSBMP2Boundaries.cmake missing required argument")
endif()

set(_mpi_command ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG})
if(DEFINED MPIEXEC_PREFLAGS AND NOT "${MPIEXEC_PREFLAGS}" STREQUAL "")
    separate_arguments(_mpi_preflags UNIX_COMMAND "${MPIEXEC_PREFLAGS}")
    list(APPEND _mpi_command ${_mpi_preflags})
endif()

function(run_boundary_case case_name input expected_low expected_high require_outflow)
    set(_root "${WORKING_DIRECTORY}/${case_name}")
    file(REMOVE_RECURSE "${_root}")
    file(MAKE_DIRECTORY "${_root}")
    foreach(_nranks IN ITEMS 1 2)
        set(_run_dir "${_root}/run_${_nranks}r")
        file(MAKE_DIRECTORY "${_run_dir}")
        set(_run_log "${_run_dir}/simulation.log")
        set(_diagnostic "${_run_dir}/run.composite")
        set(_command ${_mpi_command} ${_nranks} ${TEST_EXE} ${input}
            erf.sbm_composite_diagnostic_file=${_diagnostic})
        execute_process(
            COMMAND ${_command}
            WORKING_DIRECTORY "${_run_dir}"
            OUTPUT_FILE "${_run_log}"
            ERROR_FILE "${_run_log}"
            RESULT_VARIABLE _result)
        if(NOT _result EQUAL 0)
            message(FATAL_ERROR "SBM ${case_name} ${_nranks}-rank run failed: ${_result}")
        endif()
        if(NOT EXISTS "${_diagnostic}")
            message(FATAL_ERROR "SBM ${case_name} diagnostic is missing: ${_diagnostic}")
        endif()
        file(READ "${_diagnostic}" _diagnostic_text)
        foreach(_required IN ITEMS
                "format=erf-sbm-p2-composite-v1"
                "finest_level=0"
                "level_count=1"
                "moment_mode=2"
                "boundary_policy_xlo=${expected_low}"
                "boundary_policy_xhi=${expected_high}"
                "boundary_policy_ylo=Periodic"
                "boundary_policy_yhi=Periodic"
                "boundary_policy_zlo=Periodic"
                "boundary_policy_zhi=Periodic"
                "passed=1")
            string(FIND "${_diagnostic_text}" "${_required}" _offset)
            if(_offset EQUAL -1)
                message(FATAL_ERROR "SBM ${case_name} diagnostic is missing ${_required}")
            endif()
        endforeach()
        foreach(_comp IN ITEMS 0 1 2 3 4 5 6 7)
            string(REGEX MATCH "boundary_inventory_outward_comp_${_comp}=([-0-9eE.+]+)" _outward_match "${_diagnostic_text}")
            set(_outward_value "${CMAKE_MATCH_1}")
            string(REGEX MATCH "boundary_inventory_closure_error_comp_${_comp}=([-0-9eE.+]+)" _closure_match "${_diagnostic_text}")
            set(_closure_value "${CMAKE_MATCH_1}")
            string(REGEX MATCH "composite_tolerance_comp_${_comp}=([-0-9eE.+]+)" _tolerance_match "${_diagnostic_text}")
            set(_tolerance_value "${CMAKE_MATCH_1}")
            if(NOT _outward_match OR NOT _closure_match OR NOT _tolerance_match)
                message(FATAL_ERROR "SBM ${case_name} diagnostic is missing component ${_comp} boundary inventory fields")
            endif()
            if(_closure_value GREATER _tolerance_value)
                message(FATAL_ERROR "SBM ${case_name} component ${_comp} boundary closure exceeds tolerance: ${_closure_value} > ${_tolerance_value}")
            endif()
            if(NOT require_outflow AND NOT _outward_value EQUAL 0)
                message(FATAL_ERROR "SBM wall component ${_comp} has nonzero outward inventory: ${_outward_value}")
            endif()
            if(require_outflow AND _comp EQUAL 0 AND NOT _outward_value GREATER 0)
                message(FATAL_ERROR "SBM outflow component 0 did not carry a positive outward inventory: ${_outward_value}")
            endif()
        endforeach()
    endforeach()
endfunction()

run_boundary_case(wall "${WALL_INPUT}" ImpermeableWall ImpermeableWall FALSE)
run_boundary_case(outflow "${OUTFLOW_INPUT}" ImpermeableWall AdvectiveOutflow TRUE)

file(WRITE "${LOG}" "SBM single-level wall and outward-only outflow qualification passed at 1 and 2 MPI ranks\n")
