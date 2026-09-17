if(NOT DEFINED MPIEXEC OR NOT DEFINED MPIEXEC_NUMPROC_FLAG OR
   NOT DEFINED TEST_EXE OR NOT DEFINED INPUT OR NOT DEFINED WORKING_DIRECTORY OR
   NOT DEFINED LOG)
    message(FATAL_ERROR "RunSBMP2AMRSubcycle.cmake missing required argument")
endif()

set(_root "${WORKING_DIRECTORY}/subcycle")
file(REMOVE_RECURSE "${_root}")
file(MAKE_DIRECTORY "${_root}")

set(_mpi_command ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG})
if(DEFINED MPIEXEC_PREFLAGS AND NOT "${MPIEXEC_PREFLAGS}" STREQUAL "")
    separate_arguments(_mpi_preflags UNIX_COMMAND "${MPIEXEC_PREFLAGS}")
    list(APPEND _mpi_command ${_mpi_preflags})
endif()

foreach(_nranks IN ITEMS 1 2)
    set(_run_dir "${_root}/run_${_nranks}r")
    file(MAKE_DIRECTORY "${_run_dir}")
    set(_run_log "${_run_dir}/simulation.log")
    set(_diagnostic "${_run_dir}/run.composite")
    set(_command ${_mpi_command} ${_nranks} ${TEST_EXE} ${INPUT}
        erf.sbm_composite_diagnostic_file=${_diagnostic})
    execute_process(
        COMMAND ${_command}
        WORKING_DIRECTORY "${_run_dir}"
        OUTPUT_FILE "${_run_log}"
        ERROR_FILE "${_run_log}"
        RESULT_VARIABLE _result)
    if(NOT _result EQUAL 0)
        message(FATAL_ERROR "native AMR subcycling ${_nranks}-rank run failed: ${_result}")
    endif()
    if(NOT EXISTS "${_diagnostic}")
        message(FATAL_ERROR "native AMR subcycling diagnostic is missing: ${_diagnostic}")
    endif()
    file(READ "${_diagnostic}" _diagnostic_text)
    foreach(_required IN ITEMS
            "format=erf-sbm-p2-composite-v1"
            "finest_level=1"
            "level_count=2"
            "moment_mode=2"
            "coarse_steps=2"
            "fine_steps=4"
            "fine_substeps_per_coarse=2"
            "interface_oracle_passed=1"
            "passed=1")
        string(FIND "${_diagnostic_text}" "${_required}" _offset)
        if(_offset EQUAL -1)
            message(FATAL_ERROR "${_nranks}-rank subcycling diagnostic is missing ${_required}")
        endif()
    endforeach()
    file(READ "${_run_log}" _run_text)
    foreach(_required IN ITEMS "Coarse STEP 2 ends" "SBM layout identity")
        string(FIND "${_run_text}" "${_required}" _offset)
        if(_offset EQUAL -1)
            message(FATAL_ERROR "${_nranks}-rank subcycling log is missing ${_required}")
        endif()
    endforeach()
endforeach()

file(WRITE "${LOG}" "native factor-2 AMR subcycling passed at 1 and 2 MPI ranks\n")
