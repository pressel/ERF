if(NOT DEFINED MPIEXEC OR NOT DEFINED MPIEXEC_NUMPROC_FLAG OR
   NOT DEFINED TEST_EXE OR NOT DEFINED INPUT OR NOT DEFINED ANELASTIC_INPUT OR
   NOT DEFINED WORKING_DIRECTORY OR NOT DEFINED LOG)
    message(FATAL_ERROR "RunSBMP2DynamicRK.cmake missing required argument")
endif()

set(_mpi_command ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG})
if(DEFINED MPIEXEC_PREFLAGS AND NOT "${MPIEXEC_PREFLAGS}" STREQUAL "")
    separate_arguments(_mpi_preflags UNIX_COMMAND "${MPIEXEC_PREFLAGS}")
    list(APPEND _mpi_command ${_mpi_preflags})
endif()

function(run_dynamic_case mode nranks output_prefix)
    set(_run_dir "${WORKING_DIRECTORY}/${output_prefix}_${nranks}r")
    file(MAKE_DIRECTORY "${_run_dir}")
    set(_run_log "${_run_dir}/simulation.log")
    if(mode EQUAL 0)
        set(_input "${INPUT}")
    else()
        set(_input "${ANELASTIC_INPUT}")
    endif()
    set(_command ${_mpi_command} ${nranks} ${TEST_EXE} ${_input}
        erf.anelastic=${mode})
    execute_process(
        COMMAND ${_command}
        WORKING_DIRECTORY "${_run_dir}"
        OUTPUT_FILE "${_run_log}"
        ERROR_FILE "${_run_log}"
        RESULT_VARIABLE _result)
    if(NOT _result EQUAL 0)
        message(FATAL_ERROR "SBM dynamic ${output_prefix} ${nranks}-rank run failed: ${_result}")
    endif()
    file(READ "${_run_log}" _text)
    if(mode EQUAL 0)
        set(_temporal_mode compressible_rk3)
        set(_expected_stages 3)
    else()
        set(_temporal_mode anelastic_heun)
        set(_expected_stages 2)
    endif()
    string(REGEX MATCHALL
        "SBM actual stage low-order demand level=[0-9]+ stage=[0-9]+ temporal_mode=${_temporal_mode} acoustic_substepping=disabled tau=[0-9eE+.-]+ actual_rate=[0-9eE+.-]+ actual_advective_rate=[0-9eE+.-]+ actual_diffusive_rate=[0-9eE+.-]+ tau_actual_rate=[0-9eE+.-]+"
        _matches "${_text}")
    list(LENGTH _matches _count)
    if(NOT _count EQUAL _expected_stages)
        message(FATAL_ERROR "SBM dynamic ${output_prefix} ${nranks}-rank run reported ${_count} ${_temporal_mode} stages, expected ${_expected_stages}")
    endif()

    set(_rates)
    foreach(_stage RANGE 0 ${_expected_stages})
        if(_stage GREATER_EQUAL _expected_stages)
            break()
        endif()
        string(REGEX MATCH
            "SBM actual stage low-order demand level=[0-9]+ stage=${_stage} temporal_mode=${_temporal_mode} acoustic_substepping=disabled tau=[0-9eE+.-]+ actual_rate=[0-9eE+.-]+ actual_advective_rate=[0-9eE+.-]+ actual_diffusive_rate=[0-9eE+.-]+ tau_actual_rate=([0-9eE+.-]+)"
            _stage_match "${_text}")
        if(NOT _stage_match)
            message(FATAL_ERROR "SBM dynamic ${output_prefix} ${nranks}-rank stage ${_stage} was not parseable")
        endif()
        set(_tau_rate "${CMAKE_MATCH_1}")
        if(_tau_rate GREATER 1.0000000001)
            message(FATAL_ERROR "SBM dynamic ${output_prefix} ${nranks}-rank stage ${_stage} exceeds the exact-stage bound: ${_tau_rate}")
        endif()
        list(APPEND _rates "${_tau_rate}")
    endforeach()

    if(mode EQUAL 0)
        list(GET _rates 0 _rate0)
        list(GET _rates 1 _rate1)
        list(GET _rates 2 _rate2)
        if("${_rate0}" STREQUAL "${_rate1}" AND "${_rate1}" STREQUAL "${_rate2}")
            message(FATAL_ERROR "SBM dynamic compressible carrier demand did not change across RK stages")
        endif()
    endif()

    set(${output_prefix}_${nranks}_count "${_count}" PARENT_SCOPE)
    set(${output_prefix}_${nranks}_rates "${_rates}" PARENT_SCOPE)
endfunction()

run_dynamic_case(0 1 compressible)
run_dynamic_case(0 2 compressible)
run_dynamic_case(1 1 anelastic)
run_dynamic_case(1 2 anelastic)

if(NOT "${compressible_1_rates}" STREQUAL "${compressible_2_rates}")
    message(FATAL_ERROR "SBM dynamic compressible global stage demands differ between 1 and 2 ranks")
endif()
if(NOT "${anelastic_1_rates}" STREQUAL "${anelastic_2_rates}")
    message(FATAL_ERROR "SBM dynamic anelastic global stage demands differ between 1 and 2 ranks")
endif()

file(WRITE "${LOG}"
    "dynamic_compressible_1rank_count=${compressible_1_count}\n"
    "dynamic_compressible_1rank_tau_rates=${compressible_1_rates}\n"
    "dynamic_compressible_2rank_count=${compressible_2_count}\n"
    "dynamic_compressible_2rank_tau_rates=${compressible_2_rates}\n"
    "dynamic_anelastic_1rank_count=${anelastic_1_count}\n"
    "dynamic_anelastic_1rank_tau_rates=${anelastic_1_rates}\n"
    "dynamic_anelastic_2rank_count=${anelastic_2_count}\n"
    "dynamic_anelastic_2rank_tau_rates=${anelastic_2_rates}\n"
    "dynamic_no_acoustic_real_carrier=verified\n")
