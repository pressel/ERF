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

    set(_actual_rates)
    set(_tau_rates)
    foreach(_stage RANGE 0 ${_expected_stages})
        if(_stage GREATER_EQUAL _expected_stages)
            break()
        endif()
        string(REGEX MATCH
            "SBM actual stage low-order demand level=[0-9]+ stage=${_stage} temporal_mode=${_temporal_mode} acoustic_substepping=disabled tau=[0-9eE+.-]+ actual_rate=([0-9eE+.-]+) actual_advective_rate=[0-9eE+.-]+ actual_diffusive_rate=[0-9eE+.-]+ tau_actual_rate=([0-9eE+.-]+)"
            _stage_match "${_text}")
        if(NOT _stage_match)
            message(FATAL_ERROR "SBM dynamic ${output_prefix} ${nranks}-rank stage ${_stage} was not parseable")
        endif()
        set(_actual_rate "${CMAKE_MATCH_1}")
        set(_tau_rate "${CMAKE_MATCH_2}")
        if(_tau_rate GREATER 1.0000000001)
            message(FATAL_ERROR "SBM dynamic ${output_prefix} ${nranks}-rank stage ${_stage} exceeds the exact-stage bound: ${_tau_rate}")
        endif()
        list(APPEND _actual_rates "${_actual_rate}")
        list(APPEND _tau_rates "${_tau_rate}")
    endforeach()

    if(mode EQUAL 0)
        list(GET _actual_rates 0 _rate0)
        list(GET _actual_rates 1 _rate1)
        list(GET _actual_rates 2 _rate2)
        if("${_rate0}" STREQUAL "${_rate1}" AND "${_rate1}" STREQUAL "${_rate2}")
            message(FATAL_ERROR "SBM dynamic compressible carrier demand did not change across RK stages")
        endif()

        string(REGEX MATCHALL
            "SBM F01 variable-density views level=[0-9]+ stage=[0-9]+ predictor_target_max=([0-9eE+.-]+) predictor_eval_max=([0-9eE+.-]+) target_eval_max=([0-9eE+.-]+) predictor_density_min=[0-9eE+.-]+ predictor_density_max=[0-9eE+.-]+ target_density_min=[0-9eE+.-]+ target_density_max=[0-9eE+.-]+ eval_density_min=[0-9eE+.-]+ eval_density_max=[0-9eE+.-]+ ratio_residual=([0-9eE+.-]+) actual_carrier_max=([0-9eE+.-]+) actual_carrier_demand=([0-9eE+.-]+)"
            _view_matches "${_text}")
        list(LENGTH _view_matches _view_count)
        if(NOT _view_count EQUAL _expected_stages)
            message(FATAL_ERROR
                "SBM dynamic compressible run reported ${_view_count} F01 density-view diagnostics, expected ${_expected_stages}")
        endif()
        set(_material_density_view FALSE)
        set(_nonzero_carrier FALSE)
        foreach(_view IN LISTS _view_matches)
            string(REGEX MATCH
                "predictor_target_max=([0-9eE+.-]+).*actual_carrier_max=([0-9eE+.-]+)"
                _view_match "${_view}")
            if(_view_match)
                if(CMAKE_MATCH_1 GREATER 1.0e-10)
                    set(_material_density_view TRUE)
                endif()
                if(CMAKE_MATCH_2 GREATER 1.0e-10)
                    set(_nonzero_carrier TRUE)
                endif()
            endif()
        endforeach()
        if(NOT _material_density_view)
            message(FATAL_ERROR
                "SBM dynamic compressible run never produced materially distinct predictor/target density views")
        endif()
        if(NOT _nonzero_carrier)
            message(FATAL_ERROR
                "SBM dynamic compressible run never handed a nonzero ERF carrier to SBM")
        endif()
    endif()

    set(${output_prefix}_${nranks}_count "${_count}" PARENT_SCOPE)
    set(${output_prefix}_${nranks}_actual_rates "${_actual_rates}" PARENT_SCOPE)
    set(${output_prefix}_${nranks}_tau_rates "${_tau_rates}" PARENT_SCOPE)
endfunction()

run_dynamic_case(0 1 compressible)
run_dynamic_case(0 2 compressible)
run_dynamic_case(1 1 anelastic)
run_dynamic_case(1 2 anelastic)

# Test-only semantic mutant: the callback uses the fast state-evaluation
# density for the accepted spectrum.  It must fail on the variable-density
# ratio contract, rather than pass or fail through an unrelated CFL path.
set(_wrong_dir "${WORKING_DIRECTORY}/wrong_density_1r")
file(MAKE_DIRECTORY "${_wrong_dir}")
set(_wrong_log "${_wrong_dir}/simulation.log")
set(_wrong_command ${_mpi_command} 1 ${TEST_EXE} ${INPUT}
    erf.anelastic=0 erf.sbm_test_use_state_eval_density=true)
execute_process(
    COMMAND ${_wrong_command}
    WORKING_DIRECTORY "${_wrong_dir}"
    OUTPUT_FILE "${_wrong_log}"
    ERROR_FILE "${_wrong_log}"
    RESULT_VARIABLE _wrong_result)
if(_wrong_result EQUAL 0)
    message(FATAL_ERROR "SBM F01 state-evaluation-density mutant unexpectedly passed")
endif()
file(READ "${_wrong_log}" _wrong_text)
if(NOT _wrong_text MATCHES "SBM variable-density stage ratio contract failed")
    message(FATAL_ERROR
        "SBM F01 mutant failed without the required variable-density ratio diagnostic")
endif()

if(NOT "${compressible_1_actual_rates}" STREQUAL "${compressible_2_actual_rates}")
    message(FATAL_ERROR "SBM dynamic compressible actual-rate vectors differ between 1 and 2 ranks")
endif()
if(NOT "${compressible_1_tau_rates}" STREQUAL "${compressible_2_tau_rates}")
    message(FATAL_ERROR "SBM dynamic compressible tau-rate vectors differ between 1 and 2 ranks")
endif()
if(NOT "${anelastic_1_actual_rates}" STREQUAL "${anelastic_2_actual_rates}")
    message(FATAL_ERROR "SBM dynamic anelastic actual-rate vectors differ between 1 and 2 ranks")
endif()
if(NOT "${anelastic_1_tau_rates}" STREQUAL "${anelastic_2_tau_rates}")
    message(FATAL_ERROR "SBM dynamic anelastic tau-rate vectors differ between 1 and 2 ranks")
endif()

file(WRITE "${LOG}"
    "dynamic_compressible_1rank_count=${compressible_1_count}\n"
    "dynamic_compressible_1rank_actual_rates=${compressible_1_actual_rates}\n"
    "dynamic_compressible_1rank_tau_rates=${compressible_1_tau_rates}\n"
    "dynamic_compressible_2rank_count=${compressible_2_count}\n"
    "dynamic_compressible_2rank_actual_rates=${compressible_2_actual_rates}\n"
    "dynamic_compressible_2rank_tau_rates=${compressible_2_tau_rates}\n"
    "dynamic_anelastic_1rank_count=${anelastic_1_count}\n"
    "dynamic_anelastic_1rank_actual_rates=${anelastic_1_actual_rates}\n"
    "dynamic_anelastic_1rank_tau_rates=${anelastic_1_tau_rates}\n"
    "dynamic_anelastic_2rank_count=${anelastic_2_count}\n"
    "dynamic_anelastic_2rank_actual_rates=${anelastic_2_actual_rates}\n"
    "dynamic_anelastic_2rank_tau_rates=${anelastic_2_tau_rates}\n"
    "dynamic_no_acoustic_real_carrier=verified\n"
    "dynamic_f01_density_view=verified\n"
    "dynamic_f01_negative_ratio_mutant=verified\n")
