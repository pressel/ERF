set(BUILD_GMOCK OFF CACHE BOOL
    "Build Google Mock targets for the vendored GoogleTest dependency")
set(INSTALL_GTEST OFF CACHE BOOL
    "Install targets from the vendored GoogleTest dependency")
if(MSVC)
  set(gtest_force_shared_crt ON CACHE BOOL
      "Use the shared CRT when building the vendored GoogleTest dependency")
endif()

add_subdirectory(${PROJECT_SOURCE_DIR}/Submodules/googletest
                 ${PROJECT_BINARY_DIR}/Submodules/googletest
                 EXCLUDE_FROM_ALL)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|IntelLLVM")
  foreach(_erf_gtest_target gtest gtest_main)
    if(TARGET ${_erf_gtest_target})
      target_compile_options(${_erf_gtest_target}
        PRIVATE
          -Wno-unknown-warning-option
      )
    endif()
  endforeach()
endif()

set(erf_lib_name erf_srclib)
set(erf_exe_name erf_unit_tests)
set(erf_parallel_exe_name erf_parallel_tests)

add_executable(${erf_exe_name})
target_sources(${erf_exe_name}
  PRIVATE
    ${PROJECT_SOURCE_DIR}/Exec/ERF_Prob.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ERF_GTestMain.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ERF_GTestHelloWorld.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ERF_GTestRuntimeStubs.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Utils/EOS/ERF_GTestEOSScalar.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Utils/EOS/ERF_GTestEOSKernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Utils/Interpolation/ERF_GTestInterpolationScalar.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Utils/Interpolation/ERF_GTestInterpolationKernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Utils/Microphysics/ERF_GTestMicrophysicsUtilsScalar.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Utils/Microphysics/ERF_GTestMicrophysicsUtilsKernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Microphysics/SatAdj/ERF_GTestSatAdjThermo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Microphysics/SatAdj/ERF_GTestSatAdjCell.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Microphysics/SatAdj/ERF_GTestSatAdjMultiFab.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Microphysics/Kessler/ERF_GTestKesslerScalar.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Microphysics/Kessler/ERF_GTestKesslerKernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Microphysics/Kessler/ERF_GTestKesslerPhysicalProperties.cpp
)
build_erf_exe(${erf_exe_name} NO_ERF_MAIN)
target_link_libraries(${erf_exe_name} PRIVATE GTest::gtest)

include(GoogleTest)
gtest_discover_tests(${erf_exe_name}
  DISCOVERY_MODE PRE_TEST
  PROPERTIES
    LABELS "unit"
)

if(ERF_ENABLE_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)

  set(erf_parallel_nprocs ${ERF_TEST_NRANKS})

  add_executable(${erf_parallel_exe_name})
  target_sources(${erf_parallel_exe_name}
    PRIVATE
      ${PROJECT_SOURCE_DIR}/Exec/ERF_Prob.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/ERF_GTestParallelMain.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/ERF_GTestParallelHelloWorld.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/ERF_GTestRuntimeStubs.cpp
  )
  build_erf_exe(${erf_parallel_exe_name} NO_ERF_MAIN)
  target_link_libraries(${erf_parallel_exe_name}
    PRIVATE
      GTest::gtest
      MPI::MPI_CXX
  )
  target_compile_definitions(${erf_parallel_exe_name}
    PRIVATE
      ERF_PARALLEL_TEST_NPROCS=${erf_parallel_nprocs}
  )

  add_test(NAME ${erf_parallel_exe_name}_np${erf_parallel_nprocs}
    COMMAND
      ${MPIEXEC_EXECUTABLE}
      ${MPIEXEC_NUMPROC_FLAG} ${erf_parallel_nprocs}
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:${erf_parallel_exe_name}>
      ${MPIEXEC_POSTFLAGS}
    COMMAND_EXPAND_LISTS
  )

  set_tests_properties(${erf_parallel_exe_name}_np${erf_parallel_nprocs}
    PROPERTIES
      LABELS "parallel"
      PROCESSORS ${erf_parallel_nprocs}
  )
endif()
