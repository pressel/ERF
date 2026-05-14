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

include(GoogleTest)

function(erf_register_gtest_suite target)
  set(options)
  set(oneValueArgs WORKING_DIRECTORY)
  set(multiValueArgs LABELS)
  cmake_parse_arguments(ERF_GTEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(NOT TARGET ${target})
    message(FATAL_ERROR "Cannot register unknown gtest suite target: ${target}")
  endif()

  if(NOT ERF_GTEST_WORKING_DIRECTORY)
    set(ERF_GTEST_WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  endif()

  if(NOT ERF_GTEST_LABELS)
    set(ERF_GTEST_LABELS unit)
  endif()

  list(JOIN ERF_GTEST_LABELS ";" _erf_gtest_labels)
  string(REPLACE ";" "\\;" _erf_gtest_labels "${_erf_gtest_labels}")

  gtest_discover_tests(${target}
    DISCOVERY_MODE POST_BUILD
    WORKING_DIRECTORY ${ERF_GTEST_WORKING_DIRECTORY}
    PROPERTIES
      LABELS "${_erf_gtest_labels}"
      SKIP_RETURN_CODE 77
  )
endfunction()

set(erf_lib_name erf_srclib)
set(erf_exe_name erf_unit_tests)

add_executable(${erf_exe_name})
target_sources(${erf_exe_name}
  PRIVATE
    ${PROJECT_SOURCE_DIR}/Exec/ERF_Prob.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ERF_GTestMain.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ERF_GTestHelloWorld.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ERF_GTestRuntimeStubs.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Microphysics/SatAdj/ERF_GTestSatAdjThermo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Microphysics/SatAdj/ERF_GTestSatAdjCell.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Microphysics/SatAdj/ERF_GTestSatAdjMultiFab.cpp
)
build_erf_exe(${erf_exe_name})
target_link_libraries(${erf_exe_name} PRIVATE GTest::gtest)

erf_register_gtest_suite(${erf_exe_name}
  LABELS unit
)

if(ERF_ENABLE_SHOC)
  add_subdirectory(Shoc)
endif()
