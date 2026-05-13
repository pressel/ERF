if(NOT DEFINED BUILD_GMOCK)
  set(BUILD_GMOCK OFF CACHE BOOL "")
endif()
if(NOT DEFINED INSTALL_GTEST)
  set(INSTALL_GTEST OFF CACHE BOOL "")
endif()
if(MSVC)
  if(NOT DEFINED gtest_force_shared_crt)
    set(gtest_force_shared_crt ON CACHE BOOL "")
  endif()
endif()

add_subdirectory(${PROJECT_SOURCE_DIR}/Submodules/googletest
                 ${PROJECT_BINARY_DIR}/Submodules/googletest)

set(erf_lib_name erf_srclib)
set(erf_exe_name erf_unit_tests)

add_executable(${erf_exe_name})
target_sources(${erf_exe_name}
  PRIVATE
    ${PROJECT_SOURCE_DIR}/Exec/ERF_Prob.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ERF_GTestHelloWorld.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ERF_GTestRuntimeStubs.cpp
)
build_erf_exe(${erf_exe_name})
target_link_libraries(${erf_exe_name} PRIVATE GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(${erf_exe_name}
  PROPERTIES
    LABELS "unit"
)
