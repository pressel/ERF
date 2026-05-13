if(NOT TARGET GTest::gtest)
  message(FATAL_ERROR
    "erf_shoc_unit_tests expects the vendored googletest target from "
    "Tests/Unit/CMakeLists.txt. Configure unit tests through the top-level "
    "ERF test build so Submodules/googletest is added first.")
endif()

add_executable(erf_shoc_unit_tests
  ERF_ShocUnitMain.cpp
  ERF_ShocStructureTests.cpp
  ERF_ShocTkeTests.cpp
  ERF_ShocMomentsTests.cpp
  ERF_ShocPDFTests.cpp
  ERF_ShocImplicitTests.cpp
  ERF_ShocPhysicalPropertyTests.cpp
  ${CMAKE_SOURCE_DIR}/Source/PBL/Shoc/ERF_ShocStructure.cpp
  ${CMAKE_SOURCE_DIR}/Source/PBL/Shoc/ERF_ShocTKE.cpp
  ${CMAKE_SOURCE_DIR}/Source/PBL/Shoc/ERF_ShocMoments.cpp
  ${CMAKE_SOURCE_DIR}/Source/PBL/Shoc/ERF_ShocPDF.cpp
  ${CMAKE_SOURCE_DIR}/Source/PBL/Shoc/ERF_ShocEnergyFixer.cpp
  ${CMAKE_SOURCE_DIR}/Source/PBL/Shoc/ERF_ShocImplicit.cpp
)

target_include_directories(erf_shoc_unit_tests PRIVATE
  ${CMAKE_SOURCE_DIR}/Source
  ${CMAKE_SOURCE_DIR}/Source/DataStructs
  ${CMAKE_SOURCE_DIR}/Source/PBL
  ${CMAKE_SOURCE_DIR}/Source/PBL/Shoc
  ${CMAKE_SOURCE_DIR}/Source/Utils
)

target_compile_definitions(erf_shoc_unit_tests PRIVATE
  ERF_SHOC_UNIT_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures"
)

target_link_libraries(erf_shoc_unit_tests PRIVATE
  GTest::gtest
  amrex
)

erf_register_gtest_suite(erf_shoc_unit_tests
  LABELS unit shoc
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
