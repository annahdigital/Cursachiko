cmake_minimum_required (VERSION 3.14)

find_package(OpenMP)

add_executable (CursachikoCode "CursachikoCode.cpp" "CursachikoCode.h")
if(OpenMP_CXX_FOUND)
	target_compile_options(CursachikoCode PRIVATE -Wall ${OpenMP_CXX_FLAGS})
	target_link_libraries(CursachikoCode PRIVATE ${OpenMP_CXX_FLAGS})
endif()
