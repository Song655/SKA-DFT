cmake_minimum_required(VERSION 3.10.2)

#! Base direct fourier transform project
project(dft CXX CUDA)

#! Flags for the compiler
if ("${CMAKE_C_COMPILER_ID}" MATCHES ".*Clang.*" OR "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
	set(CMAKE_C_FLAGS "-Wall -Wextra")
	set(CMAKE_C_FLAGS_DEBUG "-g -O0")
	set(CMAKE_C_FLAGS_RELEASE "-O3")
	set(CMAKE_CXX_FLAGS "-Wall -Wextra ")
	set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
	set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()

#! Locate CUDA
find_package(CUDA REQUIRED)

find_package(PkgConfig REQUIRED)
if (PKG_CONFIG_FOUND)
	pkg_check_modules(STARPU "starpu-1.3")
	if(starpu_FOUND)
		#set(CMAKE_CXX_FLAGS ${starpu_CFLAGS})	
	endif()
endif()

set(
	CUDA_NVCC_FLAGS
	${CUDA_NVCC_FLAGS};
	-std=c++11 -arch=sm_60 -Xptxas -v -O3 -restrict -D_FORCE_INLINES -D_DEBUG -lineinfo
)

#! build and link dft
cuda_add_executable(dft main.cpp dft_cuda.cu dft_cpu.cpp dft_starpu.cu util.cpp)
target_link_libraries(dft m)
target_link_libraries(dft ${STARPU_LDFLAGS})
