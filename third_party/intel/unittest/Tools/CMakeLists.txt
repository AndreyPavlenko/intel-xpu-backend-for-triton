if (NOT DEFINED ENV{ONEAPI_ROOT})
    return()
endif ()

add_triton_ut(
        NAME KernelLauncherTest
        SRCS KernelLauncherTest.cpp
)

set(CMAKE_CXX_COMPILER icpx)
target_compile_options(KernelLauncherTest PRIVATE -fsycl -g -O0 -frtti)
target_link_options(KernelLauncherTest PRIVATE -fsycl -lze_loader)
target_include_directories(KernelLauncherTest PRIVATE "$ENV{ONEAPI_ROOT}/compiler/latest/include")
