

# safe for CMake 3.12 and later
file(GLOB HEADER_LIST CONFIGURE_DEPENDS 
    "${pmpp_SOURCE_DIR}/include/pmpp/*.h" 
    "${pmpp_SOURCE_DIR}/include/pmpp/*.hh")

add_executable(merge merge.cu)

target_include_directories(merge PUBLIC "${pmpp_SOURCE_DIR}/include")

target_compile_options(merge PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")