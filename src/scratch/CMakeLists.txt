

file(GLOB HEADER_LIST CONFIGURE_DEPENDS 
    "${pmpp_SOURCE_DIR}/include/pmpp/*.h" 
    "${pmpp_SOURCE_DIR}/include/pmpp/*.hh")

add_executable(scratch scratch.cu)

target_include_directories(scratch PUBLIC "${pmpp_SOURCE_DIR}/include")

target_compile_options(scratch PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")