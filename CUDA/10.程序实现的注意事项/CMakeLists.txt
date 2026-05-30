set(COMMON_FILES common.cpp)

add_executable(generate_data generate_data.c  ${COMMON_FILES})
add_executable(generate_userkey generate_userkey.c  ${COMMON_FILES})
add_executable(crypt crypt.c  ${COMMON_FILES})
add_executable(crypt.config crypt.config.cu ${COMMON_FILES})
add_executable(crypt.constant crypt.constant.cu  ${COMMON_FILES})
add_executable(crypt.flexible crypt.flexible.cu ${COMMON_FILES})
add_executable(crypt.flexible.book crypt.flexible.book.cu ${COMMON_FILES})
add_executable(crypt.flexible.thread crypt.flexible.thread.cu ${COMMON_FILES})
add_executable(crypt.legacy crypt.legacy.cu ${COMMON_FILES})
add_executable(crypt.openmp crypt.openmp.cu ${COMMON_FILES})
target_link_libraries(crypt.openmp PRIVATE OpenMP::OpenMP_CXX)
add_executable(crypt.overlap crypt.overlap.cu ${COMMON_FILES})
add_executable(crypt.overlap.book crypt.overlap.book.cu ${COMMON_FILES})
add_executable(crypt.parallelized crypt.parallelized.cu ${COMMON_FILES})
add_executable(debug-hazards debug-hazards.cu ${COMMON_FILES})
add_executable(debug-segfault debug-segfault.cu ${COMMON_FILES})
add_executable(debug-segfault.fixed debug-segfault.fixed.cu ${COMMON_FILES})
add_executable(sumMatrixGPU sumMatrixGPU.cu ${COMMON_FILES})
add_executable(sumMatrixGPU_nvToolsExt sumMatrixGPU_nvToolsExt.cu ${COMMON_FILES})

find_library(NVTOOLSEXT_LIBRARY nvToolsExt PATHS ${CUDAToolkit_LIBRARY_DIR} /usr/local/cuda/lib64)
    
#     if(NVTOOLSEXT_LIBRARY)
add_library(CUDAToolkit::nvToolsExt UNKNOWN IMPORTED)
set_target_properties(CUDAToolkit::nvToolsExt PROPERTIES
    IMPORTED_LOCATION "${NVTOOLSEXT_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${CUDAToolkit_INCLUDE_DIRS}"
)
#     endif()
# endif()
target_link_libraries(sumMatrixGPU_nvToolsExt PRIVATE CUDAToolkit::nvToolsExt)