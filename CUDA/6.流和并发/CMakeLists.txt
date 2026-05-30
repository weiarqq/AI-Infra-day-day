

set(COMMON_FILES common.cpp)

add_executable(simpleHyperqDepth simpleHyperqDepth.cu ${COMMON_FILES})
add_executable(simpleHyperqBreadth simpleHyperqBreadth.cu ${COMMON_FILES})
add_executable(simpleHyperqOpenmp simpleHyperqOpenmp.cu ${COMMON_FILES})
target_link_libraries(simpleHyperqOpenmp PRIVATE OpenMP::OpenMP_CXX)
add_executable(simpleHyperqDependence simpleHyperqDependence.cu ${COMMON_FILES})
add_executable(simpleMultiAddDepth simpleMultiAddDepth.cu ${COMMON_FILES})
add_executable(simpleMultiAddBreadth simpleMultiAddBreadth.cu ${COMMON_FILES})
add_executable(simpleCallback simpleCallback.cu ${COMMON_FILES})
add_executable(asyncAPI asyncAPI.cu ${COMMON_FILES})