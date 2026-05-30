
set(COMMON_FILES common.cpp)

add_executable(checkSmemSquare checkSmemSquare.cu ${COMMON_FILES})
add_executable(checkSmemRectangle checkSmemRectangle.cu ${COMMON_FILES})
add_executable(reduceInter reduceInter.cu ${COMMON_FILES})
add_executable(transposeRectangle transposeRectangle.cu ${COMMON_FILES})
add_executable(constantStencil constantStencil.cu ${COMMON_FILES})
add_executable(constantReadOnly constantReadOnly.cu ${COMMON_FILES})
add_executable(simpleShfl simpleShfl.cu ${COMMON_FILES})
add_executable(reduceIntegerShfl reduceIntegerShfl.cu ${COMMON_FILES})