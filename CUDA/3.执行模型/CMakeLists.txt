
set(COMMON_FILES common.cpp)
# 构建可执行文件
add_executable(nestedHelloWorld nestedHelloWorld.cu ${COMMON_FILES})
add_cudadevrt(nestedHelloWorld)


add_executable(nestedReduce nestedReduce.cu ${COMMON_FILES})
add_cudadevrt(nestedReduce)


add_executable(nestedReduce2 nestedReduce2.cu ${COMMON_FILES})
add_cudadevrt(nestedReduce2)


add_executable(reduceInter reduceInter.cu ${COMMON_FILES})
add_executable(simpleDivergence simpleDivergence.cu ${COMMON_FILES})
add_executable(sumMatrix sumMatrix.cu ${COMMON_FILES})
