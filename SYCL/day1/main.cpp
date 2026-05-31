#include <iostream>
#include <sycl/sycl.hpp>

int main()
{
    constexpr int N = 16;
    sycl::queue q { sycl::gpu_selector_v };
    float* x = sycl::malloc_shared<float>(N, q);
    float* y = sycl::malloc_shared<float>(N, q);
    float* z = sycl::malloc_host<float>(N, q);

    for (int i = 0; i < N; i++) {
        x[i] = i;
        y[i] = 2 * i;
    }

    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
         int i = idx[0];
         z[i] = x[i] + y[i];
     }).wait();

    std::cout << z[10] << std::endl;

    sycl::free(x, q);
    sycl::free(y, q);
    sycl::free(z, q);
}