#include "device_launch_parameters.h"

#define USE_CUDA
#include "cuda_KDtree.hpp"
#include <algorithm>
#include <iostream>
#include <stdio.h>
using namespace std;

const int MAX = 100;
int query_count = 2;
float A[5007][3];
int B[107], result[5007][MAX];
int main()
{
    cudaError_t cudaStatus;

    // 初获取设备数量
    int num = 0;
    cudaStatus = cudaGetDeviceCount(&num);
    std::cout << "Number of GPU: " << num << std::endl;

    // 获取GPU设备属性
    cudaDeviceProp prop;
    if (num > 0) {
        cudaGetDeviceProperties(&prop, 0);
        // 打印设备名称
        std::cout << "Device: " << prop.name << std::endl;
    }

    for (int i = 0; i < 100; i++) {
        A[i][0] = A[i][1] = A[i][2] = i;
        B[i] = i;
    }
    double dist = 2;
    random_shuffle(A, A + 100);
    SearchRadius((float*)A, 100, B, query_count, dist, MAX, (int*)result);
    for (int i = 0; i < query_count; i++) {
            printf("%d(%f %f %f)\n", B[i], A[B[i]][0], A[B[i]][1], A[B[i]][2]);
        for (int k = 0; k < 3; k++)
            printf("%d(%f %f %f) ", result[i][k], A[result[i][k]][0], A[result[i][k]][1], A[result[i][k]][2]);
        puts("");
    }
    //for (int i = 0; i < 100; i++) {
    //    printf("%d %d %d\n", A[i][0], A[i][1], A[i][2]);
    //}
}
