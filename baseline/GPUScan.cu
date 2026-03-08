#include "GPUScan.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/partition.h>
#include <thrust/remove.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <chrono>

using namespace std::chrono;

uint64_t  TOTALTHDCOUNT = 65536;
const uint32_t WARPSIZE = 32;
const uint32_t BLOCKSIZE = 512;



//定义捕获错误的宏
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#ifdef _DEBUG_



#endif

struct is_similar
{
    __host__ __device__
        bool operator()(const thrust::tuple<int, int, bool>& x)
    {
        return thrust::get<2>(x) == true;
    }
};
struct is_unsimilar
{
    __host__ __device__
        bool operator()(const int& x)
    {
        return x == -1;
    }
};
struct comprule_odd
{
    __host__ __device__
        bool operator()(const thrust::tuple<int, int>& x1, const thrust::tuple<int, int>& x2)
    {
        if (thrust::get<0>(x1) != thrust::get<0>(x2)) return thrust::get<0>(x1) <= thrust::get<0>(x2);
        else if (thrust::get<1>(x1) != thrust::get<1>(x2)) return thrust::get<1>(x1) <= thrust::get<1>(x2);
    }
};
struct comprule_even
{
    __host__ __device__
        bool operator()(const thrust::tuple<int, int>& x1, const thrust::tuple<int, int>& x2)
    {
        if (thrust::get<0>(x1) != thrust::get<0>(x2)) return thrust::get<0>(x1) <= thrust::get<0>(x2);
        else if (thrust::get<1>(x1) != thrust::get<1>(x2)) return thrust::get<1>(x1) >= thrust::get<1>(x2);
    }
};
struct par_rule
{
    __host__ __device__
        bool operator()(const thrust::tuple<int, int>& x1)
    {
        return thrust::get<0>(x1) != -1;
    }
};
struct par_rule1
{
    __host__ __device__
        bool operator()(const thrust::tuple<int, int>& x1)
    {
        return thrust::get<1>(x1) != -1;
    }
};

//compute similarity
//__global__ void __ID_BOUNDARIES(int* A_u, int* begin, int* end, long elength) {
//
//    __shared__ int smem[BLOCKSIZE];
//    int threadCount = blockDim.x * gridDim.x;        //thread sum
//    int tid = threadIdx.x + blockIdx.x * blockDim.x; //thread id
//    int bid = tid % BLOCKSIZE;
//    //tid++;
//    while (tid < elength)
//    {
//        smem[bid] = A_u[tid];
//        __syncthreads();
//        if (tid == 0)
//        {
//            begin[smem[bid]] = tid;
//            if (smem[bid] != smem[bid + 1]) end[smem[bid]] = tid + 1;
//        }
//        else if (tid == elength - 1)
//        {
//            if (smem[bid] != smem[bid - 1]) begin[smem[bid]] = tid;
//            end[smem[bid]] = tid + 1;
//        }
//        else
//        {
//            if (bid == 0) {
//                if (smem[bid] != A_u[tid - 1]) begin[smem[bid]] = tid;
//                if (smem[bid] != smem[bid + 1]) end[smem[bid]] = tid + 1;
//            }
//            else if (bid == BLOCKSIZE - 1) {
//                if (smem[bid] != smem[bid - 1]) begin[smem[bid]] = tid;
//                if (smem[bid] != A_u[tid + 1]) end[smem[bid]] = tid + 1;
//            }
//            else {
//                if (smem[bid] != smem[bid - 1]) begin[smem[bid]] = tid;
//                if (smem[bid] != smem[bid + 1]) end[smem[bid]] = tid + 1;
//            }
//        }
//        tid += threadCount;
//    }
//}

__global__ void __ID_BOUNDARIES(int* A_u, int* begin, int* end, long elength) {

    int threadCount = blockDim.x * gridDim.x;        //thread sum
    int tid = threadIdx.x + blockIdx.x * blockDim.x; //thread id
    while (tid < elength)
    {
        int u = A_u[tid];
        if (tid == 0)
        {
            begin[u] = tid;
            if (u != A_u[tid + 1]) end[u] = tid + 1;
        }
        else if (tid == elength - 1)
        {
            if (u != A_u[tid - 1]) begin[u] = tid;
            end[u] = tid + 1;
        }
        else
        {
            if (u != A_u[tid - 1]) begin[u] = tid;
            if (u != A_u[tid + 1]) end[u] = tid + 1;
        }
        tid += threadCount;
    }
}

__device__ int intersect(int setA[], int setB[], int begin1, int end1, int begin2, int end2, int* A_v) {

    __shared__ int total[BLOCKSIZE / WARPSIZE];
    __shared__ int lenA[BLOCKSIZE / WARPSIZE];
    __shared__ int lenB[BLOCKSIZE / WARPSIZE];
    __shared__ int supA[BLOCKSIZE / WARPSIZE];
    __shared__ int supB[BLOCKSIZE / WARPSIZE];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int bid = tid % BLOCKSIZE;
    int WarpIdx_block = bid / WARPSIZE;
    int widA = tid % WARPSIZE;
    int widB = tid % WARPSIZE;
    int w_pos = bid - tid % WARPSIZE;
    int posA = (tid % WARPSIZE) / 8;
    int posB = (tid % WARPSIZE) % 8;
    int num_posA = 8;
    int num_posB = 4;
    int elemA;
    int elemB;
    int count = 0;
    int pointerA = 0;
    int pointerB = 0;

    total[WarpIdx_block] = 0;
    lenA[WarpIdx_block] = end1 - begin1;
    lenB[WarpIdx_block] = end2 - begin2;
    supA[WarpIdx_block] = 0;
    supB[WarpIdx_block] = 0;
    while (pointerA < lenA[WarpIdx_block] && pointerB < lenB[WarpIdx_block])
    {
        if (num_posA == 8)
        {
            if (widA + begin1 <= end1 - 1)//
            {
                setA[bid] = A_v[widA + begin1];
            }
            num_posA = 0;
            widA += WARPSIZE;
        }
        if (num_posB == 4)
        {
            if (widB + begin2 <= end2 - 1)
            {
                setB[bid] = A_v[widB + begin2];
            }
            num_posB = 0;
            widB += WARPSIZE;
        }
        if (posA < lenA[WarpIdx_block] && posB < lenB[WarpIdx_block])
        {
            elemA = setA[w_pos + posA % WARPSIZE];
            elemB = setB[w_pos + posB % WARPSIZE];
            if (elemA == elemB) count++;
            if (posA % 4 == 3 || posA == lenA[WarpIdx_block] - 1)
            {
                atomicExch(&supA[WarpIdx_block], elemA);
            }
            if (posB % 8 == 7 || posB == lenB[WarpIdx_block] - 1)
            {
                atomicExch(&supB[WarpIdx_block], elemB);
            }
        }
        //__syncthreads();
        if (supA[WarpIdx_block] <= supB[WarpIdx_block])
        {
            posA += 4;
            num_posA = num_posA + 1;
            pointerA += 4;
        }
        if (supA[WarpIdx_block] >= supB[WarpIdx_block])
        {
            posB += 8;
            num_posB = num_posB + 1;
            pointerB += 8;
        }
    }
    atomicAdd(&total[WarpIdx_block], count);
    return total[WarpIdx_block];
}


__global__ void __ID_eps_NEIGHBORS(double eps, long elength, int* E_u, int* E_v, int* A_v, int* begin, int* end, bool* flag) {

    __shared__ int b1[BLOCKSIZE];
    __shared__ int b2[BLOCKSIZE];
    __shared__ int e1[BLOCKSIZE];
    __shared__ int e2[BLOCKSIZE];
    __shared__ int setA[BLOCKSIZE];
    __shared__ int setB[BLOCKSIZE];
    __shared__ int count[BLOCKSIZE];
    int threadCount = blockDim.x * gridDim.x;        //thread sum
    int tid = threadIdx.x + blockIdx.x * blockDim.x, tid_t; //thread id
    int bid = tid % BLOCKSIZE;
    int pos = tid % WARPSIZE;                        //thread id in warp
    int all;
    if ((elength / 2) % threadCount == 0) {
        all = elength / 2;
    }
    else {
        all = ((elength / 2) / threadCount + 1) * threadCount;
    }
    tid_t = tid;
    while (tid_t < all) {//
        if (tid_t < elength / 2)
        {
            b1[bid] = begin[E_u[tid_t]];
            b2[bid] = begin[E_v[tid_t]];
            e1[bid] = end[E_u[tid_t]];
            e2[bid] = end[E_v[tid_t]];
        }
        int el_pos = tid_t - pos;//begin position of each warp in edge list
        int b_pos = bid - pos;
        for (int i = 0; i < 32; ++i)
        {
            if (el_pos + i >= elength / 2) break;
            count[b_pos + i] = intersect(setA, setB, b1[b_pos + i], e1[b_pos + i], b2[b_pos + i], e2[b_pos + i], A_v);
        }
        if (tid_t < elength / 2) {
            e1[bid] = e1[bid] - b1[bid] + 1;
            e2[bid] = e2[bid] - b2[bid] + 1;
            if ((count[bid] + 2) * (count[bid] + 2) >= eps * eps * (e1[bid] * e2[bid])) {
                flag[tid_t] = true;
            }
            else {
                flag[tid_t] = false;
            }
        }
        tid_t += threadCount;
    }
}

__global__ void __ID_eps_NEIGHBORS1(double eps, long elength, int* E_u, int* E_v, int* A_v, int* begin, int* end, bool* flag) {
    
    __shared__ int setA[BLOCKSIZE];
    __shared__ int setB[BLOCKSIZE];
    __shared__ int total[BLOCKSIZE / WARPSIZE][WARPSIZE];
    __shared__ int lenA[BLOCKSIZE / WARPSIZE];
    __shared__ int lenB[BLOCKSIZE / WARPSIZE];
    __shared__ int supA[BLOCKSIZE / WARPSIZE];
    __shared__ int supB[BLOCKSIZE / WARPSIZE];
    __shared__ int pointerA[BLOCKSIZE / WARPSIZE];
    __shared__ int pointerB[BLOCKSIZE / WARPSIZE];
    __shared__ int b1[BLOCKSIZE];
    __shared__ int b2[BLOCKSIZE];
    __shared__ int e1[BLOCKSIZE];
    __shared__ int e2[BLOCKSIZE];
    int threadCount = blockDim.x * gridDim.x;        //thread sum
    int tid = threadIdx.x + blockIdx.x * blockDim.x; //thread id
    int bid = tid % BLOCKSIZE;
    int wid = tid % WARPSIZE;                        //thread id in warp
    int WarpIdx_global = tid / WARPSIZE;       //全局Warp索引
    int WarpIdx_block = threadIdx.x / WARPSIZE;    //线程块内的Warp索引
    int WarpNumb = blockDim.x * gridDim.x / WARPSIZE;    //全局的Warp数
    //int Proc_Numb, Proc_it;                           //一个Warp处理的边的数目,边表处理起始位置标识

    elength = elength / 2;
    int all = ((elength + threadCount - 1) / threadCount) * threadCount;
    int tid_t = tid;
    while (tid_t < all) {
        if (tid_t < elength)
        {
            b1[bid] = begin[E_u[tid_t]];
            b2[bid] = begin[E_v[tid_t]];
            e1[bid] = end[E_u[tid_t]];
            e2[bid] = end[E_v[tid_t]];
        }
        int el_pos = tid_t - wid;//begin position of each warp in edge list
        int b_pos = bid - wid;
        for (int i = 0; i < 32; ++i)
        {
            if (el_pos + i >= elength) break;
            total[WarpIdx_block][i] = 0;
            lenA[WarpIdx_block] = e1[b_pos + i] - b1[b_pos + i];
            lenB[WarpIdx_block] = e2[b_pos + i] - b2[b_pos + i];
            supA[WarpIdx_block] = 0;
            supB[WarpIdx_block] = 0;
            int widA = wid;
            int widB = wid;
            int posA = wid / 8;
            int posB = wid % 8;
            int elemA;
            int elemB;
            int count = 0;
            pointerA[WarpIdx_block] = 0;
            pointerB[WarpIdx_block] = 0;
            int num_posA = 8;
            int num_posB = 4;
            while (pointerA[WarpIdx_block] < lenA[WarpIdx_block] && pointerB[WarpIdx_block] < lenB[WarpIdx_block])
            {
                if (num_posA == 8)
                {
                    if (widA + b1[b_pos + i] <= e1[b_pos + i] - 1)//
                    {
                        setA[bid] = A_v[widA + b1[b_pos + i]];
                    }
                    num_posA = 0;
                    widA += WARPSIZE;
                }
                if (num_posB == 4)
                {
                    if (widB + b2[b_pos + i] <= e2[b_pos + i] - 1)
                    {
                        setB[bid] = A_v[widB + b2[b_pos + i]];
                    }
                    num_posB = 0;
                    widB += WARPSIZE;
                }
                if (posA < lenA[WarpIdx_block] && posB < lenB[WarpIdx_block]) {
                    //elemA = A_v[b1[b_pos + i] + posA];
                    //elemB = A_v[b2[b_pos + i] + posB];
                    elemA = setA[b_pos + posA % WARPSIZE];
                    elemB = setB[b_pos + posB % WARPSIZE];
                    if (elemA == elemB) count++;
                    if (posA % 4 == 3 || posA == lenA[WarpIdx_block] - 1)
                    {
                        supA[WarpIdx_block] = elemA;
                    }
                    if (posB % 8 == 7 || posB == lenB[WarpIdx_block] - 1)
                    {
                        supB[WarpIdx_block] = elemB;
                    }
                }
                if (supA[WarpIdx_block] <= supB[WarpIdx_block])
                {
                    posA += 4;
                    num_posA = num_posA + 1;
                    pointerA[WarpIdx_block] += 4;
                }
                if (supA[WarpIdx_block] >= supB[WarpIdx_block])
                {
                    posB += 8;
                    num_posB = num_posB + 1;
                    pointerB[WarpIdx_block] += 8;
                }
            }
            atomicAdd(&total[WarpIdx_block][i], count);
        }
        if (tid_t < elength) {
            e1[bid] = e1[bid] - b1[bid] + 1;
            e2[bid] = e2[bid] - b2[bid] + 1;
            if ((double)(total[WarpIdx_block][wid] + 2) * (total[WarpIdx_block][wid] + 2) 
                >= eps * eps * (e1[bid] * e2[bid])) {
                flag[tid_t] = true;
            }
            else {
                flag[tid_t] = false;
            }
        }

        tid_t += threadCount;
    }

    //if (WarpNumb < elength) {
    //    if (elength % WarpNumb == 0) {
    //        Proc_Numb = elength / WarpNumb;
    //    }
    //    else Proc_Numb = elength / WarpNumb + 1;
    //}
    //else Proc_Numb = 1;
    //Proc_it = WarpIdx_global * Proc_Numb;

    //for (int k = 0; k < Proc_Numb; ++k) {
    //    if (Proc_it > elength - 1) break;
    //    int b1 = begin[E_u[Proc_it]];
    //    int b2 = begin[E_v[Proc_it]];
    //    int e1 = end[E_u[Proc_it]];
    //    int e2 = end[E_v[Proc_it]];
    //    total[WarpIdx_block] = 0;
    //    lenA[WarpIdx_block] = e1 - b1;
    //    lenB[WarpIdx_block] = e2 - b2;
    //    supA[WarpIdx_block] = 0;
    //    supB[WarpIdx_block] = 0;
    //    int posA = wid / 8;
    //    int posB = wid % 8;
    //    int elemA;
    //    int elemB;
    //    int count = 0;
    //    pointerA[WarpIdx_block] = 0;
    //    pointerB[WarpIdx_block] = 0;
    //    while (pointerA[WarpIdx_block] < lenA[WarpIdx_block] && pointerB[WarpIdx_block] < lenB[WarpIdx_block])
    //    {
    //        if (posA < lenA[WarpIdx_block] && posB < lenB[WarpIdx_block])
    //        {
    //            elemA = A_v[b1 + posA];
    //            elemB = A_v[b2 + posB];
    //            if (elemA == elemB) count++;
    //            if (posA % 4 == 3 || posA == lenA[WarpIdx_block] - 1)
    //            {
    //                //atomicExch(&supA[WarpIdx_block], elemA);
    //                supA[WarpIdx_block] = elemA;
    //            }
    //            if (posB % 8 == 7 || posB == lenB[WarpIdx_block] - 1)
    //            {
    //                //atomicExch(&supB[WarpIdx_block], elemB);
    //                supB[WarpIdx_block] = elemB;
    //            }
    //        }
    //        if (supA[WarpIdx_block] <= supB[WarpIdx_block])
    //        {
    //            posA += 4;
    //            pointerA[WarpIdx_block] += 4;
    //        }
    //        if (supA[WarpIdx_block] >= supB[WarpIdx_block])
    //        {
    //            posB += 8;
    //            pointerB[WarpIdx_block] += 8;
    //        }
    //    }
    //    atomicAdd(&total[WarpIdx_block], count);
    //    e1 = e1 - b1 + 1;
    //    e2 = e2 - b2 + 1;
    //    if ((double)(total[WarpIdx_block] + 2) * (total[WarpIdx_block] + 2) >= 
    //        eps * eps * (e1 * e2)) {
    //        flag[Proc_it] = true;
    //    }
    //    else {
    //        flag[Proc_it] = false;
    //    }
    //    Proc_it++;//开始计算边表中下一条边
    //}
}

void finding_epsilon_Neighbors(double eps, long vlength, long elength, int& N_size, thrust::device_vector<int> &E_u,
    thrust::device_vector<int> &E_v, thrust::device_vector<int> &A_u, thrust::device_vector<int> &A_v,
    thrust::device_vector<bool> &flag) {
    auto start = high_resolution_clock::now();

    thrust::device_vector<int> begin(vlength);
    thrust::device_vector<int> end(vlength);

    int ThreadBlockCount = TOTALTHDCOUNT / BLOCKSIZE;
    int* rawA_u = thrust::raw_pointer_cast(A_u.data());
    int* rawA_v = thrust::raw_pointer_cast(A_v.data());
    int* rawbegin = thrust::raw_pointer_cast(begin.data());
    int* rawend = thrust::raw_pointer_cast(end.data());
    int* rawE_u = thrust::raw_pointer_cast(E_u.data());
    int* rawE_v = thrust::raw_pointer_cast(E_v.data());
    bool* rawflag = thrust::raw_pointer_cast(flag.data());
    
    __ID_BOUNDARIES <<<ThreadBlockCount, BLOCKSIZE>>> (rawA_u, rawbegin, rawend, elength);
    __ID_eps_NEIGHBORS1 <<<ThreadBlockCount, BLOCKSIZE>>> (eps, elength, rawE_u, rawE_v, rawA_v, rawbegin, rawend, rawflag);

    thrust::zip_iterator<thrust::tuple<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator, 
        thrust::device_vector<bool>::iterator>> first;
    first = thrust::make_zip_iterator(thrust::make_tuple(E_u.begin(), E_v.begin(), flag.begin()));
    thrust::stable_partition(first, first + elength / 2, is_similar());
    N_size = thrust::count_if(first, first + elength / 2, is_similar());
    //std::cout << "N_size: " << N_size << "\n";
    E_u.resize(2 * N_size);
    E_v.resize(2 * N_size);
    thrust::copy(E_v.begin(), E_v.begin() + N_size, E_u.begin() + N_size);
    thrust::copy(E_u.begin(), E_u.begin() + N_size, E_v.begin() + N_size);

    auto finish = high_resolution_clock::now();
    fprintf(stderr, "\nfinding_epsilon_Neighbors time: %.3lf s\n", duration_cast<milliseconds>(finish - start).count() / 1000.0);
}

//identify cores
__global__ void __IDcores(int miu, long vlength, int* begin, int* end, bool* is_core) {

    int threadCount = blockDim.x * gridDim.x;        //thread sum
    int tid = threadIdx.x + blockIdx.x * blockDim.x; //thread id
    while (tid < vlength)
    {
        is_core[tid] = ((end[tid] - begin[tid]) >= miu);
        tid += threadCount;
    }
}

__global__ void __uncore_remark(int N_size, int* E_u, int* E_v, bool* is_core) {

    int threadCount = blockDim.x * gridDim.x;        //thread sum
    int tid = threadIdx.x + blockIdx.x * blockDim.x; //thread id
    while (tid < 2 * N_size)
    {
        if (is_core[E_u[tid]] == 0 && is_core[E_v[tid]] == 0) {
            E_u[tid] = -1;
            E_v[tid] = -1;
        }
        tid += threadCount;
    }
}

void coreVertices(int miu, int N_size, long vlength, long elength, thrust::device_vector<int>& E_u,
    thrust::device_vector<int>& E_v) {
    auto start = high_resolution_clock::now();

    int ThreadBlockCount = TOTALTHDCOUNT / BLOCKSIZE;
    thrust::device_vector<int> begin(vlength);//未被初始化
    thrust::device_vector<int> end(vlength);//未被初始化
    thrust::device_vector<bool> is_core(vlength);
    thrust::sort_by_key(E_v.begin(), E_v.end(), E_u.begin());
    thrust::stable_sort_by_key(E_u.begin(), E_u.end(), E_v.begin());

    int* rawE_u = thrust::raw_pointer_cast(E_u.data());
    int* rawE_v = thrust::raw_pointer_cast(E_v.data());
    int* rawbegin = thrust::raw_pointer_cast(begin.data());
    int* rawend = thrust::raw_pointer_cast(end.data());
    bool* rawis_core = thrust::raw_pointer_cast(is_core.data());

    __ID_BOUNDARIES <<<ThreadBlockCount, BLOCKSIZE >>> (rawE_u, rawbegin, rawend, 2 * N_size);

    __IDcores <<<ThreadBlockCount, BLOCKSIZE >>> (miu, vlength, rawbegin, rawend, rawis_core);

    __uncore_remark <<<ThreadBlockCount, BLOCKSIZE >>> (N_size, rawE_u, rawE_v, rawis_core);

    
    int core_number = thrust::count(is_core.begin(), is_core.end(), true);
    
    fprintf(stderr, "core number: %d\n", core_number);//debug

    E_u.erase(thrust::remove(E_u.begin(), E_u.end(), -1), E_u.end());
    E_v.erase(thrust::remove(E_v.begin(), E_v.end(), -1), E_v.end());

    auto finish = high_resolution_clock::now();
    fprintf(stderr, "\ncoreVertices time: %.3lf s\n", duration_cast<milliseconds>(finish - start).count() / 1000.0);
}

//link core cluster
__global__ void __LINKING(int e_size, int* E_u, int* E_v, int* parents, bool odd) {

    __shared__ int u[BLOCKSIZE];
    int threadCount = blockDim.x * gridDim.x;        //thread sum
    int tid = threadIdx.x + blockIdx.x * blockDim.x; //thread id
    int bid = tid % BLOCKSIZE;
    bool do_copy = false;
    while (tid < e_size)
    {
        u[bid] = E_u[tid];
        __syncthreads();
        if (tid == 0) do_copy = true;
        else if (bid == 0) {
            if (u[bid] != E_u[tid - 1]) {
                do_copy = true;
            }
        }
        else if (u[bid] != u[bid - 1]) do_copy = true;
        if (do_copy) {
            parents[u[bid]] = odd ? min(u[bid], E_v[tid]) : max(u[bid], E_v[tid]);
        }
        tid += threadCount;
    }
}

__global__ void __REMOVE_SELF_LINKS(int e_size, int* E_u, int* E_v, int* parents) {

    int threadCount = blockDim.x * gridDim.x;        //thread sum
    int tid = threadIdx.x + blockIdx.x * blockDim.x; //thread id
    while (tid < e_size)
    {
        int u = parents[E_u[tid]];
        int v = parents[E_v[tid]];
        if (u == v) {
            E_u[tid] = -1;
        }
        else {
            E_u[tid] = u;
            E_v[tid] = v;
        }
        tid += threadCount;
    }
}

void linking(thrust::device_vector<int>& E_u, thrust::device_vector<int>& E_v, thrust::device_vector<int>& parents) {

    auto start = high_resolution_clock::now();

    int ThreadBlockCount = TOTALTHDCOUNT / BLOCKSIZE;
    thrust::fill(parents.begin(), parents.end(), -1);
    bool odd = false;
    int E_ulength = E_u.size();
    while (E_ulength != 0)
    {
        odd = !odd;
        thrust::zip_iterator<thrust::tuple<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator>> first;
        first = thrust::make_zip_iterator(thrust::make_tuple(E_u.begin(), E_v.begin()));
        //last = thrust::make_zip_iterator(thrust::make_tuple(E_u.end(), E_v.end()));
        if (odd) thrust::sort(first, first + E_ulength, comprule_odd());
        else thrust::sort(first, first + E_ulength, comprule_even());
        int* rawE_u = thrust::raw_pointer_cast(E_u.data());
        int* rawE_v = thrust::raw_pointer_cast(E_v.data());
        int* rawparents = thrust::raw_pointer_cast(parents.data());
        __LINKING <<<ThreadBlockCount, BLOCKSIZE >>> (E_ulength, rawE_u, rawE_v, rawparents, odd);
        __REMOVE_SELF_LINKS <<<ThreadBlockCount, BLOCKSIZE >>> (E_ulength, rawE_u, rawE_v, rawparents);
        E_ulength = thrust::partition(first, first + E_ulength, par_rule()) - first;
    }

    auto finish = high_resolution_clock::now();
    fprintf(stderr, "\nlinking time: %.3lf s\n", duration_cast<milliseconds>(finish - start).count() / 1000.0);
}

//contraction
__device__ bool Pd;

__global__ void __GRAPH_CONTRACTION(long vlength, int* parents) {

    int threadCount = blockDim.x * gridDim.x;        //thread sum
    int tid = threadIdx.x + blockIdx.x * blockDim.x; //thread id
    int label;
    while (tid < vlength)
    {
        label = parents[tid];
        if (label != -1) {
            int plabel = parents[label];
            if (plabel != label) {
                parents[tid] = plabel;
                Pd = true;
                //atomicExch(&Pd, true);
            }
        }
        tid += threadCount;
    }
}

void contraction(long vlength, thrust::device_vector<int>& parents) {
    
    auto start = high_resolution_clock::now();

    int ThreadBlockCount = TOTALTHDCOUNT / BLOCKSIZE;
    int* rawparents = thrust::raw_pointer_cast(parents.data());
    bool p = true;
    bool Pd_ini = false;
    while (p)
    {
        gpuErrchk(cudaMemcpyToSymbol(Pd, &Pd_ini, sizeof(bool)));
        __GRAPH_CONTRACTION << <ThreadBlockCount, BLOCKSIZE >> > (vlength, rawparents);
        gpuErrchk(cudaMemcpyFromSymbol(&p, Pd, sizeof(bool)));
    }
    auto finish = high_resolution_clock::now();
    fprintf(stderr, "\ncontraction time: %.3lf s\n", duration_cast<milliseconds>(finish - start).count() / 1000.0);
}

//classify non-member
__global__ void __ID_OUTLIERS(long elength, int* parents, int* A_u) {

    int threadCount = blockDim.x * gridDim.x;        //thread sum
    int tid = threadIdx.x + blockIdx.x * blockDim.x; //thread id
    while (tid < elength)
    {
        if (parents[A_u[tid]] != -1) {
            A_u[tid] = -1;
        }
        tid += threadCount;
    }
}

__global__ void __ID_ADJ_CLUSTERS(int A_ulength, int* parents, int* A_v) {

    int threadCount = blockDim.x * gridDim.x;        //thread sum
    int tid = threadIdx.x + blockIdx.x * blockDim.x; //thread id
    while (tid < A_ulength)
    {
        A_v[tid] = parents[A_v[tid]];
        tid += threadCount;
    }
}

__global__ void __ID_HUBS(int A_ulength, int* parents, int* A_u, int* A_v) {

    __shared__ int u[BLOCKSIZE];
    __shared__ int v[BLOCKSIZE];
    int threadCount = blockDim.x * gridDim.x;        //thread sum
    int tid = threadIdx.x + blockIdx.x * blockDim.x; //thread id
    int bid = tid % BLOCKSIZE;
    while (tid < A_ulength)
    {
        u[bid] = A_u[tid];
        v[bid] = A_v[tid];
        if (tid != 0) {
            if (bid == 0) {
                if (u[bid] == A_u[tid - 1] && v[bid] != A_u[tid - 1]) {
                    parents[u[bid]] = -2;
                }
            }
            else {
                if (u[bid] == u[bid - 1] && v[bid] != v[bid - 1]) {
                    parents[u[bid]] = -2;
                }
            }
        }
        tid += threadCount;
    }
}

void classNonMembers(long elength, thrust::device_vector<int>& A_u, thrust::device_vector<int>& A_v,
    thrust::device_vector<int>& parents) {

    auto start = high_resolution_clock::now();

    int ThreadBlockCount = TOTALTHDCOUNT / BLOCKSIZE;
    int* rawparents = thrust::raw_pointer_cast(parents.data());
    int* rawA_u = thrust::raw_pointer_cast(A_u.data());
    int* rawA_v = thrust::raw_pointer_cast(A_v.data());
    int A_ulength;
    __ID_OUTLIERS << <ThreadBlockCount, BLOCKSIZE >> > (elength, rawparents, rawA_u);
    thrust::zip_iterator<thrust::tuple<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator>> first;
    thrust::zip_iterator<thrust::tuple<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator>> last;
    first = thrust::make_zip_iterator(thrust::make_tuple(A_u.begin(), A_v.begin()));
    last = thrust::make_zip_iterator(thrust::make_tuple(A_u.end(), A_v.end()));
    A_ulength = thrust::partition(first, last, par_rule()) - first;
    __ID_ADJ_CLUSTERS <<<ThreadBlockCount, BLOCKSIZE >>> (A_ulength, rawparents, rawA_v);
    A_ulength = thrust::partition(first, first + A_ulength, par_rule1()) - first;
    __ID_HUBS <<<ThreadBlockCount, BLOCKSIZE >>> (A_ulength, rawparents, rawA_u, rawA_v);

    auto finish = high_resolution_clock::now();
    fprintf(stderr, "\nclassNonMembers time: %.3lf s\n", duration_cast<milliseconds>(finish - start).count() / 1000.0);
}

int* GPUScan(graph_t* g, int* Edge_u, int* Edge_v, double eps, int miu) {

    auto clock1 = high_resolution_clock::now();
    //std::cout << "create E_u E_v A_u A_v" << "\n";
    thrust::device_vector<int> E_u(g->m / 2);
    thrust::device_vector<int> E_v(g->m / 2);
    thrust::device_vector<int> A_u(g->m);
    thrust::device_vector<int> A_v(g->m);

    //std::cout << "copy E_u E_v" << "\n";
    thrust::copy(Edge_u, Edge_u + g->m / 2, E_u.begin());
    thrust::copy(Edge_v, Edge_v + g->m / 2, E_v.begin());

    //std::cout << "copy A_u A_v" << "\n";
    thrust::copy(E_u.begin(), E_u.end(), A_u.begin());
    thrust::copy(E_v.begin(), E_v.end(), A_v.begin());
    thrust::copy(E_u.begin(), E_u.end(), A_v.begin() + g->m / 2);
    thrust::copy(E_v.begin(), E_v.end(), A_u.begin() + g->m / 2);

    //int* cpuA_u = (int*)malloc(g->m * sizeof(int));
    //int* cpuA_v = (int*)malloc(g->m * sizeof(int));
    //thrust::copy(A_u.begin(), A_u.end(), cpuA_u);
    //thrust::copy(A_v.begin(), A_v.end(), cpuA_v);
    //std::cout << "sort A_u A_v 1" << "\n";
    //thrust::sort_by_key(thrust::host, cpuA_v, cpuA_v + g->m, cpuA_u);
    //std::cout << "sort A_u A_v 2" << "\n";
    //thrust::sort_by_key(thrust::host, cpuA_u, cpuA_u + g->m, cpuA_v);
    //thrust::copy(cpuA_u, cpuA_u + g->m, A_u.begin());
    //thrust::copy(cpuA_v, cpuA_v + g->m, A_v.begin());

    //std::cout << "sort A_u A_v 1 by key" << "\n";
    thrust::sort_by_key(thrust::device, A_v.begin(), A_v.end(), A_u.begin());
    //std::cout << "sort A_u A_v 2 by key" << "\n";
    thrust::stable_sort_by_key(thrust::device, A_u.begin(), A_u.end(), A_v.begin());
    auto clock2 = high_resolution_clock::now();
    fprintf(stderr, "\npreprocessing time: %.3lf s\n", duration_cast<milliseconds>(clock2 - clock1).count() / 1000.0);
    //int* cA_u = (int*)malloc(g->m * sizeof(int));
    //int* cA_v = (int*)malloc(g->m * sizeof(int));
    //thrust::copy(A_u.begin(), A_u.end(), cA_u);
    //thrust::copy(A_v.begin(), A_v.end(), cA_v);
    //finding epsilon neighbors
    
    //thrust::device_vector<int> begin(g->n);
    //thrust::device_vector<int> end(g->n);
    //thrust::device_vector<bool> flag(g->m / 2);
    //std::cout << "finding_epsilon_Neighbors" << "\n";

    //compute similarity
    int N_size;
    thrust::device_vector<bool> flag(g->m / 2);
    //auto clock1 = high_resolution_clock::now();
    finding_epsilon_Neighbors(eps, g->n, g->m, N_size, E_u, E_v, A_u, A_v, flag);
    //auto clock2 = high_resolution_clock::now();
    //fprintf(stderr, "finding_epsilon_neighbors time: %.3lf s\n", duration_cast<milliseconds>(clock2 - clock1).count() / 1000.0);
    //E_u.resize(2 * N_size);
    //E_v.resize(2 * N_size);
    //thrust::copy(E_v.begin(), E_v.begin() + N_size, E_u.begin() + N_size);
    //thrust::copy(E_u.begin(), E_u.begin() + N_size, E_v.begin() + N_size);

    //thrust::device_vector<bool> is_core(g->n);
    coreVertices(miu, N_size, g->n, g->m, E_u, E_v);

    //link
    thrust::device_vector<int> parents(g->n);
    linking(E_u, E_v, parents);

    //graph contraction
    contraction(g->n, parents);

    //classify non-member
    classNonMembers(g->m, A_u, A_v, parents);

    auto output_begin = high_resolution_clock::now();
    int* cluster = (int*)malloc((g->n) * sizeof(int));
    thrust::copy(parents.begin(), parents.end(), &cluster[0]);

    auto output_end = high_resolution_clock::now();
    fprintf(stderr, "\noutput time: %.3lf s\n", duration_cast<milliseconds>(output_end - output_begin).count() / 1000.0);
    return cluster;
}