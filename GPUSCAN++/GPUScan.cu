#include "GPUScan.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/partition.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <chrono>

const uint64_t  TOTALTHDCOUNT = 65536;
const uint32_t WARPSIZE = 32;
const uint32_t ii_BLOCKSIZE = 512;
const uint32_t cc_BLOCKSIZE = 128;

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

struct is_core
{
    __host__ __device__
        bool operator()(const int x)
    {
        return x >= 0;
    }
};

struct coreGraph_rule
{
    __host__ __device__
        bool operator()(const thrust::tuple<int, int>& x)
    {
        return (thrust::get<0>(x) >= 0) && (thrust::get<1>(x) >= 0);
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
        return thrust::get<0>(x1) != thrust::get<1>(x1);
    }
};

struct par_rule1
{
    __host__ __device__
        bool operator()(const thrust::tuple<vid_t, vid_t, char>& x1)
    {
        return thrust::get<2>(x1) == 'V';
    }
};

struct par_rule2
{
    __host__ __device__
        bool operator()(const thrust::tuple<vid_t, vid_t, char>& x1)
    {
        return (thrust::get<2>(x1) == 'Z') || (thrust::get<2>(x1) == 'W') ||
            (thrust::get<2>(x1) == 'z') || (thrust::get<2>(x1) == 'w');
    }
};

struct remove_rule
{
    __host__ __device__
        bool operator()(const thrust::tuple<vid_t, vid_t, bool>& x1)
    {
        return thrust::get<2>(x1) == false;
    }
};

__global__ __launch_bounds__(ii_BLOCKSIZE, 1024 / ii_BLOCKSIZE)
__global__ void __init_input(int* role, int* sd, int* ed, unsigned int* gpunum_edges, int miu, long vlength) {
    int threadCount = blockDim.x * gridDim.x;        //thread sum
    int tid = threadIdx.x + blockIdx.x * blockDim.x; //thread id
    while (tid < vlength) {
        //role[tid] = -1;
        sd[tid] = 0;
        ed[tid] = gpunum_edges[tid + 1] - gpunum_edges[tid];
        if (ed[tid] < miu) {
            role[tid] = -2;//-2为非核心点
        }
        else {
            role[tid] = -1;//-1为未知角色的点
        }
        tid += threadCount;
    }
}

__global__ __launch_bounds__(ii_BLOCKSIZE, 1024 / ii_BLOCKSIZE)
__global__ void __prep_edge(char* similar, unsigned int* gpunum_edges, int* rawE_u, int* rawE_v, double eps, int miu, 
    long elength, int* ed, int* sd, int* role) {
    int threadCount = blockDim.x * gridDim.x;        //thread sum
    int tid = threadIdx.x + blockIdx.x * blockDim.x; //thread id
    int u, v;
    double deg_u, deg_v;
    while (tid < elength / 2) {
        u = rawE_u[tid];
        v = rawE_v[tid];
        deg_u = (double)(gpunum_edges[u + 1] - gpunum_edges[u] + 1);
        deg_v = (double)(gpunum_edges[v + 1] - gpunum_edges[v] + 1);
        if (deg_u < deg_v * eps * eps) {
            similar[tid] = 'N';
            if (role[u] == -1) {
                atomicSub(&ed[u], 1);
                if (ed[u] < miu) role[u] = -2;
            }
            if (role[v] == -1) {
                atomicSub(&ed[v], 1);
                if (ed[v] < miu) role[v] = -2;
            }
        }
        else if (eps * eps * deg_u * deg_v <= 4.0) {
            similar[tid] = 'Y';
            if (role[u] == -1) {
                atomicAdd(&sd[u], 1);
                if (sd[u] >= miu) role[u] = u;
            }
            if (role[v] == -1) {
                atomicAdd(&sd[v], 1);
                if (sd[v] >= miu) role[v] = v;
            }
        }
        else {
            similar[tid] = 'U';
        }
        tid += threadCount;
    }
}

__device__ bool intersect(const double eps, const int* __restrict__ gpuadj, const int begin_u, const int end_u, 
    const int begin_v, const int end_v) {

    __shared__ int TriangleSum[cc_BLOCKSIZE / WARPSIZE];
    __shared__ int search_node[cc_BLOCKSIZE];
    //int tid = threadIdx.x + blockIdx.x * blockDim.x; //thread id
    int wid = threadIdx.x % WARPSIZE;                   //Warp内索引
    int bid = threadIdx.x;
    int WarpIdx_block = bid / WARPSIZE;
    TriangleSum[WarpIdx_block] = 0;
    //设置二分搜索标志
    unsigned int low;
    unsigned int high = end_v;
    unsigned int middle = begin_v;
    

    //开始查询
    int sum = 0;
    wid = begin_u + threadIdx.x % WARPSIZE;//待定修改
    while (wid < end_u)//待定修改
    {
        search_node[bid] = gpuadj[wid];
        //low = begin_v;
        low = middle;
        high = end_v;

        
        //若共享内存中无法找到，则在全局内存中找
        while (low < high)
        {
            middle = (low + high) >> 1;
            if (search_node[bid] < gpuadj[middle]) {
                high = middle;
            }
            else if (search_node[bid] > gpuadj[middle]) {
                low = middle + 1;
            }
            else {
                sum++;
                break;
            }
        }
        //若全局内存也找不到则线程换到下一个v的邻居点
        wid += WARPSIZE;
    }
    atomicAdd(&TriangleSum[WarpIdx_block], sum);
    return (TriangleSum[WarpIdx_block] + 2) * (TriangleSum[WarpIdx_block] + 2) < eps * eps *
        (end_u - begin_u + 1) * (end_v - begin_v + 1);
}

//__device__ bool intersect(const double eps, const int* __restrict__ gpuadj, const int begin_u, const int end_u,
//    const int begin_v, const int end_v) {
//
//    __shared__ int TriangleSum[cc_BLOCKSIZE / WARPSIZE];
//    __shared__ int search_node[cc_BLOCKSIZE];
//    //int tid = threadIdx.x + blockIdx.x * blockDim.x; //thread id
//    int wid = threadIdx.x % WARPSIZE;                   //Warp内索引
//    int bid = threadIdx.x;
//    int WarpIdx_block = bid / WARPSIZE;
//    TriangleSum[WarpIdx_block] = 0;
//    //设置二分搜索标志
//    unsigned int low;
//    unsigned int high = end_v;
//    unsigned int middle = begin_v;
//
//
//    //开始查询
//    int sum = (int)(sqrtf((end_u - begin_u + 1) * (end_v - begin_v + 1))*eps-2);
//    wid = begin_u + threadIdx.x % WARPSIZE;//待定修改
//    while (wid < end_u)//待定修改
//    {
//        search_node[bid] = gpuadj[wid];
//        //low = begin_v;
//        low = middle;
//        high = end_v;
//
//
//        //若共享内存中无法找到，则在全局内存中找
//        while (low < high)
//        {
//            middle = (low + high) >> 1;
//            if (search_node[bid] < gpuadj[middle]) {
//                high = middle;
//            }
//            else if (search_node[bid] > gpuadj[middle]) {
//                low = middle + 1;
//            }
//            else {
//                atomicAdd(&TriangleSum[WarpIdx_block], 1);
//                break;
//            }
//        }
//        
//        //若全局内存也找不到则线程换到下一个v的邻居点
//        wid += WARPSIZE;
//        if (TriangleSum[WarpIdx_block] > sum) break;
//    }
//    //atomicAdd(&TriangleSum[WarpIdx_block], sum);
//    return TriangleSum[WarpIdx_block] <= sum;
//}


__global__ __launch_bounds__(cc_BLOCKSIZE, 1024 / cc_BLOCKSIZE)
__global__ void __check_core(const double eps, const int miu, const long vlength, long elength, const unsigned int* __restrict__ gpunum_edges,
    const int* __restrict__ gpuadj, const int* __restrict__ rawE_u, const int* __restrict__ rawE_v, char* __restrict__ similar,
    int* __restrict__ role, int* __restrict__ sd, int* __restrict__ ed) {

    __shared__ int begin_u[cc_BLOCKSIZE / WARPSIZE];
    __shared__ int begin_v[cc_BLOCKSIZE / WARPSIZE];
    __shared__ int end_u[cc_BLOCKSIZE / WARPSIZE];
    __shared__ int end_v[cc_BLOCKSIZE / WARPSIZE];
    //__shared__ int TriangleSum[cc_BLOCKSIZE / WARPSIZE];
    //__shared__ int search_node[cc_BLOCKSIZE];

    int tid = threadIdx.x + blockIdx.x * blockDim.x; //thread id
    int wid = tid % WARPSIZE;                   //Warp内索引
    int Proc_it = tid / WARPSIZE;       //全局Warp索引
    int WarpIdx_block = threadIdx.x / WARPSIZE;    //线程块内的Warp索引
    int WarpNumb = blockDim.x * gridDim.x / WARPSIZE;    //全局的Warp数
    int u_vertex, v_vertex;                                 //当前warp查询的点u,v
    //int num_cache = 0;                                      //mod 32
    bool unsimilar;

    elength = elength / 2;
    
    while (Proc_it < elength) {
        u_vertex = rawE_u[Proc_it];
        v_vertex = rawE_v[Proc_it];
        //如果u和v相似度没被计算切u和v至少有一个role没被确定
        if (similar[Proc_it] == 'U' && (role[u_vertex] == -1 || role[v_vertex] == -1))
        {
            //compute similar
            begin_u[WarpIdx_block] = gpunum_edges[u_vertex];
            end_u[WarpIdx_block] = gpunum_edges[u_vertex + 1];
            begin_v[WarpIdx_block] = gpunum_edges[v_vertex];
            end_v[WarpIdx_block] = gpunum_edges[v_vertex + 1];
            unsimilar = intersect(eps, gpuadj, begin_u[WarpIdx_block],
                end_u[WarpIdx_block], begin_v[WarpIdx_block],
                end_v[WarpIdx_block]);

            //num_tri[Proc_it] = TriangleSum[WarpIdx_block];

            if (unsimilar)
            {
                similar[Proc_it] = 'N';
                if (wid == 0) {
                    if (role[u_vertex] == -1)
                    {
                        atomicSub(&ed[u_vertex], 1);
                        if (ed[u_vertex] < miu)
                        {
                            role[u_vertex] = -2;//cons
                            //atomicExch(&role[u_vertex], -2);
                        }
                    }
                    if (role[v_vertex] == -1)
                    {
                        atomicSub(&ed[v_vertex], 1);
                        if (ed[v_vertex] < miu)
                        {
                            role[v_vertex] = -2;//cons
                            //atomicExch(&role[v_vertex], -2);
                        }
                    }
                }
            }
            else
            {
                similar[Proc_it] = 'Y';
                if (wid == 0) {
                    if (role[u_vertex] == -1)
                    {
                        atomicAdd(&sd[u_vertex], 1);
                        if (sd[u_vertex] >= miu)
                        {
                            role[u_vertex] = u_vertex;//cons
                            //atomicExch(&role[u_vertex], u_vertex);
                        }
                    }
                    if (role[v_vertex] == -1)
                    {
                        atomicAdd(&sd[v_vertex], 1);
                        if (sd[v_vertex] >= miu)
                        {
                            role[v_vertex] = v_vertex;//cons
                            //atomicExch(&role[v_vertex], v_vertex);
                        }
                    }
                }
            }
        }
        Proc_it += WarpNumb;
    }
}

__global__ void __check_core0(const double eps, const int miu, const long vlength, long elength, const unsigned int* __restrict__ gpunum_edges,
    const int* __restrict__ gpuadj, const int* __restrict__ rawE_u, const int* __restrict__ rawE_v, char* __restrict__ similar,
    int* __restrict__ role, int* __restrict__ sd, int* __restrict__ ed) {

    __shared__ int begin_u[cc_BLOCKSIZE / WARPSIZE];
    __shared__ int begin_v[cc_BLOCKSIZE / WARPSIZE];
    __shared__ int end_u[cc_BLOCKSIZE / WARPSIZE];
    __shared__ int end_v[cc_BLOCKSIZE / WARPSIZE];
    //__shared__ int TriangleSum[cc_BLOCKSIZE / WARPSIZE];
    //__shared__ int search_node[cc_BLOCKSIZE];

    //int tid = threadIdx.x + blockIdx.x * blockDim.x; //thread id
    int wid = (threadIdx.x + blockIdx.x * blockDim.x) % WARPSIZE;                   //Warp内索引
    int WarpIdx_block = threadIdx.x / WARPSIZE;    //线程块内的Warp索引
    int warpsInblock = blockDim.x / WARPSIZE;      //bolck内warp数
    unsigned int Proc_Numb, Proc_it;                           //一个Warp处理的边的数目,边表处理起始位置标识
    int u_vertex, v_vertex;                                 //当前warp查询的点u,v
    bool unsimilar;
    //int blockNumb = gridDim.x;

    elength = elength / 2;
    if (gridDim.x < elength) {
        if (elength % gridDim.x == 0) {
            Proc_Numb = elength / gridDim.x;
        }
        else Proc_Numb = elength / gridDim.x + 1;
    }
    else Proc_Numb = 1;
    Proc_it = blockIdx.x * Proc_Numb + WarpIdx_block;

    Proc_Numb = (blockIdx.x + 1) * Proc_Numb;

    while (Proc_it < Proc_Numb) {
        if (Proc_it > elength - 1) break;
        u_vertex = rawE_u[Proc_it];
        v_vertex = rawE_v[Proc_it];
        if (similar[Proc_it] == 'U' && (role[u_vertex] == -1 || role[v_vertex] == -1))
        {
            begin_u[WarpIdx_block] = gpunum_edges[u_vertex];
            end_u[WarpIdx_block] = gpunum_edges[u_vertex + 1];
            begin_v[WarpIdx_block] = gpunum_edges[v_vertex];
            end_v[WarpIdx_block] = gpunum_edges[v_vertex + 1];
            //compute similar
            unsimilar = intersect(eps, gpuadj, begin_u[WarpIdx_block],
                end_u[WarpIdx_block], begin_v[WarpIdx_block],
                end_v[WarpIdx_block]);

            if (unsimilar)
            {
                similar[Proc_it] = 'N';
                if (wid == 0) {
                    if (role[u_vertex] == -1)
                    {
                        atomicSub(&ed[u_vertex], 1);
                        if (ed[u_vertex] < miu)
                        {
                            role[u_vertex] = -2;//cons
                        }
                    }
                    if (role[v_vertex] == -1)
                    {
                        atomicSub(&ed[v_vertex], 1);
                        if (ed[v_vertex] < miu)
                        {
                            role[v_vertex] = -2;//cons
                        }
                    }
                }
            }
            else
            {
                similar[Proc_it] = 'Y';
                if (wid == 0) {
                    if (role[u_vertex] == -1)
                    {
                        atomicAdd(&sd[u_vertex], 1);
                        if (sd[u_vertex] >= miu)
                        {
                            role[u_vertex] = u_vertex;//cons
                        }
                    }
                    if (role[v_vertex] == -1)
                    {
                        atomicAdd(&sd[v_vertex], 1);
                        if (sd[v_vertex] >= miu)
                        {
                            role[v_vertex] = v_vertex;//cons
                        }
                    }
                }
            }
        }
        Proc_it += warpsInblock;
    }
}

inline __device__ int findroot(const int idx, int* const __restrict__ role) {

    int curr = role[idx];
    if (curr != idx) {
        int next, prev = idx;
        while (curr > (next = role[curr])) {
            role[prev] = next;
            prev = curr;
            curr = next;
        }
    }
    return curr;
}

__global__ __launch_bounds__(cc_BLOCKSIZE, 1024 / cc_BLOCKSIZE)
__global__ void __cluster_core1(const unsigned int* __restrict__ gpunum_edges, const int* __restrict__ gpuadj, 
    const unsigned int* __restrict__ gpuEid, char* const __restrict__ similar, int* const __restrict__ role, 
    const int* __restrict__ gpucore_node, const int core_number) {

    const int offset = (threadIdx.x + blockIdx.x * blockDim.x) % WARPSIZE;                   //Warp内索引
    int WarpIdx_global = (threadIdx.x + blockIdx.x * blockDim.x) / WARPSIZE;       //全局Warp索引
    int WarpNumb = blockDim.x * gridDim.x / WARPSIZE;    //全局的Warp数

    while (WarpIdx_global < core_number)
    {
        const int u_vertex = gpucore_node[WarpIdx_global];
        int u_parent = findroot(u_vertex, role);
        for (int i = gpunum_edges[u_vertex] + offset; i < gpunum_edges[u_vertex + 1]; i += WARPSIZE)
        {
            const int v_vertex = gpuadj[i];
            if (u_vertex < v_vertex) break;
            if (role[v_vertex] >= 0 && similar[gpuEid[i]] == 'Y')
            {
                int v_parent = findroot(v_vertex, role);
                bool repeat;
                do
                {
                    repeat = false;
                    if (u_parent != v_parent)
                    {
                        int ret;
                        if (u_parent < v_parent)
                        {
                            if ((ret = atomicCAS(&role[v_parent], v_parent, u_parent)) != v_parent)
                            {
                                v_parent = ret;
                                repeat = true;
                            }
                        }
                        else
                        {
                            if ((ret = atomicCAS(&role[u_parent], u_parent, v_parent)) != u_parent)
                            {
                                u_parent = ret;
                                repeat = true;
                            }
                        }
                    }
                } while (repeat);
            }
        }
        WarpIdx_global += WarpNumb;
    }
}

__global__ __launch_bounds__(cc_BLOCKSIZE, 1024 / cc_BLOCKSIZE)
__global__ void __cluster_core2(const double eps, const unsigned int* __restrict__ gpunum_edges,
    const int* __restrict__ gpuadj, const unsigned int* __restrict__ gpuEid,
    char* const __restrict__ similar, int* const __restrict__ role, const int* __restrict__ gpucore_node,
    const int core_number) {

    __shared__ int begin_u[cc_BLOCKSIZE / WARPSIZE];
    __shared__ int end_u[cc_BLOCKSIZE / WARPSIZE];
    __shared__ int begin_v[cc_BLOCKSIZE / WARPSIZE][WARPSIZE];
    __shared__ int end_v[cc_BLOCKSIZE / WARPSIZE][WARPSIZE];
    __shared__ int role_sh[cc_BLOCKSIZE / WARPSIZE][WARPSIZE];
    __shared__ char similar_sh[cc_BLOCKSIZE / WARPSIZE][WARPSIZE];
    //__shared__ int TriangleSum[cc_BLOCKSIZE / WARPSIZE];
    //__shared__ int search_node[cc_BLOCKSIZE];

    int wid = (threadIdx.x + blockIdx.x * blockDim.x) % WARPSIZE;                   //Warp内索引
    int WarpIdx_global = (threadIdx.x + blockIdx.x * blockDim.x) / WARPSIZE;       //全局Warp索引
    int WarpIdx_block = threadIdx.x / WARPSIZE;    //线程块内的Warp索引
    int WarpNumb = blockDim.x * gridDim.x / WARPSIZE;    //全局的Warp数
    int num_cache;                                   //mod 32

    while (WarpIdx_global < core_number)
    {
        const int u_vertex = gpucore_node[WarpIdx_global];
        int u_parent = findroot(u_vertex, role);
        begin_u[WarpIdx_block] = gpunum_edges[u_vertex];
        end_u[WarpIdx_block] = gpunum_edges[u_vertex + 1];
        num_cache = 0;
        for (int offset = begin_u[WarpIdx_block]; offset < end_u[WarpIdx_block]; offset++)
        {
            const int v_vertex = gpuadj[offset];
            if (v_vertex >= u_vertex) break;
            if (num_cache % WARPSIZE == 0) {
                if (offset + wid < end_u[WarpIdx_block]) {
                    begin_v[WarpIdx_block][wid] = gpunum_edges[gpuadj[offset + wid]];
                    end_v[WarpIdx_block][wid] = gpunum_edges[gpuadj[offset + wid] + 1];
                    role_sh[WarpIdx_block][wid] = role[gpuadj[offset + wid]];
                    similar_sh[WarpIdx_block][wid] = similar[gpuEid[offset + wid]];
                }
            }
            if (role_sh[WarpIdx_block][num_cache] >= 0 && similar_sh[WarpIdx_block][num_cache] == 'U')
            {
                //is same set?
                int v_parent = findroot(v_vertex, role);
                if (u_parent != v_parent)
                {
                    bool unsimilar;
                    if (end_v[WarpIdx_block][num_cache] - begin_v[WarpIdx_block][num_cache] > end_u[WarpIdx_block] - begin_u[WarpIdx_block])
                    {
                        unsimilar = intersect(eps, gpuadj, begin_u[WarpIdx_block], end_u[WarpIdx_block], begin_v[WarpIdx_block][num_cache],
                            end_v[WarpIdx_block][num_cache]);
                    }
                    else {
                        unsimilar = intersect(eps, gpuadj, begin_v[WarpIdx_block][num_cache], end_v[WarpIdx_block][num_cache], begin_u[WarpIdx_block],
                            end_u[WarpIdx_block]);
                    }
                    if (unsimilar)
                    {
                        similar[gpuEid[offset]] = 'N';
                    }
                    else
                    {
                        similar[gpuEid[offset]] = 'Y';
                        u_parent = findroot(u_vertex, role);
                        v_parent = findroot(v_vertex, role);
                        if (wid == 0)
                        {
                            bool repeat;
                            do
                            {
                                repeat = false;
                                if (u_parent != v_parent)
                                {
                                    int ret;
                                    if (u_parent < v_parent)
                                    {
                                        if ((ret = atomicCAS(&role[v_parent], v_parent, u_parent)) != v_parent)
                                        {
                                            v_parent = ret;
                                            repeat = true;
                                        }
                                    }
                                    else
                                    {
                                        if ((ret = atomicCAS(&role[u_parent], u_parent, v_parent)) != u_parent)
                                        {
                                            u_parent = ret;
                                            repeat = true;
                                        }
                                    }
                                }
                            } while (repeat);
                        }
                    }
                }
            }
            num_cache = (num_cache + 1) % WARPSIZE;
        }
        WarpIdx_global += WarpNumb;
    }
}

__global__ __launch_bounds__(ii_BLOCKSIZE, 1024 / ii_BLOCKSIZE)
__global__ void __cluster_core(int* gpucore_node, int* role, int core_number) {

    int threadCount = blockDim.x * gridDim.x;        //thread sum
    int tid = threadIdx.x + blockIdx.x * blockDim.x; //thread id
    int Parent;
    while (tid < core_number)
    {
        int coreid = gpucore_node[tid];
        Parent = role[coreid];
        if (Parent != role[Parent])
        {
            while (Parent != role[Parent])
            {
                Parent = role[Parent];
            }
            atomicExch(&role[coreid], Parent);
        }
        tid += threadCount;
    }
}

__global__ __launch_bounds__(cc_BLOCKSIZE, 1024 / cc_BLOCKSIZE)
__global__ void __cluster_noncore(double eps, unsigned int* gpunum_edges, int* gpuadj, unsigned int* gpuEid,
    char* similar, int* role, int* gpucore_node, int core_number) {

    __shared__ int begin_u[cc_BLOCKSIZE / WARPSIZE];
    __shared__ int end_u[cc_BLOCKSIZE / WARPSIZE];
    __shared__ int begin_v[cc_BLOCKSIZE / WARPSIZE][WARPSIZE];
    __shared__ int end_v[cc_BLOCKSIZE / WARPSIZE][WARPSIZE];
    __shared__ int role_sh[cc_BLOCKSIZE / WARPSIZE][WARPSIZE];
    //__shared__ int TriangleSum[cc_BLOCKSIZE / WARPSIZE];
    //__shared__ int search_node[cc_BLOCKSIZE];

    //int threadCount = blockDim.x * gridDim.x;        //thread sum
    int tid = threadIdx.x + blockIdx.x * blockDim.x; //thread id
    int wid = tid % WARPSIZE;                   //Warp内索引
    int WarpIdx_global = tid / WARPSIZE;       //全局Warp索引
    int WarpIdx_block = threadIdx.x / WARPSIZE;    //线程块内的Warp索引
    int WarpNumb = blockDim.x * gridDim.x / WARPSIZE;    //全局的Warp数
    int num_cache;

    int u_vertex, v_vertex;
    while (WarpIdx_global < core_number)
    {
        u_vertex = gpucore_node[WarpIdx_global];
        begin_u[WarpIdx_block] = gpunum_edges[u_vertex];
        end_u[WarpIdx_block] = gpunum_edges[u_vertex + 1];
        num_cache = 0;
        for (unsigned int offset = begin_u[WarpIdx_block]; offset < end_u[WarpIdx_block]; ++offset)
        {
            v_vertex = gpuadj[offset];
            if (num_cache % WARPSIZE == 0) {
                if (offset + wid < end_u[WarpIdx_block]) {
                    begin_v[WarpIdx_block][wid] = gpunum_edges[gpuadj[offset + wid]];
                    end_v[WarpIdx_block][wid] = gpunum_edges[gpuadj[offset + wid] + 1];
                    role_sh[WarpIdx_block][wid] = role[gpuadj[offset + wid]];
                    //similar_sh[WarpIdx_block][wid] = similar[gpuEid[offset + wid]];
                }
            }
            if (role_sh[WarpIdx_block][num_cache] < 0)
            {
                //如果相似性未被计算过，则需计算
                if (similar[gpuEid[offset]] == 'U')
                {
                    bool unsimilar;
                    if (end_v[WarpIdx_block][num_cache] - begin_v[WarpIdx_block][num_cache] > 
                        end_u[WarpIdx_block] - begin_u[WarpIdx_block])
                    {
                        unsimilar = intersect(eps, gpuadj, begin_u[WarpIdx_block], end_u[WarpIdx_block], 
                            begin_v[WarpIdx_block][num_cache],end_v[WarpIdx_block][num_cache]);
                    }
                    else {
                        unsimilar = intersect(eps, gpuadj, begin_v[WarpIdx_block][num_cache], 
                            end_v[WarpIdx_block][num_cache],begin_u[WarpIdx_block], end_u[WarpIdx_block]);
                    }
                    if (unsimilar)
                    {
                        similar[gpuEid[offset]] = 'N';
                    }
                    else
                    {
                        similar[gpuEid[offset]] = 'Y';
                        if (wid == 0) {
                            atomicExch(&role[v_vertex], role[u_vertex]);
                        }
                    }
                }
                else if (similar[gpuEid[offset]] == 'Y') {
                    if (wid == 0) {
                        atomicExch(&role[v_vertex], role[u_vertex]);//如果已经计算过且相似则可以直接聚类
                    }
                }
            }
            num_cache = (num_cache + 1) % WARPSIZE;
        }
        WarpIdx_global += WarpNumb;
    }
}

__global__ __launch_bounds__(ii_BLOCKSIZE, 1024 / ii_BLOCKSIZE)
__global__ void __classify_others(int* role, unsigned int* gpunum_edges, int* gpuadj, int* gpuothers_node, int others_number) {

    __shared__ int adjrole[ii_BLOCKSIZE / WARPSIZE][WARPSIZE];
    __shared__ bool checkflag[ii_BLOCKSIZE / WARPSIZE];
    __shared__ int ne[ii_BLOCKSIZE / WARPSIZE];
    int tid = threadIdx.x + blockIdx.x * blockDim.x; //thread id
    int node;
    int begin, end;
    
    int wid = tid % WARPSIZE;                   //Warp内索引
    int WarpIdx_global = tid / WARPSIZE;       //全局Warp索引
    int WarpIdx_block = threadIdx.x / WARPSIZE;    //线程块内的Warp索引
    int WarpNumb = blockDim.x * gridDim.x / WARPSIZE;    //全局的Warp数
    while (WarpIdx_global < others_number)
    {
        node = gpuothers_node[WarpIdx_global];
        begin = gpunum_edges[node];
        end = gpunum_edges[node + 1];
        checkflag[WarpIdx_block] = false;
        ne[WarpIdx_block] = -1;
        if (end - begin > 1)
        {
            int Proc_it;
            int num_cache = 0;
            for (Proc_it = begin; Proc_it < end; Proc_it++)
            {
                if (num_cache % WARPSIZE == 0) {
                    if (Proc_it + wid <= end) {
                        adjrole[WarpIdx_block][wid] = role[gpuadj[Proc_it + wid]];
                    }
                }
                if (adjrole[WarpIdx_block][num_cache] >= 0) {
                    ne[WarpIdx_block] = adjrole[WarpIdx_block][num_cache];
                    break;
                }
                num_cache = (num_cache + 1) % WARPSIZE;
            }
            if (ne[WarpIdx_block] >= 0 && Proc_it < end - 1) {
                Proc_it++;
                for (int w = Proc_it + wid; w < end; w += WARPSIZE)
                {
                    int rol = role[gpuadj[w]];
                    if (rol >= 0 && rol != ne[WarpIdx_block]) {
                        checkflag[WarpIdx_block] = true;
                    }
                    if (checkflag[WarpIdx_block]) break;
                }
                if (checkflag[WarpIdx_block]) role[node] = -1;
            }
            //wid_t = begin + wid;
            //while (!checkflag[WarpIdx_block] && wid_t < end)
            //{
            //    adjrole[WarpIdx_block][wid] = role[gpuadj[wid_t]];
            //    if (wid > 0) {
            //        if (adjrole[WarpIdx_block][wid] >= 0 && adjrole[WarpIdx_block][wid - 1] >= 0 &&
            //            adjrole[WarpIdx_block][wid] != adjrole[WarpIdx_block][wid - 1]) {
            //            checkflag[WarpIdx_block] = true;
            //            role[node] = -1;
            //        }
            //    }
            //    wid_t += WARPSIZE;
            //}
        }
        WarpIdx_global += WarpNumb;
    }
}

int* GPUScan(graph_t* g, int* Edge_u, int* Edge_v, double eps, int miu) {
    //init GPU
    cudaDeviceProp deviceProp;
    gpuErrchk(cudaGetDeviceProperties(&deviceProp, 0));
    //cout << "使用GPU 设备 0: " << deviceProp.name << endl;
    //cout << "设备常量内存： " << deviceProp.totalConstMem << " bytes." << endl;
    //cout << "纹理内存对齐： " << deviceProp.textureAlignment << " bytes." << endl;
    //cout << "线程warp大小： " << deviceProp.warpSize << endl;
    //cout << "SM的数量：" << deviceProp.multiProcessorCount << std::endl;
    //cout << "每个线程块的共享内存大小：" << deviceProp.sharedMemPerBlock / 1024.0 << " KB" << endl;
    //cout << "每个线程块的最大线程数：" << deviceProp.maxThreadsPerBlock << endl;
    //cout << "设备上一个线程块（Block）种可用的32位寄存器数量： " << deviceProp.regsPerBlock << endl;
    //cout << "每个EM的最大线程数：" << deviceProp.maxThreadsPerMultiProcessor << endl;
    //cout << "每个EM的最大线程束数：" << deviceProp.maxThreadsPerMultiProcessor / 32 << endl;
    //cout << "设备上多处理器的数量： " << deviceProp.multiProcessorCount << endl;
    //cout << "线程块每维最大线程数： " << deviceProp.maxThreadsDim[0] << ", " << deviceProp.maxThreadsDim[1] << endl;
    //cout << "网格每维最大线程数： " << deviceProp.maxGridSize[0] << ", " << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2] << endl;
    using namespace std::chrono;

    auto start = high_resolution_clock::now();

    int* gpuadj;
    unsigned int* gpunum_edges;
    unsigned int* gpuEid;
    int* rawE_u;
    int* rawE_v;
    //thrust::device_vector<int> E_u(g->m / 2);
    //thrust::device_vector<int> E_v(g->m / 2);

    char* similar;
    int* role;
    int* sd;
    int* ed;

    //init input
   
    gpuErrchk(cudaMalloc((void**)&gpuadj, (g->m) * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&gpunum_edges, (g->n + 1) * sizeof(unsigned int)));
    gpuErrchk(cudaMalloc((void**)&gpuEid, (g->m) * sizeof(unsigned int)));
    gpuErrchk(cudaMalloc((void**)&similar, (g->m / 2) * sizeof(char)));
    gpuErrchk(cudaMalloc((void**)&role, (g->n) * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&sd, (g->n) * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&ed, (g->n) * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&rawE_u, (g->m / 2) * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&rawE_v, (g->m / 2) * sizeof(int)));
    
    
    gpuErrchk(cudaMemcpy(gpuadj, g->adj, (g->m) * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(gpunum_edges, g->num_edges, (g->n + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(gpuEid, g->eid, (g->m) * sizeof(unsigned int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(rawE_u, Edge_u, (g->m / 2) * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(rawE_v, Edge_v, (g->m / 2) * sizeof(int), cudaMemcpyHostToDevice));
    //thrust::copy(Edge_u, Edge_u + g->m / 2, E_u.begin());
    //thrust::copy(Edge_v, Edge_v + g->m / 2, E_v.begin());
    //int* rawE_u = thrust::raw_pointer_cast(E_u.data());
    //int* rawE_v = thrust::raw_pointer_cast(E_v.data());
    
    auto finish = high_resolution_clock::now();
    fprintf(stderr, "Create GPU Memory time: %.3lf s\n", duration_cast<milliseconds>(finish - start).count() / 1000.0);

    uint32_t  ii_ThreadBlockCount = TOTALTHDCOUNT / ii_BLOCKSIZE;
    uint32_t  cc_ThreadBlockCount = TOTALTHDCOUNT / cc_BLOCKSIZE;

    auto init_input_start = high_resolution_clock::now();
    __init_input <<<ii_ThreadBlockCount, ii_BLOCKSIZE>>> (role, sd, ed, gpunum_edges, miu, g->n);
    //auto init_input_finish = high_resolution_clock::now();
    //fprintf(stderr, "Initialize sd ed time: %.3lf s\n", duration_cast<milliseconds>(init_input_finish - init_input_start).count() / 1000.0);

    //auto prep_edge_start = high_resolution_clock::now();
    __prep_edge <<<ii_ThreadBlockCount, ii_BLOCKSIZE>>> (similar, gpunum_edges, rawE_u, rawE_v, eps, miu, g->m, ed, sd, role);

    thrust::device_ptr<int> role_ptr = thrust::device_pointer_cast(role);
    thrust::device_ptr<char> similar_ptr = thrust::device_pointer_cast(similar);
    int core_number = thrust::count_if(role_ptr, role_ptr + g->n, is_core());

#ifdef _DEBUG_
    int uncore_number = thrust::count(role_ptr, role_ptr + g->n, -2);//debug
    int unknow_number = thrust::count(role_ptr, role_ptr + g->n, -1);//debug
    fprintf(stderr, "\nprepocessing core number: %d\n", core_number);//debug
    fprintf(stderr, "prepocessing uncore number: %d\n", uncore_number);//debug
    fprintf(stderr, "prepocessing unknow number: %d\n", unknow_number);//debug
    int similar_edge = thrust::count(similar_ptr, similar_ptr + g->m / 2, 'Y');
    int unsimilar_edge = thrust::count(similar_ptr, similar_ptr + g->m / 2, 'N');
    int unknow_edge = thrust::count(similar_ptr, similar_ptr + g->m / 2, 'U');
    fprintf(stderr, "\nprepocessing similar number: %d\n", similar_edge);//debug
    fprintf(stderr, "prepocessing unsimilar number: %d\n", unsimilar_edge);//debug
    fprintf(stderr, "prepocessing unknow number: %d\n", unknow_edge);//debug
#endif

    auto prep_edge_finish = high_resolution_clock::now();
    fprintf(stderr, "\nPrep time: %.3lf s\n", duration_cast<milliseconds>(prep_edge_finish - init_input_start).count() / 1000.0);

    //check core
    auto check_core_start = high_resolution_clock::now();
    __check_core <<<cc_ThreadBlockCount, cc_BLOCKSIZE>>> (eps, miu, g->n, g->m, gpunum_edges, gpuadj, rawE_u, rawE_v, 
        similar, role, sd, ed);

    //free
    cudaFree(sd);
    cudaFree(ed);
    auto check_core_finish = high_resolution_clock::now();
    fprintf(stderr, "\nCheck Core time: %.3lf s\n", duration_cast<milliseconds>(check_core_finish - check_core_start).count() / 1000.0);

    //compute core number
    auto firstlink_start = high_resolution_clock::now();

    core_number = thrust::count_if(role_ptr, role_ptr + g->n, is_core());

#ifdef _DEBUG_
    uncore_number = thrust::count(role_ptr, role_ptr + g->n, -2);//debug
    unknow_number = thrust::count(role_ptr, role_ptr + g->n, -1);//debug
    fprintf(stderr, "\nuncore number: %d\n", uncore_number);//debug
    fprintf(stderr, "unknow number: %d\n", unknow_number);//debug
#endif
    fprintf(stderr, "core number: %d\n", core_number);//debug
    //output array
    int* clusters = (int*)malloc((g->n) * sizeof(int));

    int* gpucore_node;
    gpuErrchk(cudaMalloc((void**)&gpucore_node, (core_number) * sizeof(int)));
    thrust::copy_if(thrust::device, role, role + g->n, gpucore_node, is_core());

    __cluster_core1 <<<cc_ThreadBlockCount, cc_BLOCKSIZE>>> (gpunum_edges, gpuadj, gpuEid, similar, 
        role, gpucore_node, core_number);

#ifdef _DEBUG_
    gpuErrchk(cudaMemcpy(clusters, role, (g->n) * sizeof(int), cudaMemcpyDeviceToHost));
#endif

    auto firstlink_finish = high_resolution_clock::now();
    fprintf(stderr, "\nFirst Link Phase time: %.3lf s\n", duration_cast<milliseconds>(firstlink_finish - firstlink_start).count() / 1000.0);

    auto secondlink_start = high_resolution_clock::now();
    __cluster_core2 <<<cc_ThreadBlockCount, cc_BLOCKSIZE>>> (eps, gpunum_edges, gpuadj, gpuEid, similar, role,
        gpucore_node, core_number);

#ifdef _DEBUG_
    gpuErrchk(cudaMemcpy(clusters, role, (g->n) * sizeof(int), cudaMemcpyDeviceToHost));
#endif

    auto secondlink_finish = high_resolution_clock::now();
    fprintf(stderr, "Second Link Phase time: %.3lf s\n", duration_cast<milliseconds>(secondlink_finish - secondlink_start).count() / 1000.0);

    auto cluster_start = high_resolution_clock::now();
    __cluster_core <<<ii_ThreadBlockCount, ii_BLOCKSIZE>>> (gpucore_node, role, core_number);

#ifdef _DEBUG_
    gpuErrchk(cudaMemcpy(clusters, role, (g->n) * sizeof(int), cudaMemcpyDeviceToHost));
#endif

    auto cluster_finish = high_resolution_clock::now();
    fprintf(stderr, "Cluster core time: %.3lf s\n", duration_cast<milliseconds>(cluster_finish - cluster_start).count() / 1000.0);

    //cluster non-core
    auto noncore_start = high_resolution_clock::now();
    __cluster_noncore <<<cc_ThreadBlockCount, cc_BLOCKSIZE>>> (eps, gpunum_edges, gpuadj, gpuEid, similar, 
        role, gpucore_node, core_number);

#ifdef _DEBUG_
    gpuErrchk(cudaMemcpy(clusters, role, (g->n) * sizeof(int), cudaMemcpyDeviceToHost));
#endif

    auto noncore_finish = high_resolution_clock::now();
    fprintf(stderr, "Cluster non-core time: %.3lf s\n", duration_cast<milliseconds>(noncore_finish - noncore_start).count() / 1000.0);
    
    //free
    cudaFree(gpucore_node);

#ifdef _DEBUG_
    //similar_edge = thrust::count(similar_ptr, similar_ptr + g->m / 2, 'Y');
    //unsimilar_edge = thrust::count(similar_ptr, similar_ptr + g->m / 2, 'N');
    unknow_edge = thrust::count(similar_ptr, similar_ptr + g->m / 2, 'U');
    //fprintf(stderr, "prepocessing similar number: %d\n", similar_edge);//debug
    //fprintf(stderr, "prepocessing unsimilar number: %d\n", unsimilar_edge);//debug
    fprintf(stderr, "\nuncomputed similarity number: %d\n", unknow_edge);//debug
#endif

    auto cla_others_start = high_resolution_clock::now();
    int others_number = thrust::count(role_ptr, role_ptr + g->n, -2);
    int* gpuothers_node;
    gpuErrchk(cudaMalloc((void**)&gpuothers_node, (g->n) * sizeof(int)));
    thrust::sequence(thrust::device, gpuothers_node, gpuothers_node + g->n);
    thrust::remove_if(thrust::device, gpuothers_node, gpuothers_node + g->n, role, is_core());

    __classify_others <<<ii_ThreadBlockCount, ii_BLOCKSIZE>>> (role, gpunum_edges, gpuadj, gpuothers_node, others_number);
    
    gpuErrchk(cudaMemcpy(clusters, role, (g->n) * sizeof(int), cudaMemcpyDeviceToHost));

    auto cla_others_finish = high_resolution_clock::now();
    fprintf(stderr, "\nClassify Others time: %.3lf s\n", duration_cast<milliseconds>(cla_others_finish - cla_others_start).count() / 1000.0);

    auto free_start = high_resolution_clock::now();
    //free
    cudaFree(gpuothers_node);

    //output
    //int* clusters = (int*)malloc((g->n) * sizeof(int));
    //gpuErrchk(cudaMemcpy(clusters, role, (g->n) * sizeof(int), cudaMemcpyDeviceToHost));

    //free
    cudaFree(gpunum_edges);
    cudaFree(gpuadj);
    cudaFree(gpuEid);
    cudaFree(similar);
    cudaFree(role);
    auto free_finish = high_resolution_clock::now();
    fprintf(stderr, "Free and Output time: %.3lf s\n", duration_cast<milliseconds>(free_finish - free_start).count() / 1000.0);

    return clusters;
}



