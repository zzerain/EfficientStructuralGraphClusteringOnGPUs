#include "SCANoffload.h"
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
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <chrono>


uint64_t  TOTALMEMOSIZE = 0x80000000;// 1536000;//0x180000000;	//GPU memory 
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

struct par_rule1
{
    __host__ __device__
        bool operator()(const thrust::tuple<vid_t, vid_t, char>& x1)
    {
        return thrust::get<2>(x1) != 'V';
    }
};

struct par_rule2
{
    __host__ __device__
        bool operator()(const thrust::tuple<vid_t, vid_t, char>& x1)
    {
        return (thrust::get<2>(x1) != 'Z') && (thrust::get<2>(x1) != 'W')&&
            (thrust::get<2>(x1) != 'z') && (thrust::get<2>(x1) != 'w');
    }
};

__global__ void __init_input(vid_t* role, vid_t* sd, vid_t* ed, eid_t* gpunum_edges, vid_t vlength, int miu) {

    vid_t threadCount = blockDim.x * gridDim.x;        //thread sum
    vid_t tid = threadIdx.x + blockIdx.x * blockDim.x; //thread id
    while (tid < vlength) {
        //role[tid] = -1;
        sd[tid] = 0;
        eid_t deg = gpunum_edges[tid + 1] - gpunum_edges[tid];
        //ed[tid] = gpunum_edges[tid + 1] - gpunum_edges[tid];
        if (deg < miu) {
            role[tid] = -2;//-2为非核心点
        }
        else {
            role[tid] = -1;//-1为未知角色的点
        }
        ed[tid] = deg;
        tid += threadCount;
    }
}

using namespace std::chrono;

__global__ void __prep_edge(char* similar, eid_t* gpunum_edges, vid_t* E_u, vid_t* E_v, double eps, int miu,
    eid_t elength, vid_t* ed, vid_t* sd, vid_t* role, vid_t* gpu_ni) {

    __shared__ vid_t u_ni[ii_BLOCKSIZE];
    __shared__ vid_t v_ni[ii_BLOCKSIZE];
    eid_t threadCount = blockDim.x * gridDim.x;        //thread sum
    eid_t tid = threadIdx.x + blockIdx.x * blockDim.x; //thread id
    int bid = threadIdx.x;
    vid_t u, v;
    double deg_u, deg_v;
    while (tid < elength) {
        u = E_u[tid];
        v = E_v[tid];
        u_ni[bid] = gpu_ni[u];
        v_ni[bid] = gpu_ni[v];
        deg_u = (double)(gpunum_edges[u_ni[bid] + 1] - gpunum_edges[u_ni[bid]] + 1);
        deg_v = (double)(gpunum_edges[v_ni[bid] + 1] - gpunum_edges[v_ni[bid]] + 1);
        if (deg_u < deg_v * eps * eps) {
            similar[tid] = 'N';
            if (role[u_ni[bid]] == -1) {
                atomicSub(&ed[u_ni[bid]], 1);
                if (ed[u_ni[bid]] < miu) role[u_ni[bid]] = -2;
            }
            if (role[v_ni[bid]] == -1) {
                atomicSub(&ed[v_ni[bid]], 1);
                if (ed[v_ni[bid]] < miu) role[v_ni[bid]] = -2;
            }
        }
        else if (eps * eps * deg_u * deg_v <= 4.00) {
            similar[tid] = 'Y';
            if (role[u_ni[bid]] == -1) {
                atomicAdd(&sd[u_ni[bid]], 1);
                if (sd[u_ni[bid]] >= miu) role[u_ni[bid]] = u;
            }
            if (role[v_ni[bid]] == -1) {
                atomicAdd(&sd[v_ni[bid]], 1);
                if (sd[v_ni[bid]] >= miu) role[v_ni[bid]] = v;
            }
        }
        else {
            similar[tid] = 'U';
        }
        tid += threadCount;
    }
}

__device__ bool intersect(const double eps, const vid_t* __restrict__ gpuadj, const eid_t begin_u, const eid_t end_u,
    const eid_t begin_v, const eid_t end_v, vid_t* __restrict__ BT_sh, const int* __restrict__ wap_log,
    vid_t* __restrict__ TriangleSum, vid_t* __restrict__ search_node) {

    //__shared__ int TriangleSum[cc_BLOCKSIZE / WARPSIZE];
    //__shared__ int search_node[cc_BLOCKSIZE];
    //int tid = threadIdx.x + blockIdx.x * blockDim.x; //thread id
    eid_t wid = threadIdx.x % WARPSIZE;                   //Warp内索引
    int bid = threadIdx.x;
    int WarpIdx_block = bid / WARPSIZE;
    TriangleSum[WarpIdx_block] = 0;
    //设置二分搜索标志
    eid_t low = begin_v;
    eid_t high = end_v;
    eid_t middle = (low + high) >> 1;
    //在共享内存中缓存二叉搜索树的前5层
    while (wid < 31)
    {
        int n = wap_log[wid];
        for (int i = n - 1; i >= 0; --i)
        {
            if ((wid + 1) >> i & 1) low = middle + 1;
            else high = middle;

            if (low >= high) break;
            middle = (low + high) >> 1;
        }
        if (low >= high) BT_sh[wid] = -1;
        else BT_sh[wid] = gpuadj[middle];
        //__syncthreads();
        wid += WARPSIZE;
    }

    //开始查询
    vid_t sum = 0;
    int shared_count;//设置共享邻居查询索引
    wid = begin_u + threadIdx.x % WARPSIZE;//待定修改
    while (wid < end_u)//待定修改
    {
        search_node[bid] = gpuadj[wid];
        shared_count = 1;
        low = begin_v;
        high = end_v;

        //先在共享内存中找
        for (int i = 0; i < 5; ++i)     //共享内存存储了前5层二叉树
        {
            if (BT_sh[shared_count - 1] == -1)
            {
                break;
            }
            middle = (low + high) >> 1;
            if (search_node[bid] < BT_sh[shared_count - 1])
            {
                high = middle;
                shared_count *= 2;
            }
            else if (search_node[bid] > BT_sh[shared_count - 1])
            {
                low = middle + 1;
                shared_count = shared_count * 2 + 1;
            }
            else
            {
                sum++;
                low = high;
                break;
            }
        }
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

__global__ void __check_core(const eid_t elength, const eid_t* __restrict__ gpunum_edges, const double eps, const int miu, 
    const vid_t* __restrict__ gpuadj, const vid_t* __restrict__ E_u, const vid_t* __restrict__ E_v, char* __restrict__ similar,
    vid_t* __restrict__ role, vid_t* __restrict__ sd, vid_t* __restrict__ ed, const int* __restrict__ wap_log, 
    const vid_t* __restrict__ gpu_ni) {

    __shared__ vid_t Binary_Tree[cc_BLOCKSIZE / WARPSIZE][WARPSIZE];    //存储二叉搜索树的前k层，k=5
    __shared__ eid_t begin_u[cc_BLOCKSIZE / WARPSIZE][WARPSIZE];
    __shared__ eid_t begin_v[cc_BLOCKSIZE / WARPSIZE][WARPSIZE];
    __shared__ eid_t end_u[cc_BLOCKSIZE / WARPSIZE][WARPSIZE];
    __shared__ eid_t end_v[cc_BLOCKSIZE / WARPSIZE][WARPSIZE];
    __shared__ int wap_logsh[cc_BLOCKSIZE / WARPSIZE][WARPSIZE];
    __shared__ vid_t TriangleSum[cc_BLOCKSIZE / WARPSIZE];
    __shared__ vid_t search_node[cc_BLOCKSIZE];
    __shared__ vid_t u_ni[cc_BLOCKSIZE / WARPSIZE][WARPSIZE];
    __shared__ vid_t v_ni[cc_BLOCKSIZE / WARPSIZE][WARPSIZE];

    //int tid = threadIdx.x + blockIdx.x * blockDim.x; //thread id
    int wid = (threadIdx.x + blockIdx.x * blockDim.x) % WARPSIZE;                   //Warp内索引
    //int WarpIdx_global = tid / WARPSIZE;       //全局Warp索引
    int WarpIdx_block = threadIdx.x / WARPSIZE;    //线程块内的Warp索引
    int WarpNumb = blockDim.x * gridDim.x / WARPSIZE;    //全局的Warp数
    eid_t Proc_Numb, Proc_it;                           //一个Warp处理的边的数目,边表处理起始位置标识
    vid_t u_vertex, v_vertex;                                 //当前warp查询的点u,v
    int num_cache = 0;                                      //mod 32

    wap_logsh[WarpIdx_block][wid] = wap_log[wid];
    if (WarpNumb < elength) {
        if (elength % WarpNumb == 0) {
            Proc_Numb = elength / WarpNumb;
        }
        else Proc_Numb = elength / WarpNumb + 1;
    }
    else Proc_Numb = 1;
    Proc_it = ((threadIdx.x + blockIdx.x * blockDim.x) / WARPSIZE) * Proc_Numb;

    for (eid_t k = 0; k < Proc_Numb; ++k)
    {
        if (Proc_it > elength - 1) break;
        if (num_cache % WARPSIZE == 0) {
            if (Proc_it + wid <= elength - 1) {
                u_ni[WarpIdx_block][wid] = gpu_ni[E_u[Proc_it + wid]];
                v_ni[WarpIdx_block][wid] = gpu_ni[E_v[Proc_it + wid]];
                begin_u[WarpIdx_block][wid] = gpunum_edges[gpu_ni[E_u[Proc_it + wid]]];
                end_u[WarpIdx_block][wid] = gpunum_edges[gpu_ni[E_u[Proc_it + wid]] + 1];
                begin_v[WarpIdx_block][wid] = gpunum_edges[gpu_ni[E_v[Proc_it + wid]]];
                end_v[WarpIdx_block][wid] = gpunum_edges[gpu_ni[E_v[Proc_it + wid]] + 1];
            }
        }
        u_vertex = E_u[Proc_it];
        v_vertex = E_v[Proc_it];
        //如果u和v相似度没被计算切u和v至少有一个role没被确定
        if (similar[Proc_it] == 'U' && (role[u_ni[WarpIdx_block][num_cache]] == -1 ||
            role[v_ni[WarpIdx_block][num_cache]] == -1))
        {
            //compute similar
            bool unsimilar;
            unsimilar = intersect(eps, gpuadj, begin_u[WarpIdx_block][num_cache],
                end_u[WarpIdx_block][num_cache], begin_v[WarpIdx_block][num_cache],
                end_v[WarpIdx_block][num_cache], Binary_Tree[WarpIdx_block],
                wap_logsh[WarpIdx_block], TriangleSum, search_node);

            if (unsimilar)
            {
                similar[Proc_it] = 'N';
                if (wid == 0) {
                    if (role[u_ni[WarpIdx_block][num_cache]] == -1)
                    {
                        atomicSub(&ed[u_ni[WarpIdx_block][num_cache]], 1);
                        if (ed[u_ni[WarpIdx_block][num_cache]] < miu)
                        {
                            role[u_ni[WarpIdx_block][num_cache]] = -2;//cons
                            //atomicExch(&role[u_vertex], -2);
                        }
                    }
                    if (role[v_ni[WarpIdx_block][num_cache]] == -1)
                    {
                        atomicSub(&ed[v_ni[WarpIdx_block][num_cache]], 1);
                        if (ed[v_ni[WarpIdx_block][num_cache]] < miu)
                        {
                            role[v_ni[WarpIdx_block][num_cache]] = -2;//cons
                            //atomicExch(&role[v_vertex], -2);
                        }
                    }
                }
            }
            else
            {
                similar[Proc_it] = 'Y';
                if (wid == 0) {
                    if (role[u_ni[WarpIdx_block][num_cache]] == -1)
                    {
                        atomicAdd(&sd[u_ni[WarpIdx_block][num_cache]], 1);
                        if (sd[u_ni[WarpIdx_block][num_cache]] >= miu)
                        {
                            role[u_ni[WarpIdx_block][num_cache]] = u_vertex;//cons
                            //atomicExch(&role[u_vertex], u_vertex);
                        }
                    }
                    if (role[v_ni[WarpIdx_block][num_cache]] == -1)
                    {
                        atomicAdd(&sd[v_ni[WarpIdx_block][num_cache]], 1);
                        if (sd[v_ni[WarpIdx_block][num_cache]] >= miu)
                        {
                            role[v_ni[WarpIdx_block][num_cache]] = v_vertex;//cons
                            //atomicExch(&role[v_vertex], v_vertex);
                        }
                    }
                }
            }
        }
        Proc_it++;//开始计算边表中下一条边
        num_cache = (num_cache + 1) % WARPSIZE;
    }
}


//__global__ void __init_cluster(const int miu, const int core_number, const unsigned int* __restrict__ gpunum_edges,
//    const int* __restrict__ gpuadj, int* const __restrict__ role, const int* __restrict__ gpucore_node) {
//
//    int threadCount = blockDim.x * gridDim.x;        //thread sum
//    int tid = threadIdx.x + blockIdx.x * blockDim.x; //thread id
//
//    for (int v = tid; v < core_number; v += threadCount)
//    {
//        const int beg = gpunum_edges[v];
//        const int end = gpunum_edges[v + 1];
//        int r1 = role[v];
//        int i = beg;
//        while (i < end) {
//            int r2 = role[gpuadj[i]];
//            if ((r2 >= 0) && (r2 < r1)) {
//                r1 = r2;
//                role[v] = r1;
//            }
//            i++;
//        }
//    }
//}


inline __device__ vid_t findroot(const vid_t idx, vid_t* const __restrict__ role) {

    vid_t curr = role[idx];
    if (curr != idx) {
        vid_t next, prev = idx;
        while (curr > (next = role[curr])) {
            role[prev] = next;
            prev = curr;
            curr = next;
        }
    }
    return curr;
}

__global__ void __cluster_core1(char* const __restrict__ similar, vid_t* const __restrict__ role,
    const vid_t* __restrict__ E_u, const vid_t* __restrict__ E_v, const eid_t elength) {

    eid_t threadCount = blockDim.x * gridDim.x;        //thread sum
    eid_t tid = threadIdx.x + blockIdx.x * blockDim.x; //thread id
    //const int offset = (threadIdx.x + blockIdx.x * blockDim.x) % WARPSIZE;                   //Warp内索引
    //vid_t WarpIdx_global = (threadIdx.x + blockIdx.x * blockDim.x) / WARPSIZE;       //全局Warp索引
    //vid_t WarpNumb = blockDim.x * gridDim.x / WARPSIZE;    //全局的Warp数
    //WarpIdx_global += bp;
    //while (WarpIdx_global < ep)
    //{
    //    const vid_t u_vertex = gpucore_node[WarpIdx_global];
    //    vid_t u_parent = findroot(u_vertex, role);
    //    for (auto i = estart[u_vertex - cut] + offset; i < estart[u_vertex - cut + 1]; i += WARPSIZE)
    //    {
    //        const vid_t v_vertex = E_v[i];
    //        if (role[v_vertex] >= 0 && similar[i] == 'Y')
    //        {
    //            vid_t v_parent = findroot(v_vertex, role);
    //            bool repeat;
    //            do
    //            {
    //                repeat = false;
    //                if (u_parent != v_parent)
    //                {
    //                    vid_t ret;
    //                    if (u_parent < v_parent)
    //                    {
    //                        if ((ret = atomicCAS(&role[v_parent], v_parent, u_parent)) != v_parent)
    //                        {
    //                            v_parent = ret;
    //                            repeat = true;
    //                        }
    //                    }
    //                    else
    //                    {
    //                        if ((ret = atomicCAS(&role[u_parent], u_parent, v_parent)) != u_parent)
    //                        {
    //                            u_parent = ret;
    //                            repeat = true;
    //                        }
    //                    }
    //                }
    //            } while (repeat);
    //        }
    //    }
    //    WarpIdx_global += WarpNumb;
    //}
    while (tid < elength)
    {
        vid_t u_vertex = E_u[tid];
        vid_t v_vertex = E_v[tid];
        if ((role[u_vertex] >= 0) && (role[v_vertex] >= 0) && similar[tid] == 'Y')
        {
            vid_t u_parent = findroot(u_vertex, role);
            vid_t v_parent = findroot(v_vertex, role);
            bool repeat;
            do
            {
                repeat = false;
                if (u_parent != v_parent)
                {
                    vid_t ret;
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
        else if ((role[u_vertex] >= 0) && (role[v_vertex] >= 0) && similar[tid] == 'U')
        {
            similar[tid] = 'V';
        }
        else if (role[u_vertex] >= 0 && role[v_vertex] < 0 && similar[tid] == 'Y')
        {
            similar[tid] = 'Z';
        }
        else if (role[u_vertex] < 0 && role[v_vertex] >= 0 && similar[tid] == 'Y')
        {
            similar[tid] = 'z';
        }
        else if (role[u_vertex] >= 0 && role[v_vertex] < 0 && similar[tid] == 'U')
        {
            similar[tid] = 'W';
        }
        else if (role[u_vertex] < 0 && role[v_vertex] >= 0 && similar[tid] == 'U')
        {
            similar[tid] = 'w';
        }
        tid += threadCount;
    }
}

__global__ void __cluster_core2(const double eps, const eid_t* __restrict__ gpunum_edges, const eid_t elength,
    const vid_t* __restrict__ gpuadj, char* const __restrict__ similar, vid_t* const __restrict__ role, 
    const vid_t* __restrict__ E_u, const vid_t* __restrict__ E_v, const int* __restrict__ wap_log, 
    const vid_t* __restrict__ gpu_ni) {

    __shared__ vid_t Binary_Tree[cc_BLOCKSIZE / WARPSIZE][WARPSIZE];    //存储二叉搜索树的前k层，k=5
    __shared__ int wap_logsh[cc_BLOCKSIZE / WARPSIZE][WARPSIZE];
    __shared__ eid_t begin_u[cc_BLOCKSIZE / WARPSIZE][WARPSIZE];
    __shared__ eid_t begin_v[cc_BLOCKSIZE / WARPSIZE][WARPSIZE];
    __shared__ eid_t end_u[cc_BLOCKSIZE / WARPSIZE][WARPSIZE];
    __shared__ eid_t end_v[cc_BLOCKSIZE / WARPSIZE][WARPSIZE];
    __shared__ vid_t TriangleSum[cc_BLOCKSIZE / WARPSIZE];
    __shared__ vid_t search_node[cc_BLOCKSIZE];

    int wid = (threadIdx.x + blockIdx.x * blockDim.x) % WARPSIZE;                   //Warp内索引
    int WarpIdx_block = threadIdx.x / WARPSIZE;    //线程块内的Warp索引
    int WarpNumb = blockDim.x * gridDim.x / WARPSIZE;    //全局的Warp数
    eid_t Proc_Numb, Proc_it;
    vid_t u_vertex, v_vertex;
    int num_cache = 0;                                   //mod 32

    wap_logsh[WarpIdx_block][wid] = wap_log[wid];
    if (WarpNumb < elength) {
        if (elength % WarpNumb == 0) {
            Proc_Numb = elength / WarpNumb;
        }
        else Proc_Numb = elength / WarpNumb + 1;
    }
    else Proc_Numb = 1;
    Proc_it = ((threadIdx.x + blockIdx.x * blockDim.x) / WARPSIZE) * Proc_Numb;

    for (eid_t k = 0; k < Proc_Numb; ++k)
    {
        if (Proc_it > elength - 1) break;
        if (num_cache % WARPSIZE == 0) {
            if (Proc_it + wid <= elength - 1) {
                begin_u[WarpIdx_block][wid] = gpunum_edges[gpu_ni[E_u[Proc_it + wid]]];
                end_u[WarpIdx_block][wid] = gpunum_edges[gpu_ni[E_u[Proc_it + wid]] + 1];
                begin_v[WarpIdx_block][wid] = gpunum_edges[gpu_ni[E_v[Proc_it + wid]]];
                end_v[WarpIdx_block][wid] = gpunum_edges[gpu_ni[E_v[Proc_it + wid]] + 1];
            }
        }
        u_vertex = E_u[Proc_it];
        v_vertex = E_v[Proc_it];
        vid_t u_parent = findroot(u_vertex, role);
        vid_t v_parent = findroot(v_vertex, role);
        if (u_parent != v_parent)
        {
            bool unsimilar;
            unsimilar = intersect(eps, gpuadj, begin_u[WarpIdx_block][num_cache], end_u[WarpIdx_block][num_cache],
                begin_v[WarpIdx_block][num_cache], end_v[WarpIdx_block][num_cache], Binary_Tree[WarpIdx_block],
                wap_logsh[WarpIdx_block], TriangleSum, search_node);
            if (unsimilar)
            {
                similar[Proc_it] = 'N';
            }
            else
            {
                similar[Proc_it] = 'Y';
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
        Proc_it++;//开始计算边表中下一条边
        num_cache = (num_cache + 1) % WARPSIZE;
    }
}

//__global__ void __cluster_core2(const double eps, const eid_t* __restrict__ gpunum_edges,
//    const vid_t* __restrict__ gpuadj, char* const __restrict__ similar, vid_t* const __restrict__ role, 
//    const vid_t* __restrict__ gpucore_node, const eid_t* __restrict__ estart, 
//    const int* __restrict__ wap_log, const vid_t bp, const vid_t ep, const vid_t cut) {
//
//    __shared__ vid_t Binary_Tree[cc_BLOCKSIZE / WARPSIZE][WARPSIZE];    //存储二叉搜索树的前k层，k=5
//    __shared__ int wap_logsh[cc_BLOCKSIZE / WARPSIZE][WARPSIZE];
//    __shared__ eid_t begin_u[cc_BLOCKSIZE / WARPSIZE];
//    __shared__ eid_t end_u[cc_BLOCKSIZE / WARPSIZE];
//    __shared__ eid_t begin_v[cc_BLOCKSIZE / WARPSIZE][WARPSIZE];
//    __shared__ eid_t end_v[cc_BLOCKSIZE / WARPSIZE][WARPSIZE];
//    __shared__ vid_t role_sh[cc_BLOCKSIZE / WARPSIZE][WARPSIZE];
//    __shared__ char similar_sh[cc_BLOCKSIZE / WARPSIZE][WARPSIZE];
//    __shared__ vid_t TriangleSum[cc_BLOCKSIZE / WARPSIZE];
//    __shared__ vid_t search_node[cc_BLOCKSIZE];
//
//    int wid = (threadIdx.x + blockIdx.x * blockDim.x) % WARPSIZE;                   //Warp内索引
//    vid_t WarpIdx_global = (threadIdx.x + blockIdx.x * blockDim.x) / WARPSIZE;       //全局Warp索引
//    int WarpIdx_block = threadIdx.x / WARPSIZE;    //线程块内的Warp索引
//    vid_t WarpNumb = blockDim.x * gridDim.x / WARPSIZE;    //全局的Warp数
//    int num_cache;                                   //mod 32
//
//    wap_logsh[WarpIdx_block][wid] = wap_log[wid];
//    WarpIdx_global += bp;
//    while (WarpIdx_global < ep)
//    {
//        const int u_vertex = gpucore_node[WarpIdx_global];
//        int u_parent = findroot(u_vertex, role);
//        begin_u[WarpIdx_block] = gpunum_edges[u_vertex];
//        end_u[WarpIdx_block] = gpunum_edges[u_vertex + 1];
//        num_cache = 0;
//        for (int offset = begin_u[WarpIdx_block]; offset < end_u[WarpIdx_block]; offset++)
//        {
//            const int v_vertex = gpuadj[offset];
//            if (v_vertex >= u_vertex) break;
//            if (num_cache % WARPSIZE == 0) {
//                if (offset + wid < end_u[WarpIdx_block]) {
//                    begin_v[WarpIdx_block][wid] = gpunum_edges[gpuadj[offset + wid]];
//                    end_v[WarpIdx_block][wid] = gpunum_edges[gpuadj[offset + wid] + 1];
//                    role_sh[WarpIdx_block][wid] = role[gpuadj[offset + wid]];
//                    similar_sh[WarpIdx_block][wid] = similar[gpuEid[offset + wid]];
//                }
//            }
//            if (role_sh[WarpIdx_block][num_cache] >= 0 && similar_sh[WarpIdx_block][num_cache] == 'U')
//            {
//                //is same set?
//                int v_parent = findroot(v_vertex, role);
//                if (u_parent != v_parent)
//                {
//                    bool unsimilar;
//                    if (end_v[WarpIdx_block][num_cache] - begin_v[WarpIdx_block][num_cache] > end_u[WarpIdx_block] - begin_u[WarpIdx_block])
//                    {
//                        unsimilar = intersect(eps, gpuadj, begin_u[WarpIdx_block], end_u[WarpIdx_block], begin_v[WarpIdx_block][num_cache],
//                            end_v[WarpIdx_block][num_cache], Binary_Tree[WarpIdx_block], wap_logsh[WarpIdx_block], TriangleSum, search_node);
//                    }
//                    else {
//                        unsimilar = intersect(eps, gpuadj, begin_v[WarpIdx_block][num_cache], end_v[WarpIdx_block][num_cache], begin_u[WarpIdx_block],
//                            end_u[WarpIdx_block], Binary_Tree[WarpIdx_block], wap_logsh[WarpIdx_block], TriangleSum, search_node);
//                    }
//                    if (unsimilar)
//                    {
//                        similar[gpuEid[offset]] = 'N';
//                    }
//                    else
//                    {
//                        similar[gpuEid[offset]] = 'Y';
//                        u_parent = findroot(u_vertex, role);
//                        v_parent = findroot(v_vertex, role);
//                        if (wid == 0)
//                        {
//                            bool repeat;
//                            do
//                            {
//                                repeat = false;
//                                if (u_parent != v_parent)
//                                {
//                                    int ret;
//                                    if (u_parent < v_parent)
//                                    {
//                                        if ((ret = atomicCAS(&role[v_parent], v_parent, u_parent)) != v_parent)
//                                        {
//                                            v_parent = ret;
//                                            repeat = true;
//                                        }
//                                    }
//                                    else
//                                    {
//                                        if ((ret = atomicCAS(&role[u_parent], u_parent, v_parent)) != u_parent)
//                                        {
//                                            u_parent = ret;
//                                            repeat = true;
//                                        }
//                                    }
//                                }
//                            } while (repeat);
//                        }
//                    }
//                }
//            }
//            num_cache = (num_cache + 1) % WARPSIZE;
//        }
//        WarpIdx_global += WarpNumb;
//    }
//}

__global__ void __cluster_core(vid_t* const __restrict__ role, const vid_t vlength) {

    vid_t threadCount = blockDim.x * gridDim.x;        //thread sum
    vid_t tid = threadIdx.x + blockIdx.x * blockDim.x; //thread id
    vid_t Parent;
    //vid_t coreid;
    while (tid < vlength)
    {
        Parent = role[tid];
        if (Parent >= 0)
        {
            //Parent = role[coreid];
            if (Parent != role[Parent])
            {
                while (Parent != role[Parent])
                {
                    Parent = role[Parent];
                }
                atomicExch(&role[tid], Parent);
            }
        }
        tid += threadCount;
    }
}


__global__ void __cluster_noncore(const double eps, const eid_t elength, const eid_t* __restrict__ gpunum_edges, 
    const vid_t* __restrict__ gpuadj, char* const __restrict__ similar, vid_t* const __restrict__ role, 
    const vid_t* __restrict__ E_u, const vid_t* __restrict__ E_v, int* wap_log, const vid_t* __restrict__ gpu_ni) {

    __shared__ vid_t Binary_Tree[cc_BLOCKSIZE / WARPSIZE][WARPSIZE];    //存储二叉搜索树的前k层，k=5
    __shared__ int wap_logsh[cc_BLOCKSIZE / WARPSIZE][WARPSIZE];
    __shared__ eid_t begin_u[cc_BLOCKSIZE / WARPSIZE][WARPSIZE];
    __shared__ eid_t begin_v[cc_BLOCKSIZE / WARPSIZE][WARPSIZE];
    __shared__ eid_t end_u[cc_BLOCKSIZE / WARPSIZE][WARPSIZE];
    __shared__ eid_t end_v[cc_BLOCKSIZE / WARPSIZE][WARPSIZE];
    __shared__ char similar_sh[cc_BLOCKSIZE / WARPSIZE][WARPSIZE];
    __shared__ vid_t TriangleSum[cc_BLOCKSIZE / WARPSIZE];
    __shared__ vid_t search_node[cc_BLOCKSIZE];

    int wid = (threadIdx.x + blockIdx.x * blockDim.x) % WARPSIZE;                   //Warp内索引
    int WarpIdx_block = threadIdx.x / WARPSIZE;    //线程块内的Warp索引
    int WarpNumb = blockDim.x * gridDim.x / WARPSIZE;    //全局的Warp数
    eid_t Proc_Numb, Proc_it;
    vid_t u_vertex, v_vertex;
    int num_cache = 0;                                   //mod 32

    wap_logsh[WarpIdx_block][wid] = wap_log[wid];
    if (WarpNumb < elength) {
        if (elength % WarpNumb == 0) {
            Proc_Numb = elength / WarpNumb;
        }
        else Proc_Numb = elength / WarpNumb + 1;
    }
    else Proc_Numb = 1;
    Proc_it = ((threadIdx.x + blockIdx.x * blockDim.x) / WARPSIZE) * Proc_Numb;

    for (eid_t k = 0; k < Proc_Numb; ++k)
    {
        if (Proc_it > elength - 1) break;
        if (num_cache % WARPSIZE == 0) {
            if (Proc_it + wid <= elength - 1) {
                begin_u[WarpIdx_block][wid] = gpunum_edges[gpu_ni[E_u[Proc_it + wid]]];
                end_u[WarpIdx_block][wid] = gpunum_edges[gpu_ni[E_u[Proc_it + wid]] + 1];
                begin_v[WarpIdx_block][wid] = gpunum_edges[gpu_ni[E_v[Proc_it + wid]]];
                end_v[WarpIdx_block][wid] = gpunum_edges[gpu_ni[E_v[Proc_it + wid]] + 1];
                similar_sh[WarpIdx_block][wid] = similar[Proc_it + wid];
            }
        }
        u_vertex = E_u[Proc_it];
        v_vertex = E_v[Proc_it];
        vid_t core, nocore;
        if (similar_sh[WarpIdx_block][num_cache] == 'Z' || similar_sh[WarpIdx_block][num_cache] == 'W') {
            core = u_vertex;
            nocore = v_vertex;
        }
        else {
            core = v_vertex;
            nocore = u_vertex;
        }

        //if (role[nocore] >= 0 ) continue;
        if (similar_sh[WarpIdx_block][num_cache] == 'Z' || similar_sh[WarpIdx_block][num_cache] == 'z')
        {
            if (role[nocore] < 0) {
                if (wid == 0) {
                    atomicExch(&role[nocore], role[core]);//如果已经计算过且相似则可以直接聚类
                }
            }
        }
        if (similar_sh[WarpIdx_block][num_cache] == 'W' || similar_sh[WarpIdx_block][num_cache] == 'w')
        {
            if (role[nocore] < 0) {
                bool unsimilar;
                unsimilar = intersect(eps, gpuadj, begin_u[WarpIdx_block][num_cache], end_u[WarpIdx_block][num_cache],
                    begin_v[WarpIdx_block][num_cache], end_v[WarpIdx_block][num_cache], Binary_Tree[WarpIdx_block],
                    wap_logsh[WarpIdx_block], TriangleSum, search_node);
                if (unsimilar)
                {
                    similar[Proc_it] = 'N';
                }
                else
                {
                    similar[Proc_it] = 'Z';
                    if (wid == 0) {
                        atomicExch(&role[nocore], role[core]);
                    }
                }
            }
        }
        Proc_it++;//开始计算边表中下一条边
        num_cache = (num_cache + 1) % WARPSIZE;
    }
}

__global__ void __classify_others(vid_t* const __restrict__ role, const eid_t* __restrict__ gpunum_edges, 
    const vid_t* __restrict__ gpuadj, const vid_t* __restrict__ nodes, bool* const __restrict__ hc, 
    const vid_t* __restrict__ gpu_ni, vid_t vlength) {

    __shared__ vid_t adjrole[ii_BLOCKSIZE / WARPSIZE][WARPSIZE];
    __shared__ bool checkflag[ii_BLOCKSIZE / WARPSIZE];
    eid_t begin, end;

    int wid = (threadIdx.x + blockIdx.x * blockDim.x) % WARPSIZE;                   //Warp内索引
    vid_t WarpIdx_global = (threadIdx.x + blockIdx.x * blockDim.x) / WARPSIZE;       //全局Warp索引
    int WarpIdx_block = threadIdx.x / WARPSIZE;    //线程块内的Warp索引
    vid_t WarpNumb = blockDim.x * gridDim.x / WARPSIZE;    //全局的Warp数

    checkflag[WarpIdx_block] = false;
    for (vid_t i = WarpIdx_global; i < vlength; i += WarpNumb)
    {
        vid_t u = nodes[i];
        if (role[u] < 0 && hc[u]==false)
        {
            begin = gpunum_edges[gpu_ni[u]];
            end = gpunum_edges[gpu_ni[u] + 1];
            if (end - begin > 1)
            {
                eid_t wid_t = begin + wid;
                while (!checkflag[WarpIdx_block] && wid_t < end)
                {
                    adjrole[WarpIdx_block][wid] = role[gpuadj[wid_t]];
                    if (wid > 0) {
                        if (adjrole[WarpIdx_block][wid] >= 0 && adjrole[WarpIdx_block][wid - 1] >= 0 &&
                            adjrole[WarpIdx_block][wid] != adjrole[WarpIdx_block][wid - 1]) {
                            checkflag[WarpIdx_block] = true;
                            role[u] = -1;
                        }
                    }
                    wid_t += WARPSIZE;
                }
            }
            hc[u] = true;
        }
    }
}

void _prepEdge(GraphChip& gc, vid_t* cpurole, vid_t* cpusd, vid_t* cpued, vid_t vl, 
    int miu, double eps) {

    uint32_t  ii_ThreadBlockCount = TOTALTHDCOUNT / ii_BLOCKSIZE;
    //uint32_t  cc_ThreadBlockCount = TOTALTHDCOUNT / cc_BLOCKSIZE;

    vid_t* E_u;
    vid_t* E_v;
    char* similar;
    vid_t* role;
    vid_t* sd;
    vid_t* ed;
    eid_t* gpunum_edges;
    vid_t* gpu_ni;

    vid_t* role_copy = (vid_t*)malloc(gc.n * sizeof(vid_t));
    vid_t* sd_copy = (vid_t*)malloc(gc.n * sizeof(vid_t));
    vid_t* ed_copy = (vid_t*)malloc(gc.n * sizeof(vid_t));
#pragma omp parallel for
    for (auto j = 0; j < gc.n; j++)
    {
        auto u = gc.nodes[j];
        role_copy[j] = cpurole[u];
        sd_copy[j] = cpusd[u];
        ed_copy[j] = cpued[u];
    }

    gpuErrchk(cudaMalloc((void**)&E_u, gc.m * sizeof(vid_t)));
    gpuErrchk(cudaMalloc((void**)&E_v, gc.m * sizeof(vid_t)));
    gpuErrchk(cudaMalloc((void**)&similar, gc.m * sizeof(char)));
    gpuErrchk(cudaMalloc((void**)&role, gc.n * sizeof(vid_t)));
    gpuErrchk(cudaMalloc((void**)&sd, gc.n * sizeof(vid_t)));
    gpuErrchk(cudaMalloc((void**)&ed, gc.n * sizeof(vid_t)));
    gpuErrchk(cudaMalloc((void**)&gpunum_edges, (gc.n + 1) * sizeof(eid_t)));
    gpuErrchk(cudaMalloc((void**)&gpu_ni, vl * sizeof(vid_t)));

    gpuErrchk(cudaMemcpy(gpunum_edges, gc.num_edges, (gc.n + 1) * sizeof(eid_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(role, role_copy, gc.n * sizeof(vid_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(sd, sd_copy, gc.n * sizeof(vid_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(ed, ed_copy, gc.n * sizeof(vid_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(gpu_ni, gc.node_index, vl * sizeof(vid_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(E_u, gc.el_u, gc.m * sizeof(vid_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(E_v, gc.el_v, gc.m * sizeof(vid_t), cudaMemcpyHostToDevice));

    __prep_edge << <ii_ThreadBlockCount, ii_BLOCKSIZE >> > (similar, gpunum_edges, E_u, E_v, eps, miu, gc.m,
        ed, sd, role, gpu_ni);

    gpuErrchk(cudaMemcpy(gc.similar, similar, gc.m * sizeof(char), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(role_copy, role, gc.n * sizeof(vid_t), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(sd_copy, sd, gc.n * sizeof(vid_t), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(ed_copy, ed, gc.n * sizeof(vid_t), cudaMemcpyDeviceToHost));
#pragma omp parallel for
    for (auto j = 0; j < gc.n; j++)
    {
        auto u = gc.nodes[j];
        cpurole[u] = role_copy[j];
        cpusd[u] = sd_copy[j];
        cpued[u] = ed_copy[j];
    }
    gpuErrchk(cudaFree(E_u));
    gpuErrchk(cudaFree(E_v));
    gpuErrchk(cudaFree(similar));
    gpuErrchk(cudaFree(role));
    gpuErrchk(cudaFree(sd));
    gpuErrchk(cudaFree(ed));
    gpuErrchk(cudaFree(gpunum_edges));
    free(role_copy);
    free(sd_copy);
    free(ed_copy);
}

vid_t* GPUScan(graph_t* g, eid_t* num_edges_copy, vid_t* Edge_u, vid_t* Edge_v, double eps, int miu) {
    //init GPU
    cudaDeviceProp deviceProp;
    gpuErrchk(cudaGetDeviceProperties(&deviceProp, 0));
    
    uint32_t  ii_ThreadBlockCount = TOTALTHDCOUNT / ii_BLOCKSIZE;
    uint32_t  cc_ThreadBlockCount = TOTALTHDCOUNT / cc_BLOCKSIZE;

    vid_t* cpurole = (vid_t*)malloc(g->n * sizeof(vid_t));
    vid_t* cpusd = (vid_t*)malloc(g->n * sizeof(vid_t));
    vid_t* cpued = (vid_t*)malloc(g->n * sizeof(vid_t));
    vid_t* role1;
    vid_t* sd1;
    vid_t* ed1;
    eid_t* gpunum_edges1;
    gpuErrchk(cudaMalloc((void**)&gpunum_edges1, (g->n + 1) * sizeof(eid_t)));
    gpuErrchk(cudaMalloc((void**)&role1, g->n * sizeof(vid_t)));
    gpuErrchk(cudaMalloc((void**)&sd1, g->n * sizeof(vid_t)));
    gpuErrchk(cudaMalloc((void**)&ed1, g->n * sizeof(vid_t)));
    gpuErrchk(cudaMemcpy(gpunum_edges1, g->num_edges, (g->n + 1) * sizeof(eid_t), cudaMemcpyHostToDevice));
    fprintf(stderr,"begin initialize\n");
    __init_input <<<ii_ThreadBlockCount, ii_BLOCKSIZE>>> (role1, sd1, ed1, gpunum_edges1, g->n, miu);
    fprintf(stderr,"finish initialize\n");
    gpuErrchk(cudaMemcpy(cpurole, role1, g->n * sizeof(vid_t), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(cpusd, sd1, g->n * sizeof(vid_t), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(cpued, ed1, g->n * sizeof(vid_t), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(role1));
    gpuErrchk(cudaFree(sd1));
    gpuErrchk(cudaFree(ed1));
    gpuErrchk(cudaFree(gpunum_edges1));

    //preprocessing;
    vid_t* gpu_ni;
    int* wap_log;

    //init input
    int* cpuwap_log = (int*)malloc(32 * sizeof(int));
#pragma omp parallel
    {
#pragma omp for
        for (int i = 0; i < 32; i++) {
            cpuwap_log[i] = (int)(log(i + 1) / log(2));
        }
    }
    gpuErrchk(cudaMalloc((void**)&wap_log, 32 * sizeof(int)));
    gpuErrchk(cudaMemcpy(wap_log, cpuwap_log, 32 * sizeof(int), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc((void**)&gpu_ni, (g->n) * sizeof(vid_t)));

    //divide graph
    auto Dstart = high_resolution_clock::now();
    fprintf(stderr,"begin divide\n");
    vector<GraphChip*> gc;
    int pn = 0;
    size_t avail(0);
    size_t total(0);
    gpuErrchk(cudaMemGetInfo(&avail,&total));
    GraphDivide(g, Edge_u, Edge_v, avail-8388608, gc, pn);//prefrech 8m
    fprintf(stderr,"finish divide\nload graph edges and first 20 CSR\n");
    //load graphchip
    LoadGraphEdge(gc, g, Edge_u, Edge_v, pn);
    //fprintf(stderr,"finish load edges\nload graph CSR\n");
    
    auto Dend = high_resolution_clock::now();
    fprintf(stderr, "Divide time: %.3lf s\n", duration_cast<milliseconds>(Dend - Dstart).count() / 1000.0);

    //initialize
    for (auto i = 0; i < pn; i++)
    {
        vid_t* E_u;
        vid_t* E_v;
        char* similar;
        vid_t* role;
        vid_t* sd;
        vid_t* ed;
        eid_t* gpunum_edges;
        if(i>=20){
        	loadSpeGraphChip(gc,g,i);
        }

        vid_t* role_copy = (vid_t*)malloc(gc[i]->n * sizeof(vid_t));
        vid_t* sd_copy = (vid_t*)malloc(gc[i]->n * sizeof(vid_t));
        vid_t* ed_copy = (vid_t*)malloc(gc[i]->n * sizeof(vid_t));
#pragma omp parallel for
        for (auto j = 0; j < gc[i]->n; j++)
        {
            auto u = gc[i]->nodes[j];
            role_copy[j] = cpurole[u];
            sd_copy[j] = cpusd[u];
            ed_copy[j] = cpued[u];
        }

        gpuErrchk(cudaMalloc((void**)&E_u, gc[i]->m * sizeof(vid_t)));
        gpuErrchk(cudaMalloc((void**)&E_v, gc[i]->m * sizeof(vid_t)));
        gpuErrchk(cudaMalloc((void**)&similar, gc[i]->m * sizeof(char)));
        gpuErrchk(cudaMalloc((void**)&role, gc[i]->n * sizeof(vid_t)));
        gpuErrchk(cudaMalloc((void**)&sd, gc[i]->n * sizeof(vid_t)));
        gpuErrchk(cudaMalloc((void**)&ed, gc[i]->n * sizeof(vid_t)));
        gpuErrchk(cudaMalloc((void**)&gpunum_edges, (gc[i]->n + 1) * sizeof(eid_t)));

        gpuErrchk(cudaMemcpy(gpunum_edges, gc[i]->num_edges, (gc[i]->n + 1) * sizeof(eid_t), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(role, role_copy, gc[i]->n * sizeof(vid_t), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(sd, sd_copy, gc[i]->n * sizeof(vid_t), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(ed, ed_copy, gc[i]->n * sizeof(vid_t), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(gpu_ni, gc[i]->node_index, g->n * sizeof(vid_t), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(E_u, gc[i]->el_u, gc[i]->m * sizeof(vid_t), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(E_v, gc[i]->el_v, gc[i]->m * sizeof(vid_t), cudaMemcpyHostToDevice));

        __prep_edge <<<ii_ThreadBlockCount, ii_BLOCKSIZE>>> (similar, gpunum_edges, E_u, E_v, eps, miu, gc[i]->m, 
            ed, sd, role, gpu_ni);

        gpuErrchk(cudaMemcpy(gc[i]->similar, similar, gc[i]->m * sizeof(char), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(role_copy, role, gc[i]->n * sizeof(vid_t), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(sd_copy, sd, gc[i]->n * sizeof(vid_t), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(ed_copy, ed, gc[i]->n * sizeof(vid_t), cudaMemcpyDeviceToHost));
#pragma omp parallel for
        for (auto j = 0; j < gc[i]->n; j++)
        {
            auto u = gc[i]->nodes[j];
            cpurole[u] = role_copy[j];
            cpusd[u] = sd_copy[j];
            cpued[u] = ed_copy[j];
        }
        gpuErrchk(cudaFree(E_u));
        gpuErrchk(cudaFree(E_v));
        gpuErrchk(cudaFree(similar));
        gpuErrchk(cudaFree(role));
        gpuErrchk(cudaFree(sd));
        gpuErrchk(cudaFree(ed));
        gpuErrchk(cudaFree(gpunum_edges));
        free(role_copy);
        free(sd_copy);
        free(ed_copy);
        if(i>=20){
			freeSpeGraphChip(gc,i);
		}
    }
    auto Dend2 = high_resolution_clock::now();
    fprintf(stderr, "Initialize time: %.3lf s\n", duration_cast<milliseconds>(Dend2 - Dend).count() / 1000.0);
    
    //check core
    for (auto i = 0; i < pn; i++)
    {
        vid_t* gpuadj;
        vid_t* E_u;
        vid_t* E_v;
        char* similar;
        vid_t* role;
        vid_t* sd;
        vid_t* ed;
        eid_t* gpunum_edges;
        
        if(i>=20){
        	loadSpeGraphChip(gc,g,i);
        }
        
        vid_t* role_copy = (vid_t*)malloc(gc[i]->n * sizeof(vid_t));
        vid_t* sd_copy = (vid_t*)malloc(gc[i]->n * sizeof(vid_t));
        vid_t* ed_copy = (vid_t*)malloc(gc[i]->n * sizeof(vid_t));
#pragma omp parallel for
        for (auto j = 0; j < gc[i]->n; j++)
        {
            auto u = gc[i]->nodes[j];
            role_copy[j] = cpurole[u];
            sd_copy[j] = cpusd[u];
            ed_copy[j] = cpued[u];
        }

        gpuErrchk(cudaMalloc((void**)&gpuadj, gc[i]->adj_length * sizeof(vid_t)));
        gpuErrchk(cudaMalloc((void**)&E_u, gc[i]->m * sizeof(vid_t)));
        gpuErrchk(cudaMalloc((void**)&E_v, gc[i]->m * sizeof(vid_t)));
        gpuErrchk(cudaMalloc((void**)&similar, gc[i]->m * sizeof(char)));
        gpuErrchk(cudaMalloc((void**)&role, gc[i]->n * sizeof(vid_t)));
        gpuErrchk(cudaMalloc((void**)&sd, gc[i]->n * sizeof(vid_t)));
        gpuErrchk(cudaMalloc((void**)&ed, gc[i]->n * sizeof(vid_t)));
        gpuErrchk(cudaMalloc((void**)&gpunum_edges, (gc[i]->n + 1) * sizeof(eid_t)));

        gpuErrchk(cudaMemcpy(gpunum_edges, gc[i]->num_edges, (gc[i]->n + 1) * sizeof(eid_t), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(role, role_copy, gc[i]->n * sizeof(vid_t), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(sd, sd_copy, gc[i]->n * sizeof(vid_t), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(ed, ed_copy, gc[i]->n * sizeof(vid_t), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(gpuadj, gc[i]->adj, gc[i]->adj_length * sizeof(vid_t), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(gpu_ni, gc[i]->node_index, g->n * sizeof(vid_t), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(E_u, gc[i]->el_u, gc[i]->m * sizeof(vid_t), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(E_v, gc[i]->el_v, gc[i]->m * sizeof(vid_t), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(similar, gc[i]->similar, gc[i]->m * sizeof(char), cudaMemcpyHostToDevice));

        __check_core <<<cc_ThreadBlockCount, cc_BLOCKSIZE>>> (gc[i]->m, gpunum_edges, eps, miu, gpuadj, E_u, E_v, similar, role,
            sd, ed, wap_log, gpu_ni);

        gpuErrchk(cudaMemcpy(gc[i]->similar, similar, gc[i]->m * sizeof(char), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(role_copy, role, gc[i]->n * sizeof(vid_t), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(sd_copy, sd, gc[i]->n * sizeof(vid_t), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(ed_copy, ed, gc[i]->n * sizeof(vid_t), cudaMemcpyDeviceToHost));
#pragma omp parallel for
        for (auto j = 0; j < gc[i]->n; j++)
        {
            auto u = gc[i]->nodes[j];
            cpurole[u] = role_copy[j];
            cpusd[u] = sd_copy[j];
            cpued[u] = ed_copy[j];
        }
        gpuErrchk(cudaFree(gpuadj));
        gpuErrchk(cudaFree(E_u));
        gpuErrchk(cudaFree(E_v));
        gpuErrchk(cudaFree(similar));
        gpuErrchk(cudaFree(role));
        gpuErrchk(cudaFree(sd));
        gpuErrchk(cudaFree(ed));
        gpuErrchk(cudaFree(gpunum_edges));
        free(role_copy);
        free(sd_copy);
        free(ed_copy);
        if(i>=20){
        	freeSpeGraphChip(gc,i);
        }
    }
    //free
    free(cpusd);
    free(cpued);
    
    auto cc = high_resolution_clock::now();
    fprintf(stderr, "Check Core time: %.3lf s\n", duration_cast<milliseconds>(cc-Dend2).count() / 1000.0);
    
    vid_t* role;
    gpuErrchk(cudaMalloc((void**)&role, g->n * sizeof(vid_t)));
    gpuErrchk(cudaMemcpy(role, cpurole, g->n * sizeof(vid_t), cudaMemcpyHostToDevice));
    //compute core number
    thrust::device_ptr<vid_t> role_ptr = thrust::device_pointer_cast(role);
    vid_t uncore_number = thrust::count(role_ptr, role_ptr + g->n, -2);//debug
    vid_t unknow_number = thrust::count(role_ptr, role_ptr + g->n, -1);//debug
    vid_t core_number = thrust::count_if(role_ptr, role_ptr + g->n, is_core());
    
    fprintf(stderr, "\nuncore number: %d\n", uncore_number);//debug
    fprintf(stderr, "unknow number: %d\n", unknow_number);//debug
    fprintf(stderr, "core number: %d\n", core_number);//debug
    //free(cpurole);

    //cluster core phrase 1
    for (auto i = 0; i < pn; i++)
    {
        vid_t* E_u;
        vid_t* E_v;
        char* similar;

        gpuErrchk(cudaMalloc((void**)&E_u, gc[i]->m * sizeof(vid_t)));
        gpuErrchk(cudaMalloc((void**)&E_v, gc[i]->m * sizeof(vid_t)));
        gpuErrchk(cudaMalloc((void**)&similar, gc[i]->m * sizeof(char)));

        gpuErrchk(cudaMemcpy(similar, gc[i]->similar, gc[i]->m * sizeof(char), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(E_u, gc[i]->el_u, gc[i]->m * sizeof(vid_t), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(E_v, gc[i]->el_v, gc[i]->m * sizeof(vid_t), cudaMemcpyHostToDevice));

        __cluster_core1 <<<cc_ThreadBlockCount, cc_BLOCKSIZE>>> (similar, role, E_u, E_v, gc[i]->m);

        gpuErrchk(cudaMemcpy(gc[i]->similar, similar, gc[i]->m * sizeof(char), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaFree(E_u));
        gpuErrchk(cudaFree(E_v));
        gpuErrchk(cudaFree(similar));
    }
    fprintf(stderr, "cluster core 1 finished\n");

    //cluster core phrase 2
    gpuErrchk(cudaMemcpy(cpurole, role, g->n * sizeof(vid_t), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(role));
    eid_t* elength = (eid_t*)malloc(pn * sizeof(eid_t));
    for (auto i = 0; i < pn; i++){
	vid_t* E_u;
        vid_t* E_v;
        char* similar;
        gpuErrchk(cudaMalloc((void**)&E_u, gc[i]->m * sizeof(vid_t)));
        gpuErrchk(cudaMalloc((void**)&E_v, gc[i]->m * sizeof(vid_t)));
        gpuErrchk(cudaMalloc((void**)&similar, gc[i]->m * sizeof(char)));

        gpuErrchk(cudaMemcpy(E_u, gc[i]->el_u, gc[i]->m * sizeof(vid_t), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(E_v, gc[i]->el_v, gc[i]->m * sizeof(vid_t), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(similar, gc[i]->similar, gc[i]->m * sizeof(char), cudaMemcpyHostToDevice));

        thrust::device_ptr<vid_t> pE_u = thrust::device_pointer_cast(E_u);
        thrust::device_ptr<vid_t> pE_v = thrust::device_pointer_cast(E_v);
        thrust::device_ptr<char> psimilar = thrust::device_pointer_cast(similar);
        thrust::zip_iterator<thrust::tuple<thrust::device_vector<vid_t>::iterator, 
            thrust::device_vector<vid_t>::iterator, thrust::device_vector<char>::iterator>> first;
        first = thrust::make_zip_iterator(thrust::make_tuple(pE_u, pE_v, psimilar));
        //elength[i] = thrust::partition(first, first + gc[i].m, par_rule1()) - first;
        elength[i] = thrust::remove_if(first, first + gc[i]->m, par_rule1()) - first;

	gpuErrchk(cudaMemcpy(gc[i]->similar, similar, gc[i]->m * sizeof(char), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(gc[i]->el_u, E_u, gc[i]->m * sizeof(vid_t), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(gc[i]->el_v, E_v, gc[i]->m * sizeof(vid_t), cudaMemcpyDeviceToHost));

	gpuErrchk(cudaFree(E_u));
        gpuErrchk(cudaFree(E_v));
        gpuErrchk(cudaFree(similar));

    }
    fprintf(stderr,"adjust edge list finished\n");
    gpuErrchk(cudaMalloc((void**)&role, g->n * sizeof(vid_t)));
    gpuErrchk(cudaMemcpy(role, cpurole, g->n * sizeof(vid_t), cudaMemcpyHostToDevice));
    for (auto i = 0; i < pn; i++)
    {
        vid_t* gpuadj;
        vid_t* E_u;
        vid_t* E_v;
        char* similar;
        eid_t* gpunum_edges;

        /*gpuErrchk(cudaMalloc((void**)&E_u, gc[i]->m * sizeof(vid_t)));
        gpuErrchk(cudaMalloc((void**)&E_v, gc[i]->m * sizeof(vid_t)));
        gpuErrchk(cudaMalloc((void**)&similar, gc[i]->m * sizeof(char)));

        gpuErrchk(cudaMemcpy(E_u, gc[i]->el_u, gc[i]->m * sizeof(vid_t), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(E_v, gc[i]->el_v, gc[i]->m * sizeof(vid_t), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(similar, gc[i]->similar, gc[i]->m * sizeof(char), cudaMemcpyHostToDevice));

        thrust::device_ptr<vid_t> pE_u = thrust::device_pointer_cast(E_u);
        thrust::device_ptr<vid_t> pE_v = thrust::device_pointer_cast(E_v);
        thrust::device_ptr<char> psimilar = thrust::device_pointer_cast(similar);
        thrust::zip_iterator<thrust::tuple<thrust::device_vector<vid_t>::iterator, 
            thrust::device_vector<vid_t>::iterator, thrust::device_vector<char>::iterator>> first;
        first = thrust::make_zip_iterator(thrust::make_tuple(pE_u, pE_v, psimilar));
        //elength[i] = thrust::partition(first, first + gc[i].m, par_rule1()) - first;
        elength[i] = thrust::remove_if(first, first + gc[i]->m, par_rule1()) - first;

	gpuErrchk(cudaMemcpy(gc[i]->similar, similar, gc[i]->m * sizeof(char), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(gc[i]->el_u, E_u, gc[i]->m * sizeof(vid_t), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(gc[i]->el_v, E_v, gc[i]->m * sizeof(vid_t), cudaMemcpyDeviceToHost));

	gpuErrchk(cudaFree(E_u));
        gpuErrchk(cudaFree(E_v));
        gpuErrchk(cudaFree(similar));*/

        if (elength[i] > 0)
        {
	    if(i>=20){
        	loadSpeGraphChip(gc,g,i);
            }
	    gpuErrchk(cudaMalloc((void**)&E_u, elength[i] * sizeof(vid_t)));
            gpuErrchk(cudaMalloc((void**)&E_v, elength[i] * sizeof(vid_t)));
            gpuErrchk(cudaMalloc((void**)&similar, elength[i] * sizeof(char)));
	    gpuErrchk(cudaMemcpy(E_u, gc[i]->el_u, elength[i] * sizeof(vid_t), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(E_v, gc[i]->el_v, elength[i] * sizeof(vid_t), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(similar, gc[i]->similar, elength[i] * sizeof(char), cudaMemcpyHostToDevice));
	    
	    gpuErrchk(cudaMalloc((void**)&gpunum_edges, (gc[i]->n + 1) * sizeof(eid_t)));
   	    gpuErrchk(cudaMalloc((void**)&gpuadj, gc[i]->adj_length * sizeof(vid_t)));
  	    gpuErrchk(cudaMemcpy(gpunum_edges, gc[i]->num_edges, (gc[i]->n + 1) * sizeof(eid_t), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(gpuadj, gc[i]->adj, gc[i]->adj_length * sizeof(vid_t), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(gpu_ni, gc[i]->node_index, g->n * sizeof(vid_t), cudaMemcpyHostToDevice));

            __cluster_core2 <<<cc_ThreadBlockCount, cc_BLOCKSIZE>>> (eps, gpunum_edges, elength[i], gpuadj,
                similar, role, E_u, E_v, wap_log, gpu_ni);

	    gpuErrchk(cudaFree(gpuadj));
            gpuErrchk(cudaFree(E_u));
            gpuErrchk(cudaFree(E_v));
            gpuErrchk(cudaFree(similar));
            gpuErrchk(cudaFree(gpunum_edges));
            if(i>=20){
            	freeSpeGraphChip(gc,i);
            }
        }
    }
    fprintf(stderr, "cluster core 2 finished\n");

    __cluster_core <<<ii_ThreadBlockCount, ii_BLOCKSIZE>>> (role, g->n);

    fprintf(stderr, "cluster core all finished\n");

    //cluster non-core
    gpuErrchk(cudaMemcpy(cpurole, role, g->n * sizeof(vid_t), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(role));
    eid_t* elength2=(eid_t*)malloc(pn*sizeof(eid_t));
    for (auto i = 0; i < pn; i++){

	vid_t* E_u;
        vid_t* E_v;
        char* similar;
        gc[i]->m = gc[i]->m - elength[i];
        //gpuErrchk(cudaMalloc((void**)&E_u, (gc[i].m - elength[i]) * sizeof(vid_t)));
        //gpuErrchk(cudaMalloc((void**)&E_v, (gc[i].m - elength[i]) * sizeof(vid_t)));
        //gpuErrchk(cudaMalloc((void**)&similar, (gc[i].m - elength[i]) * sizeof(char)));
        gpuErrchk(cudaMalloc((void**)&E_u, gc[i]->m * sizeof(vid_t)));
        gpuErrchk(cudaMalloc((void**)&E_v, gc[i]->m * sizeof(vid_t)));
        gpuErrchk(cudaMalloc((void**)&similar, gc[i]->m * sizeof(char)));

        
        //gpuErrchk(cudaMemcpy(E_u, gc[i].el_u + elength[i], (gc[i].m - elength[i]) * sizeof(vid_t), cudaMemcpyHostToDevice));
        //gpuErrchk(cudaMemcpy(E_v, gc[i].el_v + elength[i], (gc[i].m - elength[i]) * sizeof(vid_t), cudaMemcpyHostToDevice));
        //gpuErrchk(cudaMemcpy(similar, gc[i].similar + elength[i], (gc[i].m - elength[i]) * sizeof(char), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(E_u, gc[i]->el_u + elength[i], gc[i]->m * sizeof(vid_t), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(E_v, gc[i]->el_v + elength[i], gc[i]->m * sizeof(vid_t), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(similar, gc[i]->similar + elength[i], gc[i]->m * sizeof(char), cudaMemcpyHostToDevice));

        thrust::device_ptr<vid_t> pE_u = thrust::device_pointer_cast(E_u);
        thrust::device_ptr<vid_t> pE_v = thrust::device_pointer_cast(E_v);
        thrust::device_ptr<char> psimilar = thrust::device_pointer_cast(similar);
        thrust::zip_iterator<thrust::tuple<thrust::device_vector<vid_t>::iterator,
            thrust::device_vector<vid_t>::iterator, thrust::device_vector<char>::iterator>> first;
        first = thrust::make_zip_iterator(thrust::make_tuple(pE_u, pE_v, psimilar));
        //elength2[i] = thrust::partition(first, first + gc[i].m - elength[i], par_rule2()) - first;
        elength2[i] = thrust::remove_if(first, first + gc[i]->m, par_rule2()) - first;

	gpuErrchk(cudaMemcpy(gc[i]->similar+elength[i], similar, gc[i]->m * sizeof(char), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(gc[i]->el_u+elength[i], E_u, gc[i]->m * sizeof(vid_t), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(gc[i]->el_v+elength[i], E_v, gc[i]->m * sizeof(vid_t), cudaMemcpyDeviceToHost));

	gpuErrchk(cudaFree(E_u));
        gpuErrchk(cudaFree(E_v));
        gpuErrchk(cudaFree(similar));
    }
    fprintf(stderr,"adjust edge list finished\n");
    gpuErrchk(cudaMalloc((void**)&role, g->n * sizeof(vid_t)));
    gpuErrchk(cudaMemcpy(role, cpurole, g->n * sizeof(vid_t), cudaMemcpyHostToDevice));
    for (auto i = 0; i < pn; i++)
    {
        vid_t* gpuadj;
        eid_t* gpunum_edges;
        vid_t* E_u;
        vid_t* E_v;
        char* similar;
        /*gc[i]->m = gc[i]->m - elength[i];
        //gpuErrchk(cudaMalloc((void**)&E_u, (gc[i].m - elength[i]) * sizeof(vid_t)));
        //gpuErrchk(cudaMalloc((void**)&E_v, (gc[i].m - elength[i]) * sizeof(vid_t)));
        //gpuErrchk(cudaMalloc((void**)&similar, (gc[i].m - elength[i]) * sizeof(char)));
        gpuErrchk(cudaMalloc((void**)&E_u, gc[i]->m * sizeof(vid_t)));
        gpuErrchk(cudaMalloc((void**)&E_v, gc[i]->m * sizeof(vid_t)));
        gpuErrchk(cudaMalloc((void**)&similar, gc[i]->m * sizeof(char)));

        
        //gpuErrchk(cudaMemcpy(E_u, gc[i].el_u + elength[i], (gc[i].m - elength[i]) * sizeof(vid_t), cudaMemcpyHostToDevice));
        //gpuErrchk(cudaMemcpy(E_v, gc[i].el_v + elength[i], (gc[i].m - elength[i]) * sizeof(vid_t), cudaMemcpyHostToDevice));
        //gpuErrchk(cudaMemcpy(similar, gc[i].similar + elength[i], (gc[i].m - elength[i]) * sizeof(char), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(E_u, gc[i]->el_u + elength[i], gc[i]->m * sizeof(vid_t), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(E_v, gc[i]->el_v + elength[i], gc[i]->m * sizeof(vid_t), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(similar, gc[i]->similar + elength[i], gc[i]->m * sizeof(char), cudaMemcpyHostToDevice));

        thrust::device_ptr<vid_t> pE_u = thrust::device_pointer_cast(E_u);
        thrust::device_ptr<vid_t> pE_v = thrust::device_pointer_cast(E_v);
        thrust::device_ptr<char> psimilar = thrust::device_pointer_cast(similar);
        thrust::zip_iterator<thrust::tuple<thrust::device_vector<vid_t>::iterator,
            thrust::device_vector<vid_t>::iterator, thrust::device_vector<char>::iterator>> first;
        first = thrust::make_zip_iterator(thrust::make_tuple(pE_u, pE_v, psimilar));
        //elength2[i] = thrust::partition(first, first + gc[i].m - elength[i], par_rule2()) - first;
        elength2[i] = thrust::remove_if(first, first + gc[i]->m, par_rule2()) - first;

		gpuErrchk(cudaMemcpy(gc[i]->similar+elength[i], similar, gc[i]->m * sizeof(char), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(gc[i]->el_u+elength[i], E_u, gc[i]->m * sizeof(vid_t), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(gc[i]->el_v+elength[i], E_v, gc[i]->m * sizeof(vid_t), cudaMemcpyDeviceToHost));

		gpuErrchk(cudaFree(E_u));
        gpuErrchk(cudaFree(E_v));
        gpuErrchk(cudaFree(similar));*/

        if (elength2[i] > 0)
        {
	    	if(i>=20){
        		loadSpeGraphChip(gc,g,i);
        	}
	    	gpuErrchk(cudaMalloc((void**)&E_u, elength2[i] * sizeof(vid_t)));
            gpuErrchk(cudaMalloc((void**)&E_v, elength2[i] * sizeof(vid_t)));
            gpuErrchk(cudaMalloc((void**)&similar, elength2[i] * sizeof(char)));
	    	gpuErrchk(cudaMemcpy(E_u, gc[i]->el_u+elength[i], elength2[i] * sizeof(vid_t), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(E_v, gc[i]->el_v+elength[i], elength2[i] * sizeof(vid_t), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(similar, gc[i]->similar+elength[i], elength2[i] * sizeof(char), cudaMemcpyHostToDevice));
	    
	    	gpuErrchk(cudaMalloc((void**)&gpunum_edges, (gc[i]->n + 1) * sizeof(eid_t)));
            gpuErrchk(cudaMalloc((void**)&gpuadj, gc[i]->adj_length * sizeof(vid_t)));
	    	gpuErrchk(cudaMemcpy(gpunum_edges, gc[i]->num_edges, (gc[i]->n + 1) * sizeof(eid_t), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(gpuadj, gc[i]->adj, gc[i]->adj_length * sizeof(vid_t), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(gpu_ni, gc[i]->node_index, g->n * sizeof(vid_t), cudaMemcpyHostToDevice));
            __cluster_noncore <<<cc_ThreadBlockCount, cc_BLOCKSIZE>>> (eps, elength2[i], gpunum_edges, gpuadj,
                similar, role, E_u, E_v, wap_log, gpu_ni);
	    	gpuErrchk(cudaFree(gpunum_edges));
            gpuErrchk(cudaFree(gpuadj));
            gpuErrchk(cudaFree(E_u));
            gpuErrchk(cudaFree(E_v));
            gpuErrchk(cudaFree(similar));
            if(i>=20){
        		freeSpeGraphChip(gc,i);
        	}
        }
    }
    fprintf(stderr, "cluster non core finished\n");

    //classify others
    bool* hc;
    gpuErrchk(cudaMalloc((void**)&hc, g->n * sizeof(bool)));
    gpuErrchk(cudaMemset(hc, false, g->n * sizeof(bool)));
    for (auto i = 0; i < pn; i++)
    {
        vid_t* gpuadj;
        eid_t* gpunum_edges;
        vid_t* gpunodes;
        if(i>=20){
        	loadSpeGraphChip(gc,g,i);
        }

        gpuErrchk(cudaMalloc((void**)&gpunodes, gc[i]->n * sizeof(vid_t)));
        gpuErrchk(cudaMalloc((void**)&gpunum_edges, (gc[i]->n + 1) * sizeof(eid_t)));
        gpuErrchk(cudaMalloc((void**)&gpuadj, gc[i]->adj_length * sizeof(vid_t)));
        
        gpuErrchk(cudaMemcpy(gpu_ni, gc[i]->node_index, g->n * sizeof(vid_t), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(gpunum_edges, gc[i]->num_edges, (gc[i]->n + 1) * sizeof(eid_t), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(gpunodes, gc[i]->nodes, gc[i]->n * sizeof(vid_t), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(gpuadj, gc[i]->adj, gc[i]->adj_length * sizeof(vid_t), cudaMemcpyHostToDevice));
        
        __classify_others <<<ii_ThreadBlockCount, ii_BLOCKSIZE>>> (role, gpunum_edges, gpuadj, gpunodes, hc, 
            gpu_ni, gc[i]->n);

        gpuErrchk(cudaFree(gpunum_edges));
        gpuErrchk(cudaFree(gpuadj));
    	gpuErrchk(cudaFree(gpunodes));
    	if(i>=20){
        	freeSpeGraphChip(gc,i);
        }
    }
    gpuErrchk(cudaFree(hc));
    fprintf(stderr, "classify others finished\n");
	
	//free original graph
	free(g->num_edges);
    g->num_edges = nullptr;
    free(g->adj);
    g->adj = nullptr;
	
    //output
    vid_t* cluster = (vid_t*)malloc((g->n) * sizeof(vid_t));
    gpuErrchk(cudaMemcpy(cluster, role, (g->n) * sizeof(vid_t), cudaMemcpyDeviceToHost));

    //free
    cudaFree(gpu_ni);
    cudaFree(role);
    cudaFree(wap_log);

    free_GraphChip(gc, pn);
    //free(cut_id);
    free(elength);
    free(elength2);
    free(cpurole);
    
    return cluster;
}

