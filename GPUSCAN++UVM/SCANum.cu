#include "SCANum.h"
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
#include <chrono>
#include <iostream>

using namespace std::chrono;
uint64_t  TOTALMEMOSIZE = 1536000;//0x180000000;	//GPU上可用的总内存数量: 6G
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

__global__ void __init_input(vid_t* role, vid_t* sd, vid_t* ed, eid_t* gpunum_edges, int miu, vid_t vlength) {
    vid_t threadCount = blockDim.x * gridDim.x;        //thread sum
    vid_t tid = threadIdx.x + blockIdx.x * blockDim.x; //thread id
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


__global__ void __prep_edge(char* similar, eid_t* gpunum_edges, vid_t* E_u, vid_t* E_v, double eps, int miu,
    eid_t elength, vid_t* ed, vid_t* sd, vid_t* role) {
    
    eid_t threadCount = blockDim.x * gridDim.x;        //thread sum
    eid_t tid = threadIdx.x + blockIdx.x * blockDim.x; //thread id
    vid_t u, v;
    double deg_u, deg_v;
    while (tid < elength / 2) {
        u = E_u[tid];
        v = E_v[tid];
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
        else if (eps * eps * deg_u * deg_v <= 4.00) {
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

__device__ bool intersect(double eps, vid_t* gpuadj, eid_t begin_u, eid_t end_u, eid_t begin_v, 
    eid_t end_v, vid_t* BT_sh, int* wap_log,vid_t* TriangleSum, vid_t* search_node) {

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

__global__ void __check_core(eid_t elength, eid_t* gpunum_edges, const double eps, const int miu, vid_t* gpuadj, 
    vid_t* E_u, vid_t* E_v, char* similar, vid_t* role, vid_t* sd, vid_t* ed, int* wap_log) {

    __shared__ vid_t Binary_Tree[cc_BLOCKSIZE / WARPSIZE][WARPSIZE];    //存储二叉搜索树的前k层，k=5
    __shared__ eid_t begin_u[cc_BLOCKSIZE / WARPSIZE][WARPSIZE];
    __shared__ eid_t begin_v[cc_BLOCKSIZE / WARPSIZE][WARPSIZE];
    __shared__ eid_t end_u[cc_BLOCKSIZE / WARPSIZE][WARPSIZE];
    __shared__ eid_t end_v[cc_BLOCKSIZE / WARPSIZE][WARPSIZE];
    __shared__ int wap_logsh[cc_BLOCKSIZE / WARPSIZE][WARPSIZE];
    __shared__ vid_t TriangleSum[cc_BLOCKSIZE / WARPSIZE];
    __shared__ vid_t search_node[cc_BLOCKSIZE];

    //int tid = threadIdx.x + blockIdx.x * blockDim.x; //thread id
    int wid = (threadIdx.x + blockIdx.x * blockDim.x) % WARPSIZE;                   //Warp内索引
    //int WarpIdx_global = tid / WARPSIZE;       //全局Warp索引
    int WarpIdx_block = threadIdx.x / WARPSIZE;    //线程块内的Warp索引
    int WarpNumb = blockDim.x * gridDim.x / WARPSIZE;    //全局的Warp数
    eid_t Proc_Numb, Proc_it;                           //一个Warp处理的边的数目,边表处理起始位置标识
    vid_t u_vertex, v_vertex;                                 //当前warp查询的点u,v
    int num_cache = 0;                                      //mod 32
    bool unsimilar;

    elength = elength / 2;
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
                begin_u[WarpIdx_block][wid] = gpunum_edges[E_u[Proc_it + wid]];
                end_u[WarpIdx_block][wid] = gpunum_edges[E_u[Proc_it + wid] + 1];
                begin_v[WarpIdx_block][wid] = gpunum_edges[E_v[Proc_it + wid]];
                end_v[WarpIdx_block][wid] = gpunum_edges[E_v[Proc_it + wid] + 1];
            }
        }
        u_vertex = E_u[Proc_it];
        v_vertex = E_v[Proc_it];
        //如果u和v相似度没被计算切u和v至少有一个role没被确定
        if (similar[Proc_it] == 'U' && (role[u_vertex] == -1 || role[v_vertex] == -1))
        {
            //compute similar
            unsimilar = intersect(eps, gpuadj, begin_u[WarpIdx_block][num_cache],
                end_u[WarpIdx_block][num_cache], begin_v[WarpIdx_block][num_cache],
                end_v[WarpIdx_block][num_cache], Binary_Tree[WarpIdx_block],
                wap_logsh[WarpIdx_block], TriangleSum, search_node);

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
        Proc_it++;//开始计算边表中下一条边
        num_cache = (num_cache + 1) % WARPSIZE;
    }
}


inline __device__ vid_t findroot(vid_t idx, vid_t* role) {

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

__global__ void __cluster_core1(char* similar, vid_t* role, vid_t* E_u, vid_t* E_v, eid_t elength) {

    eid_t threadCount = blockDim.x * gridDim.x;        //thread sum
    eid_t tid = threadIdx.x + blockIdx.x * blockDim.x; //thread id
    elength = elength / 2;

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

__global__ void __cluster_core2(double eps, eid_t* gpunum_edges, eid_t elength, vid_t* gpuadj, char* similar, 
    vid_t* role, vid_t* E_u, vid_t* E_v, int* wap_log) {

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
                begin_u[WarpIdx_block][wid] = gpunum_edges[E_u[Proc_it + wid]];
                end_u[WarpIdx_block][wid] = gpunum_edges[E_u[Proc_it + wid] + 1];
                begin_v[WarpIdx_block][wid] = gpunum_edges[E_v[Proc_it + wid]];
                end_v[WarpIdx_block][wid] = gpunum_edges[E_v[Proc_it + wid] + 1];
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


__global__ void __cluster_noncore(double eps, eid_t beg_el, eid_t elength, eid_t* gpunum_edges, vid_t* gpuadj, 
    char* similar, vid_t* role, vid_t* E_u, vid_t* E_v, int* wap_log) {

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
    Proc_it = ((threadIdx.x + blockIdx.x * blockDim.x) / WARPSIZE) * Proc_Numb + beg_el;

    for (eid_t k = 0; k < Proc_Numb; ++k)
    {
        if (Proc_it > elength - 1) break;
        if (num_cache % WARPSIZE == 0) {
            if (Proc_it + wid <= elength - 1) {
                begin_u[WarpIdx_block][wid] = gpunum_edges[E_u[Proc_it + wid]];
                end_u[WarpIdx_block][wid] = gpunum_edges[E_u[Proc_it + wid] + 1];
                begin_v[WarpIdx_block][wid] = gpunum_edges[E_v[Proc_it + wid]];
                end_v[WarpIdx_block][wid] = gpunum_edges[E_v[Proc_it + wid] + 1];
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

        //if (role[nocore] >= 0) continue;
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

__global__ void __classify_others(vid_t* role, eid_t* gpunum_edges, vid_t* gpuadj, vid_t vlength) {

    __shared__ vid_t adjrole[ii_BLOCKSIZE / WARPSIZE][WARPSIZE];
    __shared__ bool checkflag[ii_BLOCKSIZE / WARPSIZE];
    eid_t begin, end;

    int wid = (threadIdx.x + blockIdx.x * blockDim.x) % WARPSIZE;                   //Warp内索引
    vid_t WarpIdx_global = (threadIdx.x + blockIdx.x * blockDim.x) / WARPSIZE;       //全局Warp索引
    int WarpIdx_block = threadIdx.x / WARPSIZE;    //线程块内的Warp索引
    vid_t WarpNumb = blockDim.x * gridDim.x / WARPSIZE;    //全局的Warp数

    checkflag[WarpIdx_block] = false;
    for (vid_t u = WarpIdx_global; u < vlength; u += WarpNumb)
    {
        if (role[u] < 0)
        {
            begin = gpunum_edges[WarpIdx_global];
            end = gpunum_edges[WarpIdx_global + 1];
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
        }
    }
}

template<typename T>
void swap_t(T& a, T& b) {
    T temp;
    temp = a;
    a = b;
    b = temp;
}

eid_t partition1(vid_t* E_u, vid_t* E_v, char* similar, eid_t start, eid_t end) {
    //eid_t start, end;
    //start = (eid_t)0;
    //end = length - 1;
    while (true)
    {
        while (start < end && similar[start] == 'V') start++;
        if (start >= end) break;
        while (start < end && similar[end] != 'V') end--;
        if (start >= end) break;
        swap_t(E_u[start], E_u[end]);
        swap_t(E_v[start], E_v[end]);
        swap_t(similar[start], similar[end]);
    }
    return end;
}

eid_t partition2(vid_t* E_u, vid_t* E_v, char* similar, eid_t start, eid_t end) {
    //eid_t start, end;
    //start = (eid_t)0;
    //end = length - 1;
    while (true)
    {
        while (start < end && (similar[start] == 'Z' || similar[start] == 'z' 
            || similar[start] == 'W' || similar[start] == 'w')) start++;
        if (start >= end) break;
        while (start < end && (similar[end] != 'Z' && similar[end] != 'z' 
            && similar[end] != 'W' && similar[end] != 'w')) end--;
        if (start >= end) break;
        swap_t(E_u[start], E_u[end]);
        swap_t(E_v[start], E_v[end]);
        swap_t(similar[start], similar[end]);
    }
    return end;
}

vid_t* SCANum(graph_t* g, vid_t* Edge_u, vid_t* Edge_v, double eps, int miu) {
    //init GPU
    cudaDeviceProp deviceProp;
    gpuErrchk(cudaGetDeviceProperties(&deviceProp, 0));

    vid_t* gpuadj;
    eid_t* gpunum_edges;
    vid_t* E_u;
    vid_t* E_v;

    char* similar;
    vid_t* role;
    vid_t* sd;
    vid_t* ed;
    int* wap_log;
    
    fprintf(stderr,"malloc\n");
    //init input
    int* cpuwap_log = (int*)malloc(32 * sizeof(int));
#pragma omp parallel for
    for (int i = 0; i < 32; i++) {
        cpuwap_log[i] = (int)(log(i + 1) / log(2));
    }
    std::cout<<"number of nodes: "<<g->n<<" number of edges: "<<g->m<<endl;
    gpuErrchk(cudaMallocManaged((void**)&wap_log, 32 * sizeof(int)));

    gpuErrchk(cudaMallocManaged((void**)&gpuadj, (g->m) * sizeof(vid_t)));

    gpuErrchk(cudaMallocManaged((void**)&gpunum_edges, (g->n + 1) * sizeof(eid_t)));

    gpuErrchk(cudaMallocManaged((void**)&E_u, (g->m / 2) * sizeof(vid_t)));

    gpuErrchk(cudaMallocManaged((void**)&E_v, (g->m / 2) * sizeof(vid_t)));

    gpuErrchk(cudaMallocManaged((void**)&similar, (g->m / 2) * sizeof(char)));

    gpuErrchk(cudaMallocManaged((void**)&role, (g->n) * sizeof(vid_t)));

    gpuErrchk(cudaMallocManaged((void**)&sd, (g->n) * sizeof(vid_t)));

    gpuErrchk(cudaMallocManaged((void**)&ed, (g->n) * sizeof(vid_t)));

    
    memcpy(wap_log, cpuwap_log, 32 * sizeof(int));

    
    memcpy(gpuadj, g->adj, (g->m) * sizeof(vid_t));
    free(g->adj);
    g->adj=nullptr;

    
    memcpy(gpunum_edges, g->num_edges, (g->n + 1) * sizeof(eid_t));
    free(g->num_edges);
    g->num_edges=nullptr;

    
    memcpy(E_u, Edge_u, (g->m / 2) * sizeof(vid_t));
    free(Edge_u);
    Edge_u=nullptr;

    memcpy(E_v, Edge_v, (g->m / 2) * sizeof(vid_t));
    free(Edge_v);
    Edge_v=nullptr;

    uint32_t  ii_ThreadBlockCount = TOTALTHDCOUNT / ii_BLOCKSIZE;
    uint32_t  cc_ThreadBlockCount = TOTALTHDCOUNT / cc_BLOCKSIZE;
    fprintf(stderr,"init input\n");
    auto t0 = high_resolution_clock::now();

    __init_input <<<ii_ThreadBlockCount, ii_BLOCKSIZE>>> (role, sd, ed, gpunum_edges, miu, g->n);
    gpuErrchk(cudaDeviceSynchronize());
    
    fprintf(stderr,"prep edge\n");
    __prep_edge <<<ii_ThreadBlockCount, ii_BLOCKSIZE>>> (similar, gpunum_edges, E_u, E_v, eps, miu, g->m, ed, sd, role);
    gpuErrchk(cudaDeviceSynchronize());
    
    auto t1 = high_resolution_clock::now();
    fprintf(stderr, "init time: %.3lf s\n", duration_cast<milliseconds>(t1 - t0).count() / 1000.0);

    fprintf(stderr,"check core\n");
    __check_core <<<cc_ThreadBlockCount, cc_BLOCKSIZE>>> (g->m, gpunum_edges, eps, miu, gpuadj, E_u, E_v, similar, role,
        sd, ed, wap_log);
    gpuErrchk(cudaDeviceSynchronize());
    
    auto t2 = high_resolution_clock::now();
    fprintf(stderr, "check core time: %.3lf s\n", duration_cast<milliseconds>(t2 - t1).count() / 1000.0);

    //compute core number
    fprintf(stderr,"compute core number\n");
    thrust::device_ptr<int> role_ptr = thrust::device_pointer_cast(role);
    vid_t uncore_number = thrust::count(role_ptr, role_ptr + g->n, -2);//debug
    vid_t unknow_number = thrust::count(role_ptr, role_ptr + g->n, -1);//debug
    vid_t core_number = thrust::count_if(role_ptr, role_ptr + g->n, is_core());
    fprintf(stderr, "uncore number: %d\n", uncore_number);//debug
    fprintf(stderr, "unknow number: %d\n", unknow_number);//debug
    fprintf(stderr, "core number: %d\n", core_number);//debug

    //vid_t* gpucore_node;
    //gpuErrchk(cudaMallocManaged((void**)&gpucore_node, (core_number) * sizeof(vid_t)));
    //thrust::copy_if(thrust::device, role, role + g->n, gpucore_node, is_core());
    auto t3 = high_resolution_clock::now();
    fprintf(stderr, "compute core time: %.3lf s\n", duration_cast<milliseconds>(t3 - t2).count() / 1000.0);
    
    __cluster_core1 <<<cc_ThreadBlockCount, cc_BLOCKSIZE>>> (similar, role, E_u, E_v, g->m);
    gpuErrchk(cudaDeviceSynchronize());
    auto t4 = high_resolution_clock::now();
    fprintf(stderr, "cluster core1 time: %.3lf s\n", duration_cast<milliseconds>(t4 - t3).count() / 1000.0);
    //thrust::device_ptr<vid_t> pE_u = thrust::device_pointer_cast(E_u);
    //thrust::device_ptr<vid_t> pE_v = thrust::device_pointer_cast(E_v);
    //thrust::device_ptr<char> psimilar = thrust::device_pointer_cast(similar);
    //thrust::zip_iterator<thrust::tuple<thrust::device_vector<vid_t>::iterator,
    //    thrust::device_vector<vid_t>::iterator, thrust::device_vector<char>::iterator>> first;
    //first = thrust::make_zip_iterator(thrust::make_tuple(pE_u, pE_v, psimilar));
    //eid_t elength = thrust::partition(first, first + (g->m / 2), par_rule1()) - first;
    //gpuErrchk(cudaDeviceSynchronize());
    eid_t elength1 = partition1(E_u, E_v, similar, (eid_t)0, g->m / 2);
    if (elength1 > 0)
    {
        __cluster_core2 <<<cc_ThreadBlockCount, cc_BLOCKSIZE>>> (eps, gpunum_edges, elength1, gpuadj,
            similar, role, E_u, E_v, wap_log);
        gpuErrchk(cudaDeviceSynchronize());
    }
    auto t5 = high_resolution_clock::now();
    fprintf(stderr, "cluster core2 time: %.3lf s\n", duration_cast<milliseconds>(t5 - t4).count() / 1000.0);

    __cluster_core <<<ii_ThreadBlockCount, ii_BLOCKSIZE>>> (role, g->n);
    gpuErrchk(cudaDeviceSynchronize());
    
    auto t6 = high_resolution_clock::now();
    fprintf(stderr, "link time: %.3lf s\n", duration_cast<milliseconds>(t6 - t5).count() / 1000.0);

    //elength = thrust::partition(first, first + (g->m / 2), par_rule2()) - first;
    //gpuErrchk(cudaDeviceSynchronize());
    eid_t elength2 = partition1(E_u, E_v, similar, elength1, g->m / 2);
    if (elength2 > elength1)
    {
        __cluster_noncore <<<cc_ThreadBlockCount, cc_BLOCKSIZE>>> (eps, elength1, elength2, gpunum_edges, 
            gpuadj, similar, role, E_u, E_v, wap_log);
        gpuErrchk(cudaDeviceSynchronize());
    }
    auto t7 = high_resolution_clock::now();
    fprintf(stderr, "cluster non core time: %.3lf s\n", duration_cast<milliseconds>(t7 - t6).count() / 1000.0);

    __classify_others <<<ii_ThreadBlockCount, ii_BLOCKSIZE>>> (role, gpunum_edges, gpuadj, g->n);
    gpuErrchk(cudaDeviceSynchronize());
    auto t8 = high_resolution_clock::now();
    fprintf(stderr, "classify others time: %.3lf s\n", duration_cast<milliseconds>(t8 - t7).count() / 1000.0);
    
    //output
    vid_t* cluster = (vid_t*)malloc((g->n) * sizeof(vid_t));
    gpuErrchk(cudaMemcpy(cluster, role, (g->n) * sizeof(vid_t), cudaMemcpyDeviceToHost));

    //free
    gpuErrchk(cudaFree(sd));
    gpuErrchk(cudaFree(ed));
    gpuErrchk(cudaFree(role));
    gpuErrchk(cudaFree(gpunum_edges));
    gpuErrchk(cudaFree(gpuadj));
    gpuErrchk(cudaFree(E_u));
    gpuErrchk(cudaFree(E_v));
    gpuErrchk(cudaFree(similar));
    gpuErrchk(cudaFree(wap_log));

    return cluster;
}

