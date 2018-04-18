#include <iostream>
#include <math.h>
#include <vector>
#include <thrust/extrema.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#define BLOCK_SIZE 1024

using namespace std;

const int SPLIT_SIZE = 512;
const int POCKET_SIZE = 1024;

#define cuda_check_error() {                                          \
    cudaError_t e=cudaGetLastError();     \
    if(e!=cudaSuccess) {                                              \
        printf("Cuda failure %s\n", cudaGetErrorString(e));           \
        exit(0); \
    }                                                                 \
}


struct bucket
{
    int * buckets;
    int buckets_count;
};

struct comparator {
    __host__ __device__ bool operator()(float a, float b) {
        return a < b; 
    }
};

float get_max_elem(thrust::device_ptr<float> p_arr, int n){
    comparator comp;
    thrust::device_ptr<float> res = thrust::max_element(p_arr, p_arr + n, comp);
    return (*res);
}

float get_min_elem(thrust::device_ptr<float> p_arr, int n){
    comparator comp;
    thrust::device_ptr<float> res = thrust::min_element(p_arr, p_arr + n, comp);
    return (*res);
}

int get_split_count(int n){
    return (n - 1) / SPLIT_SIZE + 2;
}

int get_split_index(int n, float elem, float min_elem, float max_elem, int split_count){
    if (max_elem == min_elem)
        return 0;
    return (int)((elem - min_elem) / (max_elem - min_elem) * (split_count - 1));
}

int get_extend_arr_size(int n){
    bool is_power_o_two = n && !(n & (n - 1));
    if (is_power_o_two && n > 1){
        return n;
    }
    int nearest_power = 0;
    while(n > 0){
        n >>= 1;
        nearest_power++;
    }
    return pow(2, nearest_power);
}


float * to_small_bucket(float * input, int * scan, int * split_indexes, int n, int bucket_count){
    int* buckets = (int*)malloc(sizeof(int)*bucket_count);
    memset(buckets, 0, sizeof(int)*bucket_count);
    float * result = (float *)malloc(sizeof(float)*n);
    for(int i = 0; i < n; i++){
        result[scan[split_indexes[i]] + buckets[split_indexes[i]]] = input[i];
        buckets[split_indexes[i]] += 1;
    }
    return result;
}


bucket get_big_bucket(int * scan, int split_count, int n){
    bucket result;
    vector <int> indexes;
    indexes.push_back(0);
    int prev = 0;
    for (int i = 1; i < (split_count + 1); ++i){
        int index_n = scan[i];
        int diff = index_n - indexes.back();
        if ((diff > POCKET_SIZE && prev != 0)){
            indexes.push_back(prev);
        }
        else if (diff == POCKET_SIZE){
            indexes.push_back(index_n);
        }
        if (i == split_count && indexes.back() != n){
            indexes.push_back(index_n);
        }
        prev = index_n;
    }

    int pockets_index_size = indexes.size();
    int* pockets_index = (int*)malloc(sizeof(int)*pockets_index_size);
    memcpy(pockets_index, indexes.data(), sizeof(int)*pockets_index_size);
    result.buckets_count = pockets_index_size;
    result.buckets = pockets_index;
    return result;
}


__global__ void histogram_kernel(int * split_indexes, int * histogram, int n){

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = blockDim.x * gridDim.x;

    while (idx < n) {
        atomicAdd(histogram + split_indexes[idx], 1);
        idx += offset;
    }
}

__global__ void odd_even_sort(float * arr, int * buckets, int count_buckets, int n){
    int idx = threadIdx.x;
    int block_id = blockIdx.x;
    int count_elem;
    __shared__ float data[BLOCK_SIZE];
    
    for(int i = block_id; i < count_buckets - 1; i += gridDim.x){
        count_elem = buckets[i + 1] - buckets[i];
        if (count_elem > BLOCK_SIZE){
            continue;
        }
        if(idx < count_elem){
            data[idx] = arr[buckets[i] + idx];
        }

        __syncthreads();
        
        int iter_count;

        if (count_elem % 2 == 0)
            iter_count = count_elem / 2;
        else
            iter_count = count_elem / 2 + 1;

        for(int j = 0; j < iter_count; j++){
            if((idx % 2 == 0) && (idx < count_elem - 1)){
                if(data[idx] > data[idx + 1]){
                    float tmp = data[idx];
                    data[idx] = data[idx + 1];
                    data[idx + 1] = tmp;
                }
            }
        
           __syncthreads();

            if((idx % 2 != 0) && (idx < count_elem - 1)){
                if(data[idx] > data[idx + 1]){
                    float tmp = data[idx];
                    data[idx] = data[idx + 1];
                    data[idx + 1] = tmp;
                    
                }
            }
            __syncthreads();
        }

        if(idx < count_elem)
            arr[buckets[i] + idx] = data[idx];
    }        
}

void bucket_sort(float * array, bucket buckets, int n){
    float * dev_arr;
    int * gpu_buckets;
    cudaMalloc(&gpu_buckets, sizeof(int) * buckets.buckets_count);
    cudaMalloc(&dev_arr, sizeof(float) * n);
    cudaMemcpy(gpu_buckets, buckets.buckets, sizeof(int) * buckets.buckets_count, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_arr, array, sizeof(float) * n,  cudaMemcpyHostToDevice);

    cudaEvent_t event;
    cudaEventCreate(&event);
    
    odd_even_sort<<<1024, BLOCK_SIZE>>>(dev_arr, gpu_buckets, buckets.buckets_count, n);
    cudaEventSynchronize(event);
    cudaMemcpy(array, dev_arr, sizeof(float) * n,  cudaMemcpyDeviceToHost);

}

__global__ void scan_blocks(int * histogram, int * res, int * summ, bool is_summ){
    int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idx = threadIdx.x;
    int offset = 1;
    __shared__ int data[BLOCK_SIZE];

    data[2 * idx] = histogram[2 * global_idx];
    data[2 * idx + 1] = histogram[2 * global_idx + 1];
   
    for(int i = BLOCK_SIZE / 2; i > 0; i /= 2){
        __syncthreads();
        if(idx < i){
            int left = offset * (2 * idx + 1) - 1;
            int right = offset * (2 * idx + 2) - 1;
            data[right] += data[left];
        }
        offset *= 2;
    }
  
    __syncthreads();

    if(idx == 0){
        if(!is_summ){
            summ[blockIdx.x] = data[BLOCK_SIZE - 1];
        }
        data[BLOCK_SIZE - 1] = 0;
    }

    for(int i = 1; i < BLOCK_SIZE ; i *= 2){
        offset /= 2;
        __syncthreads();

        if (idx < i){
            int left = offset * (2 * idx + 1) - 1;
            int right = offset * (2 * idx + 2) - 1;
            int tmp = data[left];
            data[left] = data[right];
            data[right] += tmp;
        }
    }
   __syncthreads();

    res[2 * global_idx] = data[2 * idx];
    res[2 * global_idx + 1] = data[2 * idx + 1];
 }

 __global__ void add_summ(int * blocks, int * summ, int block_count){
    for(int i = blockIdx.x; i < block_count; i += gridDim.x){
        blocks[blockDim.x * i + threadIdx.x] += summ[i];
    }        
 }

int get_last_summ(int * summ, int n){
    int res = 0;
    for(int i = 0; i < n; i++){
        res += summ[i];
    }
    return res;
}

int * recursive_scan(int * dev_arr, int n, int is_summ){
    int * res = (int *)malloc(sizeof(int) * (n + 1));
    int * dev_blocks;
    int * dev_summ;
    int * summ;
    int * dev_scanned_summ;

    int block_count = n / BLOCK_SIZE;
    if (n >= BLOCK_SIZE){
        block_count = n / BLOCK_SIZE;
    }
    else{
        block_count = 1;
    }
    int threads_count = BLOCK_SIZE / 2;   

    summ = (int *)malloc(sizeof(int) * (block_count + 1));
    cudaMalloc(&dev_blocks, sizeof(int) * n);
    cudaMalloc(&dev_summ, sizeof(int) * block_count);
    cudaMalloc(&dev_scanned_summ, sizeof(int) * block_count);

    scan_blocks<<<block_count, threads_count>>>(dev_arr, dev_blocks, dev_summ, false);
    cudaMemcpy(summ, dev_summ, sizeof(int) * block_count, cudaMemcpyDeviceToHost);

    if(block_count > BLOCK_SIZE){
        int * scan_summ = recursive_scan(dev_summ, block_count, false);
        cudaMemcpy(dev_summ, scan_summ, sizeof(int) * block_count, cudaMemcpyHostToDevice);
         add_summ<<<16, BLOCK_SIZE>>>(dev_blocks, dev_summ, block_count);
    }

    else{
        scan_blocks<<<1, threads_count>>>(dev_summ, dev_scanned_summ, NULL, true);
        add_summ<<<16, BLOCK_SIZE>>>(dev_blocks, dev_scanned_summ, block_count);
    }
    
    cudaMemcpy(res, dev_blocks, sizeof(int) * n, cudaMemcpyDeviceToHost);
    res[n] = get_last_summ(summ, block_count);
    return res;
}

void main_sort(float * input_array, int n){
    int split_count, extended_size;
    float min_elem, max_elem;
    split_count = get_split_count(n);

    extended_size = get_extend_arr_size(split_count + 1);
    int * split_indexes = (int *)malloc(sizeof(int) * n);
    int * histogram = (int *)malloc(sizeof(int) * extended_size);

    int * gpu_split_indexes;
    int * gpu_histogram;
    int * gpu_scan;
    float * gpu_arr;
    
    cudaMalloc(&gpu_arr, sizeof(float) * n);
    cudaMalloc(&gpu_split_indexes, sizeof(int) * n);
    cudaMalloc(&gpu_histogram, sizeof(int) * extended_size);
    cudaMalloc(&gpu_scan, sizeof(int) * extended_size);

    cudaMemset(gpu_histogram, 0, sizeof(int) * extended_size);
    cudaMemset(gpu_scan, 0, sizeof(int) * extended_size);
    
    cudaMemcpy(gpu_arr, input_array, sizeof(float) * n, cudaMemcpyHostToDevice);
    thrust::device_ptr<float> p_arr = thrust::device_pointer_cast(gpu_arr);
    min_elem = get_min_elem(p_arr, n);
    max_elem = get_max_elem(p_arr, n);

    if(min_elem == max_elem){
        return;
    }

    for(int i = 0; i < n; i++){
        split_indexes[i] = get_split_index(n, input_array[i], min_elem, max_elem, split_count);
    }
    
    cudaMemcpy(gpu_split_indexes, split_indexes, sizeof(int) * n, cudaMemcpyHostToDevice);
    histogram_kernel<<<64, BLOCK_SIZE>>>(gpu_split_indexes, gpu_histogram, n);
   
    cudaMemcpy(histogram, gpu_histogram, sizeof(int) * extended_size, cudaMemcpyDeviceToHost);

    int * cpu_scan = recursive_scan(gpu_histogram, extended_size, false);
    float * to_small_buckets = to_small_bucket(input_array, cpu_scan, split_indexes, n, split_count);
        
    bucket big_buckets = get_big_bucket(cpu_scan, split_count, n);

    bucket_sort(to_small_buckets, big_buckets, n);

    for(int i = 1; i < big_buckets.buckets_count; i++){
        int size_pocket = big_buckets.buckets[i] - big_buckets.buckets[i - 1];
        if (size_pocket > POCKET_SIZE){
            int ind = big_buckets.buckets[i - 1];
            main_sort(to_small_buckets + ind, size_pocket);
        }
    }

    memcpy(input_array, to_small_buckets, sizeof(float) * n);

    free(to_small_buckets);
    free(cpu_scan);
    free(split_indexes);
    free(histogram);
    cudaFree(gpu_split_indexes);
    cudaFree(gpu_histogram);
    cudaFree(gpu_scan);
    cudaFree(gpu_arr);
}

int main() {
    int n;
    fread(&n, sizeof(int), 1, stdin);
    
    float * input_array = (float *)malloc(sizeof(float) * n);

    if (!n){
        return 0;
    }

    fread(input_array, sizeof(float), n, stdin);

    main_sort(input_array, n);
    fwrite(input_array, sizeof(float), n, stdout);

    free(input_array);
    return 0;
}