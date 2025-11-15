#include "cumesh.h"
#include "dtypes.cuh"
#include <cub/cub.cuh>


namespace cumesh {


template<typename T>
static __global__ void index_kernel(
    const T* values,
    const int* indices,
    const size_t N,
    T* output
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    output[tid] = values[indices[tid]];
}


template<typename T>
static __global__ void diff_kernel(
    const T* values,
    const size_t N,
    T* output
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    output[tid] = values[tid+1] - values[tid];
}


template<typename T>
static __global__ void set_flag_kernel(
    const int* indices,
    const size_t N,
    T* output
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    output[indices[tid]] = static_cast<T>(1);
}


template<typename CompT, typename FlagT>
static __global__ void less_than_kernel(
    const CompT* values,
    const CompT threshold,
    const size_t N,
    FlagT* flag
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    if (values[tid] < threshold) {
        flag[tid] = static_cast<FlagT>(1);
    } else {
        flag[tid] = static_cast<FlagT>(0);
    }
}


template<typename Ta, typename Tb>
static __global__ void inplace_div_kernel(
    Ta* a,
    const Tb* b,
    const size_t N
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    a[tid] = a[tid] / static_cast<float>(b[tid]);
}


static __global__ void copy_vec3f_to_float3_kernel(
    const Vec3f* vec3f,
    const size_t N,
    float3* output
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    output[tid] = make_float3(vec3f[tid].x, vec3f[tid].y, vec3f[tid].z);
}


void CuMesh::remove_faces(torch::Tensor& face_mask) {
    size_t F = this->faces.size;

    size_t temp_storage_bytes = 0;
    int *cu_new_num_faces;
    int3 *cu_new_faces;
    CUDA_CHECK(cudaMalloc(&cu_new_num_faces, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cu_new_faces, F * sizeof(int3)));
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        nullptr, temp_storage_bytes,
        this->faces.ptr, face_mask.data_ptr<bool>(), cu_new_faces, cu_new_num_faces,
        F
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        this->faces.ptr, face_mask.data_ptr<bool>(), cu_new_faces, cu_new_num_faces,
        F
    ));
    int new_num_faces;
    CUDA_CHECK(cudaMemcpy(&new_num_faces, cu_new_num_faces, sizeof(int), cudaMemcpyDeviceToHost));
    this->faces.resize(new_num_faces);
    CUDA_CHECK(cudaMemcpy(this->faces.ptr, cu_new_faces, new_num_faces * sizeof(int3), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaFree(cu_new_num_faces));
    CUDA_CHECK(cudaFree(cu_new_faces));
    
    this->remove_unreferenced_vertices();
}


static __global__ void set_vertex_is_referenced(
    const int3* faces,
    const size_t F,
    int* vertex_is_referenced
) {
    const int fid = blockIdx.x * blockDim.x + threadIdx.x;
    if (fid >= F) return;
    int3 face = faces[fid];
    vertex_is_referenced[face.x] = 1;
    vertex_is_referenced[face.y] = 1;
    vertex_is_referenced[face.z] = 1;
}


static __global__ void compress_vertices_kernel(
    const int* vertices_map,
    const float3* old_vertices,
    const int V,
    float3* new_vertices
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= V) return;
    int new_id = vertices_map[tid];
    int is_kept = vertices_map[tid + 1] == new_id + 1;
    if (is_kept) {
        new_vertices[new_id] = old_vertices[tid];
    }
}


static __global__ void remap_faces_kernel(
    const int* vertices_map,
    const int F,
    int3* faces
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= F) return;
    faces[tid].x = vertices_map[faces[tid].x];
    faces[tid].y = vertices_map[faces[tid].y];
    faces[tid].z = vertices_map[faces[tid].z];
}


void CuMesh::remove_unreferenced_vertices() {
    size_t V = this->vertices.size;
    size_t F = this->faces.size;

    // Mark referenced vertices
    int* cu_vertex_is_referenced;
    CUDA_CHECK(cudaMalloc(&cu_vertex_is_referenced, (V+1) * sizeof(int)));
    CUDA_CHECK(cudaMemset(cu_vertex_is_referenced, 0, (V+1) * sizeof(int)));
    set_vertex_is_referenced<<<(F+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        this->faces.ptr,
        F,
        cu_vertex_is_referenced
    );
    CUDA_CHECK(cudaGetLastError());

    // Get vertices map
    size_t temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, temp_storage_bytes,
        cu_vertex_is_referenced, V+1
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        cu_vertex_is_referenced, V+1
    ));
    int new_num_vertices;
    CUDA_CHECK(cudaMemcpy(&new_num_vertices, cu_vertex_is_referenced + V, sizeof(int), cudaMemcpyDeviceToHost));

    // Compress vertices
    this->temp_storage.resize(new_num_vertices * sizeof(float3));
    compress_vertices_kernel<<<(V+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        cu_vertex_is_referenced,
        this->vertices.ptr,
        V,
        reinterpret_cast<float3*>(this->temp_storage.ptr)
    );
    CUDA_CHECK(cudaGetLastError());
    swap_buffers(this->temp_storage, this->vertices);

    // Update faces
    remap_faces_kernel<<<(F+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        cu_vertex_is_referenced,
        F,
        this->faces.ptr
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(cu_vertex_is_referenced));

    // Delete all connectivity info since mesh has changed
    this->clear_connectivity();
}


static __global__ void sort_faces_kernel(
    int3* faces,
    const size_t F
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= F) return;

    int3 face = faces[tid];
    int tmp;

    // bubble sort 3 elements (x, y, z)
    if (face.x > face.y) { tmp = face.x; face.x = face.y; face.y = tmp; }
    if (face.y > face.z) { tmp = face.y; face.y = face.z; face.z = tmp; }
    if (face.x > face.y) { tmp = face.x; face.x = face.y; face.y = tmp; }

    faces[tid] = face;
}


static __global__ void select_first_in_each_group_kernel(
    const int3* faces,
    const size_t F,
    uint8_t* face_mask
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= F) return;
    if (tid == 0) {
        face_mask[tid] = 1;
    } else {
        int3 face = faces[tid];
        int3 prev_face = faces[tid-1];
        if (face.x == prev_face.x && face.y == prev_face.y && face.z == prev_face.z) {
            face_mask[tid] = 0;
        } else {
            face_mask[tid] = 1;
        }
    }
}


struct int3_decomposer
{
    __host__ __device__ ::cuda::std::tuple<int&, int&, int&> operator()(int3& key) const
    {
        return {key.x, key.y, key.z};
    }
};


void CuMesh::remove_duplicate_faces() {
    size_t F = this->faces.size;

    // Sort faces idx
    sort_faces_kernel<<<(F+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        this->faces.ptr,
        F
    );
    CUDA_CHECK(cudaGetLastError());

    // Select first in each group of duplicate faces
    // Sort faces
    size_t temp_storage_bytes = 0;
    int3 *cu_new_faces;
    CUDA_CHECK(cudaMalloc(&cu_new_faces, F * sizeof(int3)));
    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
        nullptr, temp_storage_bytes,
        this->faces.ptr, cu_new_faces,
        F,
        int3_decomposer{}
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        this->faces.ptr, cu_new_faces,
        F,
        int3_decomposer{}
    ));

    // Select first in each group of duplicate faces
    uint8_t* cu_face_mask;
    CUDA_CHECK(cudaMalloc(&cu_face_mask, F * sizeof(uint8_t)));
    select_first_in_each_group_kernel<<<(F+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        cu_new_faces,
        F,
        cu_face_mask
    );
    CUDA_CHECK(cudaGetLastError());
    temp_storage_bytes = 0;
    int *cu_new_num_faces;
    CUDA_CHECK(cudaMalloc(&cu_new_num_faces, sizeof(int)));
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        nullptr, temp_storage_bytes,
        cu_new_faces, cu_face_mask, this->faces.ptr, cu_new_num_faces,
        F
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        cu_new_faces, cu_face_mask, this->faces.ptr, cu_new_num_faces,
        F
    ));
    int new_num_faces;
    CUDA_CHECK(cudaMemcpy(&new_num_faces, cu_new_num_faces, sizeof(int), cudaMemcpyDeviceToHost));
    this->faces.resize(new_num_faces);
    CUDA_CHECK(cudaFree(cu_new_num_faces));
    CUDA_CHECK(cudaFree(cu_new_faces));
    CUDA_CHECK(cudaFree(cu_face_mask));

    this->remove_unreferenced_vertices();
}


static __global__ void compute_loop_boundary_lengths(
    const float3* vertices,
    const uint64_t* edges,
    const int* loop_boundaries,
    const size_t E,
    float* loop_boundary_lengths
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= E) return;
    uint64_t edge = edges[loop_boundaries[tid]];
    int e0 = int(edge & 0xFFFFFFFF);
    int e1 = int(edge >> 32);
    Vec3f v0 = Vec3f(vertices[e0]);
    Vec3f v1 = Vec3f(vertices[e1]);
    loop_boundary_lengths[tid] = (v1 - v0).norm();
}


static __global__ void compute_loop_boundary_midpoints(
    const float3* vertices,
    const uint64_t* edges,
    const int* loop_boundaries,
    const size_t E,
    Vec3f* loop_boundary_midpoints
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= E) return;
    uint64_t edge = edges[loop_boundaries[tid]];
    int e0 = int(edge & 0xFFFFFFFF);
    int e1 = int(edge >> 32);
    Vec3f v0 = Vec3f(vertices[e0]);
    Vec3f v1 = Vec3f(vertices[e1]);
    loop_boundary_midpoints[tid] = (v0 + v1) * 0.5f;
}


static __global__ void connect_new_vertices_kernel(
    const uint64_t* edges,
    const int* loop_boundaries,
    const int* loop_bound_loop_ids,
    const size_t L,
    const size_t V,
    int3* faces
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= L) return;
    int loop_id = loop_bound_loop_ids[tid];
    int loop_boundary = loop_boundaries[tid];
    uint64_t e = edges[loop_boundary];
    int e0 = int(e & 0xFFFFFFFF);
    int e1 = int(e >> 32);
    int new_v_id = loop_id + V;
    faces[tid] = {e0, e1, new_v_id};
}


void CuMesh::fill_holes(float max_hole_perimeter) {
    if (this->loop_boundaries.is_empty() || this->loop_boundaries_offset.is_empty()) {
        throw std::runtime_error("loop_boundaries and loop_boundaries_offset must be computed before calling fill_holes.");
    }

    size_t V = this->vertices.size;
    size_t F = this->faces.size;
    size_t L = this->num_bound_loops;
    size_t E = this->loop_boundaries.size;

    // Compute loop boundary lengths
    float* cu_loop_boundary_lengths;
    CUDA_CHECK(cudaMalloc(&cu_loop_boundary_lengths, E * sizeof(float)));
    compute_loop_boundary_lengths<<<(E+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        this->vertices.ptr,
        this->edges.ptr,
        this->loop_boundaries.ptr,
        E,
        cu_loop_boundary_lengths
    );
    CUDA_CHECK(cudaGetLastError());

    // Segment sum
    size_t temp_storage_bytes = 0;
    float *cu_bound_loop_perimeters;
    CUDA_CHECK(cudaMalloc(&cu_bound_loop_perimeters, L * sizeof(float)));
    CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(
        nullptr, temp_storage_bytes,
        cu_loop_boundary_lengths, cu_bound_loop_perimeters,
        L,
        this->loop_boundaries_offset.ptr,
        this->loop_boundaries_offset.ptr + 1
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        cu_loop_boundary_lengths, cu_bound_loop_perimeters,
        L,
        this->loop_boundaries_offset.ptr,
        this->loop_boundaries_offset.ptr + 1
    ));
    CUDA_CHECK(cudaFree(cu_loop_boundary_lengths));

    // Mask small loops
    uint8_t* cu_bound_loop_mask;
    CUDA_CHECK(cudaMalloc(&cu_bound_loop_mask, L * sizeof(uint8_t)));
    less_than_kernel<<<(L+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        cu_bound_loop_perimeters,
        max_hole_perimeter,
        L,
        cu_bound_loop_mask
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(cu_bound_loop_perimeters));

    // Compress bound loops size
    int* cu_bound_loops_cnt;
    CUDA_CHECK(cudaMalloc(&cu_bound_loops_cnt, L * sizeof(int)));
    diff_kernel<<<(L+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        this->loop_boundaries_offset.ptr,
        L,
        cu_bound_loops_cnt
    );
    CUDA_CHECK(cudaGetLastError());
    int *cu_new_loop_boundaries_cnt, *cu_new_num_bound_loops;
    CUDA_CHECK(cudaMalloc(&cu_new_loop_boundaries_cnt, (L+1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cu_new_num_bound_loops, sizeof(int)));
    temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        nullptr, temp_storage_bytes,
        cu_bound_loops_cnt, cu_bound_loop_mask, cu_new_loop_boundaries_cnt, cu_new_num_bound_loops,
        L
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        cu_bound_loops_cnt, cu_bound_loop_mask, cu_new_loop_boundaries_cnt, cu_new_num_bound_loops,
        L
    ));
    int new_num_bound_loops;
    CUDA_CHECK(cudaMemcpy(&new_num_bound_loops, cu_new_num_bound_loops, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(cu_bound_loops_cnt));
    CUDA_CHECK(cudaFree(cu_new_num_bound_loops));
    if (new_num_bound_loops == 0) {
        CUDA_CHECK(cudaFree(cu_new_loop_boundaries_cnt));
        CUDA_CHECK(cudaFree(cu_bound_loop_mask));
        return;
    }

    // Get loop ids of loop boundaries
    int* cu_loop_bound_loop_ids;
    CUDA_CHECK(cudaMalloc(&cu_loop_bound_loop_ids, E * sizeof(int)));
    CUDA_CHECK(cudaMemset(cu_loop_bound_loop_ids, 0, E * sizeof(int)));
    if (L > 1) {
        set_flag_kernel<<<(L-1+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
            this->loop_boundaries_offset.ptr + 1, L - 1,
            cu_loop_bound_loop_ids
        );
        CUDA_CHECK(cudaGetLastError());
    }
    temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        nullptr, temp_storage_bytes,
        cu_loop_bound_loop_ids,
        E
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        cu_loop_bound_loop_ids,
        E
    ));
    
    // Mask loop boundaries
    uint8_t* cu_loop_boundary_mask;
    CUDA_CHECK(cudaMalloc(&cu_loop_boundary_mask, E * sizeof(uint8_t)));
    index_kernel<<<(E+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        cu_bound_loop_mask,
        cu_loop_bound_loop_ids,
        E,
        cu_loop_boundary_mask
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(cu_bound_loop_mask));
    CUDA_CHECK(cudaFree(cu_loop_bound_loop_ids));

    // Compress loop boundaries
    int *cu_new_loop_boundaries, *cu_new_num_loop_boundaries;
    CUDA_CHECK(cudaMalloc(&cu_new_loop_boundaries, E * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cu_new_num_loop_boundaries, sizeof(int)));
    temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        nullptr, temp_storage_bytes,
        this->loop_boundaries.ptr, cu_loop_boundary_mask, cu_new_loop_boundaries, cu_new_num_loop_boundaries,
        E
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        this->loop_boundaries.ptr, cu_loop_boundary_mask, cu_new_loop_boundaries, cu_new_num_loop_boundaries,
        E
    ));
    int new_num_loop_boundaries;
    CUDA_CHECK(cudaMemcpy(&new_num_loop_boundaries, cu_new_num_loop_boundaries, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(cu_new_num_loop_boundaries));
    CUDA_CHECK(cudaFree(cu_loop_boundary_mask));

    // Reconstruct new bound loops
    int* cu_new_loop_boundaries_offset;
    CUDA_CHECK(cudaMalloc(&cu_new_loop_boundaries_offset, (new_num_loop_boundaries+1) * sizeof(int)));
    temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, temp_storage_bytes,
        cu_new_loop_boundaries_cnt, cu_new_loop_boundaries_offset,
        new_num_bound_loops + 1
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        cu_new_loop_boundaries_cnt, cu_new_loop_boundaries_offset,
        new_num_bound_loops + 1
    ));
    int* cu_new_loop_bound_loop_ids;
    CUDA_CHECK(cudaMalloc(&cu_new_loop_bound_loop_ids, new_num_loop_boundaries * sizeof(int)));
    CUDA_CHECK(cudaMemset(cu_new_loop_bound_loop_ids, 0, new_num_loop_boundaries * sizeof(int)));
    if (new_num_bound_loops > 1) {
        set_flag_kernel<<<(new_num_bound_loops-1+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
            cu_new_loop_boundaries_offset+1, new_num_bound_loops-1,
            cu_new_loop_bound_loop_ids
        );
        CUDA_CHECK(cudaGetLastError());
    }
    temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        nullptr, temp_storage_bytes,
        cu_new_loop_bound_loop_ids,
        new_num_loop_boundaries
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        cu_new_loop_bound_loop_ids,
        new_num_loop_boundaries
    ));
    
    // Calculate new vertex positions as average of loop vertices
    Vec3f* cu_new_loop_bound_centers;
    CUDA_CHECK(cudaMalloc(&cu_new_loop_bound_centers, new_num_loop_boundaries * sizeof(Vec3f)));
    compute_loop_boundary_midpoints<<<(new_num_loop_boundaries+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        this->vertices.ptr,
        this->edges.ptr,
        cu_new_loop_boundaries,
        new_num_loop_boundaries,
        cu_new_loop_bound_centers
    );
    CUDA_CHECK(cudaGetLastError());
    Vec3f* cu_new_vertices;
    CUDA_CHECK(cudaMalloc(&cu_new_vertices, new_num_bound_loops * sizeof(Vec3f)));
    temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(
        nullptr, temp_storage_bytes,
        cu_new_loop_bound_centers, cu_new_vertices,
        new_num_bound_loops,
        cu_new_loop_boundaries_offset,
        cu_new_loop_boundaries_offset + 1
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        cu_new_loop_bound_centers, cu_new_vertices,
        new_num_bound_loops,
        cu_new_loop_boundaries_offset,
        cu_new_loop_boundaries_offset + 1
    ));
    CUDA_CHECK(cudaFree(cu_new_loop_bound_centers));
    CUDA_CHECK(cudaFree(cu_new_loop_boundaries_offset));
    inplace_div_kernel<<<(new_num_bound_loops+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        cu_new_vertices,
        cu_new_loop_boundaries_cnt,
        new_num_bound_loops
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(cu_new_loop_boundaries_cnt));

    // Update mesh
    this->vertices.extend(new_num_bound_loops);
    this->faces.extend(new_num_loop_boundaries);
    copy_vec3f_to_float3_kernel<<<(new_num_bound_loops+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        cu_new_vertices,
        new_num_bound_loops,
        this->vertices.ptr + V
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(cu_new_vertices));
    connect_new_vertices_kernel<<<(new_num_loop_boundaries+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        this->edges.ptr,
        cu_new_loop_boundaries,
        cu_new_loop_bound_loop_ids,
        new_num_loop_boundaries,
        V,
        this->faces.ptr + F
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(cu_new_loop_boundaries));
    CUDA_CHECK(cudaFree(cu_new_loop_bound_loop_ids));

    // Delete all connectivity info since mesh has changed
    this->clear_connectivity();
}


// __global__ void calc_face_areas(
//     const float3* vertices,
//     const int3* faces,
//     const size_t F,
//     float* face_areas
// ) {
//     const int fid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (fid >= F) return;
//     int3 face = faces[fid];
//     Vec3f v0 = Vec3f(vertices[face.x]);
//     Vec3f v1 = Vec3f(vertices[face.y]);
//     Vec3f v2 = Vec3f(vertices[face.z]);
//     face_areas[fid] = 0.5 * (v1 - v0).cross(v2 - v0).norm();
// }


// void CuMesh::remove_small_connected_components(int min_area) {
//     size_t V = this->vertices.size;
//     size_t F = this->faces.size;

//     // Calculate face areas
//     float* cu_face_areas;
//     cudaMalloc(&cu_face_areas, F * sizeof(float));
//     calc_face_areas<<<(F+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
//         this->vertices.ptr,
//         this->faces.ptr,
//         F,
//         cu_face_areas
//     );

//     // Sort faces by connected component ids
//     size_t temp_storage_bytes = 0;
//     int *cu_sorted_conn_comp_ids;
//     float *cu_sorted_face_areas;
//     cudaMalloc(&cu_sorted_conn_comp_ids, F * sizeof(int));
//     cudaMalloc(&cu_sorted_face_areas, F * sizeof(float));
//     CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
//         nullptr, temp_storage_bytes,
//         this->conn_comp_ids.ptr, cu_sorted_conn_comp_ids,
//         cu_face_areas, cu_sorted_face_areas,
//         F
//     );
//     this->cub_temp_storage.resize(temp_storage_bytes);
//     CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
//         this->cub_temp_storage.ptr, temp_storage_bytes,
//         this->conn_comp_ids.ptr, cu_sorted_conn_comp_ids,
//         cu_face_areas, cu_sorted_face_areas,
//         F
//     );
//     cudaFree(cu_face_areas);
    
//     // Find unique connected component ids and their number of faces
//     int *cu_unique_conn_comp_ids;
//     int *cu_conn_comp_num_faces;
//     int *cu_num_conn_comps;
//     cudaMalloc(&cu_unique_conn_comp_ids, F * sizeof(int));
//     cudaMalloc(&cu_conn_comp_num_faces, (F + 1) * sizeof(int));
//     cudaMalloc(&cu_num_conn_comps, sizeof(int));
//     CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(
//         nullptr, temp_storage_bytes,
//         cu_sorted_conn_comp_ids, cu_unique_conn_comp_ids,
//         cu_conn_comp_num_faces, cu_num_conn_comps,
//         F
//     );
//     this->cub_temp_storage.resize(temp_storage_bytes);
//     CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(
//         this->cub_temp_storage.ptr, temp_storage_bytes,
//         cu_sorted_conn_comp_ids, cu_unique_conn_comp_ids,
//         cu_conn_comp_num_faces, cu_num_conn_comps,
//         F
//     );
//     int num_conn_comps;
//     cudaMemcpy(&num_conn_comps, cu_num_conn_comps, sizeof(int), cudaMemcpyDeviceToHost);
//     cudaFree(cu_num_conn_comps);
//     cudaFree(cu_sorted_conn_comp_ids);
//     cudaFree(cu_unique_conn_comp_ids);

//     // Compute connected component areas
//     float *cu_conn_comp_areas;
//     cudaMalloc(&cu_conn_comp_areas, num_conn_comps * sizeof(float));
//     temp_storage_bytes = 0;
//     CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
//         nullptr, temp_storage_bytes,
//         cu_conn_comp_num_faces,F+1,
//     );



//     cudaFree(cu_sorted_face_areas);
//     cudaFree(cu_conn_comp_num_faces);
    


// }

} // namespace cumesh