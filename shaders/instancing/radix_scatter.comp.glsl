#version 430 core

// Phase 2: Radix sort - stable scatter within each block.
// Thread 0 of each block performs the final ordered scatter to guarantee stability.

layout(local_size_x = 256) in;

const uint ITEMS_PER_THREAD = 4u;
const uint BLOCK_SIZE = 256u * ITEMS_PER_THREAD;

layout(std430, binding = 9) readonly buffer SortIn { uint data[]; } sort_in;
layout(std430, binding = 10) writeonly buffer SortOut { uint data[]; } sort_out;

struct SplatRecord {
    vec4 center_axis1;
    vec4 axis2_aabb;
    vec4 color_opacity;
    uvec4 packed_data;
};
layout(std430, binding = 5) readonly buffer SplatViewData { SplatRecord data[]; } view_data;

// Block prefix offsets (written by radix_block_prefix).
layout(std430, binding = 11) readonly buffer BlockPrefix { uint data[]; } block_prefix;

// Global base offsets per bin (written by radix_bin_prefix).
layout(std430, binding = 12) readonly buffer BinBase { uint data[]; } bin_base;

uniform uint element_count;
uniform uint pass_shift;
uniform uint block_count;

shared uint sh_ids[BLOCK_SIZE];
shared uint sh_bins[BLOCK_SIZE];
shared uint sh_counts[256];

void main()
{
    uint block_id = gl_WorkGroupID.x;
    if (block_id >= block_count) return;

    uint local_id = gl_LocalInvocationID.x;
    if (local_id < 256u) sh_counts[local_id] = 0u;

    uint block_start = block_id * BLOCK_SIZE;
    uint block_end = min(block_start + BLOCK_SIZE, element_count);
    uint count_in_block = block_end - block_start;

    // Load the block into shared memory.
    for (uint i = 0u; i < ITEMS_PER_THREAD; ++i) {
        uint idx = block_start + local_id + i * 256u;
        uint dst = local_id + i * 256u;
        if (idx < block_end) {
            uint splat_id = sort_in.data[idx];
                uint key = ~view_data.data[splat_id].packed_data.x; // invert for back-to-front order
            uint bin = (key >> pass_shift) & 0xFFu;
            sh_ids[dst] = splat_id;
            sh_bins[dst] = bin;
        }
    }
    barrier();

    // Stable scatter: one thread walks the block in order.
    if (local_id == 0u) {
        uint base_idx = block_id * 256u;
        for (uint i = 0u; i < count_in_block; ++i) {
            uint bin = sh_bins[i];
            uint out_idx = bin_base.data[bin] + block_prefix.data[base_idx + bin] + sh_counts[bin];
            sh_counts[bin] += 1u;
            sort_out.data[out_idx] = sh_ids[i];
        }
    }
}
