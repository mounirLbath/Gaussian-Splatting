#version 430 core

// Phase 2: Radix sort - block histogram pass.
// One workgroup processes a contiguous block of elements and writes a 256-bin
// histogram to the global block histogram buffer.

layout(local_size_x = 256) in;

const uint ITEMS_PER_THREAD = 4u;
const uint BLOCK_SIZE = 256u * ITEMS_PER_THREAD;

// Input visible indices (to be sorted).
layout(std430, binding = 9) readonly buffer SortIn { uint data[]; } sort_in;

// View data provides the radix key (packed.x).
struct SplatRecord {
    vec4 center_axis1;
    vec4 axis2_aabb;
    vec4 color_opacity;
    uvec4 packed_data;
};
layout(std430, binding = 5) readonly buffer SplatViewData { SplatRecord data[]; } view_data;

// Output block histograms: [block_count][256].
layout(std430, binding = 11) writeonly buffer BlockHist { uint data[]; } block_hist;

uniform uint element_count;
uniform uint pass_shift;
uniform uint block_count;

shared uint local_hist[256];

void main()
{
    uint block_id = gl_WorkGroupID.x;
    if (block_id >= block_count) return;

    uint local_id = gl_LocalInvocationID.x;
    local_hist[local_id] = 0u;
    barrier();

    uint block_start = block_id * BLOCK_SIZE;
    uint block_end = min(block_start + BLOCK_SIZE, element_count);

    for (uint idx = block_start + local_id; idx < block_end; idx += 256u) {
        uint splat_id = sort_in.data[idx];
        uint key = ~view_data.data[splat_id].packed_data.x; // invert for back-to-front order
        uint bin = (key >> pass_shift) & 0xFFu;
        atomicAdd(local_hist[bin], 1u);
    }

    barrier();
    block_hist.data[block_id * 256u + local_id] = local_hist[local_id];
}
