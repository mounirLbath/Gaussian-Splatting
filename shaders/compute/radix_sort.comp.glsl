#version 430 core

layout(local_size_x = 128) in;

// One radix pass sorts by one byte of the key, so there are always 256 bins.
// A workgroup owns one contiguous block of input items. We use 128 lanes and four chunks, giving 512 input items per block.
const uint WORKGROUP_SIZE = 128u;
const uint ITEMS_PER_THREAD = 4u;
const uint BLOCK_SIZE = WORKGROUP_SIZE * ITEMS_PER_THREAD;
const uint RADIX_BIN_COUNT = 256u;

struct SortPair {
    uint key;
    uint index;
};

layout(std430, binding = 6) readonly buffer SortIn  { SortPair data[]; } sort_in;
layout(std430, binding = 7) writeonly buffer SortOut { SortPair data[]; } sort_out;
// radix_hist layout:
//   phase 1 output: [block_count][256] bin counts
//   phase 2 output: [block_count][256] per-block exclusive offsets inside each bin
layout(std430, binding = 8) buffer RadixHist   { uint data[]; } radix_hist;
// Global exclusive start offset for each radix bin after phase 2.
layout(std430, binding = 9) buffer RadixPrefix { uint data[]; } radix_prefix;

struct DrawElementsIndirectCommand {
    uint count;
    uint instanceCount;
    uint firstIndex;
    int baseVertex;
    uint baseInstance;
};

layout(std430, binding = 10) readonly buffer IndirectCommand { DrawElementsIndirectCommand cmd; } indirect_cmd;

uniform int radix_phase; // 0=clear, 1=histogram, 2=prefix sum, 3=scatter
uniform int shift;       // bit shift for this radix pass
uniform uint block_count;

// Shared scratch arrays are 256 entries because the radix digit is 8 bits.
shared uint local_hist[256];
shared uint bin_totals[256];
shared uint local_offsets[256];
shared uint chunk_counts[256];

void main()
{
    uint gid = gl_GlobalInvocationID.x;
    uint element_count = indirect_cmd.cmd.instanceCount;

    // Phase 0: clear this pass' per-block histogram storage.
    // Each workgroup clears the 256 bins for its own block.
    if (radix_phase == 0) {
        uint block_id = gl_WorkGroupID.x;
        if (block_id < block_count) {
            for (uint bin = gl_LocalInvocationID.x; bin < RADIX_BIN_COUNT; bin += WORKGROUP_SIZE) {
                radix_hist.data[block_id * RADIX_BIN_COUNT + bin] = 0u;
            }
        }
        return;
    }

    // Phase 1: build one local 256-bin histogram per input block.
    // Atomics are only in shared memory, so many blocks can run independently without contending on one global 256-bin histogram.
    if (radix_phase == 1) {
        uint block_id = gl_WorkGroupID.x;
        if (block_id >= block_count) return;

        uint local_id = gl_LocalInvocationID.x;
        for (uint bin = local_id; bin < RADIX_BIN_COUNT; bin += WORKGROUP_SIZE) {
            local_hist[bin] = 0u;
        }
        barrier();

        uint block_start = block_id * BLOCK_SIZE;
        uint block_end = min(block_start + BLOCK_SIZE, element_count);
        for (uint idx = block_start + local_id; idx < block_end; idx += WORKGROUP_SIZE) {
            uint key = sort_in.data[idx].key;
            uint bin = (key >> uint(shift)) & (RADIX_BIN_COUNT - 1u);
            atomicAdd(local_hist[bin], 1u);
        }

        barrier();
        for (uint bin = local_id; bin < RADIX_BIN_COUNT; bin += WORKGROUP_SIZE) {
            radix_hist.data[block_id * RADIX_BIN_COUNT + bin] = local_hist[bin];
        }
        return;
    }

    // Phase 2: compute the offsets needed by scatter.
    if (radix_phase == 2) {
        uint local_id = gl_LocalInvocationID.x;
        for (uint bin = local_id; bin < RADIX_BIN_COUNT; bin += WORKGROUP_SIZE) {
            uint sum = 0u;
            for (uint b = 0u; b < block_count; ++b) {
                uint idx = b * RADIX_BIN_COUNT + bin;
                uint count = radix_hist.data[idx];
                radix_hist.data[idx] = sum;
                sum += count;
            }

            local_hist[bin] = sum;
            bin_totals[bin] = sum;
        }
        barrier();

        for (uint offset = 1u; offset < RADIX_BIN_COUNT; offset <<= 1u) {
            for (uint bin = local_id; bin < RADIX_BIN_COUNT; bin += WORKGROUP_SIZE) {
                local_offsets[bin] = bin >= offset ? bin_totals[bin - offset] : 0u;
            }
            barrier();
            for (uint bin = local_id; bin < RADIX_BIN_COUNT; bin += WORKGROUP_SIZE) {
                bin_totals[bin] += local_offsets[bin];
            }
            barrier();
        }

        for (uint bin = local_id; bin < RADIX_BIN_COUNT; bin += WORKGROUP_SIZE) {
            radix_prefix.data[bin] = bin_totals[bin] - local_hist[bin];
        }
        return;
    }

    // Phase 3: stable scatter from input to output.
    if (radix_phase == 3) {
        uint block_id = gl_WorkGroupID.x;
        if (block_id >= block_count) return;

        uint local_id = gl_LocalInvocationID.x;
        for (uint bin = local_id; bin < RADIX_BIN_COUNT; bin += WORKGROUP_SIZE) {
            local_offsets[bin] = 0u;
        }
        barrier();

        uint block_start = block_id * BLOCK_SIZE;
        uint block_end = min(block_start + BLOCK_SIZE, element_count);
        uint base_idx = block_id * RADIX_BIN_COUNT;

        for (uint chunk = 0u; chunk < ITEMS_PER_THREAD; ++chunk) {
            for (uint bin_idx = local_id; bin_idx < RADIX_BIN_COUNT; bin_idx += WORKGROUP_SIZE) {
                chunk_counts[bin_idx] = 0u;
            }
            barrier();

            uint item = block_start + chunk * WORKGROUP_SIZE + local_id;
            bool item_valid = item < block_end;
            SortPair p;
            uint bin = 0u;
            if (item_valid) {
                p = sort_in.data[item];
                bin = (p.key >> uint(shift)) & (RADIX_BIN_COUNT - 1u);
                atomicAdd(chunk_counts[bin], 1u);
            }
            barrier();

            if (item_valid) {
                uint local_rank = 0u;
                uint chunk_start = block_start + chunk * WORKGROUP_SIZE;
                for (uint j = 0u; j < local_id; ++j) {
                    uint prev = chunk_start + j;
                    if (prev < block_end) {
                        uint prev_bin = (sort_in.data[prev].key >> uint(shift)) & (RADIX_BIN_COUNT - 1u);
                        if (prev_bin == bin) ++local_rank;
                    }
                }

                // Final destination = global bin base + previous blocks in bin
                // + previous chunks in this block/bin + previous lanes in chunk/bin.
                uint dst = radix_prefix.data[bin] + radix_hist.data[base_idx + bin] + local_offsets[bin] + local_rank;
                sort_out.data[dst] = p;
            }

            barrier();
            for (uint bin_idx = local_id; bin_idx < RADIX_BIN_COUNT; bin_idx += WORKGROUP_SIZE) {
                local_offsets[bin_idx] += chunk_counts[bin_idx];
            }
            barrier();
        }
        return;
    }
}
