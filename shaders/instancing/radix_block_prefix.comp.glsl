#version 430 core

// Phase 2: Radix sort - per-bin prefix over blocks.
// For each bin, compute the prefix sum over blocks and write the per-block
// prefix back into block_hist. Also output the total count per bin.

layout(local_size_x = 256) in;

layout(std430, binding = 11) buffer BlockHist { uint data[]; } block_hist; // in/out
layout(std430, binding = 12) writeonly buffer BinTotals { uint data[]; } bin_totals;

uniform uint block_count;

void main()
{
    uint bin = gl_LocalInvocationID.x;
    uint sum = 0u;

    for (uint b = 0u; b < block_count; ++b) {
        uint idx = b * 256u + bin;
        uint count = block_hist.data[idx];
        block_hist.data[idx] = sum;
        sum += count;
    }

    bin_totals.data[bin] = sum;
}
