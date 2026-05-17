#version 430 core

// Phase 2: Radix sort - exclusive prefix over bins to get global base offsets.

layout(local_size_x = 256) in;

layout(std430, binding = 12) buffer BinTotals { uint data[]; } bin_totals; // in/out

shared uint temp[256];

void main()
{
    uint tid = gl_LocalInvocationID.x;
    uint v = bin_totals.data[tid];
    temp[tid] = v;
    barrier();

    for (uint offset = 1u; offset < 256u; offset <<= 1u) {
        uint add_val = 0u;
        if (tid >= offset) add_val = temp[tid - offset];
        barrier();
        temp[tid] += add_val;
        barrier();
    }

    uint inclusive = temp[tid];
    bin_totals.data[tid] = inclusive - v; // exclusive base
}
