#version 430 core

// Phase 1 projection compute shader.
// One invocation per splat. For each splat:
//   - reconstruct 3D covariance
//   - project to view space, build 2D screen-space covariance via the same Jacobian
//     as the original vertex shader
//   - derive the closed-form max radius from alpha_cutoff and opacity
//   - reject splats that are behind the near plane, off-screen, or too small
//   - write a compact per-splat record to splat_view_data[]
//   - if visible, append the splat id to visible_indices[] using an atomic counter
//   - the indirect-draw instanceCount is updated to match the visible count

layout(local_size_x = 256) in;

// Input SSBOs (same layout as the original vertex shader)
layout(std430, binding = 0) readonly buffer SplatPoints      { vec4  data[]; } splat_points;
layout(std430, binding = 1) readonly buffer SplatColors      { vec4  data[]; } splat_colors;
layout(std430, binding = 2) readonly buffer SplatCovariances { vec4  data[]; } splat_covariances;
layout(std430, binding = 3) readonly buffer SplatOpacities   { float data[]; } splat_opacities;

// Output: one record per splat. 48 bytes packed as 3 vec4 + 1 uvec4.
struct SplatRecord {
    vec4 center_axis1;   // xy = NDC center, zw = ellipse axis 1 in NDC
    vec4 axis2_aabb;     // xy = ellipse axis 2 in NDC, zw = NDC half-extent (rx, ry)
    vec4 color_opacity;  // rgb = color, a = opacity
    uvec4 packed_data;   // x = depth_key (float-flipped), y = floatBitsToUint(view_depth), z = visible, w = pad
};

layout(std430, binding = 5) writeonly buffer SplatViewData    { SplatRecord data[]; } view_data;
layout(std430, binding = 6) writeonly buffer VisibleIndices   { uint data[]; } visible_indices;

// Indirect draw command: layout matches glDrawElementsIndirect (5 uints).
// The compute shader directly increments `instanceCount` so it doubles as the
// visible-splat counter; no separate counter buffer needed.
struct DrawElementsIndirectCommand {
    uint count;          // index count per instance (6 for the quad), set on the CPU
    uint instanceCount;  // atomic counter == visible splat count
    uint firstIndex;
    uint baseVertex;
    uint baseInstance;
};
layout(std430, binding = 8) coherent buffer IndirectDraw { DrawElementsIndirectCommand cmd; } indirect;

uniform mat4 view;
uniform mat4 projection;
uniform float alpha_cutoff;
uniform uint  splat_count;

// Float-flip: turn a 32-bit float into a 32-bit unsigned integer so that
// numerical order on the unsigned interpretation matches the float order.
// This is the standard radix-sort key transform.
uint depth_to_key(float d)
{
    // We sort BACK-TO-FRONT, i.e. larger depth must come first. After float-flipping,
    // larger-float == larger-uint, so the final sort can use ascending order and we
    // walk it in reverse, OR we can flip the bits to sort descending. To keep things
    // simple we store the raw flipped key here; the consumer chooses the order.
    uint u = floatBitsToUint(d);
    // Flip sign bit if positive, flip all bits if negative -> monotonic in float order.
    uint mask = (u & 0x80000000u) != 0u ? 0xFFFFFFFFu : 0x80000000u;
    return u ^ mask;
}

void main()
{
    uint i = gl_GlobalInvocationID.x;
    if (i >= splat_count) return;

    vec3 instance_position = splat_points.data[i].xyz;
    vec3 instance_color    = splat_colors.data[i].xyz;
    vec4 cov_a = splat_covariances.data[2u*i + 0u];
    vec4 cov_b = splat_covariances.data[2u*i + 1u];
    float instance_opacity = splat_opacities.data[i];

    // View-space center.
    vec4 view_center = view * vec4(instance_position, 1.0);

    // Initialize the record (we will overwrite below if visible).
    SplatRecord rec;
    rec.center_axis1  = vec4(0.0);
    rec.axis2_aabb    = vec4(0.0);
    rec.color_opacity = vec4(instance_color, instance_opacity);
    rec.packed_data = uvec4(0u, floatBitsToUint(-view_center.z), 0u, 0u);

    // Reject splats behind the near plane.
    // view_center.z is negative when in front of the camera in standard right-handed view space.
    if (view_center.z >= -1e-4) {
        view_data.data[i] = rec;
        return;
    }

    // Reconstruct symmetric 3D covariance.
    mat3 sigma3D = mat3(
        cov_a.x, cov_a.w, cov_b.x,
        cov_a.w, cov_a.y, cov_b.y,
        cov_b.x, cov_b.y, cov_a.z);

    mat3 V = mat3(view);
    mat3 sigma_view = V * sigma3D * transpose(V);

    // Jacobian of the perspective projection.
    float diag1 = projection[0][0];
    float diag2 = projection[1][1];
    float invz = 1.0 / view_center.z;
    mat3x2 J = transpose(mat2x3(
        -diag1 * invz, 0.0,           view_center.x * diag1 * invz * invz,
        0.0,           -diag2 * invz, view_center.y * diag2 * invz * invz));

    mat2 sigma2D = J * sigma_view * transpose(J);

    float a = sigma2D[0][0]; float b = sigma2D[1][1]; float c = sigma2D[0][1];
    float det = a*b - c*c;
    float tr  = a + b;
    float discr = max(tr*tr - 4.0*det, 0.0);
    float lambda1 = (tr + sqrt(discr)) * 0.5;
    float lambda2 = (tr - sqrt(discr)) * 0.5;

    if (lambda1 <= 1e-12 || lambda2 <= 1e-12) {
        view_data.data[i] = rec;
        return;
    }

    vec2 dir1 = normalize(vec2(lambda1 - b, c));
    vec2 dir2 = vec2(-dir1.y, dir1.x);

    // Closed-form max evaluation radius (in standard-deviation units) so that
    //   opacity * exp(-0.5 * r^2) >= alpha_cutoff
    // This is the same cutoff used by the fragment shader, just hoisted -> non-lossy.
    float cutoff_for_bounds = max(alpha_cutoff, 1e-6);
    float ratio = cutoff_for_bounds / max(instance_opacity, 1e-8);
    float spread = (ratio < 1.0) ? sqrt(-2.0 * log(ratio)) : 0.0;

    if (spread <= 0.0) {
        view_data.data[i] = rec;
        return;
    }

    // NDC-space ellipse axes.
    vec2 axis1_ndc = spread * sqrt(lambda1) * dir1;
    vec2 axis2_ndc = spread * sqrt(lambda2) * dir2;

    // NDC center.
    vec4 clip = projection * view_center;
    vec2 ndc_center = clip.xy / clip.w;

    // Conservative NDC AABB half-extent (axis-aligned bounding box of the ellipse).
    float rx = abs(axis1_ndc.x) + abs(axis2_ndc.x);
    float ry = abs(axis1_ndc.y) + abs(axis2_ndc.y);

    // Frustum cull on x/y (with the AABB as conservative bound).
    if (ndc_center.x + rx < -1.0 || ndc_center.x - rx > 1.0 ||
        ndc_center.y + ry < -1.0 || ndc_center.y - ry > 1.0) {
        view_data.data[i] = rec;
        return;
    }

    // Visible. Fill record and append to the visible list.
    rec.center_axis1 = vec4(ndc_center, axis1_ndc);
    rec.axis2_aabb   = vec4(axis2_ndc, rx, ry);
    rec.packed_data.x = depth_to_key(-view_center.z); // larger key -> farther
    rec.packed_data.z = 1u;
    rec.packed_data.w = floatBitsToUint(spread); // pass per-splat spread to the vertex shader

    view_data.data[i] = rec;

    uint slot = atomicAdd(indirect.cmd.instanceCount, 1u);
    if (slot < splat_count) {
        visible_indices.data[slot] = i;
    }
}
