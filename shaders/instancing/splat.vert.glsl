#version 430 core

// Phase 1 thin vertex shader. All projection math has already been done by
// shaders/instancing/project.comp.glsl. This shader only:
//   - reads the precomputed splat record using gl_InstanceID
//   - reads the visible splat index from visible_indices[]
//   - expands the unit-quad vertex into the screen-aligned ellipse

layout(location = 0) in vec2 quad_position; // unit quad in [-1, 1]^2

struct SplatRecord {
    vec4 center_axis1;   // xy = NDC center, zw = ellipse axis 1 in NDC
    vec4 axis2_aabb;     // xy = ellipse axis 2 in NDC, zw = NDC AABB half-extent
    vec4 color_opacity;  // rgb = color, a = opacity
    uvec4 packed_data;   // x = depth_key, y = view_depth bits, z = visible, w = pad
};

layout(std430, binding = 5) readonly buffer SplatViewData  { SplatRecord data[]; } view_data;
layout(std430, binding = 6) readonly buffer VisibleIndices { uint data[]; } visible_indices;

flat out vec3 frag_color;
flat out float frag_opacity;
out vec2 uv;

void main()
{
    uint splat_id = visible_indices.data[gl_InstanceID];
    SplatRecord rec = view_data.data[splat_id];

    vec2 ndc_center = rec.center_axis1.xy;
    vec2 axis1      = rec.center_axis1.zw;
    vec2 axis2      = rec.axis2_aabb.xy;

    frag_color   = rec.color_opacity.rgb;
    frag_opacity = rec.color_opacity.a;

    // axis1/axis2 already include the per-splat `spread` factor, so the unit-quad
    // corner directly gives the screen position.
    vec2 ndc = ndc_center + quad_position.x * axis1 + quad_position.y * axis2;

    // The fragment's Mahalanobis falloff is exp(-0.5 * dot(uv,uv)). To match the
    // original visual semantics, |uv| at the quad edge must equal the per-splat
    // spread (which was derived from alpha_cutoff/opacity in the compute shader).
    float spread = uintBitsToFloat(rec.packed_data.w);
    uv = quad_position * spread;

    gl_Position = vec4(ndc, 0.0, 1.0);
}
