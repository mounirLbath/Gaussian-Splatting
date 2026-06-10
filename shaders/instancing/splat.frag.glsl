#version 430 core

layout(location = 0) out vec4 FragColor;

flat in vec3 frag_color;
flat in float frag_opacity;
in vec2 uv;

uniform float alpha_cutoff;

void main()
{
    float alpha = min(1.0, frag_opacity * exp(-0.5 * dot(uv, uv)));
    if (alpha < alpha_cutoff) discard;
    FragColor = vec4(frag_color, alpha);
}
