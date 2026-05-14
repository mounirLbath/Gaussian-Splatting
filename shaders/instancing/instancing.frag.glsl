#version 430 core


layout(location=0) out vec4 FragColor;

flat in vec3 frag_color;
flat in float frag_opacity;
in vec2 uv;

uniform float alpha_cutoff;

void main()
{
	float frag_exponent = -0.5 * dot(uv, uv);
	
	float alpha = min(1.0, frag_opacity * exp(frag_exponent));
	if(alpha < alpha_cutoff) discard;

	FragColor = vec4(frag_color, alpha);
}
