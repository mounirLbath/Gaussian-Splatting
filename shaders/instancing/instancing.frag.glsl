#version 330 core


layout(location=0) out vec4 FragColor;

flat in vec3 frag_color;
flat in float frag_opacity;
in vec2 frag_delta;
flat in mat2 inv_sigma2D;

void main()
{
	float frag_exponent = -0.5 * dot(frag_delta, inv_sigma2D * frag_delta);
	
	float alpha = min(1.0, frag_opacity * exp(frag_exponent));
	if(alpha < 1e-5) discard;

	FragColor = vec4(frag_color, alpha);
}
