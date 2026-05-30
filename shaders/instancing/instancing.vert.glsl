#version 430 core

layout (location = 0) in vec3 vertex_position; // vertex position in local space (x,y,z)

layout(location = 4) in uint instance_idx;

// Per-splat data uploaded once as std430 SSBOs. Each vec3 is padded to vec4
layout(std430, binding = 0) readonly buffer SplatPoints      { vec4  data[]; } splat_points;
layout(std430, binding = 1) readonly buffer SplatColors      { vec4  data[]; } splat_colors;
// 2 vec4 per splat: (Sxx, Syy, Szz, Sxy) then (Sxz, Syz, _, _)
layout(std430, binding = 2) readonly buffer SplatCovariances { vec4  data[]; } splat_covariances;
layout(std430, binding = 3) readonly buffer SplatOpacities   { float data[]; } splat_opacities;


uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float alpha_cutoff;

flat out vec3 frag_color;
flat out float frag_opacity;
out vec2 uv;


void main()
{
	int i = int(instance_idx);
	vec3  instance_position = splat_points.data[i].xyz;
	vec3  instance_color = splat_colors.data[i].xyz;
	vec4  cov_a = splat_covariances.data[2*i + 0]; // (Sxx, Syy, Szz, Sxy)
	vec4  cov_b = splat_covariances.data[2*i + 1]; // (Sxz, Syz, _, _)
	float instance_opacity  = splat_opacities.data[i];


	frag_color = instance_color;
	frag_opacity = instance_opacity;


	// center in world
	mat4 MV = view * model;
	vec4 view_center =  MV * vec4(instance_position, 1.0);

	// Early reject splats behind the near plane to avoid unstable projection.
	if (view_center.z >= -1e-4) {
		frag_opacity = 0.0;
		gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
		return;
	}


	// Reconstruct the symmetric 3D covariance from its 6 unique entries
	mat3 sigma3D = mat3(
		cov_a.x, cov_a.w, cov_b.x,  // col 0: (Sxx, Sxy, Sxz)
		cov_a.w, cov_a.y, cov_b.y,  // col 1: (Sxy, Syy, Syz)
		cov_b.x, cov_b.y, cov_a.z   // col 2: (Sxz, Syz, Szz)
	);
	mat3 V = mat3(MV);
	mat3 sigma_view = V * sigma3D * transpose(V);

	// compute the Jacobian of the projection on the 2d screen
	float diag1 = projection[0][0];
	float diag2 = projection[1][1];
	mat3x2 J = transpose(mat2x3(
		-diag1/view_center.z, 0.0                 , view_center.x * diag1 / (view_center.z*view_center.z),
		0.0                 , -diag2/view_center.z, view_center.y * diag2 / (view_center.z*view_center.z)));

	mat2 sigma2D = J * sigma_view * transpose(J); // the covariance matrix in NDC coordinates

	// compute the quad position in NDC coordinates
	float a = sigma2D[0][0]; float b = sigma2D[1][1]; float c = sigma2D[0][1];
	float det = a*b-c*c; 
	float tr = a+b;
	float discr = tr*tr-4*det;
	float lambda1 = (tr + sqrt(discr))/2.0; 
	float lambda2 = (tr - sqrt(discr))/2.0; 

	vec2 dir1 = vec2((lambda1-b), c); dir1 = normalize(dir1);
	vec2 dir2 = vec2(-dir1.y, dir1.x); 
	
	// Early reject splats that are too transparent 
	if (instance_opacity <= 1e-8) {
		frag_opacity = 0.0;
		gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
		return;
	}

	// Compute spread based on alpha_cutoff
	float ratio = alpha_cutoff / max(instance_opacity, 1e-8);
	float spread = (ratio < 1.0) ? sqrt(-2.0 * log(ratio)) : 0.0;
	if (spread <= 0.0) {
		frag_opacity = 0.0;
		gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
		return;
	}
	vec2 axis1_ndc = spread * sqrt(lambda1) * dir1;
	vec2 axis2_ndc = spread * sqrt(lambda2) * dir2;

	vec4 clip_center = projection * view_center;
	vec2 ndc_center = clip_center.xy / clip_center.w;
	float rx = abs(axis1_ndc.x) + abs(axis2_ndc.x);
	float ry = abs(axis1_ndc.y) + abs(axis2_ndc.y);

	// If out of the NDC bounds / screen, early reject the splat
	if (ndc_center.x + rx < -1.0 || ndc_center.x - rx > 1.0 ||
		ndc_center.y + ry < -1.0 || ndc_center.y - ry > 1.0) {
		frag_opacity = 0.0;
		gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
		return;
	}
	vec2 delta = axis1_ndc * vertex_position.x + axis2_ndc * vertex_position.y;
	
	uv = vertex_position.xy *spread;

	vec4 screen_position = clip_center;

	screen_position.xy += delta * screen_position.w;

	gl_Position = screen_position;
}
