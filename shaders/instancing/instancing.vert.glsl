#version 330 core

layout (location = 0) in vec3 vertex_position; // vertex position in local space (x,y,z)

layout(location = 4) in uint instance_idx;

uniform samplerBuffer splat_points_tbo;
uniform samplerBuffer splat_colors_tbo;
uniform samplerBuffer splat_scales_tbo;
uniform samplerBuffer splat_rotations_tbo;
uniform samplerBuffer splat_opacities_tbo;


uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

flat out vec3 frag_color;
flat out float frag_opacity;
out vec2 uv;


mat3 quat_to_mat3(vec4 q)
{
    float x = q.x, y = q.y, z = q.z, w = q.w;

    return transpose(mat3(
        1 - 2*y*y - 2*z*z,   2*x*y - 2*z*w,     2*x*z + 2*y*w,
        2*x*y + 2*z*w,       1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w,
        2*x*z - 2*y*w,       2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y
    ));
}

void main()
{
    int i = int(instance_idx);
	vec3 instance_position = texelFetch(splat_points_tbo,     i).rgb;
    vec3 instance_color    = texelFetch(splat_colors_tbo,   i).rgb;
    vec3 instance_scale    = texelFetch(splat_scales_tbo,   i).rgb;
    vec4 instance_rot      = texelFetch(splat_rotations_tbo,     i);
    float instance_opacity = texelFetch(splat_opacities_tbo, i).r;
	

	frag_color = instance_color;
	frag_opacity = instance_opacity;


	// center in world
	mat4 MV = view * model;
	vec4 view_center =  MV * vec4(instance_position, 1.0);

	
	// compute the 3d covariance matrix
	mat3 R = quat_to_mat3(instance_rot);   // Precompute + renvoyer en int32 // pack x half
	mat3 S_mat = mat3(                       
        instance_scale.x * instance_scale.x, 0, 0,
        0, instance_scale.y * instance_scale.y, 0,
        0, 0, instance_scale.z * instance_scale.z
    );

	mat3 sigma3D = R * S_mat * transpose(R);
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
	
	float spread = 3.0;
	vec2 delta = spread*(sqrt(lambda1)*vertex_position.x * dir1 + sqrt(lambda2)*vertex_position.y * dir2);
	
	uv = vertex_position.xy *spread;

	vec4 screen_position = projection * view_center;

	screen_position.xy += delta * screen_position.w;

	gl_Position = screen_position;
}
