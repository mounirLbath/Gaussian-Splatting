#version 330 core

layout (location = 0) in vec3 position;
layout (location = 4) in vec3 instance_position; 
layout (location = 5) in vec3 instance_color; 

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 instanceColor;


void main()
{
	instanceColor = instance_color;

	mat3 R = transpose(mat3(view));

	vec3 camera_right = R[0];
	vec3 camera_up    = R[1];

	vec3 center = (model * vec4(10.0 * instance_position, 1.0)).xyz;

    vec3 world_pos =
        center
        + position.x * camera_right
        + position.y * camera_up;



	gl_Position = projection * view * vec4(world_pos, 1.0);
}
