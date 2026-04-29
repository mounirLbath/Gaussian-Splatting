#version 330 core

layout (location = 0) in vec3 position;
layout (location = 4) in vec3 instance_position; 

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

flat out int instanceID;

void main()
{
	gl_Position = projection * view * model * vec4(position + 10*instance_position, 1.0);
}
