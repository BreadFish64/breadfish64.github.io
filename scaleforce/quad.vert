#version 300 es
precision mediump float;

out vec2 tex_coord;
flat out vec2 inv_tex_size;

uniform sampler2D input_texture;

const vec2 vertices[3] =
    vec2[3](vec2(0.0, 0.0), vec2(2.0, 0.0), vec2(0.0, 2.0));

void main() {
    vec2 vertex = vertices[gl_VertexID];
    gl_Position = vec4((2.0 * vertex) - 1.0, 0.0, 1.0);
    tex_coord = vec2(vertex.x, 1.0 - vertex.y);
    inv_tex_size = 1.0f / vec2(textureSize(input_texture, 0));
}
