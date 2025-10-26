#version 300 es
precision mediump float;

in vec2 tex_coord;

out vec4 frag_color;

uniform sampler2D input_texture;

void main() {
    vec2 coord = vec2(tex_coord.x, 1.0f - tex_coord.y);
    vec4 bl = textureOffset(input_texture, coord, ivec2(-1, -1));
    vec4 bc = textureOffset(input_texture, coord, ivec2(0, -1));
    vec4 br = textureOffset(input_texture, coord, ivec2(1, -1));
    vec4 cl = textureOffset(input_texture, coord, ivec2(-1, 0));
    vec4 cc = texture(input_texture, coord);
    vec4 cr = textureOffset(input_texture, coord, ivec2(1, 0));
    vec4 tl = textureOffset(input_texture, coord, ivec2(-1, 1));
    vec4 tc = textureOffset(input_texture, coord, ivec2(0, 1));
    vec4 tr = textureOffset(input_texture, coord, ivec2(1, 1));

    const float PI = 3.1415926538f;
    const float r = sqrt(1.0f / PI);
    const float centerFactor = (4.0f / 9.0f) * r * r;
    // I used calculus, trust me :P
    const float plusFactor = 0.137473063946f;
    const float crossFactor = (1.0f - centerFactor - plusFactor * 4.0f) / 4.0f;

    frag_color = cc * centerFactor;
    frag_color += (bc + cl + cr + tc) * plusFactor;
    frag_color += (bl + br + tl + tr) * crossFactor;
}