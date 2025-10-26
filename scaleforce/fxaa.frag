#version 300 es

/**
Basic FXAA implementation based on the code on geeks3d.com with the
modification that the texture2DLod stuff was removed since it's
unsupported by WebGL.
--
From:
https://github.com/mitsuhiko/webgl-meincraft
Copyright (c) 2011 by Armin Ronacher.
Some rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.
    * The names of the contributors may not be used to endorse or
      promote products derived from this software without specific
      prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

precision mediump float;

in vec2 tex_coord;

out vec4 frag_color;
flat in vec2 inv_tex_size;

uniform sampler2D input_texture;

const float FXAA_REDUCE_MIN = (1.0 / 128.0);
const float FXAA_REDUCE_MUL = (1.0 / 8.0);
const float FXAA_SPAN_MAX   = 8.0;

float ColorDist(vec4 a, vec4 b) {
    // https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.2020_conversion
    const vec3 K = vec3(0.2627, 0.6780, 0.0593);
    const mat3 MATRIX = mat3(K, -.5 * K.r / (1.0 - K.b), -.5 * K.g / (1.0 - K.b), .5, .5,
                             -.5 * K.g / (1.0 - K.r), -.5 * K.b / (1.0 - K.r));
    vec4 diff = a - b;
    vec3 YCbCr = diff.rgb * MATRIX;
    // LUMINANCE_WEIGHT is currently 1, otherwise y would be multiplied by it
    float d = length(YCbCr);
    return sqrt(a.a * b.a * d * d + diff.a * diff.a);
}

//optimized version for mobile, where dependent 
//texture reads can be a bottleneck
vec4 fxaa(sampler2D tex, vec2 texCoord) {
    mediump vec2 resolution = vec2(textureSize(tex, 0).xy);

    vec4 rgbNW = textureOffset(tex, texCoord, ivec2(-1, -1));
    vec4 rgbNE = textureOffset(tex, texCoord, ivec2(1, -1));
    vec4 rgbSW = textureOffset(tex, texCoord, ivec2(-1, 1));
    vec4 rgbSE = textureOffset(tex, texCoord, ivec2(1, 1));
    vec4 texColor = texture(tex, texCoord);
    vec4 rgbM = texColor;
    vec3 luma = vec3(0.299, 0.587, 0.114);
    float lumaNW = dot(rgbNW.rgb, luma);
    float lumaNE = dot(rgbNE.rgb, luma);
    float lumaSW = dot(rgbSW.rgb, luma);
    float lumaSE = dot(rgbSE.rgb, luma);
    float lumaM  = dot(rgbM.rgb,  luma);
    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));
    
    vec2 dir;
    dir.x = ColorDist(rgbNW + rgbNE, rgbSW + rgbSE);
    dir.y = ColorDist(rgbNW + rgbSW, rgbNE + rgbSE);
    dir *= sign(vec2(
        (lumaSW + lumaSE) - (lumaNW + lumaNE),
        (lumaNW + lumaSW) - (lumaNE + lumaSE)
    ));

    float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) *
    (0.25 * FXAA_REDUCE_MUL), FXAA_REDUCE_MIN);

    float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
    float total_dist = ColorDist(rgbSE, rgbM) + ColorDist(rgbSW, rgbM) +
    ColorDist(rgbNE, rgbM) + ColorDist(rgbNW, rgbM);
    float clamp_val = sqrt(total_dist * rcpDirMin);
    dir = clamp(dir * rcpDirMin, -clamp_val, clamp_val) / resolution;  
    
    vec4 rgbA = 0.5 * (
        texture(tex, texCoord + dir * (1.0 / 3.0 - 0.5)) +
        texture(tex, texCoord + dir * (2.0 / 3.0 - 0.5)));
    vec4 rgbB = rgbA * 0.5 + 0.25 * (
        texture(tex, texCoord + dir * -0.5) +
        texture(tex, texCoord + dir * 0.5));

    return rgbB;
}

void main() {
    frag_color = fxaa(input_texture, vec2(tex_coord.x, 1.0 - tex_coord.y));
}