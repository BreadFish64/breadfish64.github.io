function Shader(gl, type, source) {
    this.shader = gl.createShader(type);
    gl.shaderSource(this.shader, source);

    gl.compileShader(this.shader);
    if (!gl.getShaderParameter(this.shader, gl.COMPILE_STATUS)) {
        throw new Error(gl.getShaderInfoLog(this.shader));
    }
}

function createProgram(gl, shaders) {
    let program = gl.createProgram();

    for (let shader of shaders) {
        gl.attachShader(program, shader.shader);
    }

    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        throw new Error(gl.getProgramInfoLog(program));
    }

    let wrapper = {program: program};

    let numAttributes = gl.getProgramParameter(program, gl.ACTIVE_ATTRIBUTES);
    for (let i = 0; i < numAttributes; i++) {
        let attribute = gl.getActiveAttrib(program, i);
        wrapper[attribute.name] = gl.getAttribLocation(program, attribute.name);
    }
    let numUniforms = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
    for (let i$1 = 0; i$1 < numUniforms; i$1++) {
        let uniform = gl.getActiveUniform(program, i$1);
        wrapper[uniform.name] = gl.getUniformLocation(program, uniform.name);
    }

    return wrapper;
}

function createTexture(gl, filter, data, width, height) {
    let texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8, width, height);
    if (data != null) {
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, data);
    }
    gl.bindTexture(gl.TEXTURE_2D, null);
    return texture;
}

function bindTexture(gl, texture, unit) {
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, texture);
}

function updateTexture(gl, texture, width, height, src) {
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, src);
}

const quadVert = `#version 300 es
precision mediump float;

out vec2 tex_coord;

const vec2 vertices[4] =
    vec2[4](vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0), vec2(1.0, 1.0));

void main() {
    gl_Position = vec4(vertices[gl_VertexID], 0.0, 1.0);
    tex_coord = (vertices[gl_VertexID] + 1.0) / 2.0;
    tex_coord.y = 1.0 - tex_coord.y;
}
`;


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
const fxaaFrag = `#version 300 es
precision mediump float;

in vec2 tex_coord;

out vec4 frag_color;

uniform sampler2D input_texture;

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

    float clamp_val = length(dir);
    dir = clamp(dir, -clamp_val, clamp_val);
    dir /= resolution;
    
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
}`;

const oldfxaaFrag = `#version 300 es
precision mediump float;

in vec2 tex_coord;

out vec4 frag_color;

uniform sampler2D input_texture;

#ifndef FXAA_REDUCE_MIN
    #define FXAA_REDUCE_MIN   (1.0/ 128.0)
#endif
#ifndef FXAA_REDUCE_MUL
    #define FXAA_REDUCE_MUL   (1.0 / 8.0)
#endif
#ifndef FXAA_SPAN_MAX
    #define FXAA_SPAN_MAX     8.0
#endif

//optimized version for mobile, where dependent 
//texture reads can be a bottleneck
vec4 fxaa(sampler2D tex, vec2 texCoord) {
    vec4 color;
    mediump vec2 resolution = vec2(textureSize(tex, 0).xy);
    mediump vec2 inverseVP = vec2(1.0 / resolution.x, 1.0 / resolution.y);
    vec3 rgbNW = textureOffset(tex, texCoord, ivec2(-1, -1)).xyz;
    vec3 rgbNE = textureOffset(tex, texCoord, ivec2(1, -1)).xyz;
    vec3 rgbSW = textureOffset(tex, texCoord, ivec2(-1, 1)).xyz;
    vec3 rgbSE = textureOffset(tex, texCoord, ivec2(1, 1)).xyz;
    vec4 texColor = texture(tex, texCoord);
    vec3 rgbM  = texColor.xyz;
    vec3 luma = vec3(0.299, 0.587, 0.114);
    float lumaNW = dot(rgbNW, luma);
    float lumaNE = dot(rgbNE, luma);
    float lumaSW = dot(rgbSW, luma);
    float lumaSE = dot(rgbSE, luma);
    float lumaM  = dot(rgbM,  luma);
    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));
    
    mediump vec2 dir;
    dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
    dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));
    
    float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) *
                          (0.25 * FXAA_REDUCE_MUL), FXAA_REDUCE_MIN);
    
    float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
    dir = min(vec2(FXAA_SPAN_MAX, FXAA_SPAN_MAX),
              max(vec2(-FXAA_SPAN_MAX, -FXAA_SPAN_MAX),
              dir * rcpDirMin)) * inverseVP;    

    vec3 rgbA = 0.5 * (
        texture(tex, texCoord + dir * (1.0 / 3.0 - 0.5)).xyz +
        texture(tex, texCoord + dir * (2.0 / 3.0 - 0.5)).xyz);
    vec3 rgbB = rgbA * 0.5 + 0.25 * (
        texture(tex, texCoord + dir * -0.5).xyz +
        texture(tex, texCoord + dir * 0.5).xyz);

    float lumaB = dot(rgbB, luma);
    if ((lumaB < lumaMin) || (lumaB > lumaMax))
        color = vec4(rgbA, texColor.a);
    else
        color = vec4(rgbB, texColor.a);
    return color;
}

void main() {
    frag_color = fxaa(input_texture, vec2(tex_coord.x, 1.0 - tex_coord.y));
}
`;

const scaleforceFrag = `#version 300 es
precision mediump float;

in vec2 tex_coord;

out vec4 frag_color;

uniform sampler2D input_texture;

mat4x3 center_matrix;
vec4 center_alpha;

// Finds the distance between four colors and cc in YCbCr space
vec4 ColorDist(vec4 A, vec4 B, vec4 C, vec4 D) {
    // https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.2020_conversion
    const vec3 K = vec3(0.2627, 0.6780, 0.0593);
    const float LUMINANCE_WEIGHT = 1.0;
    const mat3 YCBCR_MATRIX =
        mat3(K * LUMINANCE_WEIGHT, -.5 * K.r / (1.0 - K.b), -.5 * K.g / (1.0 - K.b), .5, .5,
             -.5 * K.g / (1.0 - K.r), -.5 * K.b / (1.0 - K.r));

    mat4x3 colors = mat4x3(A.rgb, B.rgb, C.rgb, D.rgb) - center_matrix;
    mat4x3 YCbCr = YCBCR_MATRIX * colors;
    vec4 color_dist = vec3(1.0) * YCbCr;
    color_dist *= color_dist;
    vec4 alpha = vec4(A.a, B.a, C.a, D.a);

    return sqrt((color_dist + distance(alpha, center_alpha)) * center_alpha * alpha);
}

void main() {
    vec4 bl = textureOffset(input_texture, tex_coord, ivec2(-1, -1));
    vec4 bc = textureOffset(input_texture, tex_coord, ivec2(0, -1));
    vec4 br = textureOffset(input_texture, tex_coord, ivec2(1, -1));
    vec4 cl = textureOffset(input_texture, tex_coord, ivec2(-1, 0));
    vec4 cc = texture(input_texture, tex_coord);
    vec4 cr = textureOffset(input_texture, tex_coord, ivec2(1, 0));
    vec4 tl = textureOffset(input_texture, tex_coord, ivec2(-1, 1));
    vec4 tc = textureOffset(input_texture, tex_coord, ivec2(0, 1));
    vec4 tr = textureOffset(input_texture, tex_coord, ivec2(1, 1));

    center_matrix = mat4x3(cc.rgb, cc.rgb, cc.rgb, cc.rgb);
    center_alpha = cc.aaaa;

    vec4 offset_tl = ColorDist(tl, tc, tr, cr);
    vec4 offset_br = ColorDist(br, bc, bl, cl);

    // Calculate how different cc is from the texels around it
    const float plus_weight = 1.5;
    const float cross_weight = 1.5;
    float total_dist = dot(offset_tl + offset_br, vec4(cross_weight, plus_weight, cross_weight, plus_weight));

    if (total_dist == 0.0) {
        frag_color = cc;
    } else {
        // Add together all the distances with direction taken into account
        vec4 tmp = offset_tl - offset_br;
        vec2 total_offset = tmp.wy * plus_weight + (tmp.zz + vec2(-tmp.x, tmp.x)) * cross_weight;

        // When the image has thin points, they tend to split apart.
        // This is because the texels all around are different and total_offset reaches into clear areas.
        // This works pretty well to keep the offset in bounds for these cases.
        float clamp_val = length(total_offset) / total_dist;
        vec2 final_offset = clamp(total_offset, -clamp_val, clamp_val) / vec2(textureSize(input_texture, 0));

        frag_color = texture(input_texture, tex_coord - final_offset);
    }
}
`;

const bilinearFrag = `#version 300 es
precision mediump float;

in vec2 tex_coord;

out vec4 frag_color;

uniform sampler2D input_texture;

void main() {
    frag_color = texture(input_texture, tex_coord);
}
`;

function Scaler(gl) {
    this.gl = gl;
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.STENCIL_TEST);

    this.inputTex = null;
    this.inputMov = null;
    this.inputWidth = 0;
    this.inputHeight = 0;

    this.useScaleforce = true;
    this.useFXAA = true;

    this.intermediateFBO = gl.createFramebuffer();
    
    const quadVertShader = new Shader(gl, gl.VERTEX_SHADER, quadVert);
    this.scaleProgram = createProgram(gl, [quadVertShader, new Shader(gl, gl.FRAGMENT_SHADER, scaleforceFrag)]);
    this.fxaaProgram = createProgram(gl, [quadVertShader, new Shader(gl, gl.FRAGMENT_SHADER, fxaaFrag)]);
    this.bilinearProgram = createProgram(gl, [quadVertShader, new Shader(gl, gl.FRAGMENT_SHADER, bilinearFrag)]);
}

Scaler.prototype.inputImage = function(img) {
    const gl = this.gl;

    this.inputWidth = img.width;
    this.inputHeight = img.height;

    this.inputTex = createTexture(gl, gl.LINEAR, img, img.width, img.height);
    this.inputMov = null;
}

Scaler.prototype.inputVideo = function(mov) {
    const gl = this.gl;

    const width = mov.videoWidth;
    const height = mov.videoHeight;

    this.inputWidth = width;
    this.inputHeight = height;

    let emptyPixels = new Uint8Array(width * height * 4);
    this.inputTex = createTexture(gl, gl.LINEAR, emptyPixels, width, height);
    this.inputMov = mov;
}

Scaler.prototype.resize = function(scale) {
    const gl = this.gl;

    const width = Math.round(this.inputWidth * scale);
    const height = Math.round(this.inputHeight * scale);

    gl.canvas.width = width;
    gl.canvas.height = height;

    this.intermediateTexture = createTexture(gl, gl.LINEAR, null, this.inputWidth, this.inputHeight);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.intermediateFBO);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.intermediateTexture, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
}

Scaler.prototype.render = function() {
    if (!this.inputTex) {
        return;
    }

    const gl = this.gl;

    if (this.inputMov) {
        updateTexture(gl, this.inputTex, this.inputWidth, this.inputHeight, this.inputMov);
    }

    bindTexture(gl, this.inputTex, 0);

    if (this.useFXAA) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.intermediateFBO);
        gl.viewport(0, 0, this.inputWidth, this.inputHeight);
        gl.useProgram(this.fxaaProgram.program);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
        bindTexture(gl, this.intermediateTexture, 0);
    }

    gl.useProgram(this.useScaleforce ? this.scaleProgram.program : this.bilinearProgram.program);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}

Scaler.prototype.enableFXAA = function(enable) {
    this.useFXAA = enable;
    this.render();
}

Scaler.prototype.enableFilter = function(enable) {
    this.useScaleforce = enable;
    this.render();
}

let scaler = null;

function onLoad() {
    const movOrig = document.getElementById('movOrig');
    const txtScale = document.getElementById('txtScale');

    const board = document.getElementById('board');
    const gl = board.getContext('webgl2');
    scaler = new Scaler(gl);

    movOrig.addEventListener('canplaythrough', function() {
        movOrig.play();
    }, true);
    movOrig.addEventListener('loadedmetadata', function() {
        let scale = parseFloat(txtScale.value);

        scaler = new Scaler(gl);
        scaler.inputVideo(movOrig);
        scaler.resize(scale);
    }, true);
    movOrig.addEventListener('error', function() {
        alert("Can't load the video.");
    }, true);

    // https://www.sitepoint.com/get-url-parameters-with-javascript/
    const queryString = window.location.search;
    const urlParams = new URLSearchParams(queryString);
    sauce = urlParams.get('sauce');
    if (sauce == null) {
        sauce = "input.png";
    }
    const txtSrc = document.getElementById('txtSrc');
    txtSrc.value = sauce;

    onSourceChanged();

    function render() {
        if (scaler) {
            scaler.render();
        }

        requestAnimationFrame(render);
    }
    
    requestAnimationFrame(render);
}

function getSourceType(uri) {
    const movTypes = ['mp4', 'webm', 'ogv', 'ogg'];

    let ext = uri.split('.').pop().split(/\#|\?/)[0];

    for (let i=0; i<movTypes.length; ++i) {
        if (ext === movTypes[i]) {
            return 'mov';
        }
    }

    return 'img';
}

function changeImage(src) {
    const movOrig = document.getElementById('movOrig');
    movOrig.pause();

    const txtScale = document.getElementById('txtScale');

    const inputImg = new Image();
    inputImg.crossOrigin = "anonymous";
    inputImg.src = src;
    inputImg.onload = function() {
        let scale = parseFloat(txtScale.value);

        scaler.inputImage(inputImg);
        scaler.resize(scale);
    }
    inputImg.onerror = function() {
        alert("Can't load the image.");
    }
}

function changeVideo(src) {
    const movOrig = document.getElementById('movOrig');
    movOrig.src = src;
}

function onSourceChanged() {
    const txtSrc = document.getElementById('txtSrc');
    let uri = txtSrc.value;

    if (getSourceType(uri) == 'img') {
        changeImage(uri);
    }
    else {
        changeVideo(uri);
    }
}

function updateFXAA(toggle) {
    scaler.enableFXAA(toggle.checked);
}

function updateFilter(toggle) {
    scaler.enableFilter(toggle.checked);
}

function onSelectFile(input) {
    if (input.files && input.files[0]) {
        let reader = new FileReader();
        reader.onload = function (e) {
            let src = e.target.result;
            if (getSourceType(input.value) == 'img') {
                changeImage(src);
            }
            else {
                changeVideo(src);
            }
        };
        reader.readAsDataURL(input.files[0]);
    }
}

function onScaleChanged() {
    const txtScale = document.getElementById('txtScale');

    scaler.resize(parseFloat(txtScale.value));
}