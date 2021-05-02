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
        frag_color  = cc;
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
    
    this.scaleProgram = createProgram(gl, [new Shader(gl, gl.VERTEX_SHADER, quadVert), new Shader(gl, gl.FRAGMENT_SHADER, scaleforceFrag)]);
    this.bilinearProgram = createProgram(gl, [new Shader(gl, gl.VERTEX_SHADER, quadVert), new Shader(gl, gl.FRAGMENT_SHADER, bilinearFrag)]);
    gl.useProgram(this.scaleProgram.program);
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

    this.scaleTexture = createTexture(gl, gl.LINEAR, null, width, height);

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

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}

Scaler.prototype.enableFilter = function(enable) {
    if (enable) {
        this.gl.useProgram(this.scaleProgram.program);
    } else {
        this.gl.useProgram(this.bilinearProgram.program);
    }
    this.render();
}


let scaler = null;

function onLoad() {
    const movOrig = document.getElementById('movOrig');
    const txtScale = document.getElementById('txtScale');

    const board = document.getElementById('board');
    const gl = board.getContext('webgl2');


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


    const inputImg = new Image();
    inputImg.src = "input.png";
    inputImg.onload = function() {
        let scale = parseFloat(txtScale.value);

        scaler = new Scaler(gl);
        scaler.inputImage(inputImg);
        scaler.resize(scale);
    }


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