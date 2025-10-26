/******/ (() => { // webpackBootstrap
/******/ 	"use strict";
var __webpack_exports__ = {};
// This entry needs to be wrapped in an IIFE because it uses a non-standard name for the exports (exports).
(() => {
var exports = __webpack_exports__;
/*!****************************!*\
  !*** ./scaleforce/main.ts ***!
  \****************************/

Object.defineProperty(exports, "__esModule", ({ value: true }));
const movOrig = document.getElementById('movOrig');
const txtScale = document.getElementById('txtScale');
const txtSrc = document.getElementById('txtSrc');
const fileImg = document.getElementById('fileImg');
const enableFXAA = document.getElementById('enableFXAA');
const enableScaleforce = document.getElementById('enableScaleforce');
const loadButton = document.getElementById('loadButton');
const speedLabel = document.getElementById('speed');
const board = document.getElementById('board');
async function fetchText(input, init) {
    return (await fetch(input, init)).text();
}
;
async function loadShaders() {
    return {
        quadVert: await fetchText("quad.vert"),
        bilinearFrag: await fetchText("bilinear.frag"),
        scaleforceFrag: await fetchText("scaleforce.frag"),
        fxaaFrag: await fetchText("fxaa.frag"),
    };
}
function createShader(gl, type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        throw new Error(gl.getShaderInfoLog(shader) || "Unknown error while compiling shader");
    }
    return shader;
}
class ShaderProgram {
    program;
    attributes = {};
    uniforms = {};
    constructor(gl, shaders) {
        const program = gl.createProgram();
        this.program = program;
        for (let shader of shaders) {
            gl.attachShader(program, shader);
        }
        gl.linkProgram(program);
        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            throw new Error(gl.getProgramInfoLog(program) || "Unknown error while linking shader program");
        }
        let numAttributes = gl.getProgramParameter(program, gl.ACTIVE_ATTRIBUTES);
        for (let i = 0; i < numAttributes; i++) {
            let attribute = gl.getActiveAttrib(program, i);
            this.attributes[attribute.name] = gl.getAttribLocation(program, attribute.name);
        }
        let numUniforms = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
        for (let i = 0; i < numUniforms; i++) {
            let uniform = gl.getActiveUniform(program, i);
            this.uniforms[uniform.name] = gl.getUniformLocation(program, uniform.name);
        }
    }
}
function createTexture(gl, filter, data, width, height) {
    const texture = gl.createTexture();
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
function createTextureFromPixels(gl, filter, pixels, width, height) {
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8, width, height);
    if (pixels != null) {
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
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
class Scaler {
    gl;
    extTimerQuery;
    inputTex = null;
    inputMov = null;
    inputWidth = 0;
    inputHeight = 0;
    intermediateFBO;
    intermediateTexture = null;
    scaleProgram;
    fxaaProgram;
    bilinearProgram;
    timer = null;
    constructor(board, shaderSources) {
        const gl = board.getContext('webgl2');
        this.gl = gl;
        this.extTimerQuery = gl.getExtension("EXT_disjoint_timer_query_webgl2");
        gl.disable(gl.DEPTH_TEST);
        gl.disable(gl.STENCIL_TEST);
        this.intermediateFBO = gl.createFramebuffer();
        const quadVertShader = createShader(gl, gl.VERTEX_SHADER, shaderSources.quadVert);
        this.scaleProgram = new ShaderProgram(gl, [quadVertShader, createShader(gl, gl.FRAGMENT_SHADER, shaderSources.scaleforceFrag)]);
        this.fxaaProgram = new ShaderProgram(gl, [quadVertShader, createShader(gl, gl.FRAGMENT_SHADER, shaderSources.fxaaFrag)]);
        this.bilinearProgram = new ShaderProgram(gl, [quadVertShader, createShader(gl, gl.FRAGMENT_SHADER, shaderSources.bilinearFrag)]);
    }
    inputImage(img) {
        const gl = this.gl;
        this.inputWidth = img.width;
        this.inputHeight = img.height;
        this.inputTex = createTexture(gl, gl.LINEAR, img, img.width, img.height);
        this.inputMov = null;
    }
    inputVideo(mov) {
        const gl = this.gl;
        const width = mov.videoWidth;
        const height = mov.videoHeight;
        this.inputWidth = width;
        this.inputHeight = height;
        let emptyPixels = new Uint8Array(width * height * 4);
        this.inputTex = createTextureFromPixels(gl, gl.LINEAR, emptyPixels, width, height);
        this.inputMov = mov;
    }
    resize(scale) {
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
    render() {
        if (!this.inputTex) {
            return;
        }
        const gl = this.gl;
        if (this.inputMov) {
            updateTexture(gl, this.inputTex, this.inputWidth, this.inputHeight, this.inputMov);
        }
        if (this.timer) {
            const disjoint = gl.getParameter(this.extTimerQuery.GPU_DISJOINT_EXT);
            const available = gl.getQueryParameter(this.timer, gl.QUERY_RESULT_AVAILABLE);
            if (available && !disjoint) {
                const speed = gl.getQueryParameter(this.timer, gl.QUERY_RESULT);
                speedLabel.innerText = "Speed: " + (speed / 1000) + "µs";
            }
            gl.deleteQuery(this.timer);
            this.timer = null;
        }
        var runQuery = this.extTimerQuery && !this.timer;
        if (runQuery) {
            this.timer = gl.createQuery();
            gl.beginQuery(this.extTimerQuery.TIME_ELAPSED_EXT, this.timer);
        }
        bindTexture(gl, this.inputTex, 0);
        if (enableFXAA.checked) {
            gl.bindFramebuffer(gl.FRAMEBUFFER, this.intermediateFBO);
            gl.viewport(0, 0, this.inputWidth, this.inputHeight);
            gl.useProgram(this.fxaaProgram.program);
            gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
            gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
            bindTexture(gl, this.intermediateTexture, 0);
        }
        gl.useProgram(enableScaleforce.checked ? this.scaleProgram.program : this.bilinearProgram.program);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        if (runQuery) {
            gl.endQuery(this.extTimerQuery.TIME_ELAPSED_EXT);
        }
    }
}
let scaler = null;
function getSourceType(uriPath) {
    const movTypes = ['.mp4', '.webm', '.ogv', '.ogg'];
    for (const ext of movTypes) {
        if (uriPath.endsWith(ext)) {
            return 'mov';
        }
    }
    return 'img';
}
function changeImage(src) {
    movOrig.pause();
    const inputImg = new Image();
    inputImg.crossOrigin = "anonymous";
    inputImg.src = src;
    inputImg.addEventListener("load", () => {
        let scale = parseFloat(txtScale.value);
        scaler.inputImage(inputImg);
        scaler.resize(scale);
    });
    inputImg.addEventListener("error", () => {
        alert(`Can't load the image.`);
    });
}
function changeVideo(src) {
    movOrig.src = src;
}
function onSourceChanged() {
    if (getSourceType(txtSrc.value) == 'img') {
        changeImage(txtSrc.value);
    }
    else {
        changeVideo(txtSrc.value);
    }
}
function updateFXAA() {
    scaler.render();
}
function updateFilter() {
    scaler.render();
}
function onSelectFile() {
    if (fileImg.files && fileImg.files[0]) {
        let reader = new FileReader();
        reader.onload = function (ev) {
            let src = ev.target.result;
            if (getSourceType(fileImg.value) == 'img') {
                changeImage(src);
            }
            else {
                changeVideo(src);
            }
        };
        reader.readAsDataURL(fileImg.files[0]);
    }
}
function onScaleChanged() {
    scaler.resize(parseFloat(txtScale.value));
}
async function main() {
    const shaderSources = await loadShaders();
    scaler = new Scaler(board, shaderSources);
    txtScale.addEventListener('change', onScaleChanged);
    fileImg.addEventListener('change', onSelectFile);
    enableScaleforce.addEventListener('change', updateFilter);
    enableFXAA.addEventListener('change', updateFXAA);
    movOrig.addEventListener('canplaythrough', function () {
        movOrig.play();
    }, true);
    movOrig.addEventListener('loadedmetadata', function () {
        let scale = parseFloat(txtScale.value);
        scaler.inputVideo(movOrig);
        scaler.resize(scale);
    }, true);
    movOrig.addEventListener('error', function () {
        alert("Can't load the video.");
    }, true);
    const queryString = window.location.search;
    const urlParams = new URLSearchParams(queryString);
    let sauce = urlParams.get('sauce');
    if (sauce == null) {
        sauce = "input.png";
    }
    txtSrc.value = sauce;
    onSourceChanged();
    loadButton.addEventListener("click", onSourceChanged);
    function render() {
        if (scaler) {
            scaler.render();
        }
        requestAnimationFrame(render);
    }
    requestAnimationFrame(render);
}
main();

})();

/******/ })()
;
//# sourceMappingURL=main.js.map