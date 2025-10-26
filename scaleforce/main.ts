const movOrig = document.getElementById('movOrig')! as HTMLVideoElement;
const txtScale = document.getElementById('txtScale')! as HTMLInputElement;
const txtSrc = document.getElementById('txtSrc')! as HTMLInputElement;
const fileImg = document.getElementById('fileImg')! as HTMLInputElement;
const enableFXAA = document.getElementById('enableFXAA')! as HTMLInputElement;
const enableScaleforce = document.getElementById('enableScaleforce')! as HTMLInputElement;
const loadButton = document.getElementById('loadButton')! as HTMLButtonElement;
const speedLabel = document.getElementById('speed')! as HTMLLabelElement;
const board = document.getElementById('board')! as HTMLCanvasElement;

async function fetchText(input: string | URL | globalThis.Request,
    init?: RequestInit) {
    return (await fetch(input, init)).text()
}

interface ShaderSources {
    quadVert: string,
    bilinearFrag: string,
    scaleforceFrag: string,
    fxaaFrag: string,
};

async function loadShaders(): Promise<ShaderSources> {
    return {
        quadVert: await fetchText("quad.vert"),
        bilinearFrag: await fetchText("bilinear.frag"),
        scaleforceFrag: await fetchText("scaleforce.frag"),
        fxaaFrag: await fetchText("fxaa.frag"),
    }
}

function createShader(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
    const shader = gl.createShader(type)!;
    gl.shaderSource(shader, source);

    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        throw new Error(gl.getShaderInfoLog(shader) || "Unknown error while compiling shader");
    }
    return shader
}

class ShaderProgram {
    program: WebGLProgram
    attributes: Record<string, number> = {}
    uniforms: Record<string, WebGLUniformLocation> = {}

    constructor(gl: WebGL2RenderingContext, shaders: Iterable<WebGLShader>) {
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
            let attribute = gl.getActiveAttrib(program, i)!;
            this.attributes[attribute.name] = gl.getAttribLocation(program, attribute.name);
        }
        let numUniforms = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
        for (let i = 0; i < numUniforms; i++) {
            let uniform = gl.getActiveUniform(program, i)!;
            this.uniforms[uniform.name] = gl.getUniformLocation(program, uniform.name)!;
        }
    }
}

function createTexture(gl: WebGL2RenderingContext, filter: number, data: TexImageSource | null, width: number, height: number): WebGLTexture {
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

function createTextureFromPixels(gl: WebGL2RenderingContext, filter: number, pixels: ArrayBufferView<ArrayBufferLike> | null, width: number, height: number): WebGLTexture {
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

function bindTexture(gl: WebGL2RenderingContext, texture: WebGLTexture, unit: number) {
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, texture);
}

function updateTexture(gl: WebGL2RenderingContext, texture: WebGLTexture, width: number, height: number, src: TexImageSource) {
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, src);
}

class Scaler {
    gl: WebGL2RenderingContext
    extTimerQuery: any

    inputTex: WebGLTexture | null = null
    inputMov: any = null
    inputWidth: number = 0
    inputHeight: number = 0

    intermediateFBO: WebGLFramebuffer
    intermediateTexture: WebGLTexture | null = null

    scaleProgram: ShaderProgram
    fxaaProgram: ShaderProgram
    bilinearProgram: ShaderProgram

    timer: WebGLQuery | null = null

    constructor(board: HTMLCanvasElement, shaderSources: ShaderSources) {
        const gl = board.getContext('webgl2')!;
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

    inputImage(img: ImageBitmap | ImageData | HTMLImageElement | HTMLCanvasElement | HTMLVideoElement | OffscreenCanvas) {
        const gl = this.gl;

        this.inputWidth = img.width;
        this.inputHeight = img.height;

        this.inputTex = createTexture(gl, gl.LINEAR, img, img.width, img.height);
        this.inputMov = null;
    }

    inputVideo(mov: HTMLVideoElement) {
        const gl = this.gl;

        const width = mov.videoWidth;
        const height = mov.videoHeight;

        this.inputWidth = width;
        this.inputHeight = height;

        let emptyPixels = new Uint8Array(width * height * 4);
        this.inputTex = createTextureFromPixels(gl, gl.LINEAR, emptyPixels, width, height);
        this.inputMov = mov;
    }

    resize(scale: number) {
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
            bindTexture(gl, this.intermediateTexture!, 0);
        }

        gl.useProgram(enableScaleforce.checked ? this.scaleProgram.program : this.bilinearProgram.program);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

        if (runQuery) {
            gl.endQuery(this.extTimerQuery.TIME_ELAPSED_EXT);
        }
    }
}

let scaler: Scaler | null = null;

function getSourceType(uriPath: string) {
    const movTypes = ['.mp4', '.webm', '.ogv', '.ogg'];
    for (const ext of movTypes) {
        if (uriPath.endsWith(ext)) {
            return 'mov';
        }
    }
    return 'img';
}

function changeImage(src: string) {
    movOrig.pause();

    const inputImg = new Image();
    inputImg.crossOrigin = "anonymous";
    inputImg.src = src;
    inputImg.addEventListener("load", () => {
        let scale = parseFloat(txtScale.value);
        scaler!.inputImage(inputImg);
        scaler!.resize(scale);
    })
    inputImg.addEventListener("error", () => {
        alert(`Can't load the image.`);
    })
}

function changeVideo(src: string) {
    movOrig.src = src;
}

function onSourceChanged() {
    if (getSourceType(txtSrc.value) == 'img') {
        changeImage(txtSrc.value);
    } else {
        changeVideo(txtSrc.value);
    }
}

function updateFXAA() {
    scaler!.render();
}

function updateFilter() {
    scaler!.render();
}

function onSelectFile() {
    if (fileImg.files && fileImg.files[0]) {
        let reader = new FileReader();
        reader.onload = function (ev) {
            let src = ev.target!.result as string;
            if (getSourceType(fileImg.value) == 'img') {
                changeImage(src);
            } else {
                changeVideo(src);
            }
        };
        reader.readAsDataURL(fileImg.files[0]);
    }
}

function onScaleChanged() {
    scaler!.resize(parseFloat(txtScale.value));
}

async function main() {
    const shaderSources = await loadShaders();
    scaler = new Scaler(board, shaderSources);

    txtScale.addEventListener('change', onScaleChanged)
    fileImg.addEventListener('change', onSelectFile)
    enableScaleforce.addEventListener('change', updateFilter)
    enableFXAA.addEventListener('change', updateFXAA)

    movOrig.addEventListener('canplaythrough', function () {
        movOrig.play();
    }, true);
    movOrig.addEventListener('loadedmetadata', function () {
        let scale = parseFloat(txtScale.value);

        scaler!.inputVideo(movOrig);
        scaler!.resize(scale);
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
    loadButton.addEventListener("click", onSourceChanged)

    function render() {
        if (scaler) {
            scaler.render();
        }
        requestAnimationFrame(render);
    }

    requestAnimationFrame(render);
}

main()

