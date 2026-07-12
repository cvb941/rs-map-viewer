import Denque from "denque";
import { mat4, vec2, vec3, vec4 } from "gl-matrix";
import { folder } from "leva";
import { Schema } from "leva/dist/declarations/src/types";
import {
    DrawCall,
    Framebuffer,
    App as PicoApp,
    PicoGL,
    Program,
    Renderbuffer,
    Texture,
    Timer,
    UniformBuffer,
    VertexArray,
    VertexBuffer,
} from "picogl";

import { OsrsMenuEntry } from "../../components/rs/menu/OsrsMenu";
import { createTextureArray } from "../../picogl/PicoTexture";
import { RS_TO_RADIANS } from "../../rs/MathConstants";
import { MenuTargetType } from "../../rs/MenuEntry";
import { Scene } from "../../rs/scene/Scene";
import { isTouchDevice, isWebGL2Supported, pixelRatio } from "../../util/DeviceUtil";
import { getAxisDeadzone } from "../InputManager";
import { MapViewer } from "../MapViewer";
import { MapViewerRenderer } from "../MapViewerRenderer";
import { MapViewerRendererType, WEBGL } from "../MapViewerRenderers";
import { DrawRange, NULL_DRAW_RANGE } from "./DrawRange";
import { InteractType } from "./InteractType";
import { Interactions } from "./Interactions";
import { WebGLMapSquare } from "./WebGLMapSquare";
import { SdMapData } from "./loader/SdMapData";
import { SdMapDataLoader } from "./loader/SdMapDataLoader";
import { SdMapLoaderInput } from "./loader/SdMapLoaderInput";
import {
    FRAME_FXAA_PROGRAM,
    FRAME_PROGRAM,
    HAND_PROGRAM,
    createMainProgram,
    createNpcProgram,
} from "./shaders/Shaders";

const MAX_TEXTURES = 2048;
const TEXTURE_SIZE = 128;

const INTERACT_BUFFER_COUNT = 2;
const INTERACTION_RADIUS = 5;

const MAX_HAND_JOINTS = 50;
const MAX_HAND_JOINTS_PER_HAND = 25;
const XR_METERS_PER_SCENE_UNIT = 1.1; // A typical humanoid NPC is about 1.75 m tall.
const XR_MIN_WORLD_SCALE = 0.1;
const XR_MAX_WORLD_SCALE = 10;

interface ColorRgb {
    r: number;
    g: number;
    b: number;
}

interface XRSystemLike {
    requestSession(
        mode: "immersive-vr",
        options?: { optionalFeatures?: string[] },
    ): Promise<XRSessionLike>;
}

interface XRSessionLike extends EventTarget {
    inputSources: Iterable<XRInputSourceLike>;
    renderState: {
        baseLayer?: XRWebGLLayerLike;
    };
    requestAnimationFrame(
        callback: (time: DOMHighResTimeStamp, frame: XRFrameLike) => void,
    ): number;
    requestReferenceSpace(type: string): Promise<XRReferenceSpaceLike>;
    updateRenderState(state: {
        baseLayer?: XRWebGLLayerLike;
        depthNear?: number;
        depthFar?: number;
    }): Promise<void> | void;
    end(): Promise<void>;
}

interface XRFrameLike {
    session: XRSessionLike;
    getViewerPose(referenceSpace: XRReferenceSpaceLike): XRViewerPoseLike | null;
    getJointPose(joint: XRJointSpaceLike, referenceSpace: XRReferenceSpaceLike): XRPoseLike | null;
    fillPoses(
        spaces: Iterable<XRJointSpaceLike>,
        referenceSpace: XRReferenceSpaceLike,
        transforms: Float32Array,
    ): boolean;
}

interface XRSpaceLike {}

interface XRReferenceSpaceLike extends XRSpaceLike {}

interface XRPoseLike {
    transform: {
        position: { x: number; y: number; z: number };
    };
}

interface XRViewerPoseLike {
    views: XRViewLike[];
}

interface XRViewLike {
    projectionMatrix: Float32Array;
    transform: {
        inverse: {
            matrix: Float32Array;
        };
    };
}

interface XRWebGLLayerLike {
    framebuffer: WebGLFramebuffer | null;
    getViewport(view: XRViewLike): { x: number; y: number; width: number; height: number };
}

interface XRInputSourceLike {
    handedness?: string;
    gamepad?: Gamepad;
    hand?: XRHandLike;
}

interface XRHandLike {
    size: number;
    get(joint: "wrist"): XRJointSpaceLike;
    values(): Iterable<XRJointSpaceLike>;
}

interface XRJointSpaceLike extends XRSpaceLike {}

interface XRInputSourceEventLike extends Event {
    inputSource: XRInputSourceLike;
}

enum TextureFilterMode {
    DISABLED,
    BILINEAR,
    TRILINEAR,
    ANISOTROPIC_2X,
    ANISOTROPIC_4X,
    ANISOTROPIC_8X,
    ANISOTROPIC_16X,
}

function getMaxAnisotropy(mode: TextureFilterMode): number {
    switch (mode) {
        case TextureFilterMode.ANISOTROPIC_2X:
            return 2;
        case TextureFilterMode.ANISOTROPIC_4X:
            return 4;
        case TextureFilterMode.ANISOTROPIC_8X:
            return 8;
        case TextureFilterMode.ANISOTROPIC_16X:
            return 16;
        default:
            return 1;
    }
}

function optimizeAssumingFlatsHaveSameFirstAndLastData(gl: WebGL2RenderingContext) {
    const epv = gl.getExtension("WEBGL_provoking_vertex");
    if (epv) {
        epv.provokingVertexWEBGL(epv.FIRST_VERTEX_CONVENTION_WEBGL);
    }
}

export class WebGLMapViewerRenderer extends MapViewerRenderer<WebGLMapSquare> {
    type: MapViewerRendererType = WEBGL;

    dataLoader = new SdMapDataLoader();

    app!: PicoApp;
    gl!: WebGL2RenderingContext;

    timer!: Timer;

    hasMultiDraw: boolean = false;

    quadPositions?: VertexBuffer;
    quadArray?: VertexArray;

    // Shaders
    shadersPromise?: Promise<Program[]>;
    mainProgram?: Program;
    mainAlphaProgram?: Program;
    npcProgram?: Program;
    frameProgram?: Program;
    frameFxaaProgram?: Program;
    handProgram?: Program;

    // Uniforms
    sceneUniformBuffer?: UniformBuffer;

    cameraPosUni: vec2 = vec2.fromValues(0, 0);
    resolutionUni: vec2 = vec2.fromValues(0, 0);

    // Framebuffers
    needsFramebufferUpdate: boolean = false;

    colorTarget?: Renderbuffer;
    interactTarget?: Renderbuffer;
    depthTarget?: Renderbuffer;
    framebuffer?: Framebuffer;

    textureColorTarget?: Texture;
    textureFramebuffer?: Framebuffer;

    interactColorTarget?: Texture;
    interactFramebuffer?: Framebuffer;

    xrColorTarget?: Renderbuffer;
    xrDepthTarget?: Renderbuffer;
    xrFramebuffer?: Framebuffer;

    // Textures
    textureFilterMode: TextureFilterMode = TextureFilterMode.ANISOTROPIC_16X;

    textureArray?: Texture;
    textureMaterials?: Texture;

    textureIds: number[] = [];
    loadedTextureIds: Set<number> = new Set();

    mapsToLoad: Denque<SdMapData> = new Denque();

    frameDrawCall?: DrawCall;
    frameFxaaDrawCall?: DrawCall;
    handDrawCall?: DrawCall;
    handVertexBuffer?: VertexBuffer;
    handVertexArray?: VertexArray;

    // Settings
    maxLevel: number = Scene.MAX_LEVELS - 1;

    skyColor: vec4 = vec4.fromValues(0, 0, 0, 1);
    fogDepth: number = 16;

    brightness: number = 1.0;
    colorBanding: number = 255;

    smoothTerrain: boolean = false;

    cullBackFace: boolean = true;

    msaaEnabled: boolean = false;
    fxaaEnabled: boolean = false;

    loadObjs: boolean = true;
    loadNpcs: boolean = true;

    // State
    lastClientTick: number = 0;
    lastTick: number = 0;

    interactions: Interactions[];
    hoveredMapIds: Set<number> = new Set();
    closestInteractIndices: Map<number, number[]> = new Map();
    interactBuffer?: Float32Array;

    npcRenderCount: number = 0;
    npcRenderData: Uint16Array = new Uint16Array(16 * 4);

    npcDataTextureBuffer: (Texture | undefined)[] = new Array(5);

    xrSession?: XRSessionLike;
    xrRefSpace?: XRReferenceSpaceLike;
    xrWorldViewMatrix: mat4 = mat4.create();
    xrViewMatrix: mat4 = mat4.create();
    xrViewProjMatrix: mat4 = mat4.create();
    resumeRenderLoopAfterXR: boolean = false;
    xrFrameValidated: boolean = false;
    xrHandsValidated: boolean = false;
    xrFrameError?: string;
    xrWorldScale: number = XR_METERS_PER_SCENE_UNIT;
    xrPinchStartDistance?: number;
    xrPinchStartScale?: number;
    xrWorldScaleVector: vec3 = vec3.create();
    xrHandPinches: Set<XRInputSourceLike> = new Set();
    xrHandPositions: Map<XRInputSourceLike, vec3> = new Map();
    xrHandJointData: Float32Array = new Float32Array(MAX_HAND_JOINTS * 4);
    xrHandPoseData: Float32Array = new Float32Array(MAX_HAND_JOINTS_PER_HAND * 16);

    constructor(public mapViewer: MapViewer) {
        super(mapViewer);
        this.interactions = new Array(INTERACT_BUFFER_COUNT);
        for (let i = 0; i < INTERACT_BUFFER_COUNT; i++) {
            this.interactions[i] = new Interactions(INTERACTION_RADIUS);
        }
    }

    static isSupported(): boolean {
        return isWebGL2Supported;
    }

    async init(): Promise<void> {
        await super.init();

        this.app = PicoGL.createApp(this.canvas, { xrCompatible: true, antialias: false });
        this.gl = this.app.gl as WebGL2RenderingContext;

        // https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/WebGL_best_practices#use_webgl_provoking_vertex_when_its_available
        optimizeAssumingFlatsHaveSameFirstAndLastData(this.gl);

        this.timer = this.app.createTimer();

        // hack to get the right multi draw extension for picogl
        const state: any = this.app.state;
        const ext = this.gl.getExtension("WEBGL_multi_draw");
        PicoGL.WEBGL_INFO.MULTI_DRAW_INSTANCED = ext;
        state.extensions.multiDrawInstanced = ext;

        this.hasMultiDraw = !!PicoGL.WEBGL_INFO.MULTI_DRAW_INSTANCED;

        this.mapViewer.workerPool.initLoader(this.dataLoader);

        this.gl.getExtension("EXT_float_blend");

        this.app.enable(PicoGL.CULL_FACE);
        this.app.enable(PicoGL.DEPTH_TEST);
        this.app.depthFunc(PicoGL.LEQUAL);
        this.app.enable(PicoGL.BLEND);
        this.app.blendFunc(PicoGL.SRC_ALPHA, PicoGL.ONE_MINUS_SRC_ALPHA);
        this.app.clearColor(0.0, 0.0, 0.0, 1.0);

        this.quadPositions = this.app.createVertexBuffer(
            PicoGL.FLOAT,
            2,
            new Float32Array([-1, 1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1]),
        );
        this.quadArray = this.app.createVertexArray().vertexAttributeBuffer(0, this.quadPositions);

        this.shadersPromise = this.initShaders();

        this.sceneUniformBuffer = this.app.createUniformBuffer([
            PicoGL.FLOAT_MAT4, // mat4 u_viewProjMatrix;
            PicoGL.FLOAT_MAT4, // mat4 u_viewMatrix;
            PicoGL.FLOAT_MAT4, // mat4 u_projectionMatrix;
            PicoGL.FLOAT_VEC4, // vec4 u_skyColor;
            PicoGL.FLOAT_VEC2, // vec2 u_cameraPos;
            PicoGL.FLOAT, // float u_renderDistance;
            PicoGL.FLOAT, // float u_fogDepth;
            PicoGL.FLOAT, // float u_currentTime;
            PicoGL.FLOAT, // float u_brightness;
            PicoGL.FLOAT, // float u_colorBanding;
            PicoGL.FLOAT, // float u_isNewTextureAnim;
        ]);

        this.initFramebuffers();

        this.initTextures();

        console.log("Renderer init");
    }

    async initShaders(): Promise<Program[]> {
        const hasMultiDraw = this.hasMultiDraw;

        const programs = await this.app.createPrograms(
            createMainProgram(hasMultiDraw, false),
            createMainProgram(hasMultiDraw, true),
            createNpcProgram(hasMultiDraw, true),
            FRAME_PROGRAM,
            FRAME_FXAA_PROGRAM,
            HAND_PROGRAM,
        );

        const [
            mainProgram,
            mainAlphaProgram,
            npcProgram,
            frameProgram,
            frameFxaaProgram,
            handProgram,
        ] = programs;
        this.mainProgram = mainProgram;
        this.mainAlphaProgram = mainAlphaProgram;
        this.npcProgram = npcProgram;
        this.frameProgram = frameProgram;
        this.frameFxaaProgram = frameFxaaProgram;
        this.handProgram = handProgram;

        this.frameDrawCall = this.app.createDrawCall(frameProgram, this.quadArray);
        this.frameFxaaDrawCall = this.app.createDrawCall(frameFxaaProgram, this.quadArray);
        this.handVertexBuffer = this.app.createVertexBuffer(
            PicoGL.FLOAT,
            4,
            MAX_HAND_JOINTS * 4,
            PicoGL.DYNAMIC_DRAW,
        );
        this.handVertexArray = this.app
            .createVertexArray()
            .vertexAttributeBuffer(0, this.handVertexBuffer);
        this.handDrawCall = this.app
            .createDrawCall(handProgram, this.handVertexArray)
            .primitive(PicoGL.POINTS);

        return programs;
    }

    initFramebuffers(): void {
        this.initFramebuffer();

        this.textureColorTarget = this.app.createTexture2D(this.app.width, this.app.height, {
            minFilter: PicoGL.LINEAR,
            magFilter: PicoGL.LINEAR,
        });
        this.textureFramebuffer = this.app
            .createFramebuffer()
            .colorTarget(0, this.textureColorTarget);

        // Interact
        this.interactColorTarget = this.app.createTexture2D(this.app.width, this.app.height, {
            internalFormat: PicoGL.RGBA32F,
            type: PicoGL.FLOAT,
            minFilter: PicoGL.NEAREST,
            magFilter: PicoGL.NEAREST,
        });
        this.interactFramebuffer = this.app
            .createFramebuffer()
            .colorTarget(0, this.interactColorTarget);
    }

    initFramebuffer(): void {
        this.framebuffer?.delete();
        this.colorTarget?.delete();
        this.interactTarget?.delete();
        this.depthTarget?.delete();

        let samples = 0;
        if (this.msaaEnabled) {
            samples = this.gl.getParameter(PicoGL.MAX_SAMPLES);
        }

        this.colorTarget = this.app.createRenderbuffer(
            this.app.width,
            this.app.height,
            PicoGL.RGBA8,
            samples,
        );
        this.interactTarget = this.app.createRenderbuffer(
            this.app.width,
            this.app.height,
            PicoGL.RGBA32F,
            samples,
        );
        this.depthTarget = this.app.createRenderbuffer(
            this.app.width,
            this.app.height,
            PicoGL.DEPTH_COMPONENT24,
            samples,
        );
        this.framebuffer = this.app
            .createFramebuffer()
            .colorTarget(0, this.colorTarget)
            .colorTarget(1, this.interactTarget)
            .depthTarget(this.depthTarget);

        this.needsFramebufferUpdate = false;
    }

    override initCache(): void {
        super.initCache();
        if (this.app) {
            this.initTextures();
        }
        console.log("Renderer initCache", this.app);
    }

    initTextures(): void {
        const textureLoader = this.mapViewer.textureLoader;

        const allTextureIds = textureLoader.getTextureIds();

        this.textureIds = allTextureIds
            .filter((id) => textureLoader.isSd(id))
            .slice(0, MAX_TEXTURES - 1);

        this.initTextureArray();
        this.initMaterialsTexture();

        console.log("init textures", this.textureIds, allTextureIds.length);
    }

    initTextureArray() {
        if (this.textureArray) {
            this.textureArray.delete();
            this.textureArray = undefined;
        }
        this.loadedTextureIds.clear();

        console.time("load textures");

        const pixelCount = TEXTURE_SIZE * TEXTURE_SIZE;

        const textureCount = this.textureIds.length;
        const pixels = new Int32Array((textureCount + 1) * pixelCount);

        // White texture
        pixels.fill(0xffffffff, 0, pixelCount);

        const cacheInfo = this.mapViewer.loadedCache.info;

        let maxPreloadTextures = textureCount;
        // we should check if the texture loader is procedural instead
        if (cacheInfo.game === "runescape" && cacheInfo.revision >= 508) {
            maxPreloadTextures = 64;
        }

        for (let i = 0; i < Math.min(textureCount, maxPreloadTextures); i++) {
            const textureId = this.textureIds[i];
            try {
                const texturePixels = this.mapViewer.textureLoader.getPixelsArgb(
                    textureId,
                    TEXTURE_SIZE,
                    true,
                    1.0,
                );
                pixels.set(texturePixels, (i + 1) * pixelCount);
            } catch (e) {
                console.error("Failed loading texture", textureId, e);
            }
            this.loadedTextureIds.add(textureId);
        }

        this.textureArray = createTextureArray(
            this.app,
            new Uint8Array(pixels.buffer),
            TEXTURE_SIZE,
            TEXTURE_SIZE,
            textureCount + 1,
            {},
        );

        this.updateTextureFiltering();

        console.timeEnd("load textures");
    }

    updateTextureFiltering(): void {
        if (!this.textureArray) {
            throw new Error("Texture array is not initialized");
        }

        this.textureArray.bind(0);

        if (this.textureFilterMode === TextureFilterMode.DISABLED) {
            this.gl.texParameteri(
                PicoGL.TEXTURE_2D_ARRAY,
                PicoGL.TEXTURE_MIN_FILTER,
                PicoGL.NEAREST,
            );
            this.gl.texParameteri(
                PicoGL.TEXTURE_2D_ARRAY,
                PicoGL.TEXTURE_MAG_FILTER,
                PicoGL.NEAREST,
            );
        } else if (this.textureFilterMode === TextureFilterMode.BILINEAR) {
            this.gl.texParameteri(
                PicoGL.TEXTURE_2D_ARRAY,
                PicoGL.TEXTURE_MIN_FILTER,
                PicoGL.LINEAR_MIPMAP_NEAREST,
            );
            this.gl.texParameteri(
                PicoGL.TEXTURE_2D_ARRAY,
                PicoGL.TEXTURE_MAG_FILTER,
                PicoGL.LINEAR,
            );
        } else {
            this.gl.texParameteri(
                PicoGL.TEXTURE_2D_ARRAY,
                PicoGL.TEXTURE_MIN_FILTER,
                PicoGL.LINEAR_MIPMAP_LINEAR,
            );
            this.gl.texParameteri(
                PicoGL.TEXTURE_2D_ARRAY,
                PicoGL.TEXTURE_MAG_FILTER,
                PicoGL.LINEAR,
            );
        }

        const maxAnisotropy = Math.min(
            getMaxAnisotropy(this.textureFilterMode),
            PicoGL.WEBGL_INFO.MAX_TEXTURE_ANISOTROPY,
        );

        this.gl.texParameteri(
            PicoGL.TEXTURE_2D_ARRAY,
            PicoGL.TEXTURE_MAX_ANISOTROPY_EXT,
            maxAnisotropy,
        );
    }

    updateTextureArray(textures: Map<number, Int32Array>): void {
        if (!this.textureArray) {
            throw new Error("Texture array is not initialized");
        }
        let updatedCount = 0;
        for (const [id, pixels] of textures) {
            if (this.loadedTextureIds.has(id)) {
                continue;
            }
            const index = this.textureIds.indexOf(id) + 1;

            this.textureArray.bind(0);
            this.gl.texSubImage3D(
                PicoGL.TEXTURE_2D_ARRAY,
                0,
                0,
                0,
                index,
                TEXTURE_SIZE,
                TEXTURE_SIZE,
                1,
                PicoGL.RGBA,
                PicoGL.UNSIGNED_BYTE,
                new Uint8Array(pixels.buffer),
            );
            this.loadedTextureIds.add(id);
            updatedCount++;
        }
        if (updatedCount > 0) {
            this.gl.generateMipmap(PicoGL.TEXTURE_2D_ARRAY);
        }
    }

    initMaterialsTexture(): void {
        if (this.textureMaterials) {
            this.textureMaterials.delete();
            this.textureMaterials = undefined;
        }

        const textureCount = this.textureIds.length + 1;

        const data = new Int8Array(textureCount * 4);
        for (let i = 0; i < this.textureIds.length; i++) {
            const id = this.textureIds[i];
            try {
                const material = this.mapViewer.textureLoader.getMaterial(id);

                const index = (i + 1) * 4;
                data[index] = material.animU;
                data[index + 1] = material.animV;
                data[index + 2] = material.alphaCutOff * 255;
            } catch (e) {
                console.error("Failed loading texture", id, e);
            }
        }

        this.textureMaterials = this.app.createTexture2D(data, textureCount, 1, {
            minFilter: PicoGL.NEAREST,
            magFilter: PicoGL.NEAREST,
            internalFormat: PicoGL.RGBA8I,
        });
    }

    getControls(): Schema {
        return {
            "Max Level": {
                value: this.maxLevel,
                min: 0,
                max: 3,
                step: 1,
                onChange: (v: number) => {
                    this.setMaxLevel(v);
                },
            },
            Sky: {
                r: this.skyColor[0] * 255,
                g: this.skyColor[1] * 255,
                b: this.skyColor[2] * 255,
                onChange: (v: ColorRgb) => {
                    this.setSkyColor(v.r, v.g, v.b);
                },
            },
            "Fog Depth": {
                value: this.fogDepth,
                min: 0,
                max: 256,
                step: 8,
                onChange: (v: number) => {
                    this.fogDepth = v;
                },
            },
            Brightness: {
                value: 1,
                min: 0,
                max: 4,
                step: 1,
                onChange: (v: number) => {
                    this.brightness = 1.0 - v * 0.1;
                },
            },
            "Color Banding": {
                value: 50,
                min: 0,
                max: 100,
                step: 1,
                onChange: (v: number) => {
                    this.colorBanding = 255 - v * 2;
                },
            },
            "Texture Filtering": {
                value: this.textureFilterMode,
                options: {
                    Disabled: TextureFilterMode.DISABLED,
                    Bilinear: TextureFilterMode.BILINEAR,
                    Trilinear: TextureFilterMode.TRILINEAR,
                    "Anisotropic 2x": TextureFilterMode.ANISOTROPIC_2X,
                    "Anisotropic 4x": TextureFilterMode.ANISOTROPIC_4X,
                    "Anisotropic 8x": TextureFilterMode.ANISOTROPIC_8X,
                    "Anisotropic 16x": TextureFilterMode.ANISOTROPIC_16X,
                },
                onChange: (v: TextureFilterMode) => {
                    if (v === this.textureFilterMode) {
                        return;
                    }
                    this.textureFilterMode = v;
                    this.updateTextureFiltering();
                },
            },
            "Smooth Terrain": {
                value: this.smoothTerrain,
                onChange: (v: boolean) => {
                    this.setSmoothTerrain(v);
                },
            },
            "Cull Back-faces": {
                value: this.cullBackFace,
                onChange: (v: boolean) => {
                    this.cullBackFace = v;
                },
            },
            "Anti-Aliasing": folder(
                {
                    MSAA: {
                        value: this.msaaEnabled,
                        onChange: (v: boolean) => {
                            this.setMsaa(v);
                        },
                    },
                    FXAA: {
                        value: this.fxaaEnabled,
                        onChange: (v: boolean) => {
                            this.setFxaa(v);
                        },
                    },
                },
                { collapsed: true },
            ),
            Entity: folder(
                {
                    Items: {
                        value: this.loadObjs,
                        onChange: (v: boolean) => {
                            this.setLoadObjs(v);
                        },
                    },
                    Npcs: {
                        value: this.loadNpcs,
                        onChange: (v: boolean) => {
                            this.setLoadNpcs(v);
                        },
                    },
                },
                { collapsed: true },
            ),
        };
    }

    override async queueLoadMap(mapX: number, mapY: number): Promise<void> {
        const mapData = await this.mapViewer.workerPool.queueLoad<
            SdMapLoaderInput,
            SdMapData | undefined,
            SdMapDataLoader
        >(this.dataLoader, {
            mapX,
            mapY,
            maxLevel: this.maxLevel,
            loadObjs: this.loadObjs,
            loadNpcs: this.loadNpcs,
            smoothTerrain: this.smoothTerrain,
            minimizeDrawCalls: !this.hasMultiDraw,
            loadedTextureIds: this.loadedTextureIds,
        });

        if (mapData) {
            if (this.isValidMapData(mapData)) {
                this.mapsToLoad.push(mapData);
            }
        } else {
            this.mapManager.addInvalidMap(mapX, mapY);
        }
    }

    loadMap(
        mainProgram: Program,
        mainAlphaProgram: Program,
        npcProgram: Program,
        textureArray: Texture,
        textureMaterials: Texture,
        sceneUniformBuffer: UniformBuffer,
        mapData: SdMapData,
        time: number,
    ): void {
        const { mapX, mapY } = mapData;

        this.mapViewer.setMapImageUrl(
            mapX,
            mapY,
            URL.createObjectURL(mapData.minimapBlob),
            true,
            false,
        );

        const frameCount = this.stats.frameCount;
        this.mapManager.addMap(
            mapX,
            mapY,
            WebGLMapSquare.load(
                this.mapViewer.seqTypeLoader,
                this.mapViewer.npcTypeLoader,
                this.mapViewer.basTypeLoader,
                this.app,
                mainProgram,
                mainAlphaProgram,
                npcProgram,
                textureArray,
                textureMaterials,
                sceneUniformBuffer,
                mapData,
                time,
                frameCount,
            ),
        );

        this.updateTextureArray(mapData.loadedTextures);
    }

    isValidMapData(mapData: SdMapData): boolean {
        return (
            mapData.cacheName === this.mapViewer.loadedCache.info.name &&
            mapData.maxLevel === this.maxLevel &&
            mapData.loadObjs === this.loadObjs &&
            mapData.loadNpcs === this.loadNpcs &&
            mapData.smoothTerrain === this.smoothTerrain
        );
    }

    clearMaps(): void {
        this.mapManager.cleanUp();
        this.mapsToLoad.clear();
    }

    setMaxLevel(maxLevel: number): void {
        const updated = this.maxLevel !== maxLevel;
        this.maxLevel = maxLevel;
        if (updated) {
            this.clearMaps();
        }
    }

    setSkyColor(r: number, g: number, b: number) {
        this.skyColor[0] = r / 255;
        this.skyColor[1] = g / 255;
        this.skyColor[2] = b / 255;
    }

    setSmoothTerrain(enabled: boolean): void {
        const updated = this.smoothTerrain !== enabled;
        this.smoothTerrain = enabled;
        if (updated) {
            this.clearMaps();
        }
    }

    setMsaa(enabled: boolean): void {
        const updated = this.msaaEnabled !== enabled;
        this.msaaEnabled = enabled;
        if (updated) {
            this.needsFramebufferUpdate = true;
        }
    }

    setFxaa(enabled: boolean): void {
        this.fxaaEnabled = enabled;
    }

    setLoadObjs(enabled: boolean): void {
        const updated = this.loadObjs !== enabled;
        this.loadObjs = enabled;
        if (updated) {
            this.clearMaps();
        }
    }

    setLoadNpcs(enabled: boolean): void {
        const updated = this.loadNpcs !== enabled;
        this.loadNpcs = enabled;
        if (updated) {
            this.clearMaps();
        }
    }

    override onResize(width: number, height: number): void {
        this.app.resize(width, height);
    }

    override render(time: number, deltaTime: number, resized: boolean): void {
        if (this.xrSession) {
            return;
        }

        const showDebugTimer = this.mapViewer.inputManager.isKeyDown("KeyY");

        if (showDebugTimer) {
            this.timer.start();
        }

        const frameCount = this.stats.frameCount;

        const timeSec = time / 1000;

        const tick = Math.floor(timeSec / 0.6);
        const ticksElapsed = Math.min(tick - this.lastTick, 1);
        if (ticksElapsed > 0) {
            this.lastTick = tick;
        }

        const clientTick = Math.floor(timeSec / 0.02);
        const clientTicksElapsed = Math.min(clientTick - this.lastClientTick, 50);
        if (clientTicksElapsed > 0) {
            this.lastClientTick = clientTick;
        }

        if (this.needsFramebufferUpdate) {
            this.initFramebuffer();
        }

        if (
            !this.mainProgram ||
            !this.mainAlphaProgram ||
            !this.npcProgram ||
            !this.sceneUniformBuffer ||
            !this.framebuffer ||
            !this.textureFramebuffer ||
            !this.frameDrawCall ||
            !this.interactFramebuffer ||
            !this.textureArray ||
            !this.textureMaterials
        ) {
            return;
        }

        if (resized) {
            this.framebuffer.resize();
            this.textureFramebuffer.resize();
            this.interactFramebuffer.resize();

            this.resolutionUni[0] = this.app.width;
            this.resolutionUni[1] = this.app.height;
        }

        const inputManager = this.mapViewer.inputManager;
        const camera = this.mapViewer.camera;

        this.handleInput(deltaTime);

        camera.update(this.app.width, this.app.height);

        const renderDistance = this.mapViewer.renderDistance;

        const mapManagerStart = performance.now();
        this.mapManager.update(camera, frameCount, renderDistance, this.mapViewer.unloadDistance);
        const mapManagerTime = performance.now() - mapManagerStart;

        this.cameraPosUni[0] = camera.getPosX();
        this.cameraPosUni[1] = camera.getPosZ();

        this.updateSceneUniforms(
            camera.viewMatrix,
            camera.projectionMatrix,
            camera.viewProjMatrix,
            timeSec,
            renderDistance,
        );

        const currInteractions = this.interactions[frameCount % this.interactions.length];

        const interactionsStart = performance.now();
        if (!inputManager.isPointerLock()) {
            this.checkInteractions(currInteractions);
        } else if (this.hoveredMapIds.size > 0) {
            this.hoveredMapIds.clear();
        }
        const interactionsTime = performance.now() - interactionsStart;

        if (this.cullBackFace) {
            this.app.enable(PicoGL.CULL_FACE);
        } else {
            this.app.disable(PicoGL.CULL_FACE);
        }

        this.app.enable(PicoGL.DEPTH_TEST);
        this.app.depthMask(true);

        this.app.drawFramebuffer(this.framebuffer);

        this.app.clearColor(0.0, 0.0, 0.0, 1.0);
        this.app.clear();
        this.gl.clearBufferfv(PicoGL.COLOR, 0, this.skyColor);

        const tickStart = performance.now();
        this.tickPass(timeSec, ticksElapsed, clientTicksElapsed);
        const tickTime = performance.now() - tickStart;

        const npcDataTextureIndex = this.updateNpcDataTexture();
        const npcDataTexture = this.npcDataTextureBuffer[npcDataTextureIndex];

        this.app.disable(PicoGL.BLEND);
        const opaquePassStart = performance.now();
        this.renderOpaquePass();
        const opaquePassTime = performance.now() - opaquePassStart;
        const opaqueNpcPassStart = performance.now();
        this.renderOpaqueNpcPass(npcDataTextureIndex, npcDataTexture);
        const opaqueNpcPassTime = performance.now() - opaqueNpcPassStart;

        this.app.enable(PicoGL.BLEND);
        const transparentPassStart = performance.now();
        this.renderTransparentPass();
        const transparentPassTime = performance.now() - transparentPassStart;
        const transparentNpcPassStart = performance.now();
        this.renderTransparentNpcPass(npcDataTextureIndex, npcDataTexture);
        const transparentNpcPassTime = performance.now() - transparentNpcPassStart;

        // Can't sample from renderbuffer so blit to a texture for sampling.
        this.app.readFramebuffer(this.framebuffer);

        this.app.drawFramebuffer(this.textureFramebuffer);
        this.gl.readBuffer(PicoGL.COLOR_ATTACHMENT0);
        this.app.blitFramebuffer(PicoGL.COLOR_BUFFER_BIT);

        if (!inputManager.isPointerLock()) {
            const mouseX = inputManager.mouseX;
            const mouseY = inputManager.mouseY;
            if (mouseX !== -1 && mouseY !== -1) {
                if (this.msaaEnabled) {
                    // TODO: reading from the multisampled framebuffer is not accurate
                    this.app.drawFramebuffer(this.interactFramebuffer);
                    this.gl.readBuffer(PicoGL.COLOR_ATTACHMENT1);
                    this.app.blitFramebuffer(PicoGL.COLOR_BUFFER_BIT);

                    this.app.readFramebuffer(this.interactFramebuffer);
                    this.gl.readBuffer(PicoGL.COLOR_ATTACHMENT0);
                } else {
                    this.gl.readBuffer(PicoGL.COLOR_ATTACHMENT1);
                }

                currInteractions.read(
                    this.gl,
                    (mouseX * pixelRatio) | 0,
                    (mouseY * pixelRatio) | 0,
                );
            }
        }

        this.app.disable(PicoGL.DEPTH_TEST);
        this.app.depthMask(false);

        this.app.disable(PicoGL.BLEND);

        this.app.clearMask(PicoGL.COLOR_BUFFER_BIT | PicoGL.DEPTH_BUFFER_BIT);
        this.app.clearColor(0.0, 0.0, 0.0, 1.0);
        this.app.defaultDrawFramebuffer().clear();

        if (this.frameFxaaDrawCall && this.fxaaEnabled) {
            this.frameFxaaDrawCall.uniform("u_resolution", this.resolutionUni);
            this.frameFxaaDrawCall.texture("u_frame", this.textureFramebuffer.colorAttachments[0]);
            this.frameFxaaDrawCall.draw();
        } else {
            this.frameDrawCall.texture("u_frame", this.textureFramebuffer.colorAttachments[0]);
            this.frameDrawCall.draw();
        }

        this.loadPendingMap(timeSec);

        if (showDebugTimer) {
            this.timer.end();
        }

        if (this.mapViewer.inputManager.isKeyDown("KeyH")) {
            this.mapViewer.debugText = `MapManager: ${mapManagerTime.toFixed(2)}ms`;
        }
        if (this.mapViewer.inputManager.isKeyDown("KeyJ")) {
            this.mapViewer.debugText = `Interactions: ${interactionsTime.toFixed(2)}ms`;
        }
        if (this.mapViewer.inputManager.isKeyDown("KeyK")) {
            this.mapViewer.debugText = `Tick: ${tickTime.toFixed(2)}ms`;
        }
        if (this.mapViewer.inputManager.isKeyDown("KeyL")) {
            this.mapViewer.debugText = `Opaque Pass: ${opaquePassTime.toFixed(2)}ms`;
        }
        if (this.mapViewer.inputManager.isKeyDown("KeyB")) {
            this.mapViewer.debugText = `Opaque Npc Pass: ${opaqueNpcPassTime.toFixed(2)}ms`;
        }
        if (this.mapViewer.inputManager.isKeyDown("KeyN")) {
            this.mapViewer.debugText = `Transparent Pass: ${transparentPassTime.toFixed(2)}ms`;
        }
        if (this.mapViewer.inputManager.isKeyDown("KeyM")) {
            this.mapViewer.debugText = `Transparent Npc Pass: ${transparentNpcPassTime.toFixed(
                2,
            )}ms`;
        }

        if (showDebugTimer && this.timer.ready()) {
            this.mapViewer.debugText = `Frame Time GL: ${this.timer.gpuTime.toFixed(
                2,
            )}ms\n JS: ${this.timer.cpuTime.toFixed(2)}ms`;
        }
    }

    async enterVR(): Promise<void> {
        if (this.xrSession) {
            return;
        }

        const xr = (navigator as Navigator & { xr?: XRSystemLike }).xr;
        const XRWebGLLayer = (
            window as Window & {
                XRWebGLLayer?: new (
                    session: XRSessionLike,
                    gl: WebGL2RenderingContext,
                    options?: { alpha?: boolean; antialias?: boolean; depth?: boolean },
                ) => XRWebGLLayerLike;
            }
        ).XRWebGLLayer;

        if (!xr || !XRWebGLLayer) {
            alert(
                "WebXR VR is not available. On Quest, open this over HTTPS in Meta Quest Browser.",
            );
            return;
        }

        let session: XRSessionLike | undefined;
        try {
            session = await xr.requestSession("immersive-vr", {
                optionalFeatures: ["local-floor", "hand-tracking"],
            });

            this.resumeRenderLoopAfterXR = this.running;
            this.running = false;
            if (this.animationId !== undefined) {
                cancelAnimationFrame(this.animationId);
                this.animationId = undefined;
            }

            const xrGl = this.gl as WebGL2RenderingContext & {
                makeXRCompatible?: () => Promise<void>;
            };
            await xrGl.makeXRCompatible?.();

            const glLayer = new XRWebGLLayer(session, this.gl, {
                alpha: true,
                antialias: false,
                depth: false,
            });
            await session.updateRenderState({
                baseLayer: glLayer,
                depthNear: 0.1,
                depthFar: 4096,
            });

            this.xrSession = session;
            this.xrRefSpace = await this.requestXRReferenceSpace(session);
            this.xrFrameValidated = false;
            this.xrHandsValidated = false;
            this.xrFrameError = undefined;
            this.xrWorldScale = XR_METERS_PER_SCENE_UNIT;
            this.xrPinchStartDistance = undefined;
            this.xrPinchStartScale = undefined;
            while (this.gl.getError() !== PicoGL.NO_ERROR) {
                // Ignore errors left by the last desktop frame.
            }
            session.addEventListener("end", this.onXRSessionEnd);
            session.addEventListener("selectstart", this.onXRSelectStart);
            session.addEventListener("selectend", this.onXRSelectEnd);
            session.requestAnimationFrame(this.onXRFrame);
        } catch (e) {
            session?.removeEventListener("end", this.onXRSessionEnd);
            session?.removeEventListener("selectstart", this.onXRSelectStart);
            session?.removeEventListener("selectend", this.onXRSelectEnd);
            session?.end().catch(() => {});
            this.xrSession = undefined;
            this.xrRefSpace = undefined;
            this.resumeDesktopRenderLoop();
            alert(`Failed to enter VR: ${e instanceof Error ? e.message : String(e)}`);
        }
    }

    async exitVR(): Promise<void> {
        await this.xrSession?.end();
    }

    private async requestXRReferenceSpace(session: XRSessionLike): Promise<XRReferenceSpaceLike> {
        try {
            return await session.requestReferenceSpace("local-floor");
        } catch {
            try {
                return await session.requestReferenceSpace("local");
            } catch {
                return session.requestReferenceSpace("viewer");
            }
        }
    }

    private onXRSessionEnd = () => {
        const frameError = this.xrFrameError;
        this.xrSession?.removeEventListener("end", this.onXRSessionEnd);
        this.xrSession?.removeEventListener("selectstart", this.onXRSelectStart);
        this.xrSession?.removeEventListener("selectend", this.onXRSelectEnd);
        this.xrSession = undefined;
        this.xrRefSpace = undefined;
        this.xrFrameError = undefined;
        this.xrPinchStartDistance = undefined;
        this.xrPinchStartScale = undefined;
        this.xrHandPinches.clear();
        this.xrHandPositions.clear();
        this.deleteXRFramebuffer();
        this.resetPicoGLFramebufferState();
        this.app.viewport(0, 0, this.app.width, this.app.height);
        this.resumeDesktopRenderLoop();
        if (frameError) {
            setTimeout(() => alert(`VR rendering failed: ${frameError}`));
        }
    };

    private resumeDesktopRenderLoop(): void {
        if (this.resumeRenderLoopAfterXR) {
            this.running = true;
            this.animationId = requestAnimationFrame(this.frameCallback);
        }
        this.resumeRenderLoopAfterXR = false;
    }

    private onXRFrame = (time: DOMHighResTimeStamp, frame: XRFrameLike) => {
        const session = frame.session;
        if (session !== this.xrSession || !this.xrRefSpace) {
            return;
        }

        let failed = false;
        try {
            if (!this.xrFrameValidated) {
                while (this.gl.getError() !== PicoGL.NO_ERROR) {
                    // Quest Browser can leave an internal swapchain error before the callback.
                }
            }
            const deltaTime = this.stats.getDeltaTime(time);
            this.stats.update(time);
            this.renderXRFrame(time, deltaTime, frame);
            if (!this.xrFrameValidated) {
                const glError = this.gl.getError();
                if (glError !== PicoGL.NO_ERROR) {
                    throw new Error(`WebGL error 0x${glError.toString(16)}`);
                }
                this.xrFrameValidated = true;
            }
            this.onFrameEnd();
        } catch (e) {
            failed = true;
            console.error("WebXR frame failed", e);
            this.xrFrameError = e instanceof Error ? e.message : String(e);
            session.end().catch(() => {});
        } finally {
            if (!failed && this.xrSession === session) {
                session.requestAnimationFrame(this.onXRFrame);
            }
        }
    };

    private renderXRFrame(
        time: DOMHighResTimeStamp,
        deltaTime: DOMHighResTimeStamp,
        frame: XRFrameLike,
    ): void {
        const pose = frame.getViewerPose(this.xrRefSpace!);
        const glLayer = frame.session.renderState.baseLayer;
        if (!pose || !glLayer) {
            return;
        }

        if (
            !this.mainProgram ||
            !this.mainAlphaProgram ||
            !this.npcProgram ||
            !this.sceneUniformBuffer ||
            !this.textureArray ||
            !this.textureMaterials
        ) {
            return;
        }

        const views = pose.views.map((view) => ({ view, viewport: glLayer.getViewport(view) }));
        if (views.length === 0) {
            return;
        }
        this.ensureXRFramebuffer(
            Math.max(...views.map(({ viewport }) => viewport.width)),
            Math.max(...views.map(({ viewport }) => viewport.height)),
        );

        const timeSec = time / 1000;
        const frameCount = this.stats.frameCount;
        const tick = Math.floor(timeSec / 0.6);
        const ticksElapsed = Math.min(tick - this.lastTick, 1);
        if (ticksElapsed > 0) {
            this.lastTick = tick;
        }

        const clientTick = Math.floor(timeSec / 0.02);
        const clientTicksElapsed = Math.min(clientTick - this.lastClientTick, 50);
        if (clientTicksElapsed > 0) {
            this.lastClientTick = clientTick;
        }

        const camera = this.mapViewer.camera;
        this.handleKeyInput(deltaTime);
        this.handleJoystickInput(deltaTime);
        this.handleXRControllerInput(deltaTime, frame.session);
        this.handleXRHandInput(frame);
        camera.update(this.app.width, this.app.height);
        mat4.rotateX(this.xrWorldViewMatrix, camera.cameraMatrix, -camera.pitch * RS_TO_RADIANS);
        mat4.invert(this.xrWorldViewMatrix, this.xrWorldViewMatrix);
        vec3.set(this.xrWorldScaleVector, this.xrWorldScale, this.xrWorldScale, this.xrWorldScale);
        mat4.fromScaling(this.xrViewMatrix, this.xrWorldScaleVector);
        mat4.multiply(this.xrWorldViewMatrix, this.xrViewMatrix, this.xrWorldViewMatrix);

        const renderDistance = this.mapViewer.renderDistance;
        this.mapManager.update(
            camera,
            frameCount,
            renderDistance,
            this.mapViewer.unloadDistance,
            false,
        );

        this.cameraPosUni[0] = camera.getPosX();
        this.cameraPosUni[1] = camera.getPosZ();

        this.tickPass(timeSec, ticksElapsed, clientTicksElapsed);
        const npcDataTextureIndex = this.updateNpcDataTexture();
        const npcDataTexture = this.npcDataTextureBuffer[npcDataTextureIndex];

        if (this.cullBackFace) {
            this.app.enable(PicoGL.CULL_FACE);
        } else {
            this.app.disable(PicoGL.CULL_FACE);
        }

        this.app.enable(PicoGL.DEPTH_TEST);
        this.app.depthMask(true);
        this.app.clearColor(this.skyColor[0], this.skyColor[1], this.skyColor[2], 1);
        const handJointCount = this.updateXRHandJoints(frame);

        for (const { view, viewport } of views) {
            this.app.drawFramebuffer(this.xrFramebuffer!);
            this.app.viewport(0, 0, viewport.width, viewport.height);
            this.gl.disable(PicoGL.SCISSOR_TEST);
            this.gl.clear(PicoGL.COLOR_BUFFER_BIT | PicoGL.DEPTH_BUFFER_BIT);

            mat4.multiply(this.xrViewMatrix, view.transform.inverse.matrix, this.xrWorldViewMatrix);
            mat4.multiply(this.xrViewProjMatrix, view.projectionMatrix, this.xrViewMatrix);
            this.updateSceneUniforms(
                this.xrViewMatrix,
                view.projectionMatrix,
                this.xrViewProjMatrix,
                timeSec,
                renderDistance,
            );
            this.drawScenePasses(npcDataTextureIndex, npcDataTexture);
            this.drawXRHandJoints(view, handJointCount);

            this.app.readFramebuffer(this.xrFramebuffer!);
            this.gl.readBuffer(PicoGL.COLOR_ATTACHMENT0);
            this.gl.bindFramebuffer(PicoGL.DRAW_FRAMEBUFFER, glLayer.framebuffer);
            this.gl.blitFramebuffer(
                0,
                0,
                viewport.width,
                viewport.height,
                viewport.x,
                viewport.y,
                viewport.x + viewport.width,
                viewport.y + viewport.height,
                PicoGL.COLOR_BUFFER_BIT,
                PicoGL.NEAREST,
            );
            this.resetPicoGLFramebufferState();
        }

        this.loadPendingMap(timeSec);
    }

    private handleXRControllerInput(deltaTime: number, session: XRSessionLike): void {
        const deltaTimeSec = deltaTime / 1000;
        const moveSpeed = 16 * this.mapViewer.cameraSpeed * deltaTimeSec;
        const turnSpeed = 64 * 5 * deltaTimeSec;

        for (const inputSource of session.inputSources) {
            const axes = inputSource.gamepad?.axes;
            if (!axes || axes.length < 2) {
                continue;
            }

            const axisOffset = axes.length >= 4 ? 2 : 0;
            const x = getAxisDeadzone(axes[axisOffset], 0.15);
            const y = getAxisDeadzone(axes[axisOffset + 1], 0.15);
            if (x === 0 && y === 0) {
                continue;
            }

            if (inputSource.handedness === "right") {
                this.mapViewer.camera.updateYaw(this.mapViewer.camera.yaw, x * turnSpeed);
                this.mapViewer.camera.move(0, y * moveSpeed, 0);
            } else {
                this.mapViewer.camera.move(x * -moveSpeed, 0, y * moveSpeed, false);
            }
        }
    }

    private onXRSelectStart = (event: Event) => {
        const inputSource = (event as XRInputSourceEventLike).inputSource;
        if (inputSource.hand) {
            this.xrHandPinches.add(inputSource);
            this.xrHandPositions.delete(inputSource);
            if (this.xrHandPinches.size === 2) {
                this.xrPinchStartDistance = undefined;
                this.xrPinchStartScale = undefined;
            }
        }
    };

    private onXRSelectEnd = (event: Event) => {
        const inputSource = (event as XRInputSourceEventLike).inputSource;
        this.xrHandPinches.delete(inputSource);
        this.xrHandPositions.delete(inputSource);
        this.xrPinchStartDistance = undefined;
        this.xrPinchStartScale = undefined;
    };

    private handleXRHandInput(frame: XRFrameLike): void {
        const camera = this.mapViewer.camera;
        const speed = this.mapViewer.cameraSpeed;
        let pinchSource: XRInputSourceLike | undefined;
        let pinchPosition: XRPoseLike["transform"]["position"] | undefined;
        let secondPinchPosition: XRPoseLike["transform"]["position"] | undefined;

        for (const inputSource of this.xrHandPinches) {
            const wrist = inputSource.hand?.get("wrist");
            const pose = wrist ? frame.getJointPose(wrist, this.xrRefSpace!) : null;
            if (!pose) {
                this.xrHandPositions.delete(inputSource);
                continue;
            }
            if (!pinchSource) {
                pinchSource = inputSource;
                pinchPosition = pose.transform.position;
            } else {
                secondPinchPosition = pose.transform.position;
            }
        }

        if (this.xrHandPinches.size === 2) {
            if (pinchPosition && secondPinchPosition) {
                const distance = Math.hypot(
                    pinchPosition.x - secondPinchPosition.x,
                    pinchPosition.y - secondPinchPosition.y,
                    pinchPosition.z - secondPinchPosition.z,
                );
                if (this.xrPinchStartDistance === undefined && distance > 0.05) {
                    this.xrPinchStartDistance = distance;
                    this.xrPinchStartScale = this.xrWorldScale;
                } else if (
                    this.xrPinchStartDistance !== undefined &&
                    this.xrPinchStartScale !== undefined &&
                    distance > 0.05
                ) {
                    this.xrWorldScale = Math.min(
                        XR_MAX_WORLD_SCALE,
                        Math.max(
                            XR_MIN_WORLD_SCALE,
                            this.xrPinchStartScale * (distance / this.xrPinchStartDistance),
                        ),
                    );
                }
            } else {
                this.xrPinchStartDistance = undefined;
                this.xrPinchStartScale = undefined;
            }
            this.xrHandPositions.clear();
            return;
        }

        this.xrPinchStartDistance = undefined;
        this.xrPinchStartScale = undefined;
        if (pinchSource && pinchPosition) {
            const previous = this.xrHandPositions.get(pinchSource);
            if (previous) {
                const dx = pinchPosition.x - previous[0];
                const dy = pinchPosition.y - previous[1];
                const dz = pinchPosition.z - previous[2];

                if (Math.hypot(dx, dy, dz) < 0.15) {
                    if (pinchSource.handedness === "right") {
                        camera.updateYaw(camera.yaw, dx * 1024 * speed);
                    } else {
                        camera.move(dx * 64 * speed, dy * 32 * speed, -dz * 64 * speed);
                    }
                }
                vec3.set(previous, pinchPosition.x, pinchPosition.y, pinchPosition.z);
            } else {
                this.xrHandPositions.set(
                    pinchSource,
                    vec3.fromValues(pinchPosition.x, pinchPosition.y, pinchPosition.z),
                );
            }
        }
    }

    private updateXRHandJoints(frame: XRFrameLike): number {
        if (!this.handVertexBuffer) {
            return 0;
        }

        let jointCount = 0;
        for (const inputSource of frame.session.inputSources) {
            const hand = inputSource.hand;
            if (!hand || hand.size > MAX_HAND_JOINTS_PER_HAND) {
                continue;
            }
            if (!frame.fillPoses(hand.values(), this.xrRefSpace!, this.xrHandPoseData)) {
                continue;
            }

            const handState =
                (inputSource.handedness === "right" ? 1 : 0) +
                (this.xrHandPinches.has(inputSource) ? 2 : 0);
            for (let i = 0; i < hand.size && jointCount < MAX_HAND_JOINTS; i++) {
                const matrixOffset = i * 16;
                const jointOffset = jointCount * 4;
                this.xrHandJointData[jointOffset] = this.xrHandPoseData[matrixOffset + 12];
                this.xrHandJointData[jointOffset + 1] = this.xrHandPoseData[matrixOffset + 13];
                this.xrHandJointData[jointOffset + 2] = this.xrHandPoseData[matrixOffset + 14];
                this.xrHandJointData[jointOffset + 3] = handState;
                jointCount++;
            }
        }

        if (jointCount > 0) {
            this.handVertexBuffer.data(this.xrHandJointData.subarray(0, jointCount * 4));
        }
        return jointCount;
    }

    private drawXRHandJoints(view: XRViewLike, jointCount: number): void {
        if (!this.handDrawCall || jointCount === 0) {
            return;
        }

        mat4.multiply(this.xrViewProjMatrix, view.projectionMatrix, view.transform.inverse.matrix);
        this.app.enable(PicoGL.BLEND);
        this.app.depthMask(false);
        this.handDrawCall
            .uniform("u_viewProjMatrix", this.xrViewProjMatrix)
            .drawRanges([0, jointCount])
            .draw();
        this.app.depthMask(true);

        if (!this.xrHandsValidated) {
            const glError = this.gl.getError();
            if (glError !== PicoGL.NO_ERROR) {
                throw new Error(`WebXR hand rendering error 0x${glError.toString(16)}`);
            }
            this.xrHandsValidated = true;
        }
    }

    private ensureXRFramebuffer(width: number, height: number): void {
        if (!this.xrFramebuffer) {
            this.xrColorTarget = this.app.createRenderbuffer(width, height, PicoGL.RGBA8);
            this.xrDepthTarget = this.app.createRenderbuffer(
                width,
                height,
                PicoGL.DEPTH_COMPONENT24,
            );
            this.xrFramebuffer = this.app
                .createFramebuffer()
                .colorTarget(0, this.xrColorTarget)
                .depthTarget(this.xrDepthTarget);
        } else if (this.xrFramebuffer.width !== width || this.xrFramebuffer.height !== height) {
            this.xrFramebuffer.resize(width, height);
        } else {
            return;
        }

        const status = this.xrFramebuffer.getStatus();
        if (status !== PicoGL.FRAMEBUFFER_COMPLETE) {
            throw new Error(`WebXR framebuffer is incomplete: 0x${status.toString(16)}`);
        }
    }

    private deleteXRFramebuffer(): void {
        this.xrFramebuffer?.delete();
        this.xrFramebuffer = undefined;
        this.xrColorTarget?.delete();
        this.xrColorTarget = undefined;
        this.xrDepthTarget?.delete();
        this.xrDepthTarget = undefined;
    }

    private resetPicoGLFramebufferState(): void {
        const state = this.app.state as any;
        state.framebuffers[state.drawFramebufferBinding] = undefined;
        state.framebuffers[state.readFramebufferBinding] = undefined;
    }

    private updateSceneUniforms(
        viewMatrix: mat4,
        projectionMatrix: mat4,
        viewProjMatrix: mat4,
        timeSec: number,
        renderDistance: number,
    ): void {
        this.sceneUniformBuffer
            ?.set(0, viewProjMatrix as Float32Array)
            .set(1, viewMatrix as Float32Array)
            .set(2, projectionMatrix as Float32Array)
            .set(3, this.skyColor as Float32Array)
            .set(4, this.cameraPosUni as Float32Array)
            .set(5, renderDistance as any)
            .set(6, this.fogDepth as any)
            .set(7, timeSec as any)
            .set(8, this.brightness as any)
            .set(9, this.colorBanding as any)
            .set(10, this.mapViewer.isNewTextureAnim as any)
            .update();
    }

    private drawScenePasses(
        npcDataTextureIndex: number,
        npcDataTexture: Texture | undefined,
    ): void {
        this.app.disable(PicoGL.BLEND);
        this.renderOpaquePass();
        this.renderOpaqueNpcPass(npcDataTextureIndex, npcDataTexture);

        this.app.enable(PicoGL.BLEND);
        this.renderTransparentPass();
        this.renderTransparentNpcPass(npcDataTextureIndex, npcDataTexture);
    }

    private loadPendingMap(timeSec: number): void {
        const mapData = this.mapsToLoad.shift();
        if (
            mapData &&
            this.isValidMapData(mapData) &&
            this.mainProgram &&
            this.mainAlphaProgram &&
            this.npcProgram &&
            this.textureArray &&
            this.textureMaterials &&
            this.sceneUniformBuffer
        ) {
            this.loadMap(
                this.mainProgram,
                this.mainAlphaProgram,
                this.npcProgram,
                this.textureArray,
                this.textureMaterials,
                this.sceneUniformBuffer,
                mapData,
                timeSec,
            );
        }
    }

    tickPass(time: number, ticksElapsed: number, clientTicksElapsed: number): void {
        const cycle = time / 0.02;

        const seqFrameLoader = this.mapViewer.seqFrameLoader;
        const seqTypeLoader = this.mapViewer.seqTypeLoader;

        const pathfinder = this.mapViewer.pathfinder;

        this.npcRenderCount = 0;
        for (let i = 0; i < this.mapManager.visibleMapCount; i++) {
            const map = this.mapManager.visibleMaps[i];

            for (const loc of map.locsAnimated) {
                loc.update(seqFrameLoader, cycle);
            }

            for (let t = 0; t < ticksElapsed; t++) {
                for (const npc of map.npcs) {
                    npc.updateServerMovement(pathfinder, map.borderSize, map.collisionMaps);
                }
            }

            for (let t = 0; t < clientTicksElapsed; t++) {
                for (const npc of map.npcs) {
                    npc.updateMovement(seqTypeLoader, seqFrameLoader);
                }
            }

            this.addNpcRenderData(map);
        }
    }

    addNpcRenderData(map: WebGLMapSquare) {
        const npcs = map.npcs;

        if (npcs.length === 0) {
            return;
        }

        const frameCount = this.stats.frameCount;

        map.npcDataTextureOffsets[frameCount % map.npcDataTextureOffsets.length] =
            this.npcRenderCount;

        const newCount = this.npcRenderCount + npcs.length;

        if (this.npcRenderData.length / 4 < newCount) {
            const newData = new Uint16Array(Math.ceil((newCount * 2) / 16) * 16 * 4);
            newData.set(this.npcRenderData);
            this.npcRenderData = newData;
        }

        for (const npc of npcs) {
            let offset = this.npcRenderCount * 4;

            const tileX = npc.x >> 7;
            const tileY = npc.y >> 7;

            let renderPlane = npc.level;
            if (renderPlane < 3 && (map.getTileRenderFlag(1, tileX, tileY) & 0x2) === 2) {
                renderPlane++;
            }

            this.npcRenderData[offset++] = npc.x;
            this.npcRenderData[offset++] = npc.y;
            this.npcRenderData[offset++] = (npc.rotation << 2) | renderPlane;
            this.npcRenderData[offset++] = npc.npcType.id;

            this.npcRenderCount++;
        }
    }

    updateNpcDataTexture() {
        const frameCount = this.stats.frameCount;

        const newNpcDataTextureIndex = frameCount % this.npcDataTextureBuffer.length;
        const npcDataTextureIndex = (frameCount + 1) % this.npcDataTextureBuffer.length;
        this.npcDataTextureBuffer[newNpcDataTextureIndex]?.delete();
        this.npcDataTextureBuffer[newNpcDataTextureIndex] = this.app.createTexture2D(
            this.npcRenderData,
            16,
            Math.max(Math.ceil(this.npcRenderCount / 16), 1),
            {
                internalFormat: PicoGL.RGBA16UI,
                minFilter: PicoGL.NEAREST,
                magFilter: PicoGL.NEAREST,
            },
        );

        return npcDataTextureIndex;
    }

    draw(drawCall: DrawCall, drawRanges: number[][]) {
        if (this.hasMultiDraw) {
            drawCall.draw();
        } else {
            for (let i = 0; i < drawRanges.length; i++) {
                drawCall.uniform("u_drawId", i);
                drawCall.drawRanges(drawRanges[i]);
                drawCall.draw();
            }
        }
    }

    renderOpaquePass(): void {
        const camera = this.mapViewer.camera;
        const cameraMapX = camera.getMapX();
        const cameraMapY = camera.getMapY();

        for (let i = 0; i < this.mapManager.visibleMapCount; i++) {
            const map = this.mapManager.visibleMaps[i];
            const dist = map.getMapDistance(cameraMapX, cameraMapY);

            const isInteract = this.hoveredMapIds.has(map.id);
            const isLod = dist >= this.mapViewer.lodDistance;

            const { drawCall, drawRanges } = map.getDrawCall(false, isInteract, isLod);

            for (const loc of map.locsAnimated) {
                const frameId = loc.frame;
                const frame = loc.anim.frames[frameId | 0];

                const index = loc.getDrawRangeIndex(false, isInteract, isLod);
                if (index !== -1) {
                    drawCall.offsets[index] = frame[0];
                    (drawCall as any).numElements[index] = frame[1];

                    drawRanges[index] = frame;
                }
            }

            this.draw(drawCall, drawRanges);
        }
    }

    renderOpaqueNpcPass(npcDataTextureIndex: number, npcDataTexture: Texture | undefined): void {
        if (!npcDataTexture || !this.loadNpcs) {
            return;
        }

        for (let i = 0; i < this.mapManager.visibleMapCount; i++) {
            const map = this.mapManager.visibleMaps[i];
            const npcs = map.npcs;

            if (npcs.length === 0) {
                continue;
            }

            const dataOffset = map.npcDataTextureOffsets[npcDataTextureIndex];
            if (dataOffset === -1) {
                continue;
            }

            const { drawCall, drawRanges } = map.drawCallNpc;

            drawCall.uniform("u_npcDataOffset", dataOffset);
            drawCall.texture("u_npcDataTexture", npcDataTexture);

            for (let i = 0; i < npcs.length; i++) {
                const npc = npcs[i];
                const anim = npc.getAnimationFrames();

                const frameId = npc.movementFrame;
                const frame = anim.frames[frameId];

                (drawCall as any).offsets[i] = frame[0];
                (drawCall as any).numElements[i] = frame[1];

                drawRanges[i] = frame;
            }

            this.draw(drawCall, drawRanges);
        }
    }

    renderTransparentPass(): void {
        const camera = this.mapViewer.camera;
        const cameraMapX = camera.getMapX();
        const cameraMapY = camera.getMapY();

        for (let i = this.mapManager.visibleMapCount - 1; i >= 0; i--) {
            const map = this.mapManager.visibleMaps[i];
            const dist = map.getMapDistance(cameraMapX, cameraMapY);

            const isInteract = this.hoveredMapIds.has(map.id);
            const isLod = dist >= this.mapViewer.lodDistance;

            const { drawCall, drawRanges } = map.getDrawCall(true, isInteract, isLod);

            for (const loc of map.locsAnimated) {
                if (loc.anim.framesAlpha) {
                    const frameId = loc.frame;
                    const frame = loc.anim.framesAlpha[frameId | 0];

                    const index = loc.getDrawRangeIndex(true, isInteract, isLod);
                    if (index !== -1) {
                        drawCall.offsets[index] = frame[0];
                        (drawCall as any).numElements[index] = frame[1];

                        drawRanges[index] = frame;
                    }
                }
            }

            this.draw(drawCall, drawRanges);
        }
    }

    renderTransparentNpcPass(
        npcDataTextureIndex: number,
        npcDataTexture: Texture | undefined,
    ): void {
        if (!npcDataTexture || !this.loadNpcs) {
            return;
        }

        for (let i = this.mapManager.visibleMapCount - 1; i >= 0; i--) {
            const map = this.mapManager.visibleMaps[i];
            const npcs = map.npcs;

            if (npcs.length === 0) {
                continue;
            }

            const dataOffset = map.npcDataTextureOffsets[npcDataTextureIndex];
            if (dataOffset === -1) {
                continue;
            }

            const { drawCall, drawRanges } = map.drawCallNpc;

            drawCall.uniform("u_npcDataOffset", dataOffset);
            drawCall.texture("u_npcDataTexture", npcDataTexture);

            for (let i = 0; i < npcs.length; i++) {
                const npc = npcs[i];
                const anim = npc.getAnimationFrames();

                const frameId = npc.movementFrame;
                let frame: DrawRange = NULL_DRAW_RANGE;
                if (anim.framesAlpha) {
                    frame = anim.framesAlpha[frameId];
                }

                (drawCall as any).offsets[i] = frame[0];
                (drawCall as any).numElements[i] = frame[1];

                drawRanges[i] = frame;
            }

            this.draw(drawCall, drawRanges);
        }
    }

    checkInteractions(interactions: Interactions): void {
        const interactReady = interactions.check(
            this.gl,
            this.hoveredMapIds,
            this.closestInteractIndices,
        );
        if (interactReady) {
            this.interactBuffer = interactions.interactBuffer;
        }

        if (!this.interactBuffer) {
            return;
        }

        const frameCount = this.stats.frameCount;

        const inputManager = this.mapViewer.inputManager;
        const isMouseDown = inputManager.dragX !== -1 || inputManager.dragY !== -1;
        const picked = inputManager.pickX !== -1 && inputManager.pickY !== -1;

        if (!interactReady && !picked) {
            return;
        }

        const menuCooldown = isTouchDevice ? 50 : 10;

        if (
            inputManager.mouseX === -1 ||
            inputManager.mouseY === -1 ||
            frameCount - this.mapViewer.menuOpenedFrame < menuCooldown
        ) {
            return;
        }

        // Don't auto close menu on touch devices
        if (this.mapViewer.menuOpen && !picked && !isMouseDown && isTouchDevice) {
            return;
        }

        if (!picked && !this.mapViewer.tooltips) {
            this.mapViewer.closeMenu();
            return;
        }

        const menuEntries: OsrsMenuEntry[] = [];
        const examineEntries: OsrsMenuEntry[] = [];

        const locIds = new Set<number>();
        const objIds = new Set<number>();
        const npcIds = new Set<number>();

        for (let i = 0; i < INTERACTION_RADIUS + 1; i++) {
            const indices = this.closestInteractIndices.get(i);
            if (!indices) {
                continue;
            }
            for (const index of indices) {
                const interactId = this.interactBuffer[index];
                const interactType = this.interactBuffer[index + 2];
                if (interactType === InteractType.LOC) {
                    const locType = this.mapViewer.locTypeLoader.load(interactId);
                    if (locType.name === "null" && !this.mapViewer.debugId) {
                        continue;
                    }
                    if (locIds.has(interactId)) {
                        continue;
                    }
                    locIds.add(interactId);

                    for (const option of locType.actions) {
                        if (!option) {
                            continue;
                        }
                        menuEntries.push({
                            option,
                            targetId: locType.id,
                            targetType: MenuTargetType.LOC,
                            targetName: locType.name,
                            targetLevel: -1,
                            onClick: this.mapViewer.closeMenu,
                        });
                    }

                    examineEntries.push({
                        option: "Examine",
                        targetId: locType.id,
                        targetType: MenuTargetType.LOC,
                        targetName: locType.name,
                        targetLevel: -1,
                        onClick: this.mapViewer.onExamine,
                    });
                } else if (interactType === InteractType.OBJ) {
                    const objType = this.mapViewer.objTypeLoader.load(interactId);
                    if (objType.name === "null" && !this.mapViewer.debugId) {
                        continue;
                    }
                    if (objIds.has(interactId)) {
                        continue;
                    }
                    objIds.add(interactId);

                    for (const option of objType.groundActions) {
                        if (!option) {
                            continue;
                        }
                        menuEntries.push({
                            option,
                            targetId: objType.id,
                            targetType: MenuTargetType.OBJ,
                            targetName: objType.name,
                            targetLevel: -1,
                            onClick: this.mapViewer.closeMenu,
                        });
                    }

                    examineEntries.push({
                        option: "Examine",
                        targetId: objType.id,
                        targetType: MenuTargetType.OBJ,
                        targetName: objType.name,
                        targetLevel: -1,
                        onClick: this.mapViewer.onExamine,
                    });
                } else if (interactType === InteractType.NPC) {
                    let npcType = this.mapViewer.npcTypeLoader.load(interactId);
                    if (npcType.transforms) {
                        const transformed = npcType.transform(
                            this.mapViewer.varManager,
                            this.mapViewer.npcTypeLoader,
                        );
                        if (!transformed) {
                            continue;
                        }
                        npcType = transformed;
                    }
                    if (npcType.name === "null" && !this.mapViewer.debugId) {
                        continue;
                    }
                    if (npcIds.has(interactId)) {
                        continue;
                    }
                    npcIds.add(interactId);

                    for (const option of npcType.actions) {
                        if (!option) {
                            continue;
                        }
                        menuEntries.push({
                            option,
                            targetId: npcType.id,
                            targetType: MenuTargetType.NPC,
                            targetName: npcType.name,
                            targetLevel: npcType.combatLevel,
                            onClick: this.mapViewer.closeMenu,
                        });
                    }

                    examineEntries.push({
                        option: "Examine",
                        targetId: npcType.id,
                        targetType: MenuTargetType.NPC,
                        targetName: npcType.name,
                        targetLevel: npcType.combatLevel,
                        onClick: this.mapViewer.onExamine,
                    });
                }
            }
        }

        menuEntries.push({
            option: "Walk here",
            targetId: -1,
            targetType: MenuTargetType.NONE,
            targetName: "",
            targetLevel: -1,
            onClick: this.mapViewer.closeMenu,
        });
        menuEntries.push(...examineEntries);
        menuEntries.push({
            option: "Cancel",
            targetId: -1,
            targetType: MenuTargetType.NONE,
            targetName: "",
            targetLevel: -1,
            onClick: this.mapViewer.closeMenu,
        });

        this.mapViewer.menuOpen = picked;
        if (picked) {
            this.mapViewer.menuOpenedFrame = frameCount;
        }
        this.mapViewer.menuX = inputManager.mouseX;
        this.mapViewer.menuY = inputManager.mouseY;
        this.mapViewer.menuEntries = menuEntries;
    }

    override async cleanUp(): Promise<void> {
        const xrSession = this.xrSession;
        this.xrSession = undefined;
        this.xrRefSpace = undefined;
        this.resumeRenderLoopAfterXR = false;
        xrSession?.removeEventListener("end", this.onXRSessionEnd);
        xrSession?.removeEventListener("selectstart", this.onXRSelectStart);
        xrSession?.removeEventListener("selectend", this.onXRSelectEnd);
        xrSession?.end().catch(() => {});
        this.xrHandPinches.clear();
        this.xrHandPositions.clear();
        this.deleteXRFramebuffer();

        super.cleanUp();
        this.mapViewer.workerPool.resetLoader(this.dataLoader);

        this.quadArray?.delete();
        this.quadArray = undefined;

        this.quadPositions?.delete();
        this.quadPositions = undefined;

        this.handVertexArray?.delete();
        this.handVertexArray = undefined;

        this.handVertexBuffer?.delete();
        this.handVertexBuffer = undefined;
        this.handDrawCall = undefined;

        // Uniforms
        this.sceneUniformBuffer?.delete();
        this.sceneUniformBuffer = undefined;

        // Framebuffers
        this.framebuffer?.delete();
        this.framebuffer = undefined;

        this.colorTarget?.delete();
        this.colorTarget = undefined;

        this.interactTarget?.delete();
        this.interactTarget = undefined;

        this.depthTarget?.delete();
        this.depthTarget = undefined;

        this.textureFramebuffer?.delete();
        this.textureFramebuffer = undefined;

        this.textureColorTarget?.delete();
        this.textureColorTarget = undefined;

        this.interactFramebuffer?.delete();
        this.interactFramebuffer = undefined;

        this.interactColorTarget?.delete();
        this.interactColorTarget = undefined;

        // Textures
        this.textureArray?.delete();
        this.textureArray = undefined;

        this.textureMaterials?.delete();
        this.textureMaterials = undefined;

        for (const texture of this.npcDataTextureBuffer) {
            texture?.delete();
        }

        this.clearMaps();

        if (this.shadersPromise) {
            for (const shader of await this.shadersPromise) {
                shader.delete();
            }
            this.shadersPromise = undefined;
        }
        this.handProgram = undefined;
        console.log("Renderer cleaned up");
    }
}
