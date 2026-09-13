/**
 * InstantHMR ONNX model wrapper.
 *
 * Inputs:
 *   - `image`        float32 [N, 3, 224, 224] — ImageNet-normalized NCHW
 *   - `cliff_cond`   float32 [N, 3]           — person bbox centre (normalised) + frame fraction
 *
 * Outputs (by index, not by name — graph export order is stable):
 *   - [0] mhr_params   (N, 204)  — MHR pose parameters (34 joints × 6D rotation)
 *   - [1] shape_params (N, 45)   — identity (20 body + 20 head + 5 hand)
 *   - [2] cam_trans    (N, 3)    — body translation in metres, camera frame
 *   - [3] joints_2d    (N, 70, 2) — normalised crop coords in [-1, 1]
 *   - [4] joints_3d    (N, 70, 3) — body-centred, metres, Y-down (optional — falls back to zeros)
 *
 * Model class is intentionally thin: it knows about the two input tensors and
 * the five output indices. Crop/normalize/decode happens in the solution class.
 */

import * as ort from 'onnxruntime-web/all';

import { BaseTool } from '../core/base';
import { initOnnxRuntimeWeb } from '../core/onnxRuntime';
import { loadOnnxSession } from '../core/onnxSession';
import { createLogger } from '../core/logger';
import { INPUT_SIZE, NUM_JOINTS, float16ToFloat32 } from '../core/instanthmrGeometry';
import type { BackendType } from '../types/index';

const log = createLogger('InstantHMRModel');

/** Default URL — HuggingFace mirror of the InstantHMR checkpoint. */
export const INSTANTHMR_MODEL_URL =
  'https://huggingface.co/momolesang/InstantHMR/resolve/main/instanthmr.onnx';

export interface InstantHMRRunOutput {
  /** MHR pose parameters (length 204). */
  mhr: Float32Array;
  /** Identity / shape parameters (length 45). */
  shape: Float32Array;
  /** Body translation in metres, camera frame (length 3). */
  cam: Float32Array;
  /** joints_2d, normalised to crop [-1, 1] (length NUM_JOINTS * 2). */
  joints2dNorm: Float32Array;
  /** joints_3d, body-centred, metres (length NUM_JOINTS * 3). May be zeros if the model was exported without it. */
  joints3dLocal: Float32Array;
}

export class InstantHMRModel extends BaseTool {
  /** Renamed from `backend` to avoid colliding with the protected
   *  `BaseTool.backend` field introduced when `BaseTool` learned about
   *  execution providers. `InstantHMRModel` overrides `init()` entirely
   *  (it has its own EP ladder + cache-or-fetch path), so this is a
   *  class-local concern. */
  private readonly preferredBackend: BackendType;

  constructor(
    modelPath: string = INSTANTHMR_MODEL_URL,
    modelInputSize: [number, number] = [INPUT_SIZE, INPUT_SIZE],
    /** ONNX execution provider. The graph's two-input shape (image + cliff_cond)
     *  is incompatible with WebGL; WebGPU works in browsers with a real GPU and
     *  a cross-origin-isolated context, WASM otherwise. */
    backend: BackendType = 'wasm',
  ) {
    super(modelPath, modelInputSize, backend);
    this.preferredBackend = backend;
  }

  async init(): Promise<void> {
    // Configure ORT wasm once at module top-level (no-op under SSR).
    initOnnxRuntimeWeb();

    // Build execution provider list, falling back through the user's choice.
    // WebGPU → WebNN → WebGL → WASM: each tier is rejected at session-create
    // time if the graph is incompatible or the backend is unavailable, so we
    // hand a small ladder and keep what ORT actually loads.
    const fallback: BackendType[] | undefined =
      this.preferredBackend === 'wasm' ? undefined : ['wasm'];

    this.session = await loadOnnxSession({
      url: this.modelPath,
      backend: this.preferredBackend,
      cache: true,
      fallbackProviders: fallback,
      logPrefix: '[InstantHMRModel]',
    });

    log.log(`Ready (backend=${this.preferredBackend})`);
  }

  /** Run inference on one person. */
  async call(
    imageNCHW: Float32Array,
    cliffCond: Float32Array,
  ): Promise<InstantHMRRunOutput> {
    if (!this.session) throw new Error('InstantHMRModel: session not initialised');
    const feeds: Record<string, ort.Tensor> = {
      image: new ort.Tensor('float32', imageNCHW, [1, 3, INPUT_SIZE, INPUT_SIZE]),
      cliff_cond: new ort.Tensor('float32', cliffCond, [1, 3]),
    };
    const out = await this.session.run(feeds);

    const names = this.session.outputNames;
    // The export order is documented in the model README; we read by index, not
    // by name, so a re-export with different names (e.g. dropping joints_3d) does
    // not crash — see fallback below.
    const mhr = tensorData(out[names[0]]);
    const shape = tensorData(out[names[1]]);
    const cam = tensorData(out[names[2]]);
    const j2dAll = tensorData(out[names[3]]);
    const j3dAll = names.length >= 5 ? tensorData(out[names[4]]) : null;

    return {
      mhr: mhr.slice(0, 204),
      shape: shape.slice(0, 45),
      cam: cam.slice(0, 3),
      joints2dNorm: j2dAll.slice(0, NUM_JOINTS * 2),
      joints3dLocal: j3dAll
        ? j3dAll.slice(0, NUM_JOINTS * 3)
        : new Float32Array(NUM_JOINTS * 3),
    };
  }
}

/** Unpack an ORT tensor to Float32Array — handles fp32 (pass-through), fp16 (unpack), and other numeric arrays. */
function tensorData(t: ort.Tensor): Float32Array {
  const d = t.data as ArrayLike<number>;
  if (d instanceof Float32Array) return d;
  if (d instanceof Uint16Array) {
    // Half-float export: unpack bits to float32.
    const out = new Float32Array(d.length);
    for (let i = 0; i < d.length; i++) out[i] = float16ToFloat32(d[i]);
    return out;
  }
  return Float32Array.from(d);
}
