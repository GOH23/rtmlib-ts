/**
 * BaseTool - Abstract base class for ONNX models
 */

import * as ort from 'onnxruntime-web/all';
import type { BackendType } from '../types/index.js';
import { createLogger } from './logger';

const log = createLogger('BaseTool');

export abstract class BaseTool {
  protected session: ort.InferenceSession | null = null;
  protected modelPath: string;
  protected modelInputSize: [number, number];
  /** ONNX execution provider. Default `'wasm'` for backwards compatibility —
   * the field was hard-coded before subclasses needed a backend selector.
   * WebGL/WebGPU/WebNN each have their own graph-operator coverage caveats. */
  protected backend: BackendType;

  // ---- Cached input slot -------------------------------------------------
  // WebGPU/WebGL EPs upload the input tensor's backing buffer to GPU
  // memory on every `session.run()`. Allocating a fresh Float32Array +
  // ort.Tensor each frame means a new GPUBuffer upload + GC churn every
  // call — the dominant cost on YOLO's WebGPU hot path.
  //
  // We cache one (Float32Array, ort.Tensor) pair sized to the model's
  // NCHW input. Callers either pass their own buffer through
  // `inference()` (which copies into the cache on first use) or reuse the
  // buffer directly via `prepareInputData()` for zero-allocation
  // in-place updates. The tensor is reusable across `session.run()` calls
  // because ORT reads its data lazily at execute time — mutating the
  // backing Float32Array in place and re-feeding the same Tensor object is
  // supported (verified on onnxruntime-web 1.29.x).
  protected cachedInputData: Float32Array | null = null;
  private cachedInputTensor: ort.Tensor | null = null;
  private cachedInputShape: readonly number[] | null = null;

  constructor(
    modelPath: string,
    modelInputSize: [number, number],
    backend: BackendType = 'wasm',
  ) {
    this.modelPath = modelPath;
    this.modelInputSize = modelInputSize;
    this.backend = backend;
  }

  protected async init(): Promise<void> {
    // wasmPaths + simd are set unconditionally — ORT ignores them when the
    // active EP isn't wasm, so it's safe to keep setting them across all
    // backends (matches the previous behaviour for the wasm default).
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/';
    ort.env.wasm.simd = true;

    this.session = await ort.InferenceSession.create(this.modelPath, {
      executionProviders: [this.backend],
      graphOptimizationLevel: 'all',
    });

    log.log(`Loaded model: ${this.modelPath} (backend=${this.backend})`);
  }

  /**
   * Copy `buf` into the model's cached NCHW input buffer and return a
   * reusable `ort.Tensor` wrapping it. The first call allocates the
   * buffer + tensor; subsequent calls reuse both, so WebGPU uploads
   * re-target the same GPUBuffer instead of allocating a new one.
   *
   * Pass `dims` only if the input shape differs from the model's default
   * `[1, 3, modelInputSize[0], modelInputSize[1]]` (e.g. dynamic-input
   * models). The shape is part of the cache key — a different shape
   * triggers a one-shot reallocation.
   */
  protected prepareInputData(buf: Float32Array, dims?: readonly number[]): ort.Tensor {
    if (!this.session) throw new Error('Session not initialized');

    const targetDims = dims ?? [1, 3, this.modelInputSize[0], this.modelInputSize[1]];
    const targetLen = targetDims.reduce((a, b) => a * b, 1);

    // (Re)allocate the backing buffer when shape changes or on first use.
    if (
      !this.cachedInputData ||
      !this.cachedInputTensor ||
      !this.cachedInputShape ||
      !shapesEqual(this.cachedInputShape, targetDims)
    ) {
      this.cachedInputData = new Float32Array(targetLen);
      this.cachedInputTensor = new ort.Tensor('float32', this.cachedInputData, targetDims as number[]);
      this.cachedInputShape = targetDims;
    }

    // Copy into the cached buffer. If the caller passes the cached buffer
    // itself we skip the copy; otherwise we copy `buf` in.
    if (buf !== this.cachedInputData) {
      this.cachedInputData.set(buf.subarray(0, targetLen));
    }

    return this.cachedInputTensor;
  }

  protected async inference(img: Float32Array, inputSize?: [number, number]): Promise<any[]> {
    if (!this.session) throw new Error('Session not initialized');

    const dims = inputSize ? [1, 3, inputSize[0], inputSize[1]] as const : undefined;
    const inputTensor = this.prepareInputData(img, dims);

    const feeds: Record<string, ort.Tensor> = {};
    feeds[this.session.inputNames[0]] = inputTensor;

    const results = await this.session.run(feeds);
    return this.session.outputNames.map((name) => results[name]);
  }

  /**
   * Accessor for the cached NCHW buffer. Subclasses that fill the buffer
   * in-place during preprocessing should call this and write into the
   * returned array — then pass the same buffer back through
   * `prepareInputData()` on the inference call. The identity-check inside
   * `prepareInputData()` skips the redundant copy.
   *
   * Allocates lazily to `3 * inputH * inputW` on first call.
   */
  protected getCachedInputBuffer(): Float32Array {
    if (!this.cachedInputData) {
      this.prepareInputData(new Float32Array(0));
    }
    return this.cachedInputData as Float32Array;
  }

  abstract call(...args: unknown[]): Promise<unknown>;
}

function shapesEqual(a: readonly number[], b: readonly number[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}
