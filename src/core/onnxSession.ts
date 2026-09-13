/**
 * Shared ONNX-session loader: download (or fetch from cache), build the
 * execution-provider array, and create `ort.InferenceSession`.
 *
 * Every detector repeats the same ~25-line dance — fetch the model
 * buffer, check the cache, decide between `webnn | webgpu | backend`,
 * build the EP object, call `ort.InferenceSession.create`. Centralising
 * it means a single place to add a new EP (e.g. `webnn-ep`) and a
 * single place to evolve the WebGPU-availability fallback.
 */
import * as ort from 'onnxruntime-web/all';
import { getCachedModel, isModelCached } from './modelCache';
import { createLogger } from './logger';
import type { BackendType } from '../types';

export interface LoadOnnxSessionOptions {
  /** Model URL or local path. */
  url: string;
  /** Primary execution provider. */
  backend: BackendType;
  /** Use the Cache API to persist model bytes across sessions. */
  cache: boolean;
  /** WebNN deviceType — only consulted when `backend === 'webnn'`. */
  deviceType?: 'cpu' | 'gpu' | 'npu';
  /** WebNN powerPreference — only consulted when `backend === 'webnn'`. */
  powerPreference?: 'default' | 'low-power' | 'high-performance';
  /** Extra EP strings appended after the primary, e.g. `['wasm']` as a
   * always-available fallback (the `ObjectDetector` YOLO path needs this
   * so a session created against WebGL can still run on browsers without
   * WebGL). */
  fallbackProviders?: readonly BackendType[];
  /** Log prefix (e.g. `[AnimalDetector] Detection`). Omit to silence. */
  logPrefix?: string;
}

export async function loadOnnxSession(
  opts: LoadOnnxSessionOptions,
): Promise<ort.InferenceSession> {
  const log = opts.logPrefix ? createLogger(opts.logPrefix) : null;

  let buffer: ArrayBuffer;
  if (opts.cache) {
    const cached = await isModelCached(opts.url);
    log?.log(`cache ${cached ? 'hit' : 'miss'}`);
    buffer = await getCachedModel(opts.url);
  } else {
    const r = await fetch(opts.url);
    if (!r.ok) throw new Error(`Failed to fetch model: HTTP ${r.status}`);
    buffer = await r.arrayBuffer();
  }
  log?.log(`loaded, size: ${(buffer.byteLength / 1024 / 1024).toFixed(2)} MB`);

  const execProviders = buildExecutionProviders(
    opts.backend,
    opts.deviceType,
    opts.powerPreference,
    log,
  );

  if (opts.fallbackProviders) {
    execProviders.push(...opts.fallbackProviders);
  }

  return ort.InferenceSession.create(buffer, {
    executionProviders: execProviders,
    graphOptimizationLevel: 'all',
  });
}

/**
 * Build the execution-provider config for an ONNX session.
 *
 *   - `'webnn'`  → EP object with deviceType + powerPreference
 *   - `'webgpu'` → EP string `'webgpu'`, with WebGL fallback when no
 *                  GPU adapter is exposed (matches the per-detector
 *                  warning that was emitted before).
 *   - everything else (`'wasm' | 'webgl' | 'webnn'`) → EP string.
 */
export function buildExecutionProviders(
  backend: BackendType,
  deviceType?: 'cpu' | 'gpu' | 'npu',
  powerPreference?: 'default' | 'low-power' | 'high-performance',
  log: { warn(msg: string): void } | null = null,
): ort.InferenceSession.ExecutionProviderConfig[] {
  if (backend === 'webnn') {
    const ep: ort.InferenceSession.ExecutionProviderConfig = {
      name: 'webnn',
      deviceType: deviceType ?? 'gpu',
      powerPreference: powerPreference ?? 'high-performance',
    };
    return [ep];
  }

  if (backend === 'webgpu') {
    if (typeof navigator !== 'undefined' && (navigator as { gpu?: unknown }).gpu) {
      return ['webgpu'];
    }
    log?.warn('WebGPU not available, falling back to WebGL');
    return ['webgl'];
  }

  return [backend];
}
