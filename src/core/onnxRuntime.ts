/**
 * ONNX Runtime Web initialization utilities
 * Handles browser-only configuration safely
 */

import * as ort from 'onnxruntime-web/all';
import { isBrowser } from './environment';

/**
 * Configure ONNX Runtime Web WASM settings
 * This should only be called in browser environments
 * Returns true if configuration was applied, false if in SSR
 *
 * Idempotent: every solution/* class imports this module and calls
 * `initOnnxRuntimeWeb()` at module-load and again from constructors.
 * Touching `ort.env.wasm.*` a second time causes ONNX Runtime to throw
 * "multiple calls to 'initWasm()' detected" when a subsequent
 * `InferenceSession.create()` runs (the second detector picks a
 * different backend like webgpu, falls back to wasm, and wasm thinks
 * it's never been initialized). Gate the entire config block behind
 * a one-shot flag.
 */
let __onnxRuntimeConfigured = false;
export function initOnnxRuntimeWeb(): boolean {
  if (!isBrowser()) {
    return false;
  }
  if (__onnxRuntimeConfigured) {
    return false;
  }
  __onnxRuntimeConfigured = true;

  // Configure ONNX Runtime Web. Pinned to match the CDN <script> in the demo
  // app's layout; using `@latest` here lets the CDN deeploys drift apart.
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/';
  ort.env.wasm.simd = true;
  ort.env.wasm.proxy = false;

  // Enable wasm threads when the page is cross-origin-isolated (COOP+COEP).
  // SharedArrayBuffer is gated on `crossOriginIsolated`, which only becomes
  // true when those headers are set. Cap at hardwareConcurrency - 1, with a
  // floor of 2 so we always get a parallel speedup over single-threaded.
  if (typeof globalThis !== 'undefined' && (globalThis as any).crossOriginIsolated === true) {
    const cores = Math.max(1, (navigator as any).hardwareConcurrency || 1);
    ort.env.wasm.numThreads = Math.max(2, cores - 1);
  } else {
    ort.env.wasm.numThreads = 1;
  }

  return true;
}

/**
 * Get the ONNX Runtime Web instance (only safe to call in browser)
 * Throws error if called during SSR
 */
export function getOnnxRuntime(): typeof ort {
  if (!isBrowser()) {
    throw new Error('ONNX Runtime Web can only be used in browser environments');
  }
  return ort;
}
