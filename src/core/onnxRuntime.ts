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
 */
export function initOnnxRuntimeWeb(): boolean {
  if (!isBrowser()) {
    return false;
  }

  // Configure ONNX Runtime Web
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@latest/dist/';
  ort.env.wasm.simd = true;
  ort.env.wasm.proxy = false;

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
