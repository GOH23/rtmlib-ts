/**
 * BaseTool - Abstract base class for all models
 * Handles ONNX model loading and inference
 * Compatible with onnxruntime-web (browser) and onnxruntime-node
 */

import * as ort from 'onnxruntime-web';
import { BackendType } from '../types/index.js';

export abstract class BaseTool {
  protected session: ort.InferenceSession | null = null;
  protected modelPath: string;
  protected modelInputSize: [number, number];
  protected mean: number[] | null;
  protected std: number[] | null;
  protected backend: BackendType;

  constructor(
    modelPath: string,
    modelInputSize: [number, number],
    mean: number[] | null = null,
    std: number[] | null = null,
    backend: BackendType = 'webgpu'
  ) {
    this.modelPath = modelPath;
    this.modelInputSize = modelInputSize;
    this.mean = mean;
    this.std = std;
    this.backend = backend;
  }

  protected async init(): Promise<void> {
    // Configure ONNX Runtime Web - use CDN for WASM files
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/';
    ort.env.wasm.simd = true;
    ort.env.wasm.proxy = false;

    // Load model from path/URL
    this.session = await ort.InferenceSession.create(this.modelPath, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });

    console.log(`Loaded model: ${this.modelPath}`);
  }

  protected async inference(img: Float32Array, inputSize?: [number, number]): Promise<any[]> {
    if (!this.session) {
      throw new Error('Session not initialized. Call init() first.');
    }

    const [h, w] = inputSize || this.modelInputSize;

    // Build input tensor (1, 3, H, W)
    const inputTensor = new (await import('onnxruntime-web')).Tensor('float32', img, [1, 3, h, w]);

    const feeds: Record<string, any> = {};
    feeds[this.session.inputNames[0]] = inputTensor;

    const results = await this.session.run(feeds);

    return this.session.outputNames.map((name: string) => results[name]);
  }

  abstract call(...args: unknown[]): Promise<unknown>;
}
