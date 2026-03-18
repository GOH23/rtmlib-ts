/**
 * BaseTool - Abstract base class for ONNX models
 */

import * as ort from 'onnxruntime-web/all';
import type { BackendType } from '../types/index.js';

export abstract class BaseTool {
  protected session: ort.InferenceSession | null = null;
  protected modelPath: string;
  protected modelInputSize: [number, number];

  constructor(modelPath: string, modelInputSize: [number, number]) {
    this.modelPath = modelPath;
    this.modelInputSize = modelInputSize;
  }

  protected async init(): Promise<void> {
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/';
    ort.env.wasm.simd = true;

    this.session = await ort.InferenceSession.create(this.modelPath, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });

    console.log(`[BaseTool] Loaded model: ${this.modelPath}`);
  }

  protected async inference(img: Float32Array, inputSize?: [number, number]): Promise<any[]> {
    if (!this.session) throw new Error('Session not initialized');

    const [h, w] = inputSize || this.modelInputSize;
    const inputTensor = new ort.Tensor('float32', img, [1, 3, h, w]);

    const feeds: Record<string, ort.Tensor> = {};
    feeds[this.session.inputNames[0]] = inputTensor;

    const results = await this.session.run(feeds);
    return this.session.outputNames.map((name) => results[name]);
  }

  abstract call(...args: unknown[]): Promise<unknown>;
}
