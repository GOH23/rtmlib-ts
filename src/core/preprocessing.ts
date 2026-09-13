/**
 * Pre-processing utilities for pose estimation
 */

export function bboxXyxy2cs(
  bbox: [number, number, number, number],
  padding: number = 1.25
): { center: [number, number]; scale: [number, number] } {
  const [x1, y1, x2, y2] = bbox;

  const center: [number, number] = [(x1 + x2) / 2, (y1 + y2) / 2];

  const w = x2 - x1;
  const h = y2 - y1;

  // Python: scale = w * padding, h * padding (different values!)
  const scale: [number, number] = [w * padding, h * padding];

  return { center, scale };
}

export function topDownAffine(
  imageSize: [number, number],
  scale: [number, number],
  center: [number, number],
  img: Uint8Array,
  imgWidth: number,
  imgHeight: number
): { resizedImg: Float32Array; scale: [number, number] } {
  const [w, h] = imageSize;

  const srcW = scale[0];
  const srcH = scale[1];

  // Calculate transformation matrix
  const scaleX = w / srcW;
  const scaleY = h / srcH;

  // Create output array
  const outputSize = w * h * 3;
  const resizedImg = new Float32Array(outputSize);

  // Simple bilinear interpolation
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      // Map output coordinates to input coordinates
      const srcX = (x / w) * srcW + (center[0] - srcW / 2);
      const srcY = (y / h) * srcH + (center[1] - srcH / 2);

      // Get the four nearest pixels
      const x0 = Math.floor(srcX);
      const y0 = Math.floor(srcY);
      const x1 = x0 + 1;
      const y1 = y0 + 1;

      const dx = srcX - x0;
      const dy = srcY - y0;

      // Sample from input image with bounds checking
      for (let c = 0; c < 3; c++) {
        const p00 = getPixel(img, imgWidth, imgHeight, x0, y0, c);
        const p10 = getPixel(img, imgWidth, imgHeight, x1, y0, c);
        const p01 = getPixel(img, imgWidth, imgHeight, x0, y1, c);
        const p11 = getPixel(img, imgWidth, imgHeight, x1, y1, c);

        // Bilinear interpolation
        const value = p00 * (1 - dx) * (1 - dy) +
                      p10 * dx * (1 - dy) +
                      p01 * (1 - dx) * dy +
                      p11 * dx * dy;

        resizedImg[y * w * 3 + x * 3 + c] = value;
      }
    }
  }

  // Return original scale (not model dimensions) for postprocess
  return { resizedImg, scale };
}

function getPixel(
  img: Uint8Array,
  width: number,
  height: number,
  x: number,
  y: number,
  channel: number
): number {
  if (x < 0 || x >= width || y < 0 || y >= height) {
    return 0;
  }
  return img[y * width * 3 + x * 3 + channel];
}

export function normalizeImage(
  img: Float32Array,
  mean: number[],
  std: number[]
): Float32Array {
  const normalized = new Float32Array(img.length);

  for (let i = 0; i < img.length; i++) {
    const channel = i % 3;
    normalized[i] = (img[i] - mean[channel]) / std[channel];
  }

  return normalized;
}

export function transposeImage(
  img: Float32Array,
  height: number,
  width: number
): Float32Array {
  // HWC to CHW
  const transposed = new Float32Array(img.length);

  for (let c = 0; c < 3; c++) {
    for (let h = 0; h < height; h++) {
      for (let w = 0; w < width; w++) {
        transposed[c * height * width + h * width + w] = img[h * width * 3 + w * 3 + c];
      }
    }
  }

  return transposed;
}

/** Inverse-transform metadata returned by `letterboxCHW` to map boxes back. */
export interface Letterbox {
  paddingX: number;
  paddingY: number;
  scaleX: number;
  scaleY: number;
}

/** Letterbox a HWC uint8 image into a CHW float32 tensor in [0, 1]. */
export function letterboxCHW(
  img: Uint8Array,
  imgWidth: number,
  imgHeight: number,
  inputW: number,
  inputH: number,
): { tensor: Float32Array; meta: Letterbox } {
  const padded = new Uint8Array(inputH * inputW * 3);
  const aspectRatio = imgWidth / imgHeight;
  const targetAspect = inputW / inputH;

  let drawWidth: number;
  let drawHeight: number;
  let paddingX: number;
  let paddingY: number;
  if (aspectRatio > targetAspect) {
    drawWidth = inputW;
    drawHeight = Math.floor(inputW / aspectRatio);
    paddingX = 0;
    paddingY = (inputH - drawHeight) / 2;
  } else {
    drawHeight = inputH;
    drawWidth = Math.floor(inputH * aspectRatio);
    paddingX = (inputW - drawWidth) / 2;
    paddingY = 0;
  }
  const scaleX = imgWidth / drawWidth;
  const scaleY = imgHeight / drawHeight;

  for (let y = 0; y < drawHeight; y++) {
    for (let x = 0; x < drawWidth; x++) {
      const srcX = Math.floor(x * scaleX);
      const srcY = Math.floor(y * scaleY);
      const dstX = Math.floor(x + paddingX);
      const dstY = Math.floor(y + paddingY);
      for (let c = 0; c < 3; c++) {
        padded[(dstY * inputW + dstX) * 3 + c] = img[(srcY * imgWidth + srcX) * 3 + c];
      }
    }
  }

  const plane = inputH * inputW;
  const tensor = new Float32Array(3 * plane);
  const inv255 = 1 / 255;
  for (let h = 0, i = 0; h < inputH; h++) {
    for (let w = 0; w < inputW; w++, i++) {
      tensor[i] = padded[i * 3] * inv255;
      tensor[plane + i] = padded[i * 3 + 1] * inv255;
      tensor[2 * plane + i] = padded[i * 3 + 2] * inv255;
    }
  }

  return {
    tensor,
    meta: { paddingX, paddingY, scaleX, scaleY },
  };
}

/** ImageNet normalization constants used by RTMPose-family pose models. */
export const IMAGENET_MEAN: readonly [number, number, number] = [123.675, 116.28, 103.53];
export const IMAGENET_STD_INV: readonly [number, number, number] = [
  1 / 58.395,
  1 / 57.12,
  1 / 57.375,
];

/** Affine-crop center + source-space scale for a 1.25×-padded bbox. */
export function affineCropCenterScale(
  bbox: { x1: number; y1: number; x2: number; y2: number },
  inputW: number,
  inputH: number,
  padding = 1.25,
): { center: [number, number]; scale: [number, number] } {
  const bw = bbox.x2 - bbox.x1;
  const bh = bbox.y2 - bbox.y1;
  const center: [number, number] = [bbox.x1 + bw / 2, bbox.y1 + bh / 2];

  let scaleW = bw * padding;
  let scaleH = bh * padding;
  const modelAR = inputW / inputH;
  if (scaleW / scaleH > modelAR) scaleH = scaleW / modelAR;
  else scaleW = scaleH * modelAR;

  return { center, scale: [scaleW, scaleH] };
}

/**
 * Affine-crop `bbox` from `srcCanvas` into a NCHW ImageNet-normalized
 * tensor written into `destBuffer`. `destCanvas` must be pre-sized to
 * `[inputW, inputH]` and is `clearRect()`d before drawing.
 */
export function affineCropImageNet(
  srcCanvas: HTMLCanvasElement,
  bbox: { x1: number; y1: number; x2: number; y2: number },
  inputW: number,
  inputH: number,
  destCanvas: HTMLCanvasElement,
  destBuffer: Float32Array,
  padding = 1.25,
  mean: readonly [number, number, number] = IMAGENET_MEAN,
  stdInv: readonly [number, number, number] = IMAGENET_STD_INV,
): { tensor: Float32Array; center: [number, number]; scale: [number, number] } {
  const { center, scale } = affineCropCenterScale(bbox, inputW, inputH, padding);
  const [scaleW, scaleH] = scale;

  const ctx = destCanvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('Could not get 2D context from destination canvas');
  ctx.clearRect(0, 0, inputW, inputH);
  ctx.drawImage(
    srcCanvas,
    center[0] - scaleW / 2, center[1] - scaleH / 2, scaleW, scaleH,
    0, 0, inputW, inputH,
  );

  const cropped = ctx.getImageData(0, 0, inputW, inputH);
  const plane = inputW * inputH;
  const data = cropped.data;
  const [mean0, mean1, mean2] = mean;
  const [stdInv0, stdInv1, stdInv2] = stdInv;

  for (let i = 0; i < data.length; i += 16) {
    const p1 = i >> 2;
    const p2 = p1 + 1;
    const p3 = p1 + 2;
    const p4 = p1 + 3;

    destBuffer[p1] = (data[i] - mean0) * stdInv0;
    destBuffer[p2] = (data[i + 4] - mean0) * stdInv0;
    destBuffer[p3] = (data[i + 8] - mean0) * stdInv0;
    destBuffer[p4] = (data[i + 12] - mean0) * stdInv0;

    destBuffer[p1 + plane] = (data[i + 1] - mean1) * stdInv1;
    destBuffer[p2 + plane] = (data[i + 5] - mean1) * stdInv1;
    destBuffer[p3 + plane] = (data[i + 9] - mean1) * stdInv1;
    destBuffer[p4 + plane] = (data[i + 13] - mean1) * stdInv1;

    destBuffer[p1 + 2 * plane] = (data[i + 2] - mean2) * stdInv2;
    destBuffer[p2 + 2 * plane] = (data[i + 6] - mean2) * stdInv2;
    destBuffer[p3 + 2 * plane] = (data[i + 10] - mean2) * stdInv2;
    destBuffer[p4 + 2 * plane] = (data[i + 14] - mean2) * stdInv2;
  }

  return { tensor: destBuffer, center, scale };
}

/**
 * Draw a drawable source into a canvas and extract the RGBA buffer.
 * `targetCanvas` is reused when supplied; otherwise a fresh canvas is
 * allocated.
 */
export function drawSourceToRgba(
  source: HTMLVideoElement | HTMLImageElement | ImageBitmap,
  width: number,
  height: number,
  targetCanvas?: HTMLCanvasElement,
): { rgba: Uint8Array; width: number; height: number } {
  const canvas = targetCanvas ?? document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Could not get 2D context from canvas');
  ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  return {
    rgba: new Uint8Array(imageData.data.buffer),
    width: canvas.width,
    height: canvas.height,
  };
}

/**
 * Letterbox geometry: pure math for an aspect-ratio-preserving fit.
 * Returns the draw rectangle + inverse scale used to map back to
 * source coordinates. Used when callers already have a drawable
 * source canvas (so `letterboxToCanvas`'s RGBA staging step would be
 * wasted work).
 */
export function letterboxGeometry(
  srcW: number,
  srcH: number,
  dstW: number,
  dstH: number,
): {
  drawW: number;
  drawH: number;
  offX: number;
  offY: number;
  scaleX: number;
  scaleY: number;
} {
  const ar = srcW / srcH;
  const tar = dstW / dstH;
  let drawW: number;
  let drawH: number;
  let offX: number;
  let offY: number;
  if (ar > tar) {
    drawW = dstW;
    drawH = Math.floor(dstW / ar);
    offX = 0;
    offY = Math.floor((dstH - drawH) / 2);
  } else {
    drawH = dstH;
    drawW = Math.floor(dstH * ar);
    offX = Math.floor((dstW - drawW) / 2);
    offY = 0;
  }
  return { drawW, drawH, offX, offY, scaleX: srcW / drawW, scaleY: srcH / drawH };
}

/** Fill a target canvas with the letterbox background color. */
export function fillLetterbox(
  targetCtx: CanvasRenderingContext2D,
  dstW: number,
  dstH: number,
  color = '#000000',
): void {
  targetCtx.fillStyle = color;
  targetCtx.fillRect(0, 0, dstW, dstH);
}

/** Pack RGBA pixels (from a canvas's `getImageData`) into an NCHW
 * `Float32Array` of shape `[3, dstH, dstW]`, scaled to [0, 1]. */
export function rgbaToCHW(
  rgba: Uint8ClampedArray,
  dstW: number,
  dstH: number,
  out: Float32Array,
): void {
  const plane = dstW * dstH;
  for (let i = 0; i < rgba.length; i += 4) {
    const p = i >> 2;
    out[p] = rgba[i] / 255;
    out[p + plane] = rgba[i + 1] / 255;
    out[p + 2 * plane] = rgba[i + 2] / 255;
  }
}

/**
 * Letterbox an RGBA image into a target canvas at the given input
 * dimensions. The source is preserved aspect-ratio, scaled to fit, and
 * padded with black on the short axis. Returns the inverse-transform
 * meta used to map model-output coordinates back to source space.
 *
 * Used by every ONNX-based detector (YOLO/RTMW/RTMW3D). The detector
 * classes pre-allocate `targetCtx` + `targetCanvas` to amortise the
 * canvas creation across calls — this helper is just the geometry +
 * drawImage step.
 */
export function letterboxToCanvas(
  rgba: Uint8Array,
  imgW: number,
  imgH: number,
  inputW: number,
  inputH: number,
  targetCtx: CanvasRenderingContext2D,
): { paddingX: number; paddingY: number; scaleX: number; scaleY: number } {
  const geom = letterboxGeometry(imgW, imgH, inputW, inputH);

  // Stage the source RGBA on a scratch canvas, then let the browser do
  // the scaled drawImage — hardware-accelerated, much cheaper than the
  // manual pixel loop in `letterboxCHW`.
  const src = document.createElement('canvas');
  src.width = imgW;
  src.height = imgH;
  const srcCtx = src.getContext('2d')!;
  const img = srcCtx.createImageData(imgW, imgH);
  img.data.set(rgba);
  srcCtx.putImageData(img, 0, 0);

  fillLetterbox(targetCtx, inputW, inputH);
  targetCtx.drawImage(src, 0, 0, imgW, imgH, geom.offX, geom.offY, geom.drawW, geom.drawH);

  return {
    paddingX: geom.offX,
    paddingY: geom.offY,
    scaleX: geom.scaleX,
    scaleY: geom.scaleY,
  };
}
