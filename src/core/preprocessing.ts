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
