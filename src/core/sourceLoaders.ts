/**
 * Shared source-loaders: `File` → `HTMLImageElement`,
 * `Blob` → `ImageBitmap`. Used by every detector's
 * `detectFromFile` / `detectFromBlob` entry point.
 */

export function loadImageFromFile(file: File): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(url);
      resolve(img);
    };
    img.onerror = (err) => {
      URL.revokeObjectURL(url);
      reject(err instanceof Error ? err : new Error(String(err)));
    };
    img.src = url;
  });
}

export function loadBitmapFromBlob(blob: Blob): Promise<ImageBitmap> {
  return createImageBitmap(blob);
}
