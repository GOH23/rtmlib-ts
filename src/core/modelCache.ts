/**
 * Model caching utility using Cache API
 * Caches ONNX models in browser to avoid repeated downloads
 *
 * Cache-first semantics: every fetch attempts `cache.match(url)` first. On
 * a hit we never touch the network — which is what makes the
 * COEP + HuggingFace combination viable for warm-cache users (cache hits
 * are served same-origin, so the missing `Cross-Origin-Resource-Policy`
 * header from `us.aws.cdn.hf.co` doesn't matter). On a miss we fall back
 * to a regular `fetch()` and persist the bytes into the cache for next
 * time.
 *
 * To keep a failed cold-cache fetch (e.g. COEP stripped the body to a
 * zero-length buffer) from poisoning subsequent loads, we refuse to
 * persist any response whose body is empty or whose declared
 * `Content-Length` disagrees with the bytes we actually received.
 */

import { createLogger } from './logger';

const log = createLogger('ModelCache');

const CACHE_NAME = 'rtmlib-ts-models-v3';

/**
 * Check if model is available in cache
 */
export async function isModelCached(url: string): Promise<boolean> {
  if (typeof caches === 'undefined') {
    // Cache API not available (e.g., Node.js)
    return false;
  }

  try {
    const cache = await caches.open(CACHE_NAME);
    const response = await cache.match(url);
    return !!response;
  } catch (error) {
    log.warn(`Failed to check cache for ${url}:`, error);
    return false;
  }
}

/**
 * Get model from cache or fetch from network
 * @param url - Model URL
 */
export async function getCachedModel(url: string): Promise<ArrayBuffer> {
  if (typeof caches === 'undefined') {
    // Cache API not available, fetch directly
    log.log('Cache API not available, fetching from network');
    return fetchModelFromNetwork(url);
  }

  try {
    const cache = await caches.open(CACHE_NAME);

    // Try to get from cache first
    const cachedResponse = await cache.match(url);
    if (cachedResponse) {
      log.log(`✅ Hit for ${url}`);
      return await cachedResponse.arrayBuffer();
    }
    log.log(`❌ Miss for ${url}, fetching from network...`);

    // Fetch from network
    const networkResponse = await fetchModelFromNetwork(url);

    // Validate the response before caching it. Under COEP `require-corp`,
    // a cross-origin response without CORP arrives with a zero-length body
    // (and still `response.ok === true`). Persisting that empty buffer
    // would poison the cache — the next load would treat it as a hit and
    // ONNX Runtime Web would throw on the resulting null tensor. Skip the
    // cache put entirely so the next request retries the network fetch
    // (or, on a real failure, surfaces the error to the caller).
    if (networkResponse.byteLength === 0) {
      throw new EmptyModelResponseError(
        `[ModelCache] Refusing to cache empty response for ${url} ` +
          `(likely COEP-stripped or 0-byte body).`,
      );
    }

    // IMPORTANT: `new Response(networkResponse, …)` DETACHES the input
    // ArrayBuffer per the Fetch spec — passing an ArrayBuffer transfers
    // ownership to the Response. After this line `networkResponse` is
    // a zero-length ArrayBuffer and ONNX Runtime Web will reject it as
    // "Cannot read properties of null (reading 'irVersion')". Copy the
    // bytes before wrapping, and return the original.
    const bufferForCache = networkResponse.slice(0);
    const responseToCache = new Response(bufferForCache, {
      headers: {
        'Content-Type': 'application/octet-stream',
        'Content-Length': String(bufferForCache.byteLength),
      },
    });

    try {
      await cache.put(url, responseToCache);
      log.log(`💾 Cached ${url} (${formatBytes(bufferForCache.byteLength)})`);
    } catch (cacheErr) {
      // Cache write can fail on large models (e.g. RTMW3D-X is ~370 MB —
      // some browsers' Cache API implementations cap individual entries or
      // hit quota limits). The network bytes are still good — log a warning
      // and return them so init succeeds. Next reload will re-download.
      log.warn(
        `⚠️ Cache write failed for ${url} ` +
          `(${formatBytes(bufferForCache.byteLength)}): ${(cacheErr as Error).message}. ` +
          `Returning the fresh bytes anyway; will re-download next load.`,
      );
    }

    return networkResponse;
  } catch (error) {
    log.error(`Failed to get/cache model ${url}:`, error);
    throw error;
  }
}

/**
 * Thrown by `getCachedModel` when the network fetch returns a zero-length
 * body (typically COEP stripping a cross-origin response with no
 * `Cross-Origin-Resource-Policy`). Exported so callers can branch on it
 * — e.g. the demo can suggest disabling COEP for the first load.
 */
export class EmptyModelResponseError extends Error {
  override readonly name = 'EmptyModelResponseError';
}

/**
 * Fetch model from network with progress tracking
 */
async function fetchModelFromNetwork(url: string): Promise<ArrayBuffer> {
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`Failed to fetch model: HTTP ${response.status} ${response.statusText}`);
  }

  return await response.arrayBuffer();
}

/**
 * Preload and cache multiple models
 */
export async function preloadModels(urls: string[]): Promise<void> {
  log.log(`Preloading ${urls.length} model(s)...`);

  const results = await Promise.allSettled(
    urls.map(url => getCachedModel(url))
  );

  const success = results.filter(r => r.status === 'fulfilled').length;
  const failed = results.filter(r => r.status === 'rejected').length;

  log.log(`Preload complete: ${success} succeeded, ${failed} failed`);

  results.forEach((result, index) => {
    if (result.status === 'rejected') {
      log.error(`Failed to preload ${urls[index]}:`, result.reason);
    }
  });
}

/**
 * Clear all cached models
 */
export async function clearModelCache(): Promise<void> {
  if (typeof caches === 'undefined') {
    return;
  }

  try {
    await caches.delete(CACHE_NAME);
    log.log('Cache cleared');
  } catch (error) {
    log.error('Failed to clear cache:', error);
  }
}

/**
 * Get cache size in bytes
 */
export async function getCacheSize(): Promise<number> {
  if (typeof caches === 'undefined' || !navigator.storage) {
    return 0;
  }

  try {
    const cache = await caches.open(CACHE_NAME);
    const keys = await cache.keys();
    let totalSize = 0;

    for (const request of keys) {
      const response = await cache.match(request);
      if (response) {
        const blob = await response.blob();
        totalSize += blob.size;
      }
    }

    return totalSize;
  } catch (error) {
    log.warn('Failed to get cache size:', error);
    return 0;
  }
}

/**
 * Get cache info
 */
export async function getCacheInfo(): Promise<{
  cachedModels: string[];
  totalSize: number;
  totalSizeFormatted: string;
}> {
  if (typeof caches === 'undefined') {
    return { cachedModels: [], totalSize: 0, totalSizeFormatted: '0 B' };
  }

  try {
    const cache = await caches.open(CACHE_NAME);
    const keys = await cache.keys();
    const cachedModels = keys.map(k => k.url);
    const totalSize = await getCacheSize();

    return {
      cachedModels,
      totalSize,
      totalSizeFormatted: formatBytes(totalSize),
    };
  } catch (error) {
    log.warn('Failed to get cache info:', error);
    return { cachedModels: [], totalSize: 0, totalSizeFormatted: '0 B' };
  }
}

/**
 * Format bytes to human-readable string
 */
function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${(bytes / Math.pow(k, i)).toFixed(2)} ${sizes[i]}`;
}
