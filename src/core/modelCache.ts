/**
 * Model caching utility using Cache API
 * Caches ONNX models in browser to avoid repeated downloads
 */

const CACHE_NAME = 'rtmlib-ts-models-v1';

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
    console.warn(`[ModelCache] Failed to check cache for ${url}:`, error);
    return false;
  }
}

/**
 * Get model from cache or fetch from network
 * @param url - Model URL
 * @param forceRefresh - Force refresh from network
 */
export async function getCachedModel(url: string, forceRefresh: boolean = false): Promise<ArrayBuffer> {
  if (typeof caches === 'undefined') {
    // Cache API not available, fetch directly
    console.log(`[ModelCache] Cache API not available, fetching from network`);
    return fetchModelFromNetwork(url);
  }
  
  try {
    const cache = await caches.open(CACHE_NAME);
    
    // Try to get from cache first
    if (!forceRefresh) {
      const cachedResponse = await cache.match(url);
      if (cachedResponse) {
        console.log(`[ModelCache] ✅ Hit for ${url}`);
        return await cachedResponse.arrayBuffer();
      }
      console.log(`[ModelCache] ❌ Miss for ${url}, fetching from network...`);
    }
    
    // Fetch from network
    const networkResponse = await fetchModelFromNetwork(url);
    
    // Cache the response
    const responseToCache = new Response(networkResponse, {
      headers: {
        'Content-Type': 'application/octet-stream',
      },
    });
    
    await cache.put(url, responseToCache);
    console.log(`[ModelCache] 💾 Cached ${url}`);
    
    return networkResponse;
  } catch (error) {
    console.error(`[ModelCache] Failed to get/cache model ${url}:`, error);
    throw error;
  }
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
  console.log(`[ModelCache] Preloading ${urls.length} model(s)...`);
  
  const results = await Promise.allSettled(
    urls.map(url => getCachedModel(url))
  );
  
  const success = results.filter(r => r.status === 'fulfilled').length;
  const failed = results.filter(r => r.status === 'rejected').length;
  
  console.log(`[ModelCache] Preload complete: ${success} succeeded, ${failed} failed`);
  
  results.forEach((result, index) => {
    if (result.status === 'rejected') {
      console.error(`[ModelCache] Failed to preload ${urls[index]}:`, result.reason);
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
    console.log('[ModelCache] Cache cleared');
  } catch (error) {
    console.error('[ModelCache] Failed to clear cache:', error);
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
    console.warn('[ModelCache] Failed to get cache size:', error);
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
    console.warn('[ModelCache] Failed to get cache info:', error);
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
