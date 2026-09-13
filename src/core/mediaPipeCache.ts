/**
 * MediaPipe Model Cache
 * Persists .tflite and .task MediaPipe models in IndexedDB so that
 * subsequent loads skip the network round-trip.
 */

import { createLogger } from './logger';

const log = createLogger('MediaPipeCache');

const CACHE_NAME = 'mediapipe-model-cache-v1';
const DB_NAME = 'MediaPipeModels';
const DB_VERSION = 1;
const STORE_NAME = 'models';

/**
 * Open the IndexedDB instance, creating the `models` object store on
 * first run.
 */
function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'url' });
      }
    };
  });
}

/**
 * Store a model buffer in the cache, keyed by URL.
 */
export async function cacheMediaPipeModel(url: string, data: ArrayBuffer): Promise<void> {
  try {
    const db = await openDB();
    const tx = db.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);

    await new Promise<void>((resolve, reject) => {
      const request = store.put({ url, data, timestamp: Date.now() });
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });

    db.close();
    log.log(`Cached model: ${url}`);
  } catch (error) {
    log.warn('Failed to cache model:', error);
  }
}

/**
 * Read a cached model buffer by URL, or `null` on miss.
 */
export async function getCachedMediaPipeModel(url: string): Promise<ArrayBuffer | null> {
  try {
    const db = await openDB();
    const tx = db.transaction(STORE_NAME, 'readonly');
    const store = tx.objectStore(STORE_NAME);

    const result = await new Promise<IDBValidKey | null>((resolve, reject) => {
      const request = store.get(url);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });

    db.close();

    if (result && typeof result === 'object' && 'data' in result) {
      log.log(`Cache hit: ${url}`);
      return (result as any).data as ArrayBuffer;
    }

    log.log(`Cache miss: ${url}`);
    return null;
  } catch (error) {
    log.warn('Failed to get cached model:', error);
    return null;
  }
}

/**
 * Check whether a model is cached.
 */
export async function isMediaPipeModelCached(url: string): Promise<boolean> {
  try {
    const db = await openDB();
    const tx = db.transaction(STORE_NAME, 'readonly');
    const store = tx.objectStore(STORE_NAME);

    const result = await new Promise<IDBValidKey | null>((resolve, reject) => {
      const request = store.get(url);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });

    db.close();
    return result !== null && result !== undefined;
  } catch {
    return false;
  }
}

/**
 * Drop every entry from the cache.
 */
export async function clearMediaPipeCache(): Promise<void> {
  try {
    const db = await openDB();
    const tx = db.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);

    await new Promise<void>((resolve, reject) => {
      const request = store.clear();
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });

    db.close();
    log.log('Cache cleared');
  } catch (error) {
    log.warn('Failed to clear cache:', error);
  }
}

/**
 * Return total cached bytes + a list of cached URLs.
 */
export async function getMediaPipeCacheInfo(): Promise<{ size: number; models: string[] }> {
  try {
    const db = await openDB();
    const tx = db.transaction(STORE_NAME, 'readonly');
    const store = tx.objectStore(STORE_NAME);

    const models = await new Promise<Array<{ url: string; data: ArrayBuffer }>>((resolve, reject) => {
      const request = store.getAll();
      request.onsuccess = () => resolve(request.result || []);
      request.onerror = () => reject(request.error);
    });

    db.close();

    const totalSize = models.reduce((sum, m) => sum + m.data.byteLength, 0);

    return {
      size: totalSize,
      models: models.map(m => m.url),
    };
  } catch {
    return { size: 0, models: [] };
  }
}

/**
 * Fetch a model, falling back to the IndexedDB cache on subsequent loads.
 */
export async function loadMediaPipeModelWithCache(url: string): Promise<ArrayBuffer> {
  const cached = await getCachedMediaPipeModel(url);
  if (cached) {
    return cached;
  }

  log.log(`Fetching model from network: ${url}`);
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch model: HTTP ${response.status}`);
  }

  const data = await response.arrayBuffer();

  await cacheMediaPipeModel(url, data);

  return data;
}
