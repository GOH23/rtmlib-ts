/**
 * MediaPipe Model Cache
 * Кэширование .tflite и .task моделей MediaPipe в IndexedDB
 */

const CACHE_NAME = 'mediapipe-model-cache-v1';
const DB_NAME = 'MediaPipeModels';
const DB_VERSION = 1;
const STORE_NAME = 'models';

/**
 * Открыть IndexedDB
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
 * Сохранить модель в кэш
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
    console.log(`[MediaPipeCache] Cached model: ${url}`);
  } catch (error) {
    console.warn('[MediaPipeCache] Failed to cache model:', error);
  }
}

/**
 * Получить модель из кэша
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
      console.log(`[MediaPipeCache] Cache hit: ${url}`);
      return (result as any).data as ArrayBuffer;
    }
    
    console.log(`[MediaPipeCache] Cache miss: ${url}`);
    return null;
  } catch (error) {
    console.warn('[MediaPipeCache] Failed to get cached model:', error);
    return null;
  }
}

/**
 * Проверить, закэширована ли модель
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
 * Очистить кэш моделей
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
    console.log('[MediaPipeCache] Cache cleared');
  } catch (error) {
    console.warn('[MediaPipeCache] Failed to clear cache:', error);
  }
}

/**
 * Получить информацию о кэше
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
 * Загрузить модель с кешированием
 */
export async function loadMediaPipeModelWithCache(url: string): Promise<ArrayBuffer> {
  // Проверить кэш
  const cached = await getCachedMediaPipeModel(url);
  if (cached) {
    return cached;
  }
  
  // Загрузить из сети
  console.log(`[MediaPipeCache] Fetching model from network: ${url}`);
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch model: HTTP ${response.status}`);
  }
  
  const data = await response.arrayBuffer();
  
  // Сохранить в кэш
  await cacheMediaPipeModel(url, data);
  
  return data;
}
