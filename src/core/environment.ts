/**
 * Environment detection utilities
 * Helps determine if code is running in browser, SSR, or other environments
 */

/**
 * Check if running in a browser environment
 * Returns false during SSR (Node.js)
 */
export function isBrowser(): boolean {
  return typeof window !== 'undefined' && typeof document !== 'undefined';
}

/**
 * Check if running in server-side rendering environment
 */
export function isSSR(): boolean {
  return !isBrowser();
}

/**
 * Safely access browser-only APIs that may not exist during SSR
 * Returns null if not in browser environment
 */
export function getDocument(): typeof document | null {
  return isBrowser() ? document : null;
}

/**
 * Get canvas element safely (returns null during SSR)
 */
export function createCanvas(): HTMLCanvasElement | null {
  if (isSSR()) return null;
  return document.createElement('canvas');
}
