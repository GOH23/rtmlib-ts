/**
 * Tiny prefix logger. Returns `{ log, warn, error, debug }` bound to a
 * `[Prefix]` namespace so every detector / cache module can emit
 * consistent `[Name] msg` lines without repeating the prefix in every
 * `console.log(\`[Foo] ...\`)` call.
 *
 * Use `setDebug(false)` to silence `log`/`debug` output globally (e.g.
 * from a test runner or production build). `warn` and `error` always
 * pass through — they signal actionable conditions the user must see.
 *
 * `createLogger` is the single place to add structured logging, log
 * levels, or a transport swap (e.g. to a debug drawer in the demo).
 */

export interface Logger {
  log(...args: unknown[]): void;
  warn(...args: unknown[]): void;
  error(...args: unknown[]): void;
  debug(...args: unknown[]): void;
}

let debugEnabled = true;

export function setDebug(enabled: boolean): void {
  debugEnabled = enabled;
}

export function isDebug(): boolean {
  return debugEnabled;
}

export function createLogger(prefix: string): Logger {
  const tag = `[${prefix}]`;
  return {
    log: (...args) => {
      if (debugEnabled) console.log(tag, ...args);
    },
    warn: (...args) => console.warn(tag, ...args),
    error: (...args) => console.error(tag, ...args),
    debug: (...args) => {
      if (debugEnabled) console.debug(tag, ...args);
    },
  };
}
