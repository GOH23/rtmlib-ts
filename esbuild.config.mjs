// esbuild config for browser bundle generation.
//
// Output: dist/bundle.js (ESM, single file with all deps inlined — the
// harness at scripts/browser/*-harness.html imports from /dist/bundle.js
// via `<script type="module"> import * as lib from '/dist/bundle.js';`)
//
// Why bundle: the browser harness pages serve from this repo's static
// server. Without a bundle, the page would have to follow an import graph
// through node_modules, which Playwright can't do (no bare-specifier
// resolution, no on-the-fly TS). The single-file bundle means the harness
// page just `import`s `/dist/bundle.js` and everything works.

import { build } from 'esbuild';

await build({
  entryPoints: ['src/index.ts'],
  bundle: true,
  format: 'esm',
  target: 'es2020',
  platform: 'browser',
  outfile: 'dist/bundle.js',
  // onnxruntime-web is a WebWorker-aware package that ships its own
  // .wasm + .mjs files. Inlining the JS here is fine; the .wasm is
  // fetched at runtime from jsDelivr per `core/onnxRuntime.ts`.
  // No externals — everything is folded into one file.
  minify: false,
  sourcemap: false,
  logLevel: 'info',
});
