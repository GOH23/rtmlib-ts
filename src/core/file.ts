/**
 * File utilities for downloading and loading models
 */

import * as fs from 'fs';
import * as path from 'path';
import * as https from 'https';
import JSZip from 'jszip';

const CACHE_DIR = path.join(process.env.HOME || process.env.USERPROFILE || '.', '.rtmlib', 'models');

export async function downloadCheckpoint(url: string, localPath?: string): Promise<string> {
  // If local path provided, use it directly
  if (localPath && fs.existsSync(localPath)) {
    console.log(`Using local model: ${localPath}`);
    return localPath;
  }
  
  const fileName = path.basename(url);
  const cachePath = path.join(CACHE_DIR, fileName.replace('.zip', '.onnx'));
  
  if (fs.existsSync(cachePath)) {
    console.log(`Using cached model: ${cachePath}`);
    return cachePath;
  }
  
  console.log(`Downloading model from ${url}`);
  
  if (!fs.existsSync(CACHE_DIR)) {
    fs.mkdirSync(CACHE_DIR, { recursive: true });
  }
  
  const tempPath = path.join(CACHE_DIR, fileName);
  
  await downloadFile(url, tempPath);
  
  if (fileName.endsWith('.zip')) {
    await extractZip(tempPath, CACHE_DIR);
    fs.unlinkSync(tempPath);
  }
  
  return cachePath;
}

async function downloadFile(url: string, dest: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(dest);
    
    const download = (url: string) => {
      https.get(url, (response) => {
        if (response.statusCode === 302 || response.statusCode === 301) {
          download(response.headers.location!);
          return;
        }
        
        response.pipe(file);
        file.on('finish', () => {
          file.close();
          resolve();
        });
      }).on('error', reject);
    };
    
    download(url);
  });
}

async function extractZip(zipPath: string, dest: string): Promise<void> {
  const data = fs.readFileSync(zipPath);
  const zip = await JSZip.loadAsync(data);
  
  // Find .onnx file in zip
  for (const [filename, file] of Object.entries(zip.files)) {
    if (filename.endsWith('.onnx')) {
      const content = await file.async('nodebuffer');
      const onnxPath = path.join(dest, filename);
      
      // Create directory if needed
      const dir = path.dirname(onnxPath);
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
      
      fs.writeFileSync(onnxPath, content);
      console.log(`Extracted: ${filename}`);
      return;
    }
  }
  
  throw new Error('No .onnx file found in zip');
}

export function fileExists(filePath: string): boolean {
  return fs.existsSync(filePath);
}

export function resolveModelPath(modelPath: string): string {
  if (modelPath.startsWith('http://') || modelPath.startsWith('https://')) {
    return modelPath;
  }
  return path.resolve(modelPath);
}

/**
 * Extract local zip file and return onnx path
 */
export async function extractLocalZip(zipPath: string): Promise<string> {
  if (!fs.existsSync(zipPath)) {
    throw new Error(`Zip file not found: ${zipPath}`);
  }
  
  const destDir = path.dirname(zipPath);
  const data = fs.readFileSync(zipPath);
  const zip = await JSZip.loadAsync(data);
  
  // Find .onnx file in zip
  for (const [filename, file] of Object.entries(zip.files)) {
    if (filename.endsWith('.onnx')) {
      const onnxPath = path.join(destDir, filename);
      
      if (fs.existsSync(onnxPath)) {
        console.log(`Using extracted model: ${onnxPath}`);
        return onnxPath;
      }
      
      const content = await file.async('nodebuffer');
      
      // Create directory if needed
      const dir = path.dirname(onnxPath);
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
      
      fs.writeFileSync(onnxPath, content);
      console.log(`Extracted: ${filename} -> ${onnxPath}`);
      return onnxPath;
    }
  }
  
  throw new Error('No .onnx file found in zip');
}
