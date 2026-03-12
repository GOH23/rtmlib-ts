'use client';

import { useEffect, useState, useRef } from 'react';
import {
  ObjectDetector,
  PoseDetector,
  Pose3DDetector,
  AnimalDetector,
  VITPOSE_MODELS,
  ANIMAL_CLASSES
} from '../../dist/index.js';
import {
  clearModelCache,
  drawDetectionsOnCanvas,
  drawPoseOnCanvas,
  getCacheInfo,
} from '../../dist/index.js';

// Types
interface Detection {
  bbox: { x1: number; y1: number; x2: number; y2: number; confidence: number };
  className?: string;
  keypoints?: any[];
  keypoints3d?: number[][];
  classId?: number;
}

interface ModelStatus {
  loaded: boolean;
  loading: boolean;
  error?: string;
}

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  
  const [objectDetector, setObjectDetector] = useState<any>(null);
  const [poseDetector, setPoseDetector] = useState<any>(null);
  const [pose3DDetector, setPose3DDetector] = useState<any>(null);
  const [animalDetector, setAnimalDetector] = useState<any>(null);
  
  const [mode, setMode] = useState<'object' | 'pose' | 'pose3d' | 'animal'>('object');
  const [perfMode, setPerfMode] = useState<'performance' | 'balanced' | 'lightweight'>('balanced');
  const [backend, setBackend] = useState<'wasm' | 'webgpu'>(() => {
    // Check if WebGPU is available
    if (typeof navigator !== 'undefined' && (navigator as any).gpu) {
      return 'webgpu';
    }
    // Fallback to WASM if WebGPU is not available
    return 'wasm';
  });
  const [animalPoseModel, setAnimalPoseModel] = useState<'vitpose-s' | 'vitpose-b' | 'vitpose-l'>('vitpose-b');
  const [selectedClasses, setSelectedClasses] = useState<string[]>(['person']);
  const [selectedAnimalClasses, setSelectedAnimalClasses] = useState<string[]>([]);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [stats, setStats] = useState<{ time: number; count: number; detTime?: number; poseTime?: number } | null>(null);
  const [useCamera, setUseCamera] = useState(false);
  const [hasImage, setHasImage] = useState(false);
  const [videoSrc, setVideoSrc] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [detectionInterval, setDetectionInterval] = useState<NodeJS.Timeout | null>(null);
  const [cacheInfo, setCacheInfo] = useState<{ size: string; cached: number } | null>(null);
  const [showDocs, setShowDocs] = useState(false);
  
  // Performance optimization: process every Nth frame
  const [processEveryNFrames, setProcessEveryNFrames] = useState(3);
  const frameCountRef = useRef(0);

  const [modelStatus, setModelStatus] = useState<Record<string, ModelStatus>>({
    object: { loaded: false, loading: false },
    pose: { loaded: false, loading: false },
    pose3d: { loaded: false, loading: false },
    animal: { loaded: false, loading: false },
  });

  

  useEffect(() => {
    async function loadCacheInfo() {
      const cacheInfo = await getCacheInfo();
      setCacheInfo({
        size: cacheInfo.totalSizeFormatted,
        cached: cacheInfo.cachedModels.length,
      });
    }
    loadCacheInfo();
  }, []);

  // Fallback detector initialization
  const initDetectorWithFallback = async (
    mode: 'object' | 'pose' | 'pose3d' | 'animal',
    perfMode: 'performance' | 'balanced' | 'lightweight',
    fallbackBackend: 'wasm'
  ) => {
    try {
      let detector: any;
      const startTime = performance.now();

      if (mode === 'object') {
        detector = new ObjectDetector({
          classes: selectedClasses,
          mode: perfMode,
          backend: fallbackBackend,
          confidence: 0.3,
          cache: true,
        });
      } else if (mode === 'pose') {
        detector = new PoseDetector({
          detConfidence: 0.5,
          poseConfidence: 0.3,
          backend: fallbackBackend,
          cache: true,
        });
      } else if (mode === 'pose3d') {
        detector = new Pose3DDetector({
          detModel: "./end2end.onnx",
          detConfidence: 0.45,
          poseConfidence: 0.3,
          backend: fallbackBackend,
          cache: true,
        });
      } else if (mode === 'animal') {
        detector = new AnimalDetector({
          classes: selectedAnimalClasses.length > 0 ? selectedAnimalClasses : null,
          poseModelType: animalPoseModel,
          detConfidence: 0.5,
          poseConfidence: 0.3,
          backend: fallbackBackend,
          cache: true,
        });
      }

      await detector.init();
      const loadTime = Math.round(performance.now() - startTime);

      if (mode === 'object') setObjectDetector(detector);
      else if (mode === 'pose') setPoseDetector(detector);
      else if (mode === 'pose3d') setPose3DDetector(detector);
      else if (mode === 'animal') setAnimalDetector(detector);

      setModelStatus(prev => ({
        ...prev,
        [mode]: { loaded: true, loading: false }
      }));

      console.log(`${mode} detector initialized with WASM fallback in ${loadTime}ms`);
    } catch (error) {
      console.error(`Failed to load ${mode} detector with WASM fallback:`, error);
      setModelStatus(prev => ({
        ...prev,
        [mode]: { loaded: false, loading: false, error: (error as Error).message }
      }));
    }
  };

  useEffect(() => {
    async function initDetector() {
      const currentModel = modelStatus[mode];
      if (currentModel.loaded || currentModel.loading) return;

      setModelStatus(prev => ({ ...prev, [mode]: { loaded: false, loading: true } }));

      try {
        let detector: any;
        const startTime = performance.now();

        if (mode === 'object') {
          detector = new ObjectDetector({
            classes: selectedClasses,
            mode: perfMode,
            backend: backend,
            confidence: 0.3,
            cache: true,
          });
        } else if (mode === 'pose') {
          detector = new PoseDetector({
            detConfidence: 0.5,
            poseConfidence: 0.3,
            backend: backend,
            cache: true,
          });
        } else if (mode === 'pose3d') {
          detector = new Pose3DDetector({
            
            detConfidence: 0.1,
            poseConfidence: 0.3,
            backend: backend,
            cache: true,  // Disable cache for large 3D model (352MB)
          });
        } else if (mode === 'animal') {
          detector = new AnimalDetector({
            classes: selectedAnimalClasses.length > 0 ? selectedAnimalClasses : null,
            poseModelType: animalPoseModel,
            detConfidence: 0.5,
            poseConfidence: 0.3,
            backend: backend,
            cache: true,  // Disable cache for large ViTPose models
          });
        }

        await detector.init();
        const loadTime = Math.round(performance.now() - startTime);

        if (mode === 'object') setObjectDetector(detector);
        else if (mode === 'pose') setPoseDetector(detector);
        else if (mode === 'pose3d') setPose3DDetector(detector);
        else if (mode === 'animal') setAnimalDetector(detector);

        setModelStatus(prev => ({
          ...prev,
          [mode]: { loaded: true, loading: false }
        }));

        console.log(`${mode} detector initialized in ${loadTime}ms`);
      } catch (error) {
        // If WebGPU fails, try fallback to WASM
        const errorMsg = (error as Error).message;
        if (backend === 'webgpu' && (errorMsg.includes('WebGPU') || errorMsg.includes('not supported') || errorMsg.includes('session'))) {
          console.warn('WebGPU not available, falling back to WASM...');
          setBackend('wasm');
          
          // Retry with WASM
          await initDetectorWithFallback(mode, perfMode, 'wasm');
          return;
        }
        
        console.error(`Failed to load ${mode} detector:`, error);
        setModelStatus(prev => ({
          ...prev,
          [mode]: { loaded: false, loading: false, error: errorMsg }
        }));
      }
    }

    initDetector();
  }, [mode, perfMode, backend, animalPoseModel]);

  useEffect(() => {
    if (objectDetector) {
      objectDetector.setClasses(selectedClasses.length > 0 ? selectedClasses : null);
    }
  }, [selectedClasses, objectDetector]);

  useEffect(() => {
    if (animalDetector) {
      animalDetector.setClasses(selectedAnimalClasses.length > 0 ? selectedAnimalClasses : null);
    }
  }, [selectedAnimalClasses, animalDetector]);

  useEffect(() => {
    async function setupCamera() {
      if (useCamera && videoRef.current) {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480 },
          });
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        } catch (err) {
          console.error('Camera error:', err);
          setUseCamera(false);
        }
      } else if (videoRef.current?.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
        videoRef.current.srcObject = null;
      }
    }

    setupCamera();
  }, [useCamera]);

  const processDetection = async () => {
    if (!canvasRef.current || !modelStatus[mode].loaded) return;

    // Performance optimization: skip frames for video/camera
    if (useCamera || videoSrc) {
      frameCountRef.current++;
      if (frameCountRef.current % processEveryNFrames !== 0) {
        return;  // Skip this frame
      }
    }

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;

    let results: any[] = [];
    const startTime = performance.now();

    if (mode === 'object' && objectDetector) {
      if (useCamera && videoRef.current) {
        ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
        results = await objectDetector.detectFromVideo(videoRef.current, canvas);
        drawDetectionsOnCanvas(ctx, results);
      } else if (videoSrc && videoRef.current) {
        ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
        results = await objectDetector.detectFromVideo(videoRef.current, canvas);
        drawDetectionsOnCanvas(ctx, results);
      } else {
        results = await objectDetector.detectFromCanvas(canvas);
        drawDetectionsOnCanvas(ctx, results);
      }
    } else if (mode === 'pose' && poseDetector) {
      if (useCamera && videoRef.current) {
        ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
        results = await poseDetector.detectFromVideo(videoRef.current, canvas);
        drawPoseOnCanvas(ctx, results);
      } else if (videoSrc && videoRef.current) {
        ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
        results = await poseDetector.detectFromVideo(videoRef.current, canvas);
        drawPoseOnCanvas(ctx, results);
      } else {
        results = await poseDetector.detectFromCanvas(canvas);
        drawPoseOnCanvas(ctx, results);
      }
    } else if (mode === 'pose3d' && pose3DDetector) {
      if (useCamera && videoRef.current) {
        ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
        const result3d = await pose3DDetector.detectFromVideo(videoRef.current, canvas);
        results = process3DResult(result3d);
        drawPoseOnCanvas(ctx, results);
      } else if (videoSrc && videoRef.current) {
        ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
        const result3d = await pose3DDetector.detectFromVideo(videoRef.current, canvas);
        results = process3DResult(result3d);
        drawPoseOnCanvas(ctx, results);
      } else {
        const result3d = await pose3DDetector.detectFromCanvas(canvas);
        results = process3DResult(result3d);
        drawPoseOnCanvas(ctx, results);
      }
    } else if (mode === 'animal' && animalDetector) {
      if (useCamera && videoRef.current) {
        ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
        results = await animalDetector.detectFromVideo(videoRef.current, canvas);
        drawAnimalResults(ctx, results);
      } else if (videoSrc && videoRef.current) {
        ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
        results = await animalDetector.detectFromVideo(videoRef.current, canvas);
        drawAnimalResults(ctx, results);
      } else {
        results = await animalDetector.detectFromCanvas(canvas);
        drawAnimalResults(ctx, results);
      }
    }

    const endTime = performance.now();
    const detStats = (results as any).stats;

    setDetections(results);
    setStats({
      time: Math.round(endTime - startTime),
      count: results.length,
      detTime: detStats?.detTime,
      poseTime: detStats?.poseTime,
    });
  };

  const process3DResult = (result3d: any): Detection[] => {
    const keypoints = result3d.keypoints || [];
    const keypoints2d = result3d.keypoints2d || [];
    const scores = result3d.scores || [];
    const stats = result3d.stats;

    const detections: Detection[] = [];

    for (let i = 0; i < keypoints.length; i++) {
      const personKeypoints = keypoints[i];
      const personKeypoints2d = keypoints2d[i];
      const personScores = scores[i];

      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      for (const kpt of personKeypoints2d) {
        minX = Math.min(minX, kpt[0]);
        minY = Math.min(minY, kpt[1]);
        maxX = Math.max(maxX, kpt[0]);
        maxY = Math.max(maxY, kpt[1]);
      }

      detections.push({
        bbox: {
          x1: Math.max(0, minX - 20),
          y1: Math.max(0, minY - 20),
          x2: Math.min(canvasRef.current?.width || 640, maxX + 20),
          y2: Math.min(canvasRef.current?.height || 480, maxY + 20),
          confidence: personScores.reduce((a: number, b: number) => a + b, 0) / personScores.length,
        },
        keypoints: personKeypoints2d.map((kpt: number[], idx: number) => ({
          x: kpt[0],
          y: kpt[1],
          score: personScores[idx],
          visible: personScores[idx] > 0.3,
        })),
        keypoints3d: personKeypoints,
      });
    }

    (detections as any).stats = stats;
    return detections;
  };

  const drawAnimalResults = (ctx: CanvasRenderingContext2D, animals: any[]) => {
    const colors = ['#ff6b6b', '#51cf66', '#339af0', '#ffd43b', '#da77f2', '#ff922b'];
    
    animals.forEach((animal, idx) => {
      const color = colors[idx % colors.length];
      
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(
        animal.bbox.x1,
        animal.bbox.y1,
        animal.bbox.x2 - animal.bbox.x1,
        animal.bbox.y2 - animal.bbox.y1
      );
      
      ctx.fillStyle = color;
      ctx.font = 'bold 14px Inter, sans-serif';
      ctx.fillText(
        `${animal.className} ${(animal.bbox.confidence * 100).toFixed(0)}%`,
        animal.bbox.x1,
        animal.bbox.y1 - 8
      );
      
      animal.keypoints?.forEach((kp: any) => {
        if (kp.visible) {
          ctx.fillStyle = '#51cf66';
          ctx.beginPath();
          ctx.arc(kp.x, kp.y, 5, 0, Math.PI * 2);
          ctx.fill();
        }
      });
    });
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !canvasRef.current) return;

    if (file.type.startsWith('video/')) {
      const url = URL.createObjectURL(file);
      setVideoSrc(url);
      setHasImage(false);
      setUseCamera(false);
      setIsPlaying(false);

      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      if (ctx) {
        canvas.width = 640;
        canvas.height = 480;
        ctx.fillStyle = '#1e293b';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#94a3b8';
        ctx.font = '16px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Video loaded. Click Play to start detection', canvas.width / 2, canvas.height / 2);
      }
    } else {
      const img = new Image();
      img.onload = () => {
        const canvas = canvasRef.current!;
        const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        ctx.drawImage(img, 0, 0);
        setVideoSrc(null);
        setUseCamera(false);
        setHasImage(true);
        setIsPlaying(false);
        stopDetectionLoop();
      };
      img.src = URL.createObjectURL(file);
    }
  };

  const startVideoDetection = async () => {
    if (!videoRef.current || !videoSrc) return;

    try {
      videoRef.current.src = videoSrc;
      videoRef.current.load();

      await new Promise((resolve) => {
        const timeout = setTimeout(resolve, 5000);
        videoRef.current!.addEventListener('loadeddata', () => {
          clearTimeout(timeout);
          resolve(true);
        }, { once: true });
      });

      if (canvasRef.current && videoRef.current) {
        const videoWidth = videoRef.current.videoWidth || 640;
        const videoHeight = videoRef.current.videoHeight || 480;
        canvasRef.current.width = videoWidth;
        canvasRef.current.height = videoHeight;
      }

      await videoRef.current.play();
      setIsPlaying(true);
      setHasImage(false);

      const interval = setInterval(() => {
        if (!videoRef.current || videoRef.current.paused || videoRef.current.ended) return;
        processDetection();
      }, 100);

      setDetectionInterval(interval);
    } catch (error) {
      console.error('Error starting video:', error);
      setIsPlaying(false);
    }
  };

  const stopDetectionLoop = () => {
    if (detectionInterval) {
      clearInterval(detectionInterval);
      setDetectionInterval(null);
    }
    if (videoRef.current) {
      videoRef.current.pause();
    }
    setIsPlaying(false);
  };

  useEffect(() => {
    return () => {
      stopDetectionLoop();
      if (videoSrc) URL.revokeObjectURL(videoSrc);
    };
  }, []);

  useEffect(() => {
    if (!videoRef.current) return;
    const handleVideoEnd = () => {
      setIsPlaying(false);
      stopDetectionLoop();
    };
    videoRef.current.addEventListener('ended', handleVideoEnd);
    return () => videoRef.current?.removeEventListener('ended', handleVideoEnd);
  }, [videoSrc]);

  const toggleClass = (className: string) => {
    setSelectedClasses(prev =>
      prev.includes(className)
        ? prev.filter(c => c !== className)
        : [...prev, className]
    );
  };

  const toggleAnimalClass = (className: string) => {
    setSelectedAnimalClasses(prev =>
      prev.includes(className)
        ? prev.filter(c => c !== className)
        : [...prev, className]
    );
  };

  useEffect(() => {
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      if (ctx) {
        ctx.fillStyle = '#1e293b';
        ctx.fillRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      }
      setDetections([]);
      setStats(null);
    }
  }, [mode]);

  if (modelStatus[mode].loading) {
    return (
      <div style={styles.container}>
        <div style={styles.loadingScreen}>
          <div style={styles.spinner}></div>
          <h2 style={styles.loadingTitle}>Loading {mode === 'pose3d' ? '3D Pose' : mode.charAt(0).toUpperCase() + mode.slice(1)} Detector...</h2>
          <p style={styles.loadingText}>This may take a moment on first load.</p>
        </div>
      </div>
    );
  }

  if (modelStatus[mode].error) {
    return (
      <div style={styles.container}>
        <div style={styles.errorScreen}>
          <div style={styles.errorIcon}>❌</div>
          <h2 style={styles.errorTitle}>Failed to Load Model</h2>
          <p style={styles.errorText}>{modelStatus[mode].error}</p>
          <button 
            style={styles.retryButton}
            onClick={() => setModelStatus(prev => ({ ...prev, [mode]: { loaded: false, loading: true } }))}
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      {/* Header */}
      <header style={styles.header}>
        <div style={styles.headerContent}>
          <div>
            <h1 style={styles.title}>🎯 rtmlib-ts Playground</h1>
            <p style={styles.subtitle}>Real-time AI Vision: Object Detection, 2D/3D Pose & Animal Detection</p>
          </div>
          <button 
            style={styles.docsButton}
            onClick={() => setShowDocs(!showDocs)}
          >
            {showDocs ? '📖 Hide Docs' : '📖 Quick Docs'}
          </button>
        </div>
      </header>

      {/* Documentation */}
      {showDocs && (
        <div style={styles.docsPanel}>
          <div style={styles.docsGrid}>
            <div style={styles.docCard}>
              <h3 style={styles.docCardTitle}>🔍 Object Detection</h3>
              <p style={styles.docCardText}>Detect 80 COCO classes (person, car, dog, etc.) using YOLOv12.</p>
              <code style={styles.code}>new ObjectDetector({'{'} classes: ['person'] {'}'})</code>
            </div>
            <div style={styles.docCard}>
              <h3 style={styles.docCardTitle}>🧍 Pose Estimation (2D)</h3>
              <p style={styles.docCardText}>Detect 17 body keypoints using RTMW model.</p>
              <code style={styles.code}>new PoseDetector()</code>
            </div>
            <div style={styles.docCard}>
              <h3 style={styles.docCardTitle}>🎭 Pose Estimation (3D)</h3>
              <p style={styles.docCardText}>3D pose estimation with Z-coordinates in meters using RTMW3D-X.</p>
              <code style={styles.code}>new Pose3DDetector()</code>
            </div>
            <div style={styles.docCard}>
              <h3 style={styles.docCardTitle}>🦁 Animal Detection</h3>
              <p style={styles.docCardText}>Detect 30 animal species with ViTPose++ pose estimation.</p>
              <code style={styles.code}>new AnimalDetector({'{'} poseModelType: 'vitpose-b' {'}'})</code>
            </div>
          </div>
        </div>
      )}

      {/* Controls */}
      <div style={styles.controls}>
        <div style={styles.controlGroup}>
          <label style={styles.label}>Mode</label>
          <select
            value={mode}
            onChange={(e) => setMode(e.target.value as 'object' | 'pose' | 'pose3d' | 'animal')}
            style={styles.select}
          >
            <option value="object">🔍 Object Detection</option>
            <option value="pose">🧍 Pose Estimation (2D)</option>
            <option value="pose3d">🎭 Pose Estimation (3D)</option>
            <option value="animal">🦁 Animal Detection</option>
          </select>
        </div>

        <div style={styles.controlGroup}>
          <label style={styles.label}>Performance</label>
          <select
            value={perfMode}
            onChange={(e) => setPerfMode(e.target.value as any)}
            style={styles.select}
          >
            <option value="performance">⚡ Performance (640×640)</option>
            <option value="balanced">⚖️ Balanced (416×416)</option>
            <option value="lightweight">🚀 Lightweight (320×320)</option>
          </select>
        </div>

        <div style={styles.controlGroup}>
          <label style={styles.label}>Backend</label>
          <select
            value={backend}
            onChange={(e) => setBackend(e.target.value as any)}
            style={styles.select}
          >
            <option value="wasm">💻 WASM (CPU)</option>
            <option value="webgpu">🎮 WebGPU (GPU)</option>
          </select>
        </div>

        {/* Performance optimization: Frame skipper */}
        {(useCamera || videoSrc) && (
          <div style={styles.controlGroup}>
            <label style={styles.label}>⚡ Process Every Nth Frame</label>
            <select
              value={processEveryNFrames}
              onChange={(e) => setProcessEveryNFrames(Number(e.target.value))}
              style={styles.select}
            >
              <option value={1}>Every frame (slow)</option>
              <option value={2}>Every 2nd frame</option>
              <option value={3}>Every 3rd frame (recommended)</option>
              <option value={4}>Every 4th frame</option>
              <option value={5}>Every 5th frame (fast)</option>
            </select>
            <small style={styles.hint}>
              Higher = faster but less smooth
            </small>
          </div>
        )}

        {mode === 'animal' && (
          <div style={styles.controlGroup}>
            <label style={styles.label}>Animal Pose Model</label>
            <select
              value={animalPoseModel}
              onChange={(e) => setAnimalPoseModel(e.target.value as any)}
              style={styles.select}
            >
              {(Object.keys(VITPOSE_MODELS) as Array<keyof typeof VITPOSE_MODELS>).map((key) => (
                <option key={key} value={key}>
                  {VITPOSE_MODELS[key].name} - {VITPOSE_MODELS[key].ap} AP
                </option>
              ))}
            </select>
          </div>
        )}

        <div style={styles.controlGroup}>
          <label style={styles.label}>Input Source</label>
          <div style={styles.buttonGroup}>
            <button
              onClick={() => {
                setUseCamera(!useCamera);
                setVideoSrc(null);
                setHasImage(false);
                stopDetectionLoop();
              }}
              style={{
                ...styles.button,
                background: useCamera ? 'linear-gradient(135deg, #00d9ff, #00ff88)' : undefined,
                color: useCamera ? '#000' : '#fff',
              }}
            >
              {useCamera ? '📹 Camera On' : '📷 Camera'}
            </button>
            {videoSrc && (
              <button
                onClick={isPlaying ? stopDetectionLoop : startVideoDetection}
                style={{
                  ...styles.button,
                  background: isPlaying ? 'linear-gradient(135deg, #00ff88, #00d9ff)' : 'linear-gradient(135deg, #00d9ff, #00ff88)',
                  color: '#000',
                }}
              >
                {isPlaying ? '⏹ Stop' : '▶ Play'}
              </button>
            )}
          </div>
          <label style={styles.fileLabel}>
            📁 Upload Image/Video
            <input
              type="file"
              accept="image/*,video/*"
              onChange={handleFileUpload}
              style={{ display: 'none' }}
            />
          </label>
        </div>

        {mode === 'object' && (
          <div style={styles.controlGroup}>
            <label style={styles.label}>Classes</label>
            <div style={styles.classGrid}>
              {['person', 'car', 'dog', 'cat', 'bicycle', 'bus', 'truck'].map(cls => (
                <label key={cls} style={styles.checkbox}>
                  <input
                    type="checkbox"
                    checked={selectedClasses.includes(cls)}
                    onChange={() => toggleClass(cls)}
                  />
                  {cls}
                </label>
              ))}
            </div>
          </div>
        )}

        {mode === 'animal' && (
          <div style={styles.controlGroup}>
            <label style={styles.label}>Animal Classes</label>
            <div style={styles.classGrid}>
              {['dog', 'cat', 'horse', 'zebra', 'elephant', 'tiger', 'lion', 'panda'].map(cls => (
                <label key={cls} style={styles.checkbox}>
                  <input
                    type="checkbox"
                    checked={selectedAnimalClasses.includes(cls)}
                    onChange={() => toggleAnimalClass(cls)}
                  />
                  {cls}
                </label>
              ))}
            </div>
            <small style={styles.hint}>
              {selectedAnimalClasses.length === 0 ? 'All 30 animals' : `${selectedAnimalClasses.length} selected`}
            </small>
          </div>
        )}

        <button
          onClick={processDetection}
          style={styles.detectButton}
        >
          🚀 Run Detection
        </button>
      </div>

      {/* Canvas */}
      <div style={styles.canvasWrapper}>
        <canvas ref={canvasRef} style={styles.canvas} width={640} height={480} />
        <video ref={videoRef} muted style={styles.hiddenVideo} />
      </div>

      {/* Status badges */}
      {videoSrc && (
        <div style={{
          ...styles.badge,
          background: isPlaying ? 'linear-gradient(135deg, #00ff88, #00d9ff)' : 'linear-gradient(135deg, #ff4444, #ff6b6b)',
        }}>
          {isPlaying ? '🎬 Playing' : '⏸ Paused'}
        </div>
      )}

      {mode === 'pose3d' && (
        <div style={styles.infoBadge}>
          💡 3D Mode: Z-coordinates in meters
        </div>
      )}

      {mode === 'animal' && (
        <div style={styles.infoBadge}>
          🦁 Animal Mode: {VITPOSE_MODELS[animalPoseModel].name} ({VITPOSE_MODELS[animalPoseModel].ap} AP)
        </div>
      )}

      {/* Stats */}
      {stats && (
        <div style={styles.statsGrid}>
          <div style={styles.statCard}>
            <div style={styles.statValue}>{stats.count}</div>
            <div style={styles.statLabel}>Detections</div>
          </div>
          <div style={styles.statCard}>
            <div style={styles.statValue}>{stats.time}ms</div>
            <div style={styles.statLabel}>Total Time</div>
          </div>
          {stats.detTime && (
            <div style={styles.statCard}>
              <div style={styles.statValue}>{stats.detTime}ms</div>
              <div style={styles.statLabel}>Detection</div>
            </div>
          )}
          {stats.poseTime && (
            <div style={styles.statCard}>
              <div style={styles.statValue}>{stats.poseTime}ms</div>
              <div style={styles.statLabel}>Pose</div>
            </div>
          )}
        </div>
      )}

      {/* Cache */}
      {cacheInfo && (
        <div style={styles.cacheBar}>
          <div style={styles.cacheInfo}>
            <span>💾 Model Cache:</span>
            <span style={styles.cacheSize}>{cacheInfo.size}</span>
            <span style={styles.cacheCount}>({cacheInfo.cached} models)</span>
          </div>
          <button
            onClick={async () => {
              await clearModelCache();
              setCacheInfo({ size: '0 B', cached: 0 });
              setModelStatus({
                object: { loaded: false, loading: false },
                pose: { loaded: false, loading: false },
                pose3d: { loaded: false, loading: false },
                animal: { loaded: false, loading: false },
              });
            }}
            style={styles.clearButton}
          >
            🗑️ Clear Cache
          </button>
        </div>
      )}

      {/* Results */}
      {detections.length > 0 && (
        <div style={styles.resultsPanel}>
          <h3 style={styles.resultsTitle}>📊 Detection Results</h3>
          <div style={styles.resultsList}>
            {detections.map((det, idx) => (
              <div key={idx} style={styles.resultCard}>
                <div style={styles.resultHeader}>
                  <span style={styles.resultName}>
                    {det.className || 'Person'} #{idx + 1}
                  </span>
                  <span style={styles.resultConfidence}>
                    {(det.bbox.confidence * 100).toFixed(1)}%
                  </span>
                </div>
                <div style={styles.resultCoords}>
                  [{det.bbox.x1.toFixed(0)}, {det.bbox.y1.toFixed(0)}, {det.bbox.x2.toFixed(0)}, {det.bbox.y2.toFixed(0)}]
                </div>
                {mode === 'pose3d' && det.keypoints3d && (
                  <div style={styles.result3d}>
                    <small>🎯 3D: [{det.keypoints3d[0][0].toFixed(3)}, {det.keypoints3d[0][1].toFixed(3)}, {det.keypoints3d[0][2].toFixed(3)}]m</small>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    maxWidth: 1400,
    margin: '0 auto',
    padding: '24px',
    fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    background: 'linear-gradient(180deg, #0f172a 0%, #1e293b 100%)',
    minHeight: '100vh',
  },
  header: {
    marginBottom: '32px',
    padding: '24px 32px',
    background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(147, 51, 234, 0.15))',
    borderRadius: '20px',
    border: '1px solid rgba(255, 255, 255, 0.1)',
    backdropFilter: 'blur(10px)',
  },
  headerContent: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    flexWrap: 'wrap',
    gap: '16px',
  },
  title: {
    fontSize: '32px',
    fontWeight: '800',
    background: 'linear-gradient(135deg, #60a5fa, #c084fc)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    margin: '0 0 8px 0',
  },
  subtitle: {
    fontSize: '15px',
    color: '#94a3b8',
    margin: 0,
  },
  docsButton: {
    padding: '12px 24px',
    borderRadius: '12px',
    border: '1px solid rgba(147, 51, 234, 0.3)',
    background: 'linear-gradient(135deg, rgba(147, 51, 234, 0.2), rgba(59, 130, 246, 0.2))',
    color: '#e2e8f0',
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: '600',
    transition: 'all 0.2s',
  },
  docsPanel: {
    marginBottom: '32px',
    padding: '24px',
    background: 'rgba(30, 41, 59, 0.5)',
    borderRadius: '16px',
    border: '1px solid rgba(147, 51, 234, 0.2)',
  },
  docsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
    gap: '16px',
  },
  docCard: {
    padding: '20px',
    background: 'rgba(59, 130, 246, 0.1)',
    borderRadius: '12px',
    border: '1px solid rgba(59, 130, 246, 0.2)',
  },
  docCardTitle: {
    fontSize: '16px',
    fontWeight: '700',
    color: '#e2e8f0',
    margin: '0 0 8px 0',
  },
  docCardText: {
    fontSize: '14px',
    color: '#94a3b8',
    margin: '0 0 12px 0',
  },
  code: {
    display: 'block',
    padding: '10px 14px',
    background: 'rgba(0, 0, 0, 0.3)',
    borderRadius: '8px',
    fontSize: '12px',
    color: '#51cf66',
    fontFamily: '"Fira Code", monospace',
  },
  controls: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
    gap: '20px',
    marginBottom: '32px',
    padding: '28px',
    background: 'rgba(30, 41, 59, 0.6)',
    borderRadius: '20px',
    border: '1px solid rgba(255, 255, 255, 0.1)',
  },
  controlGroup: {
    display: 'flex',
    flexDirection: 'column',
    gap: '10px',
  },
  label: {
    fontSize: '13px',
    fontWeight: '600',
    color: '#94a3b8',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
  },
  select: {
    padding: '14px 18px',
    borderRadius: '12px',
    border: '1px solid rgba(255, 255, 255, 0.15)',
    background: 'rgba(15, 23, 42, 0.8)',
    color: '#e2e8f0',
    fontSize: '14px',
    fontWeight: '500',
    cursor: 'pointer',
    transition: 'all 0.2s',
  },
  buttonGroup: {
    display: 'flex',
    gap: '10px',
    flexWrap: 'wrap',
  },
  button: {
    padding: '12px 20px',
    borderRadius: '10px',
    border: '1px solid rgba(255, 255, 255, 0.15)',
    background: 'rgba(59, 130, 246, 0.2)',
    color: '#e2e8f0',
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: '600',
    transition: 'all 0.2s',
  },
  fileLabel: {
    display: 'inline-block',
    padding: '14px 24px',
    borderRadius: '12px',
    background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)',
    color: '#fff',
    cursor: 'pointer',
    textAlign: 'center',
    fontWeight: '600',
    fontSize: '14px',
    transition: 'transform 0.2s, box-shadow 0.2s',
    boxShadow: '0 4px 14px rgba(59, 130, 246, 0.4)',
  },
  classGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(90px, 1fr))',
    gap: '8px',
  },
  checkbox: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    fontSize: '13px',
    padding: '8px 12px',
    background: 'rgba(59, 130, 246, 0.1)',
    borderRadius: '8px',
    cursor: 'pointer',
    color: '#e2e8f0',
    transition: 'background 0.2s',
  },
  hint: {
    color: '#64748b',
    fontSize: '12px',
    marginTop: '4px',
  },
  detectButton: {
    gridColumn: '1 / -1',
    padding: '18px 32px',
    borderRadius: '14px',
    border: 'none',
    background: 'linear-gradient(135deg, #3b82f6, #8b5cf6, #ec4899)',
    color: '#fff',
    cursor: 'pointer',
    fontSize: '16px',
    fontWeight: '700',
    transition: 'transform 0.2s, box-shadow 0.2s',
    boxShadow: '0 8px 24px rgba(59, 130, 246, 0.4)',
  },
  canvasWrapper: {
    position: 'relative',
    borderRadius: '20px',
    overflow: 'hidden',
    background: '#0f172a',
    marginBottom: '24px',
    boxShadow: '0 20px 60px rgba(0, 0, 0, 0.4)',
    border: '1px solid rgba(255, 255, 255, 0.1)',
  },
  canvas: {
    width: '100%',
    height: 'auto',
    display: 'block',
  },
  hiddenVideo: {
    position: 'absolute',
    top: 0,
    left: 0,
    width: 0,
    height: 0,
    opacity: 0,
  },
  badge: {
    display: 'inline-block',
    padding: '10px 20px',
    borderRadius: '10px',
    color: '#fff',
    fontWeight: '600',
    fontSize: '14px',
    marginBottom: '16px',
    boxShadow: '0 4px 14px rgba(0, 0, 0, 0.3)',
  },
  infoBadge: {
    display: 'inline-block',
    padding: '12px 20px',
    borderRadius: '12px',
    background: 'rgba(59, 130, 246, 0.15)',
    border: '1px solid rgba(59, 130, 246, 0.3)',
    color: '#60a5fa',
    marginBottom: '16px',
    fontSize: '14px',
    fontWeight: '600',
  },
  statsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
    gap: '16px',
    marginBottom: '24px',
  },
  statCard: {
    padding: '24px',
    background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(147, 51, 234, 0.15))',
    borderRadius: '16px',
    textAlign: 'center',
    border: '1px solid rgba(255, 255, 255, 0.1)',
  },
  statValue: {
    fontSize: '32px',
    fontWeight: '800',
    background: 'linear-gradient(135deg, #51cf66, #00d9ff)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    marginBottom: '6px',
  },
  statLabel: {
    fontSize: '13px',
    color: '#94a3b8',
    fontWeight: '500',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
  },
  cacheBar: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '20px 24px',
    background: 'rgba(30, 41, 59, 0.6)',
    borderRadius: '16px',
    marginBottom: '24px',
    border: '1px solid rgba(255, 255, 255, 0.1)',
  },
  cacheInfo: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    fontSize: '14px',
    color: '#94a3b8',
  },
  cacheSize: {
    color: '#51cf66',
    fontWeight: '700',
    fontSize: '16px',
  },
  cacheCount: {
    color: '#64748b',
    fontSize: '13px',
  },
  clearButton: {
    padding: '10px 20px',
    borderRadius: '10px',
    border: '1px solid rgba(239, 68, 68, 0.3)',
    background: 'rgba(239, 68, 68, 0.2)',
    color: '#f87171',
    cursor: 'pointer',
    fontSize: '13px',
    fontWeight: '600',
  },
  resultsPanel: {
    padding: '28px',
    background: 'rgba(30, 41, 59, 0.6)',
    borderRadius: '20px',
    border: '1px solid rgba(255, 255, 255, 0.1)',
  },
  resultsTitle: {
    fontSize: '18px',
    fontWeight: '700',
    color: '#e2e8f0',
    margin: '0 0 20px 0',
  },
  resultsList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
  },
  resultCard: {
    padding: '18px 20px',
    background: 'rgba(15, 23, 42, 0.6)',
    borderRadius: '12px',
    border: '1px solid rgba(59, 130, 246, 0.2)',
    transition: 'border-color 0.2s',
  },
  resultHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '8px',
  },
  resultName: {
    fontSize: '15px',
    fontWeight: '700',
    color: '#e2e8f0',
  },
  resultConfidence: {
    fontSize: '14px',
    fontWeight: '700',
    color: '#51cf66',
    background: 'rgba(81, 207, 102, 0.15)',
    padding: '4px 12px',
    borderRadius: '6px',
  },
  resultCoords: {
    fontSize: '13px',
    color: '#64748b',
    fontFamily: '"Fira Code", monospace',
  },
  result3d: {
    marginTop: '10px',
    padding: '10px 14px',
    background: 'rgba(59, 130, 246, 0.15)',
    borderRadius: '8px',
    color: '#60a5fa',
    fontFamily: '"Fira Code", monospace',
    fontSize: '13px',
  },
  loadingScreen: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: '80vh',
    gap: '24px',
  },
  spinner: {
    width: '60px',
    height: '60px',
    border: '4px solid rgba(59, 130, 246, 0.2)',
    borderTop: '4px solid #3b82f6',
    borderRadius: '50%',
    animation: 'spin 1s linear infinite',
  },
  loadingTitle: {
    fontSize: '24px',
    fontWeight: '700',
    color: '#e2e8f0',
    margin: '0 0 8px 0',
  },
  loadingText: {
    fontSize: '15px',
    color: '#94a3b8',
    margin: 0,
  },
  errorScreen: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: '80vh',
    gap: '24px',
  },
  errorIcon: {
    fontSize: '48px',
  },
  errorTitle: {
    fontSize: '24px',
    fontWeight: '700',
    color: '#f87171',
    margin: '0 0 8px 0',
  },
  errorText: {
    fontSize: '15px',
    color: '#94a3b8',
    margin: '0 0 20px 0',
    textAlign: 'center',
    maxWidth: '400px',
  },
  retryButton: {
    padding: '14px 32px',
    borderRadius: '12px',
    border: 'none',
    background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)',
    color: '#fff',
    cursor: 'pointer',
    fontSize: '16px',
    fontWeight: '700',
    boxShadow: '0 8px 24px rgba(59, 130, 246, 0.4)',
  },
};
