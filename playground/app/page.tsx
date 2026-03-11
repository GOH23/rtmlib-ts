'use client';

import { useEffect, useState, useRef } from 'react';
import type { ObjectDetector as ObjectType, PoseDetector as PoseType } from '../../src/index';
import { drawDetectionsOnCanvas, drawPoseOnCanvas, drawResultsOnCanvas, getCacheInfo, clearModelCache } from '../../src/index';

// Types
interface Detection {
  bbox: { x1: number; y1: number; x2: number; y2: number; confidence: number };
  className?: string;
  keypoints?: any[];
}

// Lazy load detectors to avoid SSR issues
let ObjectDetector: typeof ObjectType;
let PoseDetector: typeof PoseType;

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [objectDetector, setObjectDetector] = useState<any>(null);
  const [poseDetector, setPoseDetector] = useState<any>(null);
  const [mode, setMode] = useState<'object' | 'pose'>('object');
  const [detModel, setDetModel] = useState<'yolov12n' | 'yolov26n'>('yolov12n');
  const [perfMode, setPerfMode] = useState<'performance' | 'balanced' | 'lightweight'>('balanced');
  const [backend, setBackend] = useState<'wasm' | 'webgpu'>('wasm');
  const [selectedClasses, setSelectedClasses] = useState<string[]>(['person']);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [stats, setStats] = useState<{ time: number; count: number } | null>(null);
  const [loading, setLoading] = useState(true);
  const [useCamera, setUseCamera] = useState(false);
  const [hasImage, setHasImage] = useState(false);
  const [videoSrc, setVideoSrc] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [detectionInterval, setDetectionInterval] = useState<NodeJS.Timeout | null>(null);
  const [cacheInfo, setCacheInfo] = useState<{ size: string; cached: number } | null>(null);

  // Initialize detectors
  useEffect(() => {
    async function init() {
      // Dynamic import to avoid SSR issues
      const { ObjectDetector: ObjDet, PoseDetector: PoseDet } = await import('../../src/index');
      ObjectDetector = ObjDet;
      PoseDetector = PoseDet;

      const modelPath = `https://huggingface.co/demon2233/rtmlib-ts/resolve/main/yolo/${detModel}.onnx`;

      // Object Detector with mode preset
      const objDet = new ObjectDetector({
        model: modelPath,
        classes: selectedClasses,
        mode: perfMode,  // Use performance mode preset
        backend: backend,
        confidence: 0.3,  // Lower threshold for video
        cache: true,  // Enable caching
      });
      const detStart = performance.now();
      await objDet.init();
      console.log(`ObjectDetector initialized in ${Math.round(performance.now() - detStart)}ms`);
      setObjectDetector(objDet);

      // Pose Detector
      const poseDet = new PoseDetector({
        detModel: modelPath,
        poseModel: 'https://huggingface.co/demon2233/rtmlib-ts/resolve/main/rtmpose/end2end.onnx',
        detConfidence: 0.5,
        poseConfidence: 0.3,
        backend: backend,
        cache: true,  // Enable caching
      });
      const poseStart = performance.now();
      await poseDet.init();
      console.log(`PoseDetector initialized in ${Math.round(performance.now() - poseStart)}ms`);
      setPoseDetector(poseDet);

      // Load cache info
      const cacheInfo = await getCacheInfo();
      setCacheInfo({
        size: cacheInfo.totalSizeFormatted,
        cached: cacheInfo.cachedModels.length,
      });

      setLoading(false);
      console.log(`Detectors initialized with ${detModel}, mode: ${perfMode}, backend: ${backend}`);
    }

    init();
  }, [detModel, perfMode, backend]);

  // Update classes when changed
  useEffect(() => {
    if (objectDetector) {
      objectDetector.setClasses(selectedClasses.length > 0 ? selectedClasses : null);
    }
  }, [selectedClasses, objectDetector]);

  // Camera handling
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

  // Process detection
  const processDetection = async () => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;

    let results: any[] = [];
    const startTime = performance.now();

    console.log('🔍 processDetection called:', { mode, useCamera, videoSrc, isPlaying, hasImage });

    if (mode === 'object' && objectDetector) {
      if (useCamera && videoRef.current) {
        // Draw current video frame FIRST
        ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
        // Then detect
        results = await objectDetector.detectFromVideo(videoRef.current, canvas);
        // Draw detections on top
        drawDetectionsOnCanvas(ctx, results);
        console.log('📹 Camera frame drawn');
      } else if (videoSrc && videoRef.current) {
        console.log('🎬 Processing video, videoRef:', !!videoRef.current, 'paused:', videoRef.current.paused, 'readyState:', videoRef.current.readyState);
        
        // Check if video is actually playing
        if (videoRef.current.readyState < 2) {
          console.log('⚠️ Video not ready, readyState:', videoRef.current.readyState);
          return;
        }
        if (videoRef.current.videoWidth === 0 || videoRef.current.videoHeight === 0) {
          console.log('⚠️ Video dimensions are 0');
          return;
        }
        
        // Draw current video frame FIRST
        ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
        
        // Verify draw worked
        const checkData = ctx.getImageData(0, 0, 10, 10);
        const hasVideoData = checkData.data.some(pixel => pixel > 0);
        console.log('🎬 Video frame drawn:', hasVideoData ? '✅' : '❌', 'readyState:', videoRef.current.readyState, 'time:', videoRef.current.currentTime.toFixed(2));
        
        // Then detect
        results = await objectDetector.detectFromVideo(videoRef.current, canvas);
        // Draw detections on top
        drawDetectionsOnCanvas(ctx, results);
      } else {
        console.log('🖼️ Image mode or no video');
        // Image mode - image already on canvas
        results = await objectDetector.detectFromCanvas(canvas);
        drawDetectionsOnCanvas(ctx, results);
      }
    } else if (mode === 'pose' && poseDetector) {
      if (useCamera && videoRef.current) {
        // Draw current video frame FIRST
        ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
        // Then detect
        results = await poseDetector.detectFromVideo(videoRef.current, canvas);
        // Draw pose on top
        drawPoseOnCanvas(ctx, results);
      } else if (videoSrc && videoRef.current) {
        // Check if video is actually playing
        if (videoRef.current.paused || videoRef.current.ended) {
          console.log('⚠️ Video paused or ended');
          return;
        }
        
        // Draw current video frame FIRST
        ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
        
        // Verify draw worked
        const checkData = ctx.getImageData(0, 0, 10, 10);
        const hasVideoData = checkData.data.some(pixel => pixel > 0);
        console.log('🎬 Video frame drawn:', hasVideoData ? '✅' : '❌', 'readyState:', videoRef.current.readyState, 'time:', videoRef.current.currentTime.toFixed(2));
        
        // Then detect
        results = await poseDetector.detectFromVideo(videoRef.current, canvas);
        // Draw pose on top
        drawPoseOnCanvas(ctx, results);
      } else {
        // Image mode - image already on canvas
        results = await poseDetector.detectFromCanvas(canvas);
        drawPoseOnCanvas(ctx, results);
      }
    }

    const endTime = performance.now();
    const detStats = (results as any).stats;

    setDetections(results);
    setStats({
      time: Math.round(endTime - startTime),
      count: results.length,
    });
    
    console.log(`[Performance] Total: ${Math.round(endTime - startTime)}ms (Det: ${detStats?.detTime || 'N/A'}ms, Pose: ${detStats?.poseTime || 'N/A'}ms)`);
  };

  // Handle file upload
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !canvasRef.current) return;

    if (file.type.startsWith('video/')) {
      // Video file
      const url = URL.createObjectURL(file);
      setVideoSrc(url);
      setHasImage(false);
      setUseCamera(false);
      setIsPlaying(false);
      
      // Clear canvas
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      if (ctx) {
        canvas.width = 640;
        canvas.height = 480;
        ctx.fillStyle = '#1a1a2e';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#8892b0';
        ctx.font = '16px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Video loaded. Click Play to start detection', canvas.width / 2, canvas.height / 2);
      }
    } else {
      // Image file
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

  // Start video detection loop
  const startVideoDetection = async () => {
    if (!videoRef.current || !videoSrc) return;
    
    try {
      console.log('🎬 Starting video playback...');
      console.log('Video src:', videoSrc.substring(0, 50) + '...');
      
      // Restart video if it ended
      if (videoRef.current.ended) {
        videoRef.current.currentTime = 0;
        console.log('🔄 Restarting video from beginning');
      }
      
      videoRef.current.src = videoSrc;
      videoRef.current.load();
      
      // Wait for video to be ready with more events
      await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          console.warn('⚠️ Video loading timeout');
          resolve(true);
        }, 5000);
        
        videoRef.current!.addEventListener('loadeddata', () => {
          console.log('✅ Video loadeddata event');
          clearTimeout(timeout);
          resolve(true);
        }, { once: true });
        
        videoRef.current!.addEventListener('error', (e) => {
          console.error('❌ Video loading error:', e);
          clearTimeout(timeout);
          reject(e);
        }, { once: true });
      });
      
      // Check video dimensions
      console.log('Video dimensions:', videoRef.current.videoWidth, 'x', videoRef.current.videoHeight);
      console.log('Video readyState:', videoRef.current.readyState);
      console.log('Video duration:', videoRef.current.duration);
      
      // Set canvas size before playing
      if (canvasRef.current && videoRef.current) {
        const videoWidth = videoRef.current.videoWidth || 640;
        const videoHeight = videoRef.current.videoHeight || 480;
        canvasRef.current.width = videoWidth;
        canvasRef.current.height = videoHeight;
        console.log(`✅ Canvas size set to ${canvasRef.current.width}x${canvasRef.current.height}`);
      }
      
      // Play video
      await videoRef.current.play();
      console.log('✅ Video playback started, readyState:', videoRef.current.readyState);
      
      // Wait a bit for first frame to render
      await new Promise(resolve => setTimeout(resolve, 100));
      
      setIsPlaying(true);
      setHasImage(false);
      
      // Draw first frame immediately
      if (canvasRef.current && videoRef.current) {
        const ctx = canvasRef.current.getContext('2d', { willReadFrequently: true });
        if (ctx) {
          // Clear canvas first
          ctx.fillStyle = '#000';
          ctx.fillRect(0, 0, canvasRef.current.width, canvasRef.current.height);
          
          // Try to draw
          console.log('Drawing first frame, video dimensions:', videoRef.current.videoWidth, 'x', videoRef.current.videoHeight);
          ctx.drawImage(videoRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height);
          
          // Verify draw worked
          const imageData = ctx.getImageData(0, 0, 10, 10);
          const hasData = imageData.data.some(pixel => pixel > 0);
          console.log('First frame drawn:', hasData ? '✅ Success' : '❌ Canvas still black');
        }
      }
      
      // Run detection loop synced with video
      let lastTime = videoRef.current.currentTime;
      
      const interval = setInterval(() => {
        if (!videoRef.current || videoRef.current.paused || videoRef.current.ended) {
          return;
        }
        
        const currentTime = videoRef.current.currentTime;
        // Only process if video time has changed (skip if detection is too slow)
        if (currentTime > lastTime) {
          lastTime = currentTime;
          processDetection();
        }
      }, 100);  // Check more frequently but only process when video advances
      
      setDetectionInterval(interval);
      console.log('✅ Video detection loop started (synced)');
    } catch (error) {
      console.error('❌ Error starting video:', error);
      setIsPlaying(false);
      alert('Error loading video: ' + (error as Error).message);
    }
  };

  // Stop video detection loop
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

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopDetectionLoop();
      if (videoSrc) {
        URL.revokeObjectURL(videoSrc);
      }
    };
  }, []);

  // Handle video end
  useEffect(() => {
    if (!videoRef.current) return;
    
    const handleVideoEnd = () => {
      console.log('🏁 Video ended');
      setIsPlaying(false);
      stopDetectionLoop();
    };
    
    videoRef.current.addEventListener('ended', handleVideoEnd);
    return () => {
      videoRef.current?.removeEventListener('ended', handleVideoEnd);
    };
  }, [videoSrc]);

  // Handle class toggle
  const toggleClass = (className: string) => {
    setSelectedClasses(prev =>
      prev.includes(className)
        ? prev.filter(c => c !== className)
        : [...prev, className]
    );
  };

  if (loading) {
    return (
      <div style={styles.container}>
        <h1>Loading models...</h1>
        <p>This may take a moment on first load.</p>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <h1>🎯 rtmlib-ts Playground</h1>
        <p>Test Object Detection & Pose Estimation</p>
      </header>

      <div style={styles.controls}>
        <div style={styles.controlGroup}>
          <label>Mode:</label>
          <select
            value={mode}
            onChange={(e) => setMode(e.target.value as 'object' | 'pose')}
            style={styles.select}
          >
            <option value="object">Object Detection</option>
            <option value="pose">Pose Estimation</option>
          </select>
        </div>

        <div style={styles.controlGroup}>
          <label>Performance Mode:</label>
          <select
            value={perfMode}
            onChange={(e) => setPerfMode(e.target.value as any)}
            style={styles.select}
          >
            <option value="performance">Performance (640×640, ~500ms)</option>
            <option value="balanced">Balanced (416×416, ~200ms)</option>
            <option value="lightweight">Lightweight (320×320, ~100ms)</option>
          </select>
        </div>

        <div style={styles.controlGroup}>
          <label>Backend:</label>
          <select
            value={backend}
            onChange={(e) => setBackend(e.target.value as any)}
            style={styles.select}
          >
            <option value="wasm">WASM (CPU)</option>
            <option value="webgpu">WebGPU (GPU) - Experimental</option>
          </select>
        </div>

        <div style={styles.controlGroup}>
          <label>Detection Model:</label>
          <select
            value={detModel}
            onChange={(e) => setDetModel(e.target.value as 'yolov12n' | 'yolov26n')}
            style={styles.select}
          >
            <option value="yolov12n">YOLO12n (Fast - Recommended)</option>
            <option value="yolov26n">YOLOv26n (Experimental)</option>
          </select>
          <small style={{color: '#8892b0', fontSize: '11px', marginTop: '4px'}}>
            {detModel === 'yolov12n' ? '✅ Works perfectly' : '⚠️ May need model re-export'}
          </small>
        </div>

        <div style={styles.controlGroup}>
          <label>Input:</label>
          <div style={{display: 'flex', gap: '8px', flexWrap: 'wrap'}}>
            <button
              onClick={() => {
                setUseCamera(!useCamera);
                setVideoSrc(null);
                setHasImage(false);
                stopDetectionLoop();
              }}
              style={{
                ...styles.button,
                background: useCamera ? '#00d9ff' : undefined,
                color: useCamera ? '#000' : '#fff',
              }}
            >
              {useCamera ? '📹 Camera Active' : '📷 Use Camera'}
            </button>
            {videoSrc && (
              <>
                <button
                  onClick={isPlaying ? stopDetectionLoop : startVideoDetection}
                  style={{
                    ...styles.button,
                    background: isPlaying ? '#00ff88' : '#00d9ff',
                    color: '#000',
                  }}
                >
                  {videoRef.current?.ended ? '🔄 Restart' : isPlaying ? '⏹ Stop' : '▶ Play'}
                </button>
              </>
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
          {videoSrc && (
            <small style={{color: '#00ff88', fontSize: '11px', marginTop: '4px'}}>
              🎬 Video loaded - Click Play to start detection
            </small>
          )}
        </div>

        {mode === 'object' && (
          <div style={styles.controlGroup}>
            <label>Classes:</label>
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
            <button
              onClick={() => setSelectedClasses([])}
              style={{ ...styles.button, marginTop: '8px' }}
            >
              Select All (80 classes)
            </button>
          </div>
        )}

        <button
          onClick={processDetection}
          style={{ ...styles.button, ...styles.primaryButton }}
        >
          🚀 Detect
        </button>
      </div>

      <div style={styles.canvasContainer}>
        <canvas ref={canvasRef} style={styles.canvas} width={640} height={480} />
        {/* Hidden video element for processing */}
        <video
          ref={videoRef}
          muted
        />
      </div>

      {/* Video preview indicator */}
      {videoSrc && (
        <div style={{...styles.videoPreview, background: isPlaying ? '#00ff88' : '#ff4444'}}>
          {isPlaying ? '🎬 Playing' : '⏸ Paused'} - {isPlaying ? 'Detection active' : 'Click Play to start'}
        </div>
      )}
      
      {/* Video ended message */}
      {videoSrc && !isPlaying && videoRef.current?.ended && (
        <div style={{...styles.videoPreview, background: '#ff6b6b', marginTop: '8px'}}>
          🏁 Video ended - Upload new video or click Play to restart
        </div>
      )}

      {stats && (
        <div style={styles.stats}>
          <div style={styles.statItem}>
            <span style={styles.statValue}>{stats.count}</span>
            <span style={styles.statLabel}>Detections</span>
          </div>
          <div style={styles.statItem}>
            <span style={styles.statValue}>{stats.time}ms</span>
            <span style={styles.statLabel}>Inference Time</span>
          </div>
          <div style={styles.statItem}>
            <span style={{ ...styles.statValue, fontSize: '14px' }}>{detModel}</span>
            <span style={styles.statLabel}>Model</span>
          </div>
        </div>
      )}

      {cacheInfo && (
        <div style={styles.cacheInfo}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <span style={{ fontSize: '14px', color: '#8892b0' }}>💾 Model Cache:</span>
              <span style={{ marginLeft: '8px', color: '#00ff88', fontWeight: '600' }}>{cacheInfo.size}</span>
              <span style={{ marginLeft: '4px', color: '#8892b0', fontSize: '12px' }}>({cacheInfo.cached} models)</span>
            </div>
            <button
              onClick={async () => {
                await clearModelCache();
                setCacheInfo({ size: '0 B', cached: 0 });
              }}
              style={styles.clearCacheBtn}
            >
              🗑️ Clear Cache
            </button>
          </div>
        </div>
      )}

      {detections.length > 0 && (
        <div style={styles.results}>
          <h3>Results:</h3>
          <div style={styles.resultsList}>
            {detections.map((det, idx) => (
              <div key={idx} style={styles.resultItem}>
                <strong>{mode === 'object' ? det.className : 'Person'} {idx + 1}</strong>
                <span>{(det.bbox.confidence * 100).toFixed(1)}%</span>
                <code>
                  [{det.bbox.x1.toFixed(0)}, {det.bbox.y1.toFixed(0)}, {det.bbox.x2.toFixed(0)}, {det.bbox.y2.toFixed(0)}]
                </code>
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
    maxWidth: 1200,
    margin: '0 auto',
    padding: '20px',
    fontFamily: 'system-ui, -apple-system, sans-serif',
  },
  header: {
    textAlign: 'center',
    marginBottom: '30px',
    padding: '20px',
    background: 'linear-gradient(135deg, rgba(0,217,255,0.1), rgba(0,255,136,0.1))',
    borderRadius: '16px',
    border: '1px solid rgba(255,255,255,0.1)',
  },
  controls: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
    gap: '16px',
    marginBottom: '20px',
    padding: '20px',
    background: 'rgba(255,255,255,0.05)',
    borderRadius: '12px',
    backdropFilter: 'blur(10px)',
    border: '1px solid rgba(255,255,255,0.1)',
  },
  controlGroup: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  },
  select: {
    padding: '10px 14px',
    borderRadius: '8px',
    border: '1px solid rgba(255,255,255,0.2)',
    background: 'rgba(0,0,0,0.3)',
    color: '#fff',
    fontSize: '14px',
    cursor: 'pointer',
  },
  button: {
    padding: '10px 20px',
    borderRadius: '8px',
    border: 'none',
    background: 'rgba(255,255,255,0.1)',
    color: '#fff',
    cursor: 'pointer',
    transition: 'all 0.2s',
    fontSize: '14px',
    fontWeight: '500',
  },
  fileLabel: {
    padding: '10px 20px',
    borderRadius: '8px',
    background: 'linear-gradient(90deg, #00d9ff, #00ff88)',
    color: '#000',
    cursor: 'pointer',
    textAlign: 'center',
    fontWeight: '600',
    transition: 'transform 0.2s',
  },
  classGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(100px, 1fr))',
    gap: '8px',
  },
  checkbox: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    fontSize: '13px',
    padding: '6px 10px',
    background: 'rgba(0,0,0,0.2)',
    borderRadius: '6px',
    cursor: 'pointer',
  },
  canvasContainer: {
    position: 'relative',
    borderRadius: '16px',
    overflow: 'hidden',
    background: '#000',
    boxShadow: '0 20px 60px rgba(0,0,0,0.5)',
    border: '1px solid rgba(255,255,255,0.1)',
  },
  canvas: {
    width: '100%',
    display: 'block',
  },
  stats: {
    display: 'grid',
    gridTemplateColumns: 'repeat(3, 1fr)',
    gap: '16px',
    marginTop: '20px',
    padding: '20px',
    background: 'rgba(255,255,255,0.05)',
    borderRadius: '12px',
    backdropFilter: 'blur(10px)',
    border: '1px solid rgba(255,255,255,0.1)',
  },
  statItem: {
    textAlign: 'center',
    padding: '12px',
    background: 'rgba(0,0,0,0.2)',
    borderRadius: '10px',
  },
  statValue: {
    display: 'block',
    fontSize: '28px',
    fontWeight: 'bold',
    background: 'linear-gradient(90deg, #00d9ff, #00ff88)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    backgroundClip: 'text',
  },
  statLabel: {
    fontSize: '12px',
    color: '#8892b0',
    marginTop: '6px',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
  },
  results: {
    marginTop: '20px',
    padding: '20px',
    background: 'rgba(255,255,255,0.05)',
    borderRadius: '12px',
    backdropFilter: 'blur(10px)',
    border: '1px solid rgba(255,255,255,0.1)',
  },
  resultsList: {
    display: 'grid',
    gap: '10px',
    marginTop: '12px',
  },
  resultItem: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '12px 16px',
    background: 'rgba(0,0,0,0.3)',
    borderRadius: '8px',
    border: '1px solid rgba(255,255,255,0.05)',
    transition: 'transform 0.2s',
  },
  videoPreview: {
    padding: '12px 20px',
    borderRadius: '8px',
    color: '#000',
    fontWeight: '600',
    textAlign: 'center',
    marginTop: '16px',
    animation: 'pulse 2s infinite',
  },
  cacheInfo: {
    marginTop: '16px',
    padding: '16px 20px',
    background: 'rgba(0,217,255,0.1)',
    borderRadius: '12px',
    border: '1px solid rgba(0,217,255,0.2)',
  },
  clearCacheBtn: {
    padding: '8px 16px',
    borderRadius: '8px',
    border: 'none',
    background: 'rgba(255,0,0,0.2)',
    color: '#ff4444',
    cursor: 'pointer',
    fontSize: '13px',
    fontWeight: '500',
    transition: 'all 0.2s',
  },
};
