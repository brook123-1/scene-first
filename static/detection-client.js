(function () {
  'use strict';

  const MAX_DETECTION_EDGE = 1600;
  const DETECTION_TIMEOUT_MS = 45000;
  // Temporary safety ceiling.  Real upload progress and server-stage timings,
  // not a larger timeout, are the P0.5 performance signal.
  const MASTER_UPLOAD_TIMEOUT_MS = 240000;
  const MASTER_RECOVERY_TIMEOUT_MS = 150000;

  class MasterUploadError extends Error {
    constructor(code, message, metrics = {}) {
      super(message);
      this.name = 'MasterUploadError';
      this.code = code;
      this.metrics = metrics;
    }
  }

  function canvasBlob(canvas, type = 'image/jpeg', quality = 0.86) {
    return new Promise((resolve, reject) => {
      canvas.toBlob(blob => blob ? resolve(blob) : reject(new Error('无法创建检测副本。')), type, quality);
    });
  }

  async function createDetectionFile(image, sourceName = 'photo') {
    const sourceWidth = image.naturalWidth || image.width;
    const sourceHeight = image.naturalHeight || image.height;
    const ratio = Math.min(1, MAX_DETECTION_EDGE / Math.max(sourceWidth, sourceHeight));
    const detectionWidth = Math.max(1, Math.round(sourceWidth * ratio));
    const detectionHeight = Math.max(1, Math.round(sourceHeight * ratio));
    const canvas = document.createElement('canvas');
    canvas.width = detectionWidth;
    canvas.height = detectionHeight;
    const context = canvas.getContext('2d', { alpha: false });
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = 'high';
    context.drawImage(image, 0, 0, detectionWidth, detectionHeight);
    const blob = await canvasBlob(canvas);
    const safeStem = String(sourceName).replace(/\.[^.]+$/, '').replace(/[^a-z0-9_-]+/gi, '-').slice(0, 48) || 'photo';
    return {
      file: new File([blob], `${safeStem}-detection.jpg`, { type: 'image/jpeg' }),
      sourceWidth,
      sourceHeight,
      detectionWidth,
      detectionHeight,
      scaleX: sourceWidth / detectionWidth,
      scaleY: sourceHeight / detectionHeight,
    };
  }

  function scaleBox(box, scaleX, scaleY, sourceWidth, sourceHeight) {
    const [x, y, width, height] = box;
    const left = Math.max(0, Math.min(sourceWidth - 1, Math.round(x * scaleX)));
    const top = Math.max(0, Math.min(sourceHeight - 1, Math.round(y * scaleY)));
    return [
      left,
      top,
      Math.max(1, Math.min(sourceWidth - left, Math.round(width * scaleX))),
      Math.max(1, Math.min(sourceHeight - top, Math.round(height * scaleY))),
    ];
  }

  function scaleDetections(detections, payload) {
    return detections.map(item => ({
      ...item,
      box: scaleBox(item.box, payload.scaleX, payload.scaleY, payload.sourceWidth, payload.sourceHeight),
      head_box: scaleBox(item.head_box || item.box, payload.scaleX, payload.scaleY, payload.sourceWidth, payload.sourceHeight),
      detection_box: item.box,
      detection_head_box: item.head_box || item.box,
      ...(item.face_landmarks ? { face_landmarks: scaleLandmarks(item.face_landmarks, payload) } : {}),
      ...(item.body_landmarks ? { body_landmarks: scaleLandmarks(item.body_landmarks, payload) } : {}),
    }));
  }

  function scaleLandmarks(landmarks, payload) {
    if (!landmarks || typeof landmarks !== 'object') return {};
    return Object.fromEntries(Object.entries(landmarks).map(([key, point]) => [
      key,
      [
        Math.max(0, Math.min(payload.sourceWidth - 1, Math.round(point[0] * payload.scaleX))),
        Math.max(0, Math.min(payload.sourceHeight - 1, Math.round(point[1] * payload.scaleY))),
      ],
    ]));
  }

  async function fetchWithTimeout(url, options = {}, timeoutMs = DETECTION_TIMEOUT_MS) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      return await fetch(url, { ...options, signal: controller.signal });
    } finally {
      clearTimeout(timer);
    }
  }

  async function responseMessage(response, fallback) {
    const text = await response.text();
    try {
      const parsed = JSON.parse(text);
      return parsed.detail || parsed.message || fallback;
    } catch {
      return text || fallback;
    }
  }

  async function detect(image, sourceFile, onStage = () => {}) {
    onStage('compressing');
    const payload = await createDetectionFile(image, sourceFile?.name);
    onStage('uploading', payload);
    const body = new FormData();
    body.append('file', payload.file);
    const response = await fetchWithTimeout('/api/detect', { method: 'POST', body });
    if (!response.ok) throw new Error(await responseMessage(response, '人物检查失败，请重试。'));
    onStage('mapping', payload);
    const result = await response.json();
    result.detections = scaleDetections(result.detections || [], payload);
    result.source_width = payload.sourceWidth;
    result.source_height = payload.sourceHeight;
    result.client_detection_width = payload.detectionWidth;
    result.client_detection_height = payload.detectionHeight;
    return result;
  }

  async function getMasterStatus(imageId) {
    const response = await fetchWithTimeout(`/api/images/${encodeURIComponent(imageId)}/master`, {}, 15000);
    if (!response.ok) throw new MasterUploadError('status_unavailable', '暂时无法确认高清照片状态，请检查网络后重试。');
    return response.json();
  }

  async function waitForMaster(imageId, onStage, timeoutMs = MASTER_RECOVERY_TIMEOUT_MS) {
    const started = performance.now();
    while (performance.now() - started < timeoutMs) {
      const state = await getMasterStatus(imageId);
      if (state.status === 'ready') return state;
      if (state.status === 'failed' || state.status === 'missing') return state;
      onStage('server-processing', { status: state.status });
      await new Promise(resolve => setTimeout(resolve, 1200));
    }
    throw new MasterUploadError('processing_timeout', '照片已经上传，但服务器准备高清副本时间过长。请稍后安全重试，系统会先查询已有结果。');
  }

  function uploadMasterRequest(imageId, file, onStage, signal) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      const started = performance.now();
      let uploadedAt = null;
      let terminalReason = null;
      const metrics = {
        format: file.type || 'unknown', bytes: file.size,
        width: null, height: null, pixels: null,
        request_started_ms: Math.round(performance.timeOrigin + started),
      };
      const rejectWith = (code, message) => {
        metrics.total_ms = Math.round(performance.now() - started);
        metrics.terminal_reason = code;
        reject(new MasterUploadError(code, message, metrics));
      };
      xhr.open('POST', `/api/images/${encodeURIComponent(imageId)}/master`);
      xhr.timeout = MASTER_UPLOAD_TIMEOUT_MS;
      xhr.upload.addEventListener('loadstart', () => onStage('reading', metrics));
      xhr.upload.addEventListener('progress', event => {
        if (!event.lengthComputable) return onStage('uploading', { ...metrics, percent: null });
        onStage('uploading', { ...metrics, percent: Math.min(100, Math.round(event.loaded / event.total * 100)) });
      });
      xhr.upload.addEventListener('load', () => {
        uploadedAt = performance.now();
        metrics.upload_ms = Math.round(uploadedAt - started);
        onStage('server-processing', metrics);
      });
      xhr.addEventListener('load', () => {
        metrics.total_ms = Math.round(performance.now() - started);
        metrics.server_wait_ms = Math.max(0, metrics.total_ms - (metrics.upload_ms || metrics.total_ms));
        let result = {};
        try { result = JSON.parse(xhr.responseText || '{}'); } catch {}
        if (xhr.status >= 200 && xhr.status < 300) return resolve({ ...result, client_upload_timings: metrics });
        const detail = result.detail || result.message;
        if (xhr.status === 413) return rejectWith(detail?.includes('像素') ? 'pixel_limit' : 'file_too_large', detail || '照片尺寸超过当前安全上限。');
        if (xhr.status === 415) return rejectWith('unsupported_format', detail || '无法读取这种照片格式。');
        if (xhr.status >= 500) return rejectWith('server_processing', detail || '服务器准备高清副本失败，请安全重试。');
        rejectWith('request_rejected', detail || '原尺寸照片上传失败，请重试。');
      });
      xhr.addEventListener('timeout', () => rejectWith('upload_timeout', '上传或准备高清副本超过4分钟。系统将先查询服务端状态，不会直接重复上传。'));
      xhr.addEventListener('error', () => rejectWith(navigator.onLine === false ? 'network_offline' : 'connection_interrupted', navigator.onLine === false ? '手机当前处于离线状态，请恢复网络后重试。' : '上传连接意外断开，请检查网络后安全重试。'));
      xhr.addEventListener('abort', () => rejectWith(terminalReason || 'user_cancelled', terminalReason === 'user_cancelled' ? '已取消上传。' : '上传已中止。'));
      if (signal) {
        if (signal.aborted) { terminalReason = 'user_cancelled'; return xhr.abort(); }
        signal.addEventListener('abort', () => { terminalReason = 'user_cancelled'; xhr.abort(); }, { once: true });
      }
      const body = new FormData();
      const extension = (file.type || '').includes('heic') || (file.type || '').includes('heif') ? 'heic' : (file.type || '').includes('png') ? 'png' : 'jpg';
      body.append('file', file, `master.${extension}`);
      xhr.send(body);
    });
  }

  async function uploadMaster(imageId, file, onStage = () => {}, signal = null) {
    const existing = await getMasterStatus(imageId);
    if (existing.status === 'ready') {
      onStage('ready', { reused: true });
      return { ...existing, reused: true };
    }
    if (existing.status === 'uploading' || existing.status === 'processing') {
      const recovered = await waitForMaster(imageId, onStage);
      if (recovered.status === 'ready') return { ...recovered, reused: true };
    }
    try {
      const result = await uploadMasterRequest(imageId, file, onStage, signal);
      onStage('ready', result.client_upload_timings || {});
      return result;
    } catch (error) {
      if (!['upload_timeout', 'connection_interrupted'].includes(error.code)) throw error;
      try {
        const recovered = await waitForMaster(imageId, onStage);
        if (recovered.status === 'ready') return { ...recovered, reused: true, recovered_after: error.code };
      } catch {}
      throw error;
    }
  }

  window.SceneFirstDetection = {
    detect,
    uploadMaster,
    getMasterStatus,
    fetchWithTimeout,
    testing: { createDetectionFile, scaleBox, scaleDetections, uploadMasterRequest },
    constants: { MAX_DETECTION_EDGE, DETECTION_TIMEOUT_MS, MASTER_UPLOAD_TIMEOUT_MS, MASTER_RECOVERY_TIMEOUT_MS },
  };
})();
