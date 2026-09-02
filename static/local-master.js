(function () {
  'use strict';

  const CROP_PADDING = 0.42;
  const MAX_CROP_EDGE = 2048;
  const MAX_PARALLEL = 2;

  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

  function clampBox(box, width, height) {
    let [x, y, w, h] = box.map(value => Math.round(value));
    x = clamp(x, 0, width - 1); y = clamp(y, 0, height - 1);
    w = clamp(w, 1, width - x); h = clamp(h, 1, height - y);
    return [x, y, w, h];
  }

  function paddedCropBox(headBox, width, height, padding = CROP_PADDING) {
    const [x, y, w, h] = headBox;
    return clampBox([x - w * padding, y - h * padding, w * (1 + 2 * padding), h * (1 + 2 * padding)], width, height);
  }

  function canvasBlob(canvas, type = 'image/jpeg', quality = 0.94) {
    return new Promise((resolve, reject) => canvas.toBlob(
      blob => blob ? resolve(blob) : reject(new Error('无法在本机创建人物局部。')), type, quality,
    ));
  }

  function sourceCanvas(image) {
    const canvas = document.createElement('canvas');
    canvas.width = image.naturalWidth || image.width;
    canvas.height = image.naturalHeight || image.height;
    canvas.getContext('2d', { alpha: false }).drawImage(image, 0, 0);
    return canvas;
  }

  function buildMask(width, height, headBoxInCrop) {
    const mask = document.createElement('canvas'); mask.width = width; mask.height = height;
    const context = mask.getContext('2d');
    const [x, y, w, h] = headBoxInCrop;
    context.fillStyle = '#fff';
    context.beginPath(); context.ellipse(x + w / 2, y + h * .415, w / 2, h * .415, 0, 0, Math.PI * 2); context.fill();
    const neckW = w * .32, neckX = x + (w - neckW) / 2, neckY = y + h * .58;
    context.beginPath(); context.roundRect(neckX, neckY, neckW, h * .42, Math.max(2, w * .08)); context.fill();
    const radius = Math.max(2, Math.min(18, Math.floor(Math.min(w, h) * .035)));
    if (radius > 0) {
      const blurred = document.createElement('canvas'); blurred.width = width; blurred.height = height;
      const blurredContext = blurred.getContext('2d');
      blurredContext.filter = `blur(${radius}px)`; blurredContext.drawImage(mask, 0, 0);
      const pixels = blurredContext.getImageData(0, 0, width, height);
      for (let index = 3; index < pixels.data.length; index += 4) {
        const alpha = pixels.data[index];
        pixels.data[index - 3] = pixels.data[index - 2] = pixels.data[index - 1] = 255;
        pixels.data[index] = alpha < 3 ? 0 : alpha;
      }
      blurredContext.putImageData(pixels, 0, 0);
      return blurred;
    }
    return mask;
  }

  async function prepareCrop(master, region, imageId, retryNonce) {
    const headBox = clampBox(region.head_box || region.box, master.width, master.height);
    const cropBox = paddedCropBox(headBox, master.width, master.height);
    const [x, y, width, height] = cropBox;
    const scale = Math.min(1, MAX_CROP_EDGE / Math.max(width, height));
    const uploadWidth = Math.max(1, Math.round(width * scale));
    const uploadHeight = Math.max(1, Math.round(height * scale));
    const crop = document.createElement('canvas'); crop.width = uploadWidth; crop.height = uploadHeight;
    crop.getContext('2d', { alpha: false }).drawImage(master, x, y, width, height, 0, 0, uploadWidth, uploadHeight);
    const headBoxInCrop = [headBox[0] - x, headBox[1] - y, headBox[2], headBox[3]];
    const uploadHead = headBoxInCrop.map(value => Math.round(value * scale));
    const blob = await canvasBlob(crop);
    return {
      personId: region.id, region: structuredClone(region), blob, cropBox, headBox,
      headBoxInCrop, uploadHeadBoxInCrop: uploadHead, uploadSize: [uploadWidth, uploadHeight],
      metadata: {
        image_id: imageId, person_id: region.id, original_size: [master.width, master.height],
        head_box: headBox, crop_box: cropBox, head_box_in_crop: uploadHead,
        upload_scale: scale, retry_nonce: retryNonce, prompt_profile: 'balanced_portrait',
        source: region.source || 'automatic', selected: true,
      },
    };
  }

  async function createPersonJob(item, provider) {
    const body = new FormData();
    body.append('crop', item.blob, 'person-crop.jpg');
    body.append('metadata', JSON.stringify(item.metadata));
    body.append('provider', provider);
    const response = await fetch('/api/local-person-jobs', { method: 'POST', body });
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(detail.detail || '人物局部任务创建失败。');
    }
    return response.json();
  }

  async function pollPersonJob(jobId) {
    for (;;) {
      const response = await fetch(`/api/local-person-jobs/${encodeURIComponent(jobId)}`);
      if (!response.ok) throw new Error('人物局部任务状态暂时不可用。');
      const job = await response.json();
      if (job.status === 'completed' || job.status === 'failed') return job;
      await new Promise(resolve => setTimeout(resolve, 850));
    }
  }

  function loadImage(url) {
    return new Promise((resolve, reject) => {
      const image = new Image(); image.onload = () => resolve(image); image.onerror = reject;
      image.src = `${url}${url.includes('?') ? '&' : '?'}t=${Date.now()}`;
    });
  }

  function compositePatch(working, patch, item) {
    const [x, y, width, height] = item.cropBox;
    const localMask = buildMask(width, height, item.headBoxInCrop);
    const layer = document.createElement('canvas'); layer.width = width; layer.height = height;
    const context = layer.getContext('2d');
    context.drawImage(patch, 0, 0, width, height);
    context.globalCompositeOperation = 'destination-in'; context.drawImage(localMask, 0, 0);
    working.getContext('2d').drawImage(layer, x, y);
    return localMask;
  }

  function depthScore(item) {
    const [x, y, width, height] = item.headBox;
    // In ordinary perspective, a lower and larger head is more likely to be
    // in front. This only makes concurrent completion deterministic; semantic
    // instance masks are still needed for genuinely interleaved occlusion.
    return y + height + Math.sqrt(width * height) * .25 + x * 1e-6;
  }

  function safeFallback(working, master, item) {
    const [x, y, width, height] = item.cropBox;
    const localMask = buildMask(width, height, item.headBoxInCrop);
    const source = document.createElement('canvas'); source.width = width; source.height = height;
    const sourceContext = source.getContext('2d');
    sourceContext.drawImage(master, x, y, width, height, 0, 0, width, height);
    sourceContext.filter = `blur(${Math.max(8, Math.min(38, Math.round(Math.min(width, height) * .12)))}px)`;
    sourceContext.drawImage(source, 0, 0);
    sourceContext.globalCompositeOperation = 'destination-in'; sourceContext.drawImage(localMask, 0, 0);
    working.getContext('2d').drawImage(source, x, y);
    return localMask;
  }

  async function run(options) {
    const runStarted = performance.now();
    const sourceStarted = performance.now();
    const master = sourceCanvas(options.image);
    const working = sourceCanvas(options.image);
    const sourceCanvasMs = Math.round(performance.now() - sourceStarted);
    const selected = options.regions.filter(region => region.selected).map(region => structuredClone(region));
    // Large/near people produce the most useful first visible result.
    selected.sort((a, b) => ((b.head_box || b.box)[2] * (b.head_box || b.box)[3]) - ((a.head_box || a.box)[2] * (a.head_box || a.box)[3]));
    const preparationStarted = performance.now();
    const prepared = [];
    for (const region of selected) prepared.push(await prepareCrop(master, region, options.imageId, options.retryNonce || 0));
    const metrics = {
      original_bytes_uploaded: 0,
      crop_bytes_uploaded: prepared.reduce((sum, item) => sum + item.blob.size, 0),
      source_canvas_ms: sourceCanvasMs,
      crop_prepare_ms: Math.round(performance.now() - preparationStarted),
      people: [],
    };
    const generationStarted = performance.now();
    let cursor = 0, completed = 0;
    const rendered = [];
    const worker = async () => {
      while (cursor < prepared.length) {
        const item = prepared[cursor++];
        const started = performance.now();
        let job = null, actualMode = 'anime', fallbackReason = null;
        try {
          if (options.mode === 'safe') throw new Error('用户选择安全遮挡。');
          const created = await createPersonJob(item, options.provider);
          job = await pollPersonJob(created.job_id);
          if (job.status !== 'completed') throw new Error(job.error || '生成人物局部失败。');
          const downloadStarted = performance.now();
          const patch = await loadImage(job.output_url);
          const downloadMs = Math.round(performance.now() - downloadStarted);
          const compositeStarted = performance.now();
          compositePatch(working, patch, item);
          rendered.push({ item, patch, fallback: false });
          metrics.people.push({ person_id: item.personId, queue_ms: job.queue_ms, provider_ms: job.provider_ms, download_decode_ms: downloadMs, composite_ms: Math.round(performance.now() - compositeStarted), fallback: false });
        } catch (error) {
          actualMode = 'safe'; fallbackReason = error.message;
          const compositeStarted = performance.now();
          safeFallback(working, master, item);
          rendered.push({ item, patch: null, fallback: true });
          metrics.people.push({ person_id: item.personId, composite_ms: Math.round(performance.now() - compositeStarted), fallback: true, error: error.message });
        }
        completed += 1;
        options.onProgress?.({ personId: item.personId, actualMode, fallbackReason, completed, total: prepared.length, canvas: working, metrics });
      }
    };
    await Promise.all(Array.from({ length: Math.min(MAX_PARALLEL, Math.max(1, prepared.length)) }, worker));
    metrics.generation_and_composite_ms = Math.round(performance.now() - generationStarted);
    const recomposeStarted = performance.now();
    const workingContext = working.getContext('2d'); workingContext.clearRect(0, 0, working.width, working.height); workingContext.drawImage(master, 0, 0);
    const finalOrder = rendered.sort((a, b) => depthScore(a.item) - depthScore(b.item));
    for (const value of finalOrder) {
      if (value.fallback) safeFallback(working, master, value.item);
      else compositePatch(working, value.patch, value.item);
    }
    metrics.final_recompose_ms = Math.round(performance.now() - recomposeStarted);
    metrics.final_composite_order = finalOrder.map(value => value.item.personId);
    metrics.total_run_ms = Math.round(performance.now() - runStarted);
    return { canvas: working, sourceCanvas: master, metrics, people: metrics.people, completed, total: prepared.length };
  }

  async function canvasToBlob(canvas, format = 'png', quality = .94) {
    return canvasBlob(canvas, format === 'jpeg' ? 'image/jpeg' : 'image/png', quality);
  }

  window.SceneFirstLocalMaster = {
    run, canvasToBlob,
    testing: { clampBox, paddedCropBox, buildMask, prepareCrop, sourceCanvas, compositePatch, safeFallback, depthScore },
    constants: { CROP_PADDING, MAX_CROP_EDGE, MAX_PARALLEL },
  };
})();
