const state = {
  rawFile: null, rawImage: null, croppedFile: null, imageId: null, imageUrl: null, image: null,
  width: 0, height: 0, detections: [], manual: false, adjusting: false, adjustTarget: null,
  dragStart: null, dragCurrent: null, dragMoved: false, editPointerId: null, retryNonce: 0, jobId: null,
  phase: 'empty', resultImage: null, resultMasterImage: null, beforeResultImage: null, exportCrop: null, completedJob: null,
  crop: null, safeCoverId: 'bald-bearded', masterUploaded: false,
  localResultCanvas: null, localSourceCanvas: null, localObjectUrls: [], processingPreview: null,
  viewer: { zoom: 1, panX: 0, panY: 0, pointers: new Map(), drag: null, pinch: null },
};
let appFeatures = { local_master: false, traditional_master_fallback: true, max_parallel_people: 2 };

const $ = (id) => document.getElementById(id);
const canvas = $('editorCanvas');
const ctx = canvas.getContext('2d');
const cropCanvas = $('cropCanvas');
const cropCtx = cropCanvas.getContext('2d');
const ratios = { free: null, original: 'original', '1': 1, '0.8': .8, '0.75': .75, '0.5625': .5625, '1.7777778': 1.7777778 };
const labParameter = new URLSearchParams(location.search).get('lab');
const labMode = labParameter === '1' || (labParameter !== '0' && ['127.0.0.1', 'localhost'].includes(location.hostname));
document.body.classList.toggle('lab-mode', labMode);

let deferredInstallPrompt = null;
const installedStandalone = window.matchMedia('(display-mode: standalone)').matches || window.navigator.standalone === true;
const isIOSBrowser = /iPad|iPhone|iPod/.test(navigator.userAgent) && !window.MSStream;
const installButton = $('installApp');

window.addEventListener('beforeinstallprompt', event => {
  event.preventDefault(); deferredInstallPrompt = event;
  if (!installedStandalone) installButton.classList.remove('hidden');
});
window.addEventListener('appinstalled', () => { deferredInstallPrompt = null; installButton.classList.add('hidden'); });
if (isIOSBrowser && !installedStandalone) {
  installButton.textContent = '添加到主屏幕';
  installButton.classList.remove('hidden');
}
installButton.onclick = async () => {
  if (deferredInstallPrompt) {
    deferredInstallPrompt.prompt();
    await deferredInstallPrompt.userChoice; deferredInstallPrompt = null; installButton.classList.add('hidden');
    return;
  }
  alert('在 Safari 中点“分享”，再选择“添加到主屏幕”。');
};

const embeddedBrowser = /Instagram|FBAN|FBAV|MicroMessenger|XiaoHongShu|XHS\//i.test(navigator.userAgent);
if (embeddedBrowser && sessionStorage.getItem('scene-first-browser-notice') !== 'dismissed') $('browserNotice').classList.remove('hidden');
$('dismissBrowserNotice').onclick = () => { sessionStorage.setItem('scene-first-browser-notice', 'dismissed'); $('browserNotice').classList.add('hidden'); };

if ('serviceWorker' in navigator && (location.protocol === 'https:' || location.hostname === '127.0.0.1' || location.hostname === 'localhost')) {
  navigator.serviceWorker?.register('/sw.js').catch(() => {});
}

function setStep(step, enabled) {
  const element = document.querySelector(`[data-step="${step}"]`);
  element.classList.toggle('disabled-step', !enabled);
}

function cloudAllowed() { return document.querySelector('input[name=mode]:checked')?.value === 'safe' || $('provider').value === 'local' || $('cloudConsentInput').checked; }

function selectedCount() { return state.detections.filter(item => item.selected).length; }

function updateCount() {
  if (state.phase === 'empty') $('countLabel').textContent = '尚未检查路人';
  else if (state.phase === 'ready') $('countLabel').textContent = '等待检查路人';
  else $('countLabel').textContent = `${selectedCount()} 人将被保护 · 共 ${state.detections.length} 个区域`;
  $('processButton').disabled = state.phase !== 'detected' || selectedCount() === 0 || !cloudAllowed();
  refreshMobileDock();
}

function refreshMobileDock() {
  const primary = $('mobilePrimary'), secondary = $('mobileSecondary');
  const set = (primaryText, primaryDisabled, primaryAction, secondaryText, secondaryDisabled, secondaryAction) => {
    primary.innerHTML = `${primaryText} <span>→</span>`; primary.disabled = primaryDisabled; primary.dataset.action = primaryAction;
    secondary.textContent = secondaryText; secondary.disabled = secondaryDisabled; secondary.dataset.action = secondaryAction;
  };
  if (state.phase === 'empty') set('选择照片', true, '', '调整构图', true, '');
  else if (state.phase === 'ready') set('开始检查路人', false, 'detect', '调整构图', false, 'pre-crop');
  else if (state.phase === 'detected') set('确认并开始保护', selectedCount() === 0 || !cloudAllowed(), 'process', '重新裁剪', false, 'pre-crop');
  else if (state.phase === 'completed') set('保存 PNG', false, 'png', '调整构图', false, 'result-crop');
  else set('处理中…', true, '', '调整构图', true, '');
}

async function loadProviders() {
  const response = await fetch('/api/providers');
  const { providers } = await response.json();
  const preferred = providers.find(item => item.id === 'ark' && item.configured) || providers.find(item => item.id === 'local');
  $('provider').innerHTML = providers.map(p => `<option value="${p.id}" ${p.configured ? '' : 'disabled'} ${p.id === preferred?.id ? 'selected' : ''}>${p.label}${p.configured ? '' : ' · 未配置'}</option>`).join('');
  updateProviderConsent();
}

async function loadFeatures() {
  try {
    const response = await fetch('/api/features');
    if (response.ok) appFeatures = await response.json();
  } catch {}
}

function updateProviderConsent() {
  const safe = document.querySelector('input[name=mode]:checked')?.value === 'safe';
  const cloud = !safe && $('provider').value !== 'local';
  $('providerField').classList.toggle('hidden', safe);
  $('cloudConsent').classList.toggle('hidden', !cloud);
  updateCount();
}

function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }

function resetViewer() {
  Object.assign(state.viewer, { zoom: 1, panX: 0, panY: 0, pointers: new Map(), drag: null, pinch: null });
  $('viewerReset').classList.add('hidden');
}

function constrainViewer(baseScale) {
  const stage = $('stage');
  const displayWidth = state.width * baseScale * state.viewer.zoom;
  const displayHeight = state.height * baseScale * state.viewer.zoom;
  const maxX = Math.max(0, (displayWidth - stage.clientWidth) / 2);
  const maxY = Math.max(0, (displayHeight - stage.clientHeight) / 2);
  state.viewer.panX = clamp(state.viewer.panX, -maxX, maxX);
  state.viewer.panY = clamp(state.viewer.panY, -maxY, maxY);
}

function draw() {
  if (!state.image) return;
  const stage = $('stage');
  const maxWidth = stage.clientWidth - 24;
  const maxHeight = Math.min(690, Math.max(260, window.innerHeight * .62));
  const baseScale = Math.min(maxWidth / state.width, maxHeight / state.height, 1);
  constrainViewer(baseScale);
  const scale = baseScale * state.viewer.zoom;
  canvas.width = Math.max(1, Math.round(state.width * scale)); canvas.height = Math.max(1, Math.round(state.height * scale)); canvas.dataset.scale = scale; canvas.dataset.zoom = state.viewer.zoom;
  canvas.style.width = `${canvas.width}px`; canvas.style.height = `${canvas.height}px`;
  canvas.style.transform = `translate(calc(-50% + ${state.viewer.panX}px), calc(-50% + ${state.viewer.panY}px))`;
  ctx.fillStyle = '#07111e'; ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(state.processingPreview || state.image, 0, 0, canvas.width, canvas.height);
  if (state.phase !== 'detected' && state.phase !== 'processing') return;
  state.detections.forEach(item => {
    const replacing = state.adjusting && state.adjustTarget === item && state.dragMoved;
    if (!replacing) drawRegion(item, scale, state.adjusting && state.adjustTarget === item);
  });
  drawBoxDraft();
  $('viewerReset').classList.toggle('hidden', state.viewer.zoom <= 1.01 && Math.abs(state.viewer.panX) < 1 && Math.abs(state.viewer.panY) < 1);
}

function drawRegion(item, scale, active = false) {
  const [x, y, w, h] = (item.head_box || item.box).map(value => value * scale);
  ctx.save(); ctx.lineWidth = active ? 4 : 2.4; ctx.strokeStyle = active ? '#72e7ff' : item.selected ? '#cbff46' : 'rgba(255,255,255,.9)'; ctx.fillStyle = active ? 'rgba(114,231,255,.22)' : item.selected ? 'rgba(203,255,70,.14)' : 'rgba(20,25,22,.2)';
  ctx.setLineDash(item.source === 'manual' ? [8, 5] : []); ctx.beginPath(); ctx.roundRect(x, y, w, h, Math.min(14, w / 6)); ctx.fill(); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = item.selected ? '#17201c' : '#fff'; ctx.beginPath(); ctx.arc(x + 13, y + 13, 11, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = item.selected ? '#cbff46' : '#17201c'; ctx.font = '700 10px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(item.selected ? '✓' : '—', x + 13, y + 13); ctx.restore();
}

function normalizedDraftBox(start, end) {
  const x = Math.min(start.x, end.x), y = Math.min(start.y, end.y);
  return { x, y, w: Math.abs(end.x - start.x), h: Math.abs(end.y - start.y) };
}

function drawBoxDraft() {
  if (!state.dragStart || !state.dragCurrent || !state.dragMoved) return;
  const { x, y, w, h } = normalizedDraftBox(state.dragStart, state.dragCurrent);
  ctx.save(); ctx.fillStyle = 'rgba(114,231,255,.18)'; ctx.strokeStyle = '#72e7ff'; ctx.lineWidth = 3; ctx.setLineDash([9, 5]);
  ctx.beginPath(); ctx.roundRect(x, y, w, h, Math.min(14, w / 6)); ctx.fill(); ctx.stroke(); ctx.restore();
}

function canvasPoint(event, target = canvas) { const rect = target.getBoundingClientRect(); return { x: event.clientX - rect.left, y: event.clientY - rect.top }; }

async function imageFromFile(file) {
  const url = URL.createObjectURL(file);
  const image = await new Promise((resolve, reject) => { const item = new Image(); item.onload = () => resolve(item); item.onerror = reject; item.src = url; });
  URL.revokeObjectURL(url); return image;
}

async function handleFile(file) {
  if (!file) return;
  try {
    $('jobStatus').textContent = '正在准备无元数据工作副本…';
    const image = await imageFromFile(file);
    revokeLocalUrls(); Object.assign(state, { rawFile: file, rawImage: image, croppedFile: null, image: image, width: image.naturalWidth, height: image.naturalHeight, imageId: null, imageUrl: null, detections: [], retryNonce: 0, jobId: null, resultImage: null, resultMasterImage: null, beforeResultImage: null, exportCrop: null, masterUploaded: false, localResultCanvas: null, localSourceCanvas: null, processingPreview: null, completedJob: null, phase: 'ready' }); resetViewer();
    document.body.classList.add('has-image'); $('emptyState').classList.add('hidden'); canvas.classList.remove('hidden'); $('fileLabel').textContent = file.name; $('uploadBox').classList.add('has-file'); $('replacePhoto').classList.remove('hidden'); $('openPreCrop').disabled = false; $('detectButton').disabled = false;
    $('selectionHint').textContent = '开始检查后，所有检测到的人会默认受到保护。'; $('cropHint').textContent = '建议先裁剪：不需要展示的人，也不会进入自动检测范围。'; $('stageNote').textContent = '当前是原图预览。你可以直接检查路人，或先点击“调整构图”。';
    setStep(1, true); $('reviewControls').classList.add('hidden'); $('resultSection').classList.add('hidden'); draw(); updateCount(); $('jobStatus').textContent = '';
  } catch { showError('无法读取这张图片；请使用 JPEG、PNG、WebP 或 HEIC。'); }
}

async function detectPhoto() {
  if (!state.rawFile) return;
  state.phase = 'detecting'; updateCount(); $('detectButton').disabled = true; $('openPreCrop').disabled = true; $('jobStatus').textContent = '正在创建无元数据工作副本并检查画面中的人物…';
  const file = state.croppedFile || state.rawFile;
  try {
    const result = await SceneFirstDetection.detect(state.image, file, stage => {
      if (stage === 'compressing') $('jobStatus').textContent = '正在本机创建轻量检查副本…';
      if (stage === 'uploading') $('jobStatus').textContent = '正在上传检查副本并识别人头…';
      if (stage === 'mapping') $('jobStatus').textContent = '正在把检测结果对齐到原图…';
    });
    Object.assign(state, { imageId: result.image_id, imageUrl: null, width: result.source_width, height: result.source_height, detections: result.detections, retryNonce: 0, masterUploaded: false, phase: 'detected' });
    resetViewer(); draw();
    $('fileLabel').textContent = file.name; $('selectionHint').textContent = result.warnings.join(' '); ['selectAll', 'clearAll', 'manualBox', 'adjustBox'].forEach(id => $(id).disabled = false);
    $('cropHint').textContent = '如需重新裁剪，请在开始保护前操作；系统会重新检查裁后画面。'; $('stageNote').textContent = '默认已保护全部检测结果。点击人物框可恢复自己或朋友；远处、侧脸或反射请放大检查。';
    $('reviewControls').classList.remove('hidden'); setStep(2, true); setStep(3, true); $('jobStatus').textContent = result.detections.length ? `已找出 ${result.detections.length} 个疑似人物。确认例外后即可开始保护。` : '没有自动找到人脸；如果画面中有人，请手动补框。'; updateCount();
  } catch (error) {
    state.phase = 'ready';
    const message = error?.name === 'AbortError' ? '人物检查超过45秒，服务可能正在启动或网络不稳定。请点击重试；照片仍保留在本机。' : (error?.message || '人物检查连接中断，请重试。');
    showError(message);
  } finally {
    if (state.phase === 'ready') { $('detectButton').disabled = false; $('openPreCrop').disabled = false; }
    updateCount();
  }
}

function toggleAt(point) {
  if (state.phase !== 'detected') return;
  const scale = Number(canvas.dataset.scale || 1), original = { x: point.x / scale, y: point.y / scale };
  const hits = state.detections.filter(item => { const [x, y, w, h] = item.head_box || item.box; return original.x >= x && original.x <= x + w && original.y >= y && original.y <= y + h; });
  if (hits.length) hits.at(-1).selected = !hits.at(-1).selected;
  draw(); updateCount();
}

function viewerDistance(pair) { return Math.hypot(pair[0].clientX - pair[1].clientX, pair[0].clientY - pair[1].clientY); }

canvas.addEventListener('pointerdown', event => {
  if (!state.image) return;
  event.preventDefault(); canvas.setPointerCapture(event.pointerId);
  const point = canvasPoint(event);
  if (state.manual && state.phase === 'detected') {
    Object.assign(state, { dragStart: point, dragCurrent: point, dragMoved: false, editPointerId: event.pointerId });
    $('stageNote').textContent = '拖动时蓝色框会实时跟随；松开即完成补框。'; draw(); return;
  }
  if (state.adjusting && state.phase === 'detected') {
    const scale = Number(canvas.dataset.scale || 1), original = { x: point.x / scale, y: point.y / scale };
    const hits = state.detections.filter(item => { const [x, y, w, h] = item.head_box || item.box; return original.x >= x && original.x <= x + w && original.y >= y && original.y <= y + h; });
    state.adjustTarget = hits.sort((a, b) => { const aa = (a.head_box || a.box), bb = (b.head_box || b.box); return aa[2] * aa[3] - bb[2] * bb[3]; })[0] || null;
    if (state.adjustTarget) {
      Object.assign(state, { dragStart: point, dragCurrent: point, dragMoved: false, editPointerId: event.pointerId });
      $('stageNote').textContent = '已高亮原框。开始拖动后原框会隐藏，蓝色新框实时跟随；松开即完成。'; draw();
    } else $('stageNote').textContent = '没有点中人物框，请放大后再点住需要调整的框。';
    return;
  }
  state.viewer.pointers.set(event.pointerId, { clientX: event.clientX, clientY: event.clientY });
  if (state.viewer.pointers.size === 2) {
    const pair = [...state.viewer.pointers.values()];
    state.viewer.pinch = { distance: viewerDistance(pair), zoom: state.viewer.zoom, panX: state.viewer.panX, panY: state.viewer.panY };
    state.viewer.drag = null;
  } else state.viewer.drag = { clientX: event.clientX, clientY: event.clientY, panX: state.viewer.panX, panY: state.viewer.panY, moved: false };
});

canvas.addEventListener('pointermove', event => {
  if (!state.image) return;
  if ((state.manual || state.adjusting) && state.dragStart && state.editPointerId === event.pointerId) {
    event.preventDefault(); state.dragCurrent = canvasPoint(event);
    state.dragMoved = Math.hypot(state.dragCurrent.x - state.dragStart.x, state.dragCurrent.y - state.dragStart.y) > 3;
    draw(); return;
  }
  if (!state.viewer.pointers.has(event.pointerId)) return;
  event.preventDefault(); state.viewer.pointers.set(event.pointerId, { clientX: event.clientX, clientY: event.clientY });
  if (state.viewer.pointers.size === 2 && state.viewer.pinch) {
    const scale = viewerDistance([...state.viewer.pointers.values()]) / state.viewer.pinch.distance;
    state.viewer.zoom = clamp(state.viewer.pinch.zoom * scale, 1, 4);
    state.viewer.panX = state.viewer.pinch.panX; state.viewer.panY = state.viewer.pinch.panY; draw(); return;
  }
  const drag = state.viewer.drag;
  if (!drag) return;
  const dx = event.clientX - drag.clientX, dy = event.clientY - drag.clientY;
  drag.moved ||= Math.hypot(dx, dy) > 5;
  if (state.viewer.zoom > 1.01) { state.viewer.panX = drag.panX + dx; state.viewer.panY = drag.panY + dy; draw(); }
});

function finishCanvasPointer(event) {
  if (!state.image) return;
  if (state.manual && state.dragStart && state.editPointerId === event.pointerId) {
    const end = canvasPoint(event), scale = Number(canvas.dataset.scale || 1), draft = normalizedDraftBox(state.dragStart, end), x = draft.x / scale, y = draft.y / scale, w = draft.w / scale, h = draft.h / scale;
    if (w > 12 && h > 12) state.detections.push({ id: `manual-${crypto.randomUUID().slice(0, 8)}`, box: [Math.round(x), Math.round(y), Math.round(w), Math.round(h)], head_box: [Math.round(x), Math.round(y), Math.round(w), Math.round(h)], confidence: 1, source: 'manual', selected: true });
    Object.assign(state, { dragStart: null, dragCurrent: null, dragMoved: false, editPointerId: null, manual: false }); canvas.classList.remove('manual'); $('manualBox').textContent = '＋ 补一个漏检头部'; $('stageNote').textContent = w > 12 && h > 12 ? '补框已加入并默认保护。仍可点击人物框取消保护。' : '框选范围太小，未添加；需要时请重新补框。'; draw(); updateCount(); return;
  }
  if (state.adjusting && state.dragStart && state.adjustTarget && state.editPointerId === event.pointerId) {
    const end = canvasPoint(event), scale = Number(canvas.dataset.scale || 1), draft = normalizedDraftBox(state.dragStart, end), x = draft.x / scale, y = draft.y / scale, w = draft.w / scale, h = draft.h / scale;
    if (w > 12 && h > 12) {
      const finalBox = [Math.round(x), Math.round(y), Math.round(w), Math.round(h)];
      state.adjustTarget.box = [...finalBox]; state.adjustTarget.head_box = [...finalBox]; state.adjustTarget.adjusted = true;
    }
    Object.assign(state, { dragStart: null, dragCurrent: null, dragMoved: false, editPointerId: null, adjustTarget: null, adjusting: false }); canvas.classList.remove('manual'); $('adjustBox').textContent = '调整一个人物框'; $('stageNote').textContent = w > 12 && h > 12 ? '人物框已更新。请确认范围覆盖头发至颈部。' : '新范围太小，保留原框；需要时请重新调整。'; draw(); updateCount(); return;
  }
  const drag = state.viewer.drag; state.viewer.pointers.delete(event.pointerId);
  if (state.viewer.pointers.size < 2) state.viewer.pinch = null;
  if (drag && !drag.moved && state.viewer.pointers.size === 0) toggleAt(canvasPoint(event));
  state.viewer.drag = null;
}
function cancelCanvasPointer(event) {
  if (state.editPointerId === event.pointerId) {
    Object.assign(state, { dragStart: null, dragCurrent: null, dragMoved: false, editPointerId: null, adjustTarget: null });
    $('stageNote').textContent = '操作被中断，原人物框未改变。'; draw(); return;
  }
  state.viewer.pointers.delete(event.pointerId); state.viewer.drag = null; if (state.viewer.pointers.size < 2) state.viewer.pinch = null;
}
canvas.addEventListener('pointerup', finishCanvasPointer); canvas.addEventListener('pointercancel', cancelCanvasPointer);
canvas.addEventListener('dblclick', event => { event.preventDefault(); resetViewer(); draw(); });

async function processImage() {
  if (!cloudAllowed() || state.phase !== 'detected') return;
  state.phase = 'processing'; updateCount(); $('processButton').disabled = true; $('jobStatus').textContent = '任务已排队…';
  const mode = document.querySelector('input[name=mode]:checked').value;
  try {
    if (appFeatures.local_master && window.SceneFirstLocalMaster) {
      return await processLocalMaster(mode);
    }
    return await processTraditionalMaster(mode);
  } catch (error) {
    handleProcessingError(error);
  }
}

async function processTraditionalMaster(mode) {
  try {
    if (!state.masterUploaded) {
      $('jobStatus').textContent = '正在读取原尺寸照片…';
      const master = await SceneFirstDetection.uploadMaster(state.imageId, state.croppedFile || state.rawFile, (stage, detail = {}) => {
        if (stage === 'reading') $('jobStatus').textContent = '正在读取原尺寸照片…';
        if (stage === 'uploading') $('jobStatus').textContent = detail.percent == null ? '正在上传原尺寸照片…' : `正在上传原尺寸照片：${detail.percent}%`;
        if (stage === 'server-processing') $('jobStatus').textContent = '上传完成，正在准备高清工作副本…';
        if (stage === 'ready') $('jobStatus').textContent = '高清照片已准备完成。';
      });
      state.masterUploaded = true;
      state.imageUrl = master.preview_url || master.image_url;
    }
    $('jobStatus').textContent = '正在创建人物保护任务…';
    const response = await SceneFirstDetection.fetchWithTimeout('/api/edit', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ image_id: state.imageId, regions: state.detections, mode, provider: mode === 'safe' ? 'local' : $('provider').value, cloud_scope: 'crop', retry_nonce: state.retryNonce, safe_cover_id: state.safeCoverId, selection_confirmed: true }) }, 30000);
    if (!response.ok) throw new Error(await response.text());
    const { job_id } = await response.json(); state.jobId = job_id;
    await pollJob(job_id);
  }
  catch (error) {
    handleProcessingError(error);
  }
}

function handleProcessingError(error) {
    state.phase = 'detected'; state.processingPreview = null; updateCount();
    const friendly = {
      upload_timeout: '上传或高清副本准备时间过长。再次点击时会先查询已有结果，不会直接重复上传。',
      network_offline: '手机网络已断开；恢复网络后可直接重试，已确认的人物不会丢失。',
      connection_interrupted: '上传连接意外断开；请检查网络后安全重试，已确认的人物不会丢失。',
      file_too_large: '照片文件超过当前30MB上限，请换一张照片或先在相册中缩小。',
      pixel_limit: '照片超过当前1600万像素安全上限；48MP照片将在后续版本支持智能降级。',
      unsupported_format: '暂时无法读取这种照片格式；请使用 JPEG、PNG、WebP 或 HEIC。',
      server_processing: '服务器准备高清副本失败；可以安全重试，人物确认结果仍然保留。',
      user_cancelled: '已取消上传；人物确认结果仍然保留。',
    }[error?.code];
    showError(friendly || error.message || '处理进度连接中断，请检查网络后重试。');
}

async function processLocalMaster(mode) {
  const started = performance.now();
  $('jobStatus').textContent = `正在本机准备 ${selectedCount()} 个人物局部…`;
  try {
    const result = await SceneFirstLocalMaster.run({
      image: state.image, imageId: state.imageId, regions: state.detections,
      provider: mode === 'safe' ? 'local' : $('provider').value, mode,
      retryNonce: state.retryNonce,
      onProgress: value => {
        $('jobStatus').textContent = value.completed < value.total
          ? `已完成 ${value.completed}/${value.total} · 下一个人物处理中 · 双并行`
          : `${value.completed}/${value.total} 完成 · 正在本机准备成品`;
        state.processingPreview = value.canvas; draw();
      },
    });
    state.processingPreview = null; state.localResultCanvas = result.canvas;
    state.localSourceCanvas = result.sourceCanvas;
    await showLocalResult(result, started, mode);
  } catch (error) {
    state.processingPreview = null;
    if (appFeatures.traditional_master_fallback && confirm('当前浏览器无法可靠完成本机高清裁剪或合成。是否改用传统兼容路线？继续后，整张原图会上传到 Scene First 服务端以生成无元数据工作副本。')) {
      state.phase = 'processing'; updateCount(); return processTraditionalMaster(mode);
    }
    state.phase = 'detected'; updateCount(); showError(error.message || '本机高清合成失败；人物确认结果仍然保留。');
  }
}

function revokeLocalUrls() {
  state.localObjectUrls.forEach(url => URL.revokeObjectURL(url));
  state.localObjectUrls = [];
}

async function showLocalResult(result, processStarted, mode) {
  revokeLocalUrls();
  const encodeStarted = performance.now();
  const [beforeBlob, afterBlob] = await Promise.all([
    SceneFirstLocalMaster.canvasToBlob(result.sourceCanvas, 'png'),
    SceneFirstLocalMaster.canvasToBlob(result.canvas, 'png'),
  ]);
  result.metrics.png_encode_ms = Math.round(performance.now() - encodeStarted);
  const beforeUrl = URL.createObjectURL(beforeBlob), afterUrl = URL.createObjectURL(afterBlob);
  state.localObjectUrls.push(beforeUrl, afterUrl);
  const previewLoadStarted = performance.now();
  state.beforeResultImage = await imageFromUrl(beforeUrl);
  state.resultImage = await imageFromUrl(afterUrl);
  result.metrics.preview_load_ms = Math.round(performance.now() - previewLoadStarted);
  const elapsedMs = Math.round(performance.now() - processStarted);
  result.metrics.total_user_wait_ms = elapsedMs;
  state.resultMasterImage = state.resultImage;
  state.completedJob = {
    id: `local-${state.imageId}`, status: 'completed', local_master: true,
    people: result.people.map(person => ({ id: person.person_id, fallback_reason: person.fallback ? person.error : null })),
    elapsed_ms: elapsedMs, outside_mask_exact: true, local_metrics: result.metrics,
  };
  state.phase = 'completed'; state.exportCrop = null;
  $('beforeImage').src = beforeUrl; $('afterImage').src = afterUrl;
  $('resultSection').classList.remove('hidden');
  const fallback = result.people.filter(person => person.fallback).length;
  const aiFailures = localAiFailures(result, mode);
  const labDetail = labMode ? ` · 裁局部 ${result.metrics.crop_prepare_ms}ms · 合回 ${result.metrics.people.reduce((sum, person) => sum + (person.composite_ms || 0), 0)}ms · PNG ${result.metrics.png_encode_ms}ms` : '';
  $('resultSummary').textContent = `本机合成 ${result.total} 人 · ${Math.round(elapsedMs / 1000)} 秒 · 整张原图未到达服务器${labDetail}`;
  $('publishCheck').innerHTML = `<span class="check-pill ok">已保护 ${result.total} 人</span><span class="check-pill ok">原图留在本机</span><span class="check-pill ${fallback ? '' : 'ok'}">${aiFailures.length ? `${aiFailures.length} 人 AI 插画失败，已安全遮挡` : fallback ? `${fallback} 人使用安全回退` : '全部完成 AI 插画匿名化'}</span>`;
  renderAiFallback(aiFailures);
  $('publishDecision').classList.add('hidden');
  $('jobStatus').textContent = '保护完成。PNG 只在本机编码；请放大复查后保存。';
  updateCount(); $('resultSection').scrollIntoView({ behavior: 'smooth' });
}

async function pollJob(jobId) {
  for (;;) {
    const response = await fetch(`/api/jobs/${jobId}`);
    if (!response.ok) throw new Error('无法读取处理进度，请检查网络后重试。');
    const job = await response.json();
    const progressDetail = job.total_people
      ? ` · 已完成 ${job.completed_people || 0}/${job.total_people}${job.parallelism > 1 ? ' · 双并行' : ''}`
      : '';
    $('jobStatus').textContent = job.status === 'processing' ? `正在保护人物 · ${job.progress}%${progressDetail}` : job.status;
    if (job.status === 'completed') return showResult(job);
    if (job.status === 'failed') { state.phase = 'detected'; updateCount(); return showError(job.error); }
    await new Promise(resolve => setTimeout(resolve, 650));
  }
}

async function loadResultImages(job) {
  const [before, after] = await Promise.all([imageFromUrl(state.imageUrl), imageFromUrl(job.output_preview_url || job.output_url)]);
  state.beforeResultImage = before; state.resultImage = after;
}
function imageFromUrl(url) { return new Promise((resolve, reject) => { const image = new Image(); image.onload = () => resolve(image); image.onerror = () => reject(new Error('图片资源加载失败')); image.src = url.startsWith('blob:') || url.startsWith('data:') ? url : `${url}?t=${Date.now()}`; }); }

async function showResult(job) {
  state.completedJob = job; state.phase = 'loading-result'; updateCount(); $('jobStatus').textContent = '生成已完成，正在加载成品图…';
  try { await loadResultImages(job); }
  catch (error) {
    state.phase = 'detected'; updateCount();
    $('reloadResult').classList.remove('hidden');
    return showError('生成已经完成，但成品图暂时没有加载出来。请点“重新加载成品”；若仍失败，请刷新一次页面后再试。');
  }
  $('reloadResult').classList.add('hidden'); state.phase = 'completed'; state.exportCrop = null; state.resultMasterImage = null;
  $('beforeImage').src = state.imageUrl; $('afterImage').src = `${job.output_preview_url || job.output_url}?t=${Date.now()}`; $('resultSection').classList.remove('hidden');
  const fallback = job.people.filter(person => person.fallback_reason).length, animated = job.people.length - fallback;
  const aiFailures = traditionalAiFailures(job.people);
  $('resultSummary').textContent = `处理 ${job.people.length} 人 · ${Math.round(job.elapsed_ms / 1000)} 秒 · ${job.outside_mask_exact ? '场景与服装保持原像素' : '需要复查像素验证'}`;
  $('publishCheck').innerHTML = `<span class="check-pill ok">已保护 ${job.people.length} 人</span><span class="check-pill ${fallback ? '' : 'ok'}">${aiFailures.length ? `${aiFailures.length} 人 AI 插画失败，已安全遮挡` : fallback ? `${fallback} 人使用安全遮挡` : `${animated} 人完成 AI 插画匿名化`}</span><span class="check-pill">请放大复查侧脸、反射与远处人物</span>`;
  renderAiFallback(aiFailures);
  $('publishDecision').classList.remove('hidden');
  $('jobStatus').textContent = '保护完成。建议滑动对比并放大复查，然后保存无元数据成品。'; updateCount(); $('resultSection').scrollIntoView({ behavior: 'smooth' });
  if (navigator.share && navigator.canShare) $('shareButton').classList.remove('hidden');
}

function canvasFromImage(image) { const output = document.createElement('canvas'); output.width = image.naturalWidth || image.width; output.height = image.naturalHeight || image.height; output.getContext('2d').drawImage(image, 0, 0); return output; }
function rotateCanvas(source) { const output = document.createElement('canvas'); output.width = source.height; output.height = source.width; const outputCtx = output.getContext('2d'); outputCtx.translate(output.width, 0); outputCtx.rotate(Math.PI / 2); outputCtx.drawImage(source, 0, 0); return output; }
function flipCanvas(source) { const output = document.createElement('canvas'); output.width = source.width; output.height = source.height; const outputCtx = output.getContext('2d'); outputCtx.translate(output.width, 0); outputCtx.scale(-1, 1); outputCtx.drawImage(source, 0, 0); return output; }

function newCropState(source, target) { const canvasSource = canvasFromImage(source); return { target, sourceCanvas: canvasSource, rect: { x: 0, y: 0, w: canvasSource.width, h: canvasSource.height }, aspect: null, drag: null, pointers: new Map(), pinch: null, operations: [] }; }
async function openCrop(target) {
  if (target === 'before' && !state.rawImage) return;
  if (target === 'before' && state.phase === 'detected' && !confirm('重新裁剪会重新检查路人，当前已选人物将清空。继续吗？')) return;
  if (target === 'result' && !state.resultMasterImage) {
    const masterUrl = state.completedJob?.output_url;
    if (!masterUrl) return showError('找不到高清成品，请重新加载成品后再试。');
    $('jobStatus').textContent = '正在为高清裁剪载入原尺寸成品…';
    try { state.resultMasterImage = await imageFromUrl(masterUrl); }
    catch { return showError('高清成品加载失败，请检查网络后重试。'); }
  }
  const source = target === 'before' ? state.rawImage : state.resultMasterImage;
  state.crop = newCropState(source, target); $('cropTitle').textContent = target === 'before' ? '调整构图后再检查路人' : '调整成品构图'; $('cropSubhead').textContent = target === 'before' ? '裁后范围才会进入自动检测' : '此裁剪只影响导出成品，不会重新生成'; $('applyCrop').textContent = target === 'before' ? '使用构图' : '应用到成品';
  $('cropSheet').classList.remove('hidden'); setCropRatio('free'); requestAnimationFrame(drawCrop);
}
function closeCrop() { $('cropSheet').classList.add('hidden'); state.crop = null; }
function cropView() {
  const crop = state.crop, box = $('cropStage').getBoundingClientRect(), width = Math.max(1, Math.floor(box.width)), height = Math.max(1, Math.floor(box.height));
  cropCanvas.width = width; cropCanvas.height = height; const scale = Math.min(width / crop.sourceCanvas.width, height / crop.sourceCanvas.height), imageWidth = crop.sourceCanvas.width * scale, imageHeight = crop.sourceCanvas.height * scale, imageX = (width - imageWidth) / 2, imageY = (height - imageHeight) / 2;
  crop.view = { scale, imageX, imageY, imageWidth, imageHeight }; return crop.view;
}
function drawCrop() {
  if (!state.crop) return; const crop = state.crop, view = cropView(), r = crop.rect;
  cropCtx.clearRect(0, 0, cropCanvas.width, cropCanvas.height); cropCtx.drawImage(crop.sourceCanvas, view.imageX, view.imageY, view.imageWidth, view.imageHeight);
  const x = view.imageX + r.x * view.scale, y = view.imageY + r.y * view.scale, w = r.w * view.scale, h = r.h * view.scale;
  cropCtx.fillStyle = 'rgba(0,0,0,.56)'; cropCtx.fillRect(view.imageX, view.imageY, view.imageWidth, y - view.imageY); cropCtx.fillRect(view.imageX, y + h, view.imageWidth, view.imageY + view.imageHeight - (y + h)); cropCtx.fillRect(view.imageX, y, x - view.imageX, h); cropCtx.fillRect(x + w, y, view.imageX + view.imageWidth - (x + w), h);
  cropCtx.strokeStyle = '#cbff46'; cropCtx.lineWidth = 2; cropCtx.strokeRect(x, y, w, h); cropCtx.strokeStyle = 'rgba(255,255,255,.52)'; cropCtx.lineWidth = 1; for (let i = 1; i < 3; i += 1) { cropCtx.beginPath(); cropCtx.moveTo(x + w * i / 3, y); cropCtx.lineTo(x + w * i / 3, y + h); cropCtx.moveTo(x, y + h * i / 3); cropCtx.lineTo(x + w, y + h * i / 3); cropCtx.stroke(); }
  const marker = window.innerWidth <= 600 ? 14 : 10;
  cropCtx.fillStyle = '#cbff46'; [[x,y],[x+w,y],[x,y+h],[x+w,y+h]].forEach(([px,py]) => cropCtx.fillRect(px - marker / 2, py - marker / 2, marker, marker));
}
function setCropRatio(name) {
  if (!state.crop) return; const crop = state.crop, ratio = ratios[name]; crop.aspect = ratio === 'original' ? crop.sourceCanvas.width / crop.sourceCanvas.height : ratio;
  if (crop.aspect) { const maxW = crop.sourceCanvas.width, maxH = crop.sourceCanvas.height; let w = maxW, h = w / crop.aspect; if (h > maxH) { h = maxH; w = h * crop.aspect; } crop.rect = { x: (maxW - w) / 2, y: (maxH - h) / 2, w, h }; }
  document.querySelectorAll('.ratio').forEach(button => button.classList.toggle('active', button.dataset.ratio === name)); drawCrop();
}
function clampCropRect(rect) { const source = state.crop.sourceCanvas, minimum = Math.min(48, source.width, source.height); rect.w = Math.max(minimum, Math.min(rect.w, source.width)); rect.h = Math.max(minimum, Math.min(rect.h, source.height)); rect.x = Math.max(0, Math.min(rect.x, source.width - rect.w)); rect.y = Math.max(0, Math.min(rect.y, source.height - rect.h)); return rect; }
function cropPoint(event) { const view = state.crop.view, point = canvasPoint(event, cropCanvas); return { x: (point.x - view.imageX) / view.scale, y: (point.y - view.imageY) / view.scale }; }
function cropHandle(point, pointerType = 'mouse') {
  // A 56×56 CSS-pixel touch target is easier to acquire on a phone while the
  // visible crop marker stays compact. Mouse input remains precise.
  const r = state.crop.rect, cssTolerance = pointerType === 'touch' || pointerType === 'pen' ? 28 : 22, tolerance = cssTolerance / state.crop.view.scale; const near = (a, b) => Math.abs(a - b) < tolerance;
  if (near(point.x, r.x) && near(point.y, r.y)) return 'nw'; if (near(point.x, r.x + r.w) && near(point.y, r.y)) return 'ne'; if (near(point.x, r.x) && near(point.y, r.y + r.h)) return 'sw'; if (near(point.x, r.x + r.w) && near(point.y, r.y + r.h)) return 'se';
  if (near(point.x, r.x) && point.y >= r.y && point.y <= r.y + r.h) return 'w'; if (near(point.x, r.x + r.w) && point.y >= r.y && point.y <= r.y + r.h) return 'e'; if (near(point.y, r.y) && point.x >= r.x && point.x <= r.x + r.w) return 'n'; if (near(point.y, r.y + r.h) && point.x >= r.x && point.x <= r.x + r.w) return 's';
  return point.x >= r.x && point.x <= r.x + r.w && point.y >= r.y && point.y <= r.y + r.h ? 'move' : 'move';
}
function resizedRect(start, point, handle) {
  let { x, y, w, h } = start, left = x, top = y, right = x + w, bottom = y + h;
  if (handle.includes('w')) left = point.x; if (handle.includes('e')) right = point.x; if (handle.includes('n')) top = point.y; if (handle.includes('s')) bottom = point.y;
  if (state.crop.aspect && handle !== 'move') { const ratio = state.crop.aspect; if (handle === 'e' || handle === 'w') { const nextH = Math.abs(right - left) / ratio; if (handle === 'w') top = bottom - nextH; else bottom = top + nextH; } else if (handle === 'n' || handle === 's') { const nextW = Math.abs(bottom - top) * ratio; if (handle === 'n') left = right - nextW; else right = left + nextW; } else { const width = Math.abs(right - left), height = Math.abs(bottom - top); if (width / height > ratio) { const nextW = height * ratio; if (handle.includes('w')) left = right - nextW; else right = left + nextW; } else { const nextH = width / ratio; if (handle.includes('n')) top = bottom - nextH; else bottom = top + nextH; } } }
  const result = { x: Math.min(left, right), y: Math.min(top, bottom), w: Math.abs(right - left), h: Math.abs(bottom - top) }; return clampCropRect(result);
}
function distance(a,b) { return Math.hypot(a.x - b.x, a.y - b.y); }
cropCanvas.addEventListener('pointerdown', event => { if (!state.crop) return; event.preventDefault(); cropCanvas.setPointerCapture(event.pointerId); const point = cropPoint(event); state.crop.pointers.set(event.pointerId, point); if (state.crop.pointers.size === 2) { const pair = [...state.crop.pointers.values()]; state.crop.pinch = { distance: distance(pair[0], pair[1]), rect: { ...state.crop.rect } }; state.crop.drag = null; return; } state.crop.drag = { point, rect: { ...state.crop.rect }, handle: cropHandle(point, event.pointerType) }; });
cropCanvas.addEventListener('pointermove', event => { if (!state.crop || !state.crop.pointers.has(event.pointerId)) return; event.preventDefault(); const point = cropPoint(event); state.crop.pointers.set(event.pointerId, point); if (state.crop.pointers.size === 2 && state.crop.pinch) { const pair = [...state.crop.pointers.values()], scale = Math.max(.25, Math.min(4, distance(pair[0], pair[1]) / state.crop.pinch.distance)), start = state.crop.pinch.rect, w = start.w / scale, h = start.h / scale; state.crop.rect = clampCropRect({ x: start.x + (start.w - w) / 2, y: start.y + (start.h - h) / 2, w, h }); drawCrop(); return; }
  const drag = state.crop.drag; if (!drag) return; if (drag.handle === 'move') { const dx = point.x - drag.point.x, dy = point.y - drag.point.y; state.crop.rect = clampCropRect({ ...drag.rect, x: drag.rect.x + dx, y: drag.rect.y + dy }); } else state.crop.rect = resizedRect(drag.rect, point, drag.handle); drawCrop(); });
function endCropPointer(event) { if (!state.crop) return; state.crop.pointers.delete(event.pointerId); if (state.crop.pointers.size < 2) state.crop.pinch = null; state.crop.drag = null; }
cropCanvas.addEventListener('pointerup', endCropPointer); cropCanvas.addEventListener('pointercancel', endCropPointer);
function cropOutput(source, rect) { const output = document.createElement('canvas'); output.width = Math.round(rect.w); output.height = Math.round(rect.h); output.getContext('2d').drawImage(source, rect.x, rect.y, rect.w, rect.h, 0, 0, output.width, output.height); return output; }
function canvasBlob(source, type = 'image/png', quality) { return new Promise(resolve => source.toBlob(resolve, type, quality)); }
async function applyCrop() {
  const crop = state.crop; if (!crop) return; const output = cropOutput(crop.sourceCanvas, crop.rect);
  if (crop.target === 'before') {
    const blob = await canvasBlob(output); state.croppedFile = new File([blob], `scene-first-crop-${Date.now()}.png`, { type: 'image/png' }); state.image = await imageFromFile(state.croppedFile); state.width = state.image.naturalWidth; state.height = state.image.naturalHeight; state.detections = []; state.phase = 'ready'; state.imageId = null; state.imageUrl = null; state.masterUploaded = false; state.exportCrop = null; resetViewer();
    $('fileLabel').textContent = '已调整构图 · 等待检查'; $('cropHint').textContent = `当前构图 ${state.width} × ${state.height}。开始检查后，系统只处理这个范围。`; $('stageNote').textContent = '构图已更新。确认后开始检查画面中的路人。'; ['selectAll', 'clearAll', 'manualBox', 'adjustBox'].forEach(id => $(id).disabled = true); $('reviewControls').classList.add('hidden'); $('openPreCrop').disabled = false; $('detectButton').disabled = false; draw(); updateCount(); closeCrop(); return;
  }
  const beforeSource = applyOperations(canvasFromImage(state.beforeResultImage), crop.operations); const beforeOutput = cropOutput(beforeSource, crop.rect); state.exportCrop = { after: output, before: beforeOutput };
  $('beforeImage').src = beforeOutput.toDataURL('image/png'); $('afterImage').src = output.toDataURL('image/png'); $('resultSummary').textContent += ' · 已应用导出裁剪'; closeCrop();
}
function applyOperations(source, operations) { return operations.reduce((value, operation) => operation === 'rotate' ? rotateCanvas(value) : flipCanvas(value), source); }
function resetCrop() { if (!state.crop) return; const base = state.crop.target === 'before' ? state.rawImage : state.resultImage; state.crop = newCropState(base, state.crop.target); setCropRatio('free'); }
function rotateCrop() { if (!state.crop) return; state.crop.sourceCanvas = rotateCanvas(state.crop.sourceCanvas); state.crop.operations.push('rotate'); state.crop.rect = { x: 0, y: 0, w: state.crop.sourceCanvas.width, h: state.crop.sourceCanvas.height }; setCropRatio('free'); }
function flipCrop() { if (!state.crop) return; state.crop.sourceCanvas = flipCanvas(state.crop.sourceCanvas); state.crop.operations.push('flip'); drawCrop(); }

async function exportImage(format) {
  if (state.completedJob?.local_master) {
    const source = state.exportCrop?.after || state.localResultCanvas;
    const blob = await SceneFirstLocalMaster.canvasToBlob(source, format, .94);
    downloadBlob(blob, `scene-first-local-${Date.now()}.${format === 'jpeg' ? 'jpg' : 'png'}`);
    return;
  }
  if (!state.jobId) return;
  if (state.exportCrop) { const blob = await canvasBlob(state.exportCrop.after, format === 'jpeg' ? 'image/jpeg' : 'image/png', .94); downloadBlob(blob, `scene-first-${state.jobId}-crop.${format === 'jpeg' ? 'jpg' : 'png'}`); return; }
  const response = await fetch('/api/export', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ job_id: state.jobId, format }) }); const result = await response.json(); if (!response.ok) return showError(result.detail || '导出失败'); const anchor = document.createElement('a'); anchor.href = result.url; anchor.download = `scene-first-${state.jobId}.${format === 'jpeg' ? 'jpg' : 'png'}`; anchor.click();
}
async function markPublishable(publishable) {
  if (!state.jobId) return;
  const response = await fetch(`/api/costs/jobs/${state.jobId}/publishable`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ publishable }) });
  if (!response.ok) return showError('未能记录这次成品判断。');
  $('publishDecision').innerHTML = publishable ? '<span>已记为“愿意发布”，将用于计算每张可发布照片成本。</span>' : '<span>已记为“还需重试”，将用于分析失败与重试成本。</span>';
}
function downloadBlob(blob, name) { const anchor = document.createElement('a'); const url = URL.createObjectURL(blob); anchor.href = url; anchor.download = name; anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); }
async function shareImage() { const source = state.exportCrop?.after || (state.completedJob?.local_master ? state.localResultCanvas : state.resultImage); const blob = await canvasBlob(source, 'image/png'); const file = new File([blob], 'scene-first.png', { type: 'image/png' }); if (navigator.canShare?.({ files: [file] })) await navigator.share({ files: [file], title: 'Scene First 成品' }); else exportImage('png'); }
function showError(message) { $('jobStatus').textContent = `错误：${String(message).replace(/[{}"]+/g, ' ').slice(0, 300)}`; }

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[character]));
}

function cleanFallbackReason(reason) {
  const value = String(reason || '').trim();
  if (!value) return '未提供具体错误原因。';
  const marker = 'safe fallback used:';
  const index = value.toLowerCase().indexOf(marker);
  return index >= 0 ? value.slice(index + marker.length).trim() : value;
}

function traditionalAiFailures(people) {
  return (people || []).filter(person => person.generation_attempted && person.fallback_reason)
    .map((person, index) => ({ label: person.id || `人物 ${index + 1}`, reason: cleanFallbackReason(person.fallback_reason) }));
}

function localAiFailures(result, mode) {
  if (mode === 'safe') return [];
  return (result.people || []).filter(person => person.fallback && person.error && person.error !== '用户选择安全遮挡。')
    .map((person, index) => ({ label: person.person_id || `人物 ${index + 1}`, reason: cleanFallbackReason(person.error) }));
}

function renderAiFallback(aiFailures) {
  const banner = $('aiFallbackBanner');
  const hasFailures = aiFailures.length > 0;
  banner.classList.toggle('hidden', !hasFailures);
  $('retryButton').classList.toggle('hidden', hasFailures);
  if (!hasFailures) { $('aiFallbackReasons').innerHTML = ''; return; }
  $('aiFallbackReasons').innerHTML = aiFailures.map(item => `<li><span>${escapeHtml(item.label)}</span><small>${escapeHtml(item.reason).slice(0, 240)}</small></li>`).join('');
}

function retryAi() {
  state.retryNonce += 1;
  state.phase = 'detected';
  $('aiFallbackBanner').classList.add('hidden');
  $('retryButton').classList.remove('hidden');
  processImage();
}

$('fileInput').addEventListener('change', event => handleFile(event.target.files[0]));
$('emptyUpload').onclick = () => $('fileInput').click();
$('viewerReset').onclick = () => { resetViewer(); draw(); };
['dragenter','dragover'].forEach(name => $('uploadBox').addEventListener(name, event => { event.preventDefault(); $('uploadBox').classList.add('dragging'); }));
['dragleave','drop'].forEach(name => $('uploadBox').addEventListener(name, event => { event.preventDefault(); $('uploadBox').classList.remove('dragging'); })); $('uploadBox').addEventListener('drop', event => handleFile(event.dataTransfer.files[0]));
$('replacePhoto').onclick = () => $('fileInput').click();
$('openPreCrop').onclick = () => openCrop('before'); $('detectButton').onclick = detectPhoto; $('selectAll').onclick = () => { state.detections.forEach(item => item.selected = true); draw(); updateCount(); }; $('clearAll').onclick = () => { state.detections.forEach(item => item.selected = false); draw(); updateCount(); };
$('manualBox').onclick = () => { state.manual = !state.manual; Object.assign(state, { adjusting: false, adjustTarget: null, dragStart: null, dragCurrent: null, dragMoved: false, editPointerId: null }); canvas.classList.toggle('manual', state.manual); $('manualBox').textContent = state.manual ? '在照片上拖出头部范围' : '＋ 补一个漏检头部'; $('adjustBox').textContent = '调整一个人物框'; $('stageNote').textContent = state.manual ? '按住并拖动，蓝色选择框会实时出现；松开即完成。' : '补框已取消。'; draw(); };
$('adjustBox').onclick = () => { state.adjusting = !state.adjusting; Object.assign(state, { manual: false, adjustTarget: null, dragStart: null, dragCurrent: null, dragMoved: false, editPointerId: null }); canvas.classList.toggle('manual', state.adjusting); $('adjustBox').textContent = state.adjusting ? '点住原框并拖出新范围' : '调整一个人物框'; $('manualBox').textContent = '＋ 补一个漏检头部'; $('stageNote').textContent = state.adjusting ? '点住需要修改的原框；它会先高亮，拖动后蓝色新框会实时跟随。' : '调整人物框已取消。'; draw(); };
document.querySelectorAll('.mode-card input').forEach(input => input.addEventListener('change', () => {
  document.querySelectorAll('.mode-card').forEach(card => card.classList.toggle('selected', card.querySelector('input').checked));
  $('safeCoverPicker').classList.toggle('hidden', document.querySelector('input[name=mode]:checked').value !== 'safe');
  updateProviderConsent();
}));
document.querySelectorAll('.safe-cover-option').forEach(button => button.addEventListener('click', () => {
  state.safeCoverId = button.dataset.coverId;
  document.querySelectorAll('.safe-cover-option').forEach(option => option.classList.toggle('selected', option === button));
}));
$('provider').addEventListener('change', updateProviderConsent); $('cloudConsentInput').addEventListener('change', updateCount); $('processButton').onclick = processImage; $('retryButton').onclick = retryAi; $('retryAiButton').onclick = retryAi; $('reloadResult').onclick = () => state.completedJob && showResult(state.completedJob); $('openResultCrop').onclick = () => openCrop('result'); $('downloadPng').onclick = () => exportImage('png'); $('downloadJpeg').onclick = () => exportImage('jpeg'); $('shareButton').onclick = shareImage;
$('markPublishable').onclick = () => markPublishable(true); $('markNeedsRetry').onclick = () => markPublishable(false);
$('cancelCrop').onclick = closeCrop; $('applyCrop').onclick = applyCrop; $('rotateCrop').onclick = rotateCrop; $('flipCrop').onclick = flipCrop; $('resetCrop').onclick = resetCrop; document.querySelectorAll('.ratio').forEach(button => button.onclick = () => setCropRatio(button.dataset.ratio));
$('mobilePrimary').onclick = () => ({ detect: detectPhoto, process: processImage, png: () => exportImage('png') }[$('mobilePrimary').dataset.action] || (() => {}))(); $('mobileSecondary').onclick = () => ({ 'pre-crop': () => openCrop('before'), 'result-crop': () => openCrop('result') }[$('mobileSecondary').dataset.action] || (() => {}))();
$('compareRange').addEventListener('input', event => { const value = event.target.value; $('afterLayer').style.width = `${value}%`; $('compareLine').style.left = `${value}%`; });
function redrawAfterLayout() { requestAnimationFrame(() => { draw(); requestAnimationFrame(draw); drawCrop(); }); }
window.addEventListener('resize', redrawAfterLayout); window.addEventListener('orientationchange', redrawAfterLayout); window.visualViewport?.addEventListener('resize', redrawAfterLayout); document.addEventListener('visibilitychange', () => { if (!document.hidden) redrawAfterLayout(); });
// Localhost-only observability for deterministic interaction regression tests.
// It is never exposed by the production domain.
if (['127.0.0.1', 'localhost'].includes(location.hostname)) window.SceneFirstAppTesting = { state, cropHandle, normalizedDraftBox };
Promise.all([loadFeatures(), loadProviders()]).finally(updateCount);
