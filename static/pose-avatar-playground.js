const $ = (id) => document.getElementById(id);
const state = { items: [], families: [], index: 0, renderToken: 0, timer: null };
const sceneLabels = {
  S01_FRONT_NEUTRAL: 'FRONT · S01',
  S04_L34_NEUTRAL: 'LEFT 3/4 · S04',
  S07_R34_NEUTRAL: 'RIGHT 3/4 · S07',
  S10_L_PROFILE: 'LEFT PROFILE · S10',
  S11_R_PROFILE: 'RIGHT PROFILE · S11',
};

function current() { return state.items[state.index]; }
function number(id) { return Number($(id).value); }
function fitting() { return document.querySelector('input[name="fitting"]:checked').value; }
function setText(id, value) { $(id).textContent = value ?? '—'; }

function updateOutputs() {
  setText('scaleValue', `${number('scale').toFixed(2)}×`);
  setText('xValue', `${number('xOffset')} px`);
  setText('yValue', `${number('yOffset')} px`);
  setText('rotationValue', `${number('rotation').toFixed(1).replace('.0', '')}°`);
}

function resetTuning(render = true) {
  $('scale').value = 1;
  $('xOffset').value = 0;
  $('yOffset').value = 0;
  $('rotation').value = 0;
  updateOutputs();
  if (render) scheduleRender(0);
}

function setMessage(value = '') { $('message').textContent = value; }

function renderMeta(meta) {
  setText('yaw', `${meta.yaw.toFixed(1)}°`);
  setText('roll', `${meta.roll.toFixed(1)}°`);
  setText('sceneValue', sceneLabels[meta.scene] || meta.scene);
  setText('bbox', meta.head_bbox.map((value) => Math.round(value)).join(' · '));
  setText('route', meta.route);
  setText('previewRoute', meta.preview_route);
  setText('headStage', meta.fitting === 'two-stage' ? 'face-primary' : meta.fitting);
  setText('neckStage', meta.neck_stage);
  $('forcedFlag').hidden = !meta.forced_preview;
  $('coverage').textContent = JSON.stringify(meta.coverage, null, 2);
  setMessage((meta.warnings || []).join(' · '));
}

async function renderComposite() {
  const item = current();
  const family = $('family').value;
  if (!item || !family) return;
  const token = ++state.renderToken;
  $('rendering').hidden = false;
  setMessage('');
  const payload = {
    case_id: item.case_id,
    family_id: family,
    fitting: fitting(),
    scene_override: $('scene').value || null,
    scale: number('scale'),
    x_offset: number('xOffset'),
    y_offset: number('yOffset'),
    rotation: number('rotation'),
  };
  try {
    const response = await fetch('/api/pose-avatar-playground/render', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Render failed');
    if (token !== state.renderToken) return;
    $('compositeImage').src = data.image_data_url;
    renderMeta(data.meta);
  } catch (error) {
    if (token === state.renderToken) setMessage(error.message);
  } finally {
    if (token === state.renderToken) $('rendering').hidden = true;
  }
}

function scheduleRender(delay = 90) {
  clearTimeout(state.timer);
  state.timer = setTimeout(renderComposite, delay);
}

function showCase() {
  const item = current();
  if (!item) return;
  setText('caseId', item.case_id);
  setText('caseCounter', `${state.index + 1} / ${state.items.length}`);
  $('originalImage').src = `${item.original_url}?v=${encodeURIComponent(item.case_id)}`;
  resetTuning(false);
  $('scene').value = '';
  scheduleRender(0);
}

function move(amount) {
  if (!state.items.length) return;
  state.index = (state.index + amount + state.items.length) % state.items.length;
  showCase();
}

function setView(view) {
  $('imageStage').className = `image-stage view-${view}`;
  document.querySelectorAll('[data-view]').forEach((button) => button.classList.toggle('active', button.dataset.view === view));
}

async function initialize() {
  try {
    const response = await fetch('/api/pose-avatar-playground');
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Playground unavailable');
    state.items = data.items || [];
    state.families = data.families || [];
    $('family').innerHTML = state.families.map((family) => `<option value="${family.id}">${family.id}${family.fixture ? ' · fixture' : ''}</option>`).join('');
    $('scene').insertAdjacentHTML('beforeend', (data.scenes || []).map((scene) => `<option value="${scene}">${sceneLabels[scene] || scene}</option>`).join(''));
    if (data.registry_error) setMessage(data.registry_error);
    if (!state.items.length) throw new Error('没有找到 .local/app/pose-avatar-p1 的真实照片 case。');
    if (!state.families.length) throw new Error('没有可用 avatar family；请先完成 manifest 与 metadata。');
    showCase();
  } catch (error) {
    setMessage(error.message);
    setText('caseId', 'UNAVAILABLE');
  }
}

$('previous').addEventListener('click', () => move(-1));
$('next').addEventListener('click', () => move(1));
$('reset').addEventListener('click', () => resetTuning());
$('family').addEventListener('change', () => { $('scene').value = ''; scheduleRender(0); });
$('scene').addEventListener('change', () => scheduleRender(0));
document.querySelectorAll('input[name="fitting"]').forEach((input) => input.addEventListener('change', () => scheduleRender(0)));
['scale', 'xOffset', 'yOffset', 'rotation'].forEach((id) => $(id).addEventListener('input', () => { updateOutputs(); scheduleRender(); }));
document.querySelectorAll('[data-view]').forEach((button) => button.addEventListener('click', () => setView(button.dataset.view)));
window.addEventListener('keydown', (event) => {
  if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
  if (event.key === 'ArrowLeft') move(-1);
  if (event.key === 'ArrowRight') move(1);
  if (event.key.toLowerCase() === 'r') resetTuning();
});

updateOutputs();
initialize();
