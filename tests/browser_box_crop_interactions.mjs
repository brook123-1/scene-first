import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const root = fileURLToPath(new URL('..', import.meta.url));
const port = 8769;
const server = spawn(`${root}\\.venv\\Scripts\\python.exe`, ['-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', String(port)], {
  cwd: root, stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true,
});

async function waitForServer() {
  for (let index = 0; index < 80; index += 1) {
    try { const response = await fetch(`http://127.0.0.1:${port}/api/health`); if (response.ok) return; } catch {}
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  throw new Error('Local server did not start.');
}

const browser = await chromium.launch({ headless: true, executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe' });
try {
  await waitForServer();
  const page = await browser.newPage({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, hasTouch: true });
  await page.route('**/api/detect', route => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({
    image_id: 'b'.repeat(32), width: 1600, height: 1200,
    detections: [
      { id: 'first', box: [300, 200, 220, 300], head_box: [300, 200, 220, 300], confidence: .9, source: 'test', selected: true },
      { id: 'second', box: [900, 300, 180, 260], head_box: [900, 300, 180, 260], confidence: .9, source: 'test', selected: true },
    ], warnings: ['Test candidates.'],
  }) }));
  await page.goto(`http://127.0.0.1:${port}/`);
  await page.evaluate(async () => {
    const source = document.createElement('canvas'); source.width = 1600; source.height = 1200;
    const context = source.getContext('2d'); context.fillStyle = '#183149'; context.fillRect(0, 0, source.width, source.height);
    const blob = await new Promise(resolve => source.toBlob(resolve, 'image/jpeg', .9));
    const transfer = new DataTransfer(); transfer.items.add(new File([blob], 'interaction.jpg', { type: 'image/jpeg' }));
    const input = document.querySelector('#fileInput'); input.files = transfer.files; input.dispatchEvent(new Event('change', { bubbles: true }));
  });
  await page.waitForFunction(() => !document.querySelector('#detectButton').disabled);
  await page.evaluate(() => document.querySelector('#detectButton').click());
  await page.waitForFunction(() => !document.querySelector('#reviewControls').classList.contains('hidden'));

  const manual = await page.evaluate(() => {
    document.querySelector('#manualBox').click();
    const canvas = document.querySelector('#editorCanvas'), rect = canvas.getBoundingClientRect(); canvas.setPointerCapture = () => {};
    const fire = (name, x, y) => canvas.dispatchEvent(new PointerEvent(name, { pointerId: 7, pointerType: 'touch', clientX: rect.left + x, clientY: rect.top + y, bubbles: true }));
    fire('pointerdown', 18, 22); fire('pointermove', 105, 126);
    const pixels = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
    let cyan = 0; for (let i = 0; i < pixels.length; i += 4) if (pixels[i] === 114 && pixels[i + 1] === 231 && pixels[i + 2] === 255) cyan += 1;
    const during = { moved: SceneFirstAppTesting.state.dragMoved, cyan };
    fire('pointerup', 105, 126);
    return { during, total: SceneFirstAppTesting.state.detections.length, manualMode: SceneFirstAppTesting.state.manual };
  });
  assert.equal(manual.during.moved, true);
  assert.ok(manual.during.cyan > 20, 'live draft rectangle should be visible');
  assert.equal(manual.total, 3);
  assert.equal(manual.manualMode, false);

  const adjusted = await page.evaluate(() => {
    document.querySelector('#adjustBox').click();
    const canvas = document.querySelector('#editorCanvas'), rect = canvas.getBoundingClientRect(); canvas.setPointerCapture = () => {};
    const start = { x: rect.width * .255, y: rect.height * .29 }, end = { x: rect.width * .48, y: rect.height * .56 };
    const fire = (name, point) => canvas.dispatchEvent(new PointerEvent(name, { pointerId: 8, pointerType: 'touch', clientX: rect.left + point.x, clientY: rect.top + point.y, bubbles: true }));
    fire('pointerdown', start);
    const highlighted = SceneFirstAppTesting.state.adjustTarget?.id === 'first' && document.querySelector('#stageNote').textContent.includes('已高亮');
    fire('pointermove', end);
    const liveReplacement = SceneFirstAppTesting.state.dragMoved;
    fire('pointerup', end);
    const first = SceneFirstAppTesting.state.detections.find(item => item.id === 'first');
    return { highlighted, liveReplacement, adjusted: first.adjusted, box: first.head_box, mode: SceneFirstAppTesting.state.adjusting };
  });
  assert.equal(adjusted.highlighted, true);
  assert.equal(adjusted.liveReplacement, true);
  assert.equal(adjusted.adjusted, true);
  assert.equal(adjusted.mode, false);
  assert.ok(adjusted.box[2] > 12 && adjusted.box[3] > 12);

  await page.evaluate(() => { window.confirm = () => true; openCrop('before'); });
  await page.waitForSelector('#cropSheet:not(.hidden)');
  const cropTargets = await page.evaluate(() => {
    const { rect, view } = SceneFirstAppTesting.state.crop;
    const touchOffset = 26 / view.scale;
    const point = { x: rect.x + touchOffset, y: rect.y + touchOffset };
    return {
      touch: SceneFirstAppTesting.cropHandle(point, 'touch'),
      mouse: SceneFirstAppTesting.cropHandle(point, 'mouse'),
    };
  });
  assert.equal(cropTargets.touch, 'nw');
  assert.equal(cropTargets.mouse, 'move');
  console.log(JSON.stringify({ manual, adjusted, cropTargets }));
} finally {
  await browser.close();
  server.kill();
}
