import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const root = fileURLToPath(new URL('..', import.meta.url));
const port = 8768;
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
  const page = await browser.newPage({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2 });
  await page.route('**/api/features', route => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ local_master: true, traditional_master_fallback: true, max_parallel_people: 2 }) }));
  await page.route('**/api/detect', route => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({
    image_id: 'a'.repeat(32), filename: 'detection.jpg', width: 1600, height: 1200,
    detections: [
      { id: 'keep-one', box: [300, 200, 220, 300], head_box: [300, 200, 220, 300], confidence: .9, source: 'test', selected: true },
      { id: 'cancel-one', box: [900, 300, 180, 260], head_box: [900, 300, 180, 260], confidence: .9, source: 'test', selected: true },
    ], warnings: ['Test candidate set.'],
  }) }));
  await page.goto(`http://127.0.0.1:${port}/`);
  await page.evaluate(async () => {
    const canvas = document.createElement('canvas'); canvas.width = 4032; canvas.height = 3024;
    const context = canvas.getContext('2d');
    const gradient = context.createLinearGradient(0, 0, canvas.width, canvas.height);
    gradient.addColorStop(0, '#17314f'); gradient.addColorStop(1, '#d3a56d'); context.fillStyle = gradient; context.fillRect(0, 0, canvas.width, canvas.height);
    const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', .9));
    const input = document.querySelector('#fileInput');
    const transfer = new DataTransfer(); transfer.items.add(new File([blob], 'phone-original.jpg', { type: 'image/jpeg' })); input.files = transfer.files; input.dispatchEvent(new Event('change', { bubbles: true }));
  });
  await page.waitForFunction(() => !document.querySelector('#detectButton').disabled);
  await page.evaluate(() => document.querySelector('#detectButton').click());
  await page.waitForFunction(() => document.querySelector('#processButton') && !document.querySelector('#reviewControls').classList.contains('hidden'));
  // Cancel one candidate through the same authoritative UI state that the user
  // manipulates; the safe-mode run must then process only the remaining one.
  await page.evaluate(() => { window.stateForTest = null; const canvas = document.querySelector('#editorCanvas'); const rect = canvas.getBoundingClientRect(); canvas.dispatchEvent(new PointerEvent('pointerdown', { pointerId: 1, clientX: rect.left + rect.width * .62, clientY: rect.top + rect.height * .36, bubbles: true })); canvas.dispatchEvent(new PointerEvent('pointerup', { pointerId: 1, clientX: rect.left + rect.width * .62, clientY: rect.top + rect.height * .36, bubbles: true })); });
  await page.evaluate(() => { const input = document.querySelector('input[name="mode"][value="safe"]'); input.checked = true; input.dispatchEvent(new Event('change', { bubbles: true })); });
  await page.evaluate(() => document.querySelector('#processButton').click());
  await page.waitForSelector('#resultSection:not(.hidden)', { timeout: 30000 });
  const summary = await page.textContent('#resultSummary');
  const status = await page.textContent('#jobStatus');
  assert.match(summary, /本机合成/);
  assert.match(summary, /整张原图未到达服务器/);
  assert.match(status, /PNG 只在本机编码/);
  console.log(JSON.stringify({ summary, status }));
} finally {
  await browser.close();
  server.kill();
}
