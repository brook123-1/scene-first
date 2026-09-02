import { chromium } from 'playwright';
import fs from 'node:fs/promises';
import path from 'node:path';

const chrome = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const browser = await chromium.launch({ headless: true, executablePath: chrome });
const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });
const renders = [];
await page.route('**/api/pose-avatar-playground', async (route) => route.fulfill({
  contentType: 'application/json',
  body: JSON.stringify({
    items: [
      { case_id: 'p1-001', original_url: 'data:image/gif;base64,R0lGODlhAQABAAAAACw=' },
      { case_id: 'p1-002', original_url: 'data:image/gif;base64,R0lGODlhAQABAAAAACw=' },
    ],
    families: [{ id: 'playground-demo-01', fixture: false, scenes: ['S01_FRONT_NEUTRAL'] }],
    scenes: ['S01_FRONT_NEUTRAL', 'S10_L_PROFILE'],
  }),
}));
await page.route('**/api/pose-avatar-playground/render', async (route) => {
  const request = route.request().postDataJSON();
  renders.push(request);
  await route.fulfill({ contentType: 'application/json', body: JSON.stringify({
    image_data_url: 'data:image/gif;base64,R0lGODlhAQABAAAAACw=',
    meta: { yaw: 8.4, roll: -2, scene: request.scene_override || 'S01_FRONT_NEUTRAL', head_bbox: [1, 2, 30, 40],
      route: 'BLUR_FALLBACK', preview_route: 'FORCED_STANDARD_PREVIEW', forced_preview: true,
      fitting: request.fitting, neck_stage: request.fitting === 'two-stage' ? 'independent' : 'not_used', coverage: {}, warnings: [] },
  }) });
});
let html = await fs.readFile(path.resolve('static/pose-avatar-playground.html'), 'utf8');
html = html.replace(/<link[^>]+pose-avatar-playground\.css[^>]*>/, '').replace(/<script[^>]+pose-avatar-playground\.js[^>]*><\/script>/, '');
html = html.replace('<head>', '<head><base href="http://playground.local/">');
await page.setContent(html);
await page.addScriptTag({ path: path.resolve('static/pose-avatar-playground.js') });
await page.waitForFunction(() => document.querySelector('#caseId')?.textContent === 'p1-001');
await page.locator('#next').click();
await page.waitForFunction(() => document.querySelector('#caseId')?.textContent === 'p1-002');
await page.locator('input[value="two-stage"]').check();
await page.locator('#scale').evaluate((node) => { node.value = '1.25'; node.dispatchEvent(new Event('input', { bubbles: true })); });
await page.waitForTimeout(180);
await page.locator('#reset').click();
await page.locator('[data-view="composite"]').click();
await page.waitForTimeout(80);
if (renders.length < 4) throw new Error(`expected multiple interactive renders, got ${renders.length}`);
if (!renders.some((value) => value.fitting === 'two-stage')) throw new Error('two-stage toggle did not render');
if (!renders.some((value) => value.scale === 1.25)) throw new Error('manual scale did not render');
if (await page.locator('#forcedFlag').isHidden()) throw new Error('forced preview state is not visible');
if (!await page.locator('#imageStage').evaluate((node) => node.classList.contains('view-composite'))) throw new Error('composite view toggle failed');
await browser.close();
console.log('pose-avatar playground browser interactions: OK');
