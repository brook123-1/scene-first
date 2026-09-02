import assert from 'node:assert/strict';
import fs from 'node:fs';
import { chromium } from 'playwright';

const executablePath = 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const source = fs.readFileSync(new URL('../static/local-master.js', import.meta.url), 'utf8');
const browser = await chromium.launch({ headless: true, executablePath });
try {
  const page = await browser.newPage();
  await page.setContent('<!doctype html><html><body></body></html>');
  await page.addScriptTag({ content: source });
  const result = await page.evaluate(async () => {
    const image = document.createElement('canvas'); image.width = 4032; image.height = 3024;
    const context = image.getContext('2d');
    const pixels = context.createImageData(image.width, image.height);
    for (let i = 0; i < pixels.data.length; i += 4) {
      const p = i / 4, x = p % image.width, y = Math.floor(p / image.width);
      pixels.data[i] = x % 251; pixels.data[i + 1] = y % 241; pixels.data[i + 2] = (x + y) % 239; pixels.data[i + 3] = 255;
    }
    context.putImageData(pixels, 0, 0);
    const regions = [
      { id: 'auto-keep', head_box: [1000, 500, 300, 420], selected: true, source: 'yunet' },
      { id: 'auto-cancel', head_box: [2100, 600, 280, 390], selected: false, source: 'yunet' },
      { id: 'manual-added', head_box: [3000, 800, 240, 330], selected: true, source: 'manual' },
      { id: 'adjusted', head_box: [500, 1400, 260, 360], selected: true, source: 'yunet', adjusted: true },
    ];
    const heapBefore = performance.memory?.usedJSHeapSize || null;
    const prepStarted = performance.now();
    const prepared = [];
    for (const region of regions.filter(item => item.selected)) {
      prepared.push(await SceneFirstLocalMaster.testing.prepareCrop(image, region, 'a'.repeat(32), 0));
    }
    const prepMs = Math.round(performance.now() - prepStarted);
    const detectionScaleX = 4032 / 1600, detectionScaleY = 3024 / 1200;
    const detectionBox = [100, 200, 300, 400];
    const remapped = detectionBox.map((value, index) => Math.round(value * (index % 2 === 0 ? detectionScaleX : detectionScaleY)));
    const before = context.getImageData(0, 0, image.width, image.height).data;
    const working = SceneFirstLocalMaster.testing.sourceCanvas(image);
    const target = prepared[0];
    const patch = document.createElement('canvas'); patch.width = target.cropBox[2]; patch.height = target.cropBox[3];
    const patchContext = patch.getContext('2d'); patchContext.fillStyle = '#ff0066'; patchContext.fillRect(0, 0, patch.width, patch.height);
    const compositeStarted = performance.now();
    const mask = SceneFirstLocalMaster.testing.compositePatch(working, patch, target);
    const compositeMs = Math.round(performance.now() - compositeStarted);
    const after = working.getContext('2d').getImageData(0, 0, working.width, working.height).data;
    const maskData = mask.getContext('2d').getImageData(0, 0, mask.width, mask.height).data;
    const [cx, cy, cw, ch] = target.cropBox;
    let outsideExact = true, insideChanged = false;
    for (let y = 0; y < image.height; y++) for (let x = 0; x < image.width; x++) {
      const index = (y * image.width + x) * 4;
      const insideCrop = x >= cx && x < cx + cw && y >= cy && y < cy + ch;
      const alpha = insideCrop ? maskData[((y - cy) * cw + (x - cx)) * 4 + 3] : 0;
      const same = before[index] === after[index] && before[index + 1] === after[index + 1] && before[index + 2] === after[index + 2] && before[index + 3] === after[index + 3];
      if (alpha === 0 && !same) outsideExact = false;
      if (alpha > 0 && !same) insideChanged = true;
    }
    const exportStarted = performance.now();
    const blob = await SceneFirstLocalMaster.canvasToBlob(working, 'png');
    const exportMs = Math.round(performance.now() - exportStarted);
    return {
      ids: prepared.map(item => item.personId),
      cropBytes: prepared.map(item => item.blob.size),
      cropBoxes: prepared.map(item => item.cropBox),
      uploadSizes: prepared.map(item => item.uploadSize),
      outsideExact, insideChanged, exportType: blob.type, exportBytes: blob.size,
      originalPixels: image.width * image.height,
      maxCropPixels: Math.max(...prepared.map(item => item.uploadSize[0] * item.uploadSize[1])),
      remapped,
      prepMs, compositeMs, exportMs, heapBefore, heapAfter: performance.memory?.usedJSHeapSize || null,
    };
  });
  assert.deepEqual(result.ids, ['auto-keep', 'manual-added', 'adjusted']);
  assert.ok(!result.ids.includes('auto-cancel'));
  assert.ok(result.cropBytes.every(value => value > 0));
  assert.ok(result.maxCropPixels < result.originalPixels * .2);
  assert.deepEqual(result.remapped, [252, 504, 756, 1008]);
  assert.ok(result.uploadSizes.every(([w, h]) => Math.max(w, h) <= 2048));
  assert.equal(result.outsideExact, true);
  assert.equal(result.insideChanged, true);
  assert.equal(result.exportType, 'image/png');
  assert.ok(result.exportBytes > 0);
  console.log(JSON.stringify(result));
} finally {
  await browser.close();
}
