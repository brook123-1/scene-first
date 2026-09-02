import assert from 'node:assert/strict';
import fs from 'node:fs';
import { chromium } from 'playwright';

const executablePath = 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const source = fs.readFileSync(new URL('../static/detection-client.js', import.meta.url), 'utf8');
const browser = await chromium.launch({ headless: true, executablePath });
try {
  const page = await browser.newPage();
  await page.setContent('<!doctype html><html><body></body></html>');
  await page.addScriptTag({ content: source });
  const result = await page.evaluate(async () => {
    const canvas = document.createElement('canvas');
    canvas.width = 4032;
    canvas.height = 3024;
    canvas.getContext('2d').fillRect(0, 0, canvas.width, canvas.height);
    const payload = await SceneFirstDetection.testing.createDetectionFile(canvas, 'iphone.heic');
    const mapped = SceneFirstDetection.testing.scaleDetections([{
      id: 'one', box: [100, 200, 300, 400], head_box: [90, 180, 330, 450], selected: true,
      face_landmarks: { left_eye: [120, 250], right_eye: [210, 255], nose: [170, 300] },
      body_landmarks: { neck_center: [180, 590] },
    }], payload)[0];
    const legacyMapped = SceneFirstDetection.testing.scaleDetections([{
      id: 'legacy', box: [10, 20, 30, 40], head_box: [9, 18, 33, 45], selected: true,
    }], payload)[0];
    return {
      payload: {
        sourceWidth: payload.sourceWidth,
        sourceHeight: payload.sourceHeight,
        detectionWidth: payload.detectionWidth,
        detectionHeight: payload.detectionHeight,
        fileName: payload.file.name,
        fileType: payload.file.type,
        fileSize: payload.file.size,
      },
      mapped,
      legacyHasPoseFields: Object.hasOwn(legacyMapped, 'face_landmarks') || Object.hasOwn(legacyMapped, 'body_landmarks'),
    };
  });
  assert.deepEqual(result.payload, {
    sourceWidth: 4032,
    sourceHeight: 3024,
    detectionWidth: 1600,
    detectionHeight: 1200,
    fileName: 'iphone-detection.jpg',
    fileType: 'image/jpeg',
    fileSize: result.payload.fileSize,
  });
  assert.ok(result.payload.fileSize > 0);
  assert.equal(result.legacyHasPoseFields, false);
  assert.deepEqual(result.mapped.box, [252, 504, 756, 1008]);
  assert.deepEqual(result.mapped.head_box, [227, 454, 832, 1134]);
  assert.deepEqual(result.mapped.face_landmarks, {
    left_eye: [302, 630], right_eye: [529, 643], nose: [428, 756],
  });
  assert.deepEqual(result.mapped.body_landmarks, { neck_center: [454, 1487] });
  console.log(JSON.stringify(result));
} finally {
  await browser.close();
}
