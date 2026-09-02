import assert from 'node:assert/strict';
import { chromium } from 'playwright';

const base = process.argv[2] || process.env.SCENE_FIRST_STAGING_URL;
const password = process.env.SCENE_FIRST_STAGING_ACCESS_CODE;
if (!base || !password) {
  throw new Error('Set SCENE_FIRST_STAGING_ACCESS_CODE and SCENE_FIRST_STAGING_URL, or pass the URL as the first argument.');
}

const browser = await chromium.launch({
  headless: true,
  executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
  proxy: process.env.SCENE_FIRST_TEST_PROXY
    ? {
        server: process.env.SCENE_FIRST_TEST_PROXY,
        username: process.env.SCENE_FIRST_TEST_PROXY_USERNAME,
        password: process.env.SCENE_FIRST_TEST_PROXY_PASSWORD,
      }
    : undefined,
});

try {
  const context = await browser.newContext({
    viewport: { width: 390, height: 844 },
    deviceScaleFactor: 2,
    httpCredentials: { username: 'scene', password },
  });
  const page = await context.newPage();
  await page.goto(base, { waitUntil: 'domcontentloaded', timeout: 180_000 });
  const result = await page.evaluate(async () => {
    const features = await fetch('/api/features').then((response) => response.json());
    const full = document.createElement('canvas');
    full.width = 4032;
    full.height = 3024;
    const context = full.getContext('2d');
    const gradient = context.createLinearGradient(0, 0, full.width, full.height);
    gradient.addColorStop(0, '#14273d');
    gradient.addColorStop(0.5, '#8f7359');
    gradient.addColorStop(1, '#d9b374');
    context.fillStyle = gradient;
    context.fillRect(0, 0, full.width, full.height);
    const masterBlob = await new Promise((resolve) => full.toBlob(resolve, 'image/jpeg', 0.88));

    const detect = document.createElement('canvas');
    detect.width = 1600;
    detect.height = 1200;
    detect.getContext('2d').drawImage(full, 0, 0, detect.width, detect.height);
    const detectBlob = await new Promise((resolve) => detect.toBlob(resolve, 'image/jpeg', 0.82));
    const form = new FormData();
    form.append('file', new File([detectBlob], 'detect.jpg', { type: 'image/jpeg' }));
    const detectionResponse = await fetch('/api/detect', { method: 'POST', body: form });
    if (!detectionResponse.ok) throw new Error(`detect:${detectionResponse.status}`);
    const detection = await detectionResponse.json();

    const stages = [];
    const uploaded = await window.SceneFirstDetection.testing.uploadMasterRequest(
      detection.image_id,
      new File([masterBlob], 'synthetic-12mp.jpg', { type: 'image/jpeg' }),
      (stage, detail) => stages.push({ stage, percent: detail.percent ?? null }),
    );
    const status = await fetch(`/api/images/${detection.image_id}/master`).then((response) => response.json());
    return {
      features,
      masterBytes: masterBlob.size,
      detectBytes: detectBlob.size,
      detectHttpMs: detection.detection_timings.request_total_ms,
      stages,
      clientTimings: uploaded.client_upload_timings,
      serverTimings: uploaded.timings,
      status: status.status,
      reused: status.reused,
      dimensions: [status.width, status.height],
    };
  });
  assert.equal(result.features.local_master, false);
  assert.equal(result.features.local_master_mode, 'traditional-master');
  assert.equal(result.status, 'ready');
  assert.deepEqual(result.dimensions, [4032, 3024]);
  assert.ok(result.stages.some((item) => item.stage === 'uploading'));
  assert.ok(result.stages.some((item) => item.stage === 'server-processing'));
  console.log(JSON.stringify(result, null, 2));
} finally {
  await browser.close();
}
