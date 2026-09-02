import assert from 'node:assert/strict';
import fs from 'node:fs';
import { chromium } from 'playwright';

const executablePath = 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const source = fs.readFileSync(new URL('../static/detection-client.js', import.meta.url), 'utf8');
const browser = await chromium.launch({ headless: true, executablePath });

try {
  const page = await browser.newPage();
  await page.setContent('<!doctype html><html><body></body></html>');
  await page.addScriptTag({ content: `
    class FakeXHR extends EventTarget {
      constructor() {
        super(); this.upload = new EventTarget(); this.status = 0; this.responseText = '';
        window.__lastXHR = this;
      }
      open(method, url) { this.method = method; this.url = url; }
      send(body) {
        this.body = body;
        const mode = window.__xhrMode || 'success';
        this.upload.dispatchEvent(new Event('loadstart'));
        const progress = new ProgressEvent('progress', { lengthComputable: true, loaded: 512, total: 1024 });
        this.upload.dispatchEvent(progress);
        if (mode === 'network') return this.dispatchEvent(new Event('error'));
        if (mode === 'timeout') return this.dispatchEvent(new Event('timeout'));
        this.upload.dispatchEvent(new Event('load'));
        this.status = mode === 'pixels' ? 413 : 200;
        this.responseText = mode === 'pixels'
          ? JSON.stringify({ detail: '原图像素过大' })
          : JSON.stringify({ status: 'ready', preview_url: '/preview.jpg' });
        this.dispatchEvent(new Event('load'));
      }
      abort() { this.dispatchEvent(new Event('abort')); }
      addEventListener(...args) { return super.addEventListener(...args); }
    }
    window.XMLHttpRequest = FakeXHR;
  ` });
  await page.addScriptTag({ content: source });

  const success = await page.evaluate(async () => {
    window.__xhrMode = 'success';
    const stages = [];
    const file = new File([new Uint8Array(1024)], 'identifiable-name.jpeg', { type: 'image/jpeg' });
    const result = await SceneFirstDetection.testing.uploadMasterRequest('a'.repeat(32), file, (stage, detail) => stages.push([stage, detail.percent]));
    return {
      status: result.status,
      stages,
      sentName: window.__lastXHR.body.get('file').name,
      metrics: result.client_upload_timings,
    };
  });
  assert.equal(success.status, 'ready');
  assert.equal(success.sentName, 'master.jpg');
  assert.deepEqual(success.stages.map(item => item[0]), ['reading', 'uploading', 'server-processing']);
  assert.equal(success.stages[1][1], 50);
  assert.equal(success.metrics.bytes, 1024);
  assert.equal(success.metrics.terminal_reason, undefined);

  const errors = await page.evaluate(async () => {
    const result = {};
    for (const mode of ['network', 'timeout', 'pixels']) {
      window.__xhrMode = mode;
      try {
        await SceneFirstDetection.testing.uploadMasterRequest('b'.repeat(32), new File(['x'], 'private.png', { type: 'image/png' }), () => {});
      } catch (error) {
        result[mode] = { name: error.name, code: error.code, terminal: error.metrics.terminal_reason };
      }
    }
    return result;
  });
  assert.deepEqual(errors.network, { name: 'MasterUploadError', code: 'connection_interrupted', terminal: 'connection_interrupted' });
  assert.deepEqual(errors.timeout, { name: 'MasterUploadError', code: 'upload_timeout', terminal: 'upload_timeout' });
  assert.deepEqual(errors.pixels, { name: 'MasterUploadError', code: 'pixel_limit', terminal: 'pixel_limit' });
  console.log(JSON.stringify({ success, errors }));
} finally {
  await browser.close();
}
