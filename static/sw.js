const SHELL_CACHE = 'scene-first-shell-v3';
const SHELL_ASSETS = [
  '/static/styles.css',
  '/static/detection-client.js',
  '/static/local-master.js',
  '/static/app.js',
  '/static/manifest.webmanifest',
  '/static/assets/app-icon-192.png',
  '/static/assets/app-icon-512.png',
  '/static/offline.html'
];

self.addEventListener('install', event => {
  event.waitUntil(caches.open(SHELL_CACHE).then(cache => cache.addAll(SHELL_ASSETS)));
  self.skipWaiting();
});

self.addEventListener('activate', event => {
  event.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(key => key !== SHELL_CACHE).map(key => caches.delete(key)))));
  self.clients.claim();
});

self.addEventListener('fetch', event => {
  const request = event.request;
  const url = new URL(request.url);
  if (request.method !== 'GET' || url.origin !== self.location.origin) return;

  // Never cache user photos, generated results, API responses or job state.
  if (url.pathname.startsWith('/api/') || url.pathname.startsWith('/media/')) return;

  if (request.mode === 'navigate') {
    event.respondWith(fetch(request).catch(() => caches.match('/static/offline.html')));
    return;
  }
  if (SHELL_ASSETS.includes(url.pathname)) {
    event.respondWith(caches.match(request).then(cached => cached || fetch(request)));
  }
});
