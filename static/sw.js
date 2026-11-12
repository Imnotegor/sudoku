const CACHE_NAME = "sudoku-offline-v12";
const ASSETS = [
  "/",
  "/index.html",
  "/quick.html",
  "/styles.css",
  "/city.css",
  "/dist/app.js",
  "/dist/city.js",
  "/dist/worker.js",
  "/fonts/DINASTI-Regular.otf",
  "/manifest.webmanifest",
  "/icon-192.png",
  "/icon-512.png",
  "/apple-touch-icon.png",
  "/favicon-32.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches
      .open(CACHE_NAME)
      .then((cache) => cache.addAll(ASSETS))
      .then(() => self.skipWaiting()),
  );
});
self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter(
              (key) => key.startsWith("sudoku-offline-") && key !== CACHE_NAME,
            )
            .map((key) => caches.delete(key)),
        ),
      )
      .then(() => self.clients.claim()),
  );
});
self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (
    event.request.method !== "GET" ||
    url.origin !== self.location.origin ||
    !ASSETS.includes(url.pathname)
  )
    return;
  event.respondWith(
    caches.open(CACHE_NAME).then(async (cache) => {
      try {
        const response = await fetch(event.request);
        if (response.ok) await cache.put(url.pathname, response.clone());
        return response;
      } catch (error) {
        const cached = await cache.match(url.pathname);
        if (cached) return cached;
        throw error;
      }
    }),
  );
});
