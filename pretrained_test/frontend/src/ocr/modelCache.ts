// Cache bajtów modeli ONNX w IndexedDB — przy drugim otwarciu aplikacji nie robimy
// fetch ~15 MB, tylko odczyt z IDB (typowo 5–20 ms).

const DB_NAME = "ocr-models";
const STORE = "files";
const DB_VERSION = 1;

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) db.createObjectStore(STORE);
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error ?? new Error("idb open failed"));
  });
}

async function idbGet(key: string): Promise<ArrayBuffer | null> {
  const db = await openDb();
  try {
    return await new Promise<ArrayBuffer | null>((resolve, reject) => {
      const tx = db.transaction(STORE, "readonly");
      const req = tx.objectStore(STORE).get(key);
      req.onsuccess = () => resolve((req.result as ArrayBuffer | undefined) ?? null);
      req.onerror = () => reject(req.error);
    });
  } finally {
    db.close();
  }
}

async function idbPut(key: string, buf: ArrayBuffer): Promise<void> {
  const db = await openDb();
  try {
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE, "readwrite");
      tx.objectStore(STORE).put(buf, key);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
      tx.onabort = () => reject(tx.error);
    });
  } finally {
    db.close();
  }
}

// Pobiera bajty modelu (z IDB jeśli jest, inaczej fetch + zapis). Zwraca Uint8Array,
// który można podać wprost do `ort.InferenceSession.create(bytes, ...)`.
export async function fetchModelCached(url: string): Promise<Uint8Array> {
  try {
    const cached = await idbGet(url);
    if (cached) return new Uint8Array(cached);
  } catch {
    // IDB niedostępne (np. tryb prywatny Safari) — trudno, lecimy bez cache
  }

  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${url}: ${res.status}`);
  const buf = await res.arrayBuffer();

  // Zapis niekrytyczny — błąd nie przerywa ładowania modelu
  idbPut(url, buf).catch(() => { /* ignore */ });

  return new Uint8Array(buf);
}
