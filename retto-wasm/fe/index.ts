// @ts-ignore
import initWASI from "./retto_wasm.js";
import axios from "axios";

export interface Point {
  x: number;
  y: number;
}

export interface PointBox {
  topLeft: Point;
  bottomRight: Point;
}

export interface DetProcessorResult {
  items: Array<{ box: PointBox; score: number }>;
}

export interface ClsPostProcessLabel {
  label: number;
  score: number;
}

export interface ClsProcessorSingleResult {
  label: ClsPostProcessLabel;
}

export interface ClsProcessorResult {
  items: ClsProcessorSingleResult[];
}

export interface RecProcessorSingleResult {
  text: string;
  score: number;
}

export interface RecProcessorResult {
  items: RecProcessorSingleResult[];
}

export type RettoWorkerStage =
  | { stage: "det"; result: DetProcessorResult }
  | { stage: "cls"; result: ClsProcessorResult }
  | { stage: "rec"; result: RecProcessorResult };

declare const RettoInner: {
  HEAPU8: Uint8Array;
  UTF8ToString(ptr: number): string;
  _alloc(n: number): number;
  _dealloc(p: number, n: number): void;
  _retto_init(
    det_ptr: number,
    det_len: number,
    cls_ptr: number,
    cls_len: number,
    rec_ptr: number,
    rec_len: number,
    rec_dict_ptr: number,
    rec_dict_len: number,
  ): void;
  _retto_embed_init(): void;
  _retto_rec(ptr: number, len: number): number;
  onRettoNotifyDetDone(sessionId: string, msg: string): void;
  onRettoNotifyClsDone(sessionId: string, msg: string): void;
  onRettoNotifyRecDone(sessionId: string, msg: string): void;
};

type BufferData = Uint8Array | ArrayBuffer;

interface BufferRegion {
  ptr: number;
  len: number;
}

class WasmBufferManager {
  constructor(private module: typeof RettoInner) {}

  private allocOne(data: BufferData): BufferRegion {
    const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : data;
    const ptr = this.module._alloc(bytes.length);
    this.module.HEAPU8.set(bytes, ptr);
    return { ptr, len: bytes.length };
  }

  private deallocOne({ ptr, len }: BufferRegion): void {
    this.module._dealloc(ptr, len);
  }

  async ctx<T>(
    data: BufferData,
    fn: (ptr: number, len: number) => Promise<T> | T,
  ): Promise<T> {
    const region = this.allocOne(data);
    try {
      const res = fn(region.ptr, region.len);
      return res instanceof Promise ? await res : res;
    } finally {
      this.deallocOne(region);
    }
  }

  async *ctxGen<T>(
    data: BufferData,
    fn: (ptr: number, len: number) => AsyncGenerator<T, void, unknown>,
  ): AsyncGenerator<T, void, unknown> {
    const region = this.allocOne(data);
    try {
      const gen = fn(region.ptr, region.len);
      for await (const v of gen) {
        yield v;
      }
    } finally {
      this.deallocOne(region);
    }
  }

  async ctxRegions<
    T extends BufferData[],
    R,
  >(
    datas: [...T],
    fn: (...regions: { [K in keyof T]: BufferRegion }) => Promise<R> | R,
  ): Promise<R> {
    const regions: BufferRegion[] = [];
    try {
      for (const d of datas) {
        regions.push(this.allocOne(d));
      }
      const tupleRegions = regions as { [K in keyof T]: BufferRegion };
      const result = fn(...tupleRegions);
      return result instanceof Promise ? await result : result;
    } finally {
      for (const reg of regions.reverse()) {
        this.deallocOne(reg);
      }
    }
  }
}

export interface RettoModel {
  det_model: ArrayBuffer;
  cls_model: ArrayBuffer;
  rec_model: ArrayBuffer;
  rec_dict: ArrayBuffer;
}

export class Retto {
  private bufferManager: WasmBufferManager;
  private static inner: Promise<Retto> | null = null;
  private emitter = new EventTarget();

  private constructor(private module: typeof RettoInner) {
    this.bufferManager = new WasmBufferManager(module);
    this.registerCallbacks();
  }

  static load(onProgress?: (ratio: number) => void): Promise<Retto> {
    if (!this.inner) {
      this.inner = (async () => {
        const wasmUrl = new URL("public/retto_wasm.wasm", import.meta.url).href;
        const resp = await axios.get<ArrayBuffer>(wasmUrl, {
          responseType: "arraybuffer",
          onDownloadProgress: (e) => {
            if (e.total && onProgress) onProgress(e.loaded / e.total);
          },
        });
        const mod = (await initWASI({
          wasmBinary: resp.data,
          locateFile: () => "",
        })) as typeof RettoInner;
        return new Retto(mod);
      })();
    }
    return this.inner;
  }

  get is_embed_build(): boolean {
    return this.module._retto_embed_init !== undefined;
  }

  private registerCallbacks() {
    this.module.onRettoNotifyDetDone = (sessionId, msg) => {
      try {
        console.log("Det done:", sessionId, msg);
        const data = JSON.parse(msg) as DetProcessorResult;
        this.emitter.dispatchEvent(
          new CustomEvent(`${sessionId}:det`, { detail: data }),
        );
      } catch {}
    };
    this.module.onRettoNotifyClsDone = (sessionId, msg) => {
      try {
        console.log("Cls done:", sessionId, msg);
        const data = JSON.parse(msg) as ClsProcessorResult;
        this.emitter.dispatchEvent(
          new CustomEvent(`${sessionId}:cls`, { detail: data }),
        );
      } catch {}
    };
    this.module.onRettoNotifyRecDone = (sessionId, msg) => {
      try {
        console.log("Rec done:", sessionId, msg);
        const data = JSON.parse(msg) as RecProcessorResult;
        this.emitter.dispatchEvent(
          new CustomEvent(`${sessionId}:rec`, { detail: data }),
        );
      } catch {}
    };
  }

  async init(models?: RettoModel) {
    if (!this.is_embed_build && !models) {
      throw new Error("Models are required for this build.");
    }
    if (models) {
      const { det_model, cls_model, rec_model, rec_dict } = models;
      await this.bufferManager.ctxRegions([
        det_model,
        cls_model,
        rec_model,
        rec_dict,
      ], (det_r, cls_r, rec_r, rec_dict_r) => {
        this.module._retto_init(
          det_r.ptr,
          det_r.len,
          cls_r.ptr,
          cls_r.len,
          rec_r.ptr,
          rec_r.len,
          rec_dict_r.ptr,
          rec_dict_r.len,
        );
      });
    } else {
      this.module._retto_embed_init();
    }
  }

  // TODO: concurrency
  async *recognize(
    data: Uint8Array | ArrayBuffer,
  ): AsyncGenerator<RettoWorkerStage, void, unknown> {
    const module = this.module;
    const emitter = this.emitter;
    yield* this.bufferManager.ctxGen(
      data,
      async function* (ptr: number, len: number) {
        const sessionPtr = module._retto_rec(ptr, len);
        const sessionId = module.UTF8ToString(sessionPtr);
        function once<T>(stage: string): Promise<T> {
          return new Promise((resolve) => {
            const eventName = `${sessionId}:${stage}`;
            const handler = (e: Event) => {
              const data = (e as CustomEvent).detail as T;
              resolve(data);
              emitter.removeEventListener(eventName, handler);
            };
            emitter.addEventListener(eventName, handler);
          });
        }
        const det = await once<DetProcessorResult>("det");
        yield { stage: "det", result: det };
        const cls = await once<ClsProcessorResult>("cls");
        yield { stage: "cls", result: cls };
        const rec = await once<RecProcessorResult>("rec");
        yield { stage: "rec", result: rec };
      },
    );
  }
}
