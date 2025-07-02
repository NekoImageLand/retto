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
  items: Array<{
    box: PointBox;
    score: number;
  }>;
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

export interface RettoWorkerResult {
  detResult: DetProcessorResult;
  clsResult: ClsProcessorResult;
  recResult: RecProcessorResult;
}

declare const RettoInner: {
  _alloc(n: number): number;
  _dealloc(p: number, n: number): void;
  _retto(p: number, n: number): void;
  HEAPU8: Uint8Array;
  onRettoNotifyDone(res: string): void;
};

export class Retto {
  private constructor(private module: typeof RettoInner) {}
  private static inner: Promise<Retto> | null = null;

  static init(onProgress?: (ratio: number) => void): Promise<Retto> {
    if (!this.inner) {
      this.inner = (async () => {
        const wasmUrl = new URL("public/retto_wasm.wasm", import.meta.url).href;
        const resp = await axios.get<ArrayBuffer>(wasmUrl, {
          responseType: "arraybuffer",
          onDownloadProgress: (e) => {
            if (e.total && onProgress) {
              onProgress(e.loaded / e.total);
            }
          },
        });
        const wasmBinary = resp.data;
        const mod = await initWASI({
          wasmBinary,
          locateFile: () => "",
        }) as typeof RettoInner;
        return new Retto(mod);
      })();
    }
    return this.inner!;
  }

  async recognize(data: Uint8Array | ArrayBuffer) {
    const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : data;
    const len = bytes.length;
    const alloc = this.module._alloc as (n: number) => number;
    const dealloc = this.module._dealloc as (p: number, n: number) => void;
    const wasmRet = this.module._retto as (p: number, n: number) => void;
    const ptr = alloc(len);
    this.module.HEAPU8.set(bytes, ptr);
    const result = await new Promise<RettoWorkerResult>((resolve) => {
      this.module.onRettoNotifyDone = (res: string) => resolve(JSON.parse(res));
      wasmRet(ptr, len);
    });
    dealloc(ptr, len);
    return result;
  }
}

export const retto = {
  init: (onProgress?: (ratio: number) => void) => Retto.init(onProgress),
  recognize: async (data: Uint8Array | ArrayBuffer) => {
    const engine = await Retto.init();
    return engine.recognize(data);
  },
};
