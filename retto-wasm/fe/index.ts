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
  _alloc(n: number): number;
  _dealloc(p: number, n: number): void;
  _retto(ptr: number, len: number): void;
  onRettoNotifyDetDone(res: string): void;
  onRettoNotifyClsDone(res: string): void;
  onRettoNotifyRecDone(res: string): void;
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

  async *recognize(
    data: Uint8Array | ArrayBuffer,
  ): AsyncGenerator<RettoWorkerStage, void, unknown> {
    const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : data;
    const len = bytes.length;
    const ptr = this.module._alloc(len);
    this.module.HEAPU8.set(bytes, ptr);
    const detP = new Promise<DetProcessorResult>((resolve) => {
      this.module.onRettoNotifyDetDone = (res) =>
        resolve(JSON.parse(res) as DetProcessorResult);
    });
    const clsP = new Promise<ClsProcessorResult>((resolve) => {
      this.module.onRettoNotifyClsDone = (res) =>
        resolve(JSON.parse(res) as ClsProcessorResult);
    });
    const recP = new Promise<RecProcessorResult>((resolve) => {
      this.module.onRettoNotifyRecDone = (res) =>
        resolve(JSON.parse(res) as RecProcessorResult);
    });
    this.module._retto(ptr, len);
    const det = await detP;
    yield { stage: "det", result: det };
    const cls = await clsP;
    yield { stage: "cls", result: cls };
    const rec = await recP;
    yield { stage: "rec", result: rec };
    this.module._dealloc(ptr, len);
  }
}

export const retto = {
  init: (onProgress?: (ratio: number) => void) => Retto.init(onProgress),
  recognize: (data: Uint8Array | ArrayBuffer) =>
    Retto.init().then((engine) => engine.recognize(data)),
};
