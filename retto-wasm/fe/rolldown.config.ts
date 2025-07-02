import { defineConfig } from "rolldown";
import { dts } from "rolldown-plugin-dts";
import copy from "rollup-plugin-copy";

const OUT_DIR = "dist";

export default defineConfig([
  {
    input: ["index.ts", "retto_wasm.js"],
    output: {
      dir: OUT_DIR,
      format: "es",
      sourcemap: true,
      assetFileNames: "[name][extname]",
      minify: {
        compress: true,
        mangle: true,
        removeWhitespace: true,
      },
    },
    plugins: [
      copy({
        targets: [
          { src: "retto_wasm.wasm", dest: "dist/public" },
        ],
      }),
    ],
  },
  {
    input: "index.ts",
    plugins: [dts()],
    output: {
      dir: OUT_DIR,
      format: "es",
    },
  },
]);
