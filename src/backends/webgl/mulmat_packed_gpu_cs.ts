/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {GPGPUProgram} from './gpgpu_math';

export class MatMulPackedProgramCSV1 implements GPGPUProgram {
  variableNames = ['matrixA', 'matrixB'];
  usesPackedTextures = true;
  outputShape: number[];
  userCode: string;
  localGroupSize: number[];

  constructor(
      aShape: [number, number, number], outputShape: [number, number, number],
      transposeA = false, transposeB = false, TS: number, addBias = false,
      activation: string = null) {
    this.outputShape = outputShape;

    const sharedDim = transposeA ? aShape[1] : aShape[2];
    const sharedDimensionPacked = Math.ceil(sharedDim / 2);

    const aSample = transposeA ? 'tileCol * 2, globalRow * 2' :
                                 'globalRow * 2, tileCol * 2';
    const bSample = transposeB ? 'globalCol * 2, tileRow * 2' :
                                 'tileRow * 2, globalCol * 2';
    const aSwizzle = transposeA ? ['a.xxyy', 'a.zzww'] : ['a.xxzz', 'a.yyww'];
    const bSwizzle = transposeB ? ['b.xzxz', 'b.ywyw'] : ['b.xyxy', 'b.zwzw'];

    let activationSnippet = '', applyActivationSnippet = '';
    if (activation) {
      activationSnippet = `vec4 activation(vec4 x) {
        ${activation}
      }`;

      applyActivationSnippet = `result = activation(result);`;
    }

    const addBiasSnippet = addBias ? 'result += getBiasAtOutCoords();' : '';
    if (addBias) {
      this.variableNames.push('bias');
    }
    this.localGroupSize = [TS, TS];
    this.userCode = `
      ${activationSnippet}

      const float sharedDimension = ${sharedDimensionPacked}.0;
      shared vec4 Asub[${TS}][${TS}];
      shared vec4 Bsub[${TS}][${TS}];
      void main() {
        ivec3 rc = getOutputCoords();
        int row = int(gl_LocalInvocationID.y);
        int col = int(gl_LocalInvocationID.x);
        int globalRow = int(gl_GlobalInvocationID.y);
        int globalCol = int(gl_GlobalInvocationID.x);

        // Loop over all tiles
        int numTiles = ${Math.ceil(sharedDimensionPacked / TS)};
        vec4 result = vec4(0);
        for (int t = 0; t < numTiles; t++) {
          // Load one tile of A and B into local memory
          int tileRow = ${TS} * t + row;
          int tileCol = ${TS} * t + col;
          Asub[row][col] = getMatrixA(rc.x, ${aSample});
          Bsub[row][col] = getMatrixB(rc.x, ${bSample});

          memoryBarrierShared();
          barrier();

          // If the tile size is larger than the shared dimension, we should
          // limit the size to |sharedDimensionPacked|.
          int sizeTS = (t == (numTiles - 1) &&
                        ${sharedDimensionPacked % TS} != 0) ?
                       ${sharedDimensionPacked % TS} : ${TS};
          for (int i = 0; i < sizeTS; i++) {
            vec4 a = Asub[row][i];
            vec4 b = Bsub[i][col];
            result += (${aSwizzle[0]} * ${bSwizzle[0]}) + (${aSwizzle[1]} * ${
        bSwizzle[1]});
          }

          // Synchronize before loading the next tile.
          barrier();
        }
        ${addBiasSnippet}

        ${applyActivationSnippet}

        setOutput(result);
      }
    `;
  }
}

export class MatMulPackedProgramCSV2 implements GPGPUProgram {
  variableNames = ['matrixA', 'matrixB'];
  usesPackedTextures = true;
  outputShape: number[];
  userCode: string;
  localGroupSize: number[];
  workPerThread: number[];

  constructor(
      aShape: [number, number, number], outputShape: [number, number, number],
      transposeA = false, transposeB = false, TS: number, WPT: number,
      addBias = false, activation: string = null) {
    this.outputShape = outputShape;

    const sharedDim = transposeA ? aShape[1] : aShape[2];
    const sharedDimensionPacked = Math.ceil(sharedDim / 2);
    const RTS = TS / WPT;
    this.localGroupSize = [TS, RTS];
    this.workPerThread = [1, WPT];

    const aSample = transposeA ? `tileCol * 2, (globalRow + w * ${RTS}) * 2` :
                                 `(globalRow + w * ${RTS}) * 2, tileCol * 2`;
    const bSample = transposeB ? `globalCol * 2, (tileRow + w * ${RTS}) * 2` :
                                 `(tileRow + w * ${RTS}) * 2, globalCol * 2`;
    const aSwizzle = transposeA ? ['a.xxyy', 'a.zzww'] : ['a.xxzz', 'a.yyww'];
    const bSwizzle = transposeB ? ['b.xzxz', 'b.ywyw'] : ['b.xyxy', 'b.zwzw'];

    let activationSnippet = '', applyActivationSnippet = '';
    if (activation) {
      activationSnippet = `vec4 activation(vec4 x) {
        ${activation}
      }`;

      applyActivationSnippet = `result = activation(result);`;
    }

    const addBiasSnippet = addBias ? 'result += getBiasAtOutCoords();' : '';
    if (addBias) {
      this.variableNames.push('bias');
    }

    this.userCode = `
      ${activationSnippet}

      const float sharedDimension = ${sharedDimensionPacked}.0;
      shared vec4 Asub[${TS}][${TS}];
      shared vec4 Bsub[${TS}][${TS}];
      void main() {
        ivec3 rc = getOutputCoords();
        int row = int(gl_LocalInvocationID.y);
        int col = int(gl_LocalInvocationID.x);
        int globalRow = ${TS} * int(gl_WorkGroupID.y) + row;
        int globalCol = ${TS} * int(gl_WorkGroupID.x) + col;

        // Loop over all tiles
        int numTiles = ${Math.ceil(sharedDimensionPacked / TS)};
        vec4 result[${WPT}];
        for (int t = 0; t < numTiles; t++) {
          // Load one tile of A and B into local memory
          int tileRow = ${TS} * t + row;
          int tileCol = ${TS} * t + col;
          for (int w = 0; w < ${WPT}; w++) {
          Asub[row + w*${RTS}][col] = getMatrixA(rc.x, ${aSample});
          Bsub[row + w*${RTS}][col] = getMatrixB(rc.x, ${bSample});
          }
          memoryBarrierShared();
          barrier();

          // If the tile size is larger than the shared dimension, we should
          // limit the size to |sharedDimensionPacked|.
          int sizeTS = (t == (numTiles - 1) &&
                        ${sharedDimensionPacked % TS} != 0) ?
                       ${sharedDimensionPacked % TS} : ${TS};
          for (int i = 0; i < sizeTS; i++) {
            vec4 b = Bsub[i][col];
            for (int w = 0; w < ${WPT}; w++) {
            vec4 a = Asub[row + w * ${RTS}][i];
            result[w] += (${aSwizzle[0]} * ${bSwizzle[0]}) +
                         (${aSwizzle[1]} * ${bSwizzle[1]});
            }
          }

          // Synchronize before loading the next tile.
          barrier();
        }
        ${addBiasSnippet}

        ${applyActivationSnippet}
        for (int w = 0; w < ${WPT}; w++) {
        imageStore(outputColor, ivec2(globalCol, globalRow + w*${RTS}),
           result[w]);
        }
      }
    `;
  }
}

export class MatMulPackedProgramCSV3 implements GPGPUProgram {
  variableNames = ['matrixA', 'matrixB'];
  usesPackedTextures = true;
  outputShape: number[];
  userCode: string;
  localGroupSize: number[];
  workPerThread: number[];

  constructor(
      aShape: [number, number, number], outputShape: [number, number, number],
      transposeA = false, transposeB = false, TS: number, WPT: number,
      addBias = false, activation: string = null) {
    this.outputShape = outputShape;

    const sharedDim = transposeA ? aShape[1] : aShape[2];
    const sharedDimensionPacked = Math.ceil(sharedDim / 2);
    const RTS = TS / WPT;
    this.localGroupSize = [RTS, RTS];
    this.workPerThread = [WPT, WPT];

    const aSample = transposeA ?
        `(tileCol + innerCol * ${RTS})* 2, (globalRow + innerRow * ${RTS})* 2` :
        `(globalRow + innerRow * ${RTS})* 2, (tileCol + innerCol * ${RTS})* 2`;
    const bSample = transposeB ?
        `(globalCol + innerCol * ${RTS})* 2, (tileRow + innerRow * ${RTS})* 2` :
        `(tileRow + innerRow * ${RTS})* 2, (globalCol + innerCol * ${RTS})* 2`;
    const aSwizzle = transposeA ? ['a.xxyy', 'a.zzww'] : ['a.xxzz', 'a.yyww'];
    const bSwizzle = transposeB ? ['b.xzxz', 'b.ywyw'] : ['b.xyxy', 'b.zwzw'];

    let activationSnippet = '', applyActivationSnippet = '';
    if (activation) {
      activationSnippet = `vec4 activation(vec4 x) {
        ${activation}
      }`;
      applyActivationSnippet = `
        result[innerRow][innerCol] = activation(result[innerRow][innerCol]);`;
    }

    if (addBias) {
      console.error('bias is not supported');
    }

    this.userCode = `
      ${activationSnippet}

      shared vec4 Asub[${TS}][${TS}];
      shared vec4 Bsub[${TS}][${TS}];
      void main() {
        ivec3 rc = getOutputCoords();
        int row = int(gl_LocalInvocationID.y);
        int col = int(gl_LocalInvocationID.x);
        int globalRow = ${TS} * int(gl_WorkGroupID.y) + row;
        int globalCol = ${TS} * int(gl_WorkGroupID.x) + col;

        vec4 Breg[${WPT}];
        vec4 result[${WPT}][${WPT}];

        for (int innerRow = 0; innerRow < ${WPT}; innerRow++) {
          for (int innerCol = 0; innerCol < ${WPT}; innerCol++) {
            result[innerRow][innerCol] = vec4(0);
          }
        }

        // Loop over all tiles
        int numTiles = ${Math.ceil(sharedDimensionPacked / TS)};
        for (int t = 0; t < numTiles; t++) {
          // Load one tile of A and B into local memory
          int tileRow = ${TS} * t + row;
          int tileCol = ${TS} * t + col;
          for (int innerRow = 0; innerRow < ${WPT}; innerRow++) {
            int inputRow = row + innerRow * ${RTS};
            for (int innerCol = 0; innerCol < ${WPT}; innerCol++) {
              int inputCol = col + innerCol * ${RTS};
              Asub[inputRow][inputCol] = getMatrixA(rc.x, ${aSample});
              Bsub[inputRow][inputCol] = getMatrixB(rc.x, ${bSample});
            }
          }

          memoryBarrierShared();
          barrier();

          // Loop over the values of a single tile
          int sizeTS = (t == (numTiles - 1) &&
                        ${sharedDimensionPacked % TS} != 0) ?
                        ${sharedDimensionPacked % TS} : ${TS};
          for (int k = 0; k < sizeTS; k++) {
            for (int inner = 0; inner < ${WPT}; inner++) {
              Breg[inner] = Bsub[k][col + inner * ${RTS}];
            }

            for (int innerRow = 0; innerRow < ${WPT}; innerRow++) {
              vec4 a = Asub[row + innerRow * ${RTS}][k];
              for (int innerCol = 0; innerCol < ${WPT}; innerCol++) {
                vec4 b = Breg[innerCol];
                result[innerRow][innerCol] +=
                    (${aSwizzle[0]} * ${bSwizzle[0]}) +
                    (${aSwizzle[1]} * ${bSwizzle[1]});
              }
            }
          }

          // Synchronize before loading the next tile.
          barrier();
        }

        // Store the final result
        for (int innerRow = 0; innerRow < ${WPT}; innerRow++) {
          int outputRow = globalRow + innerRow * ${RTS};
          if (outputRow >= ${Math.ceil(this.outputShape[1] / 2)}) {
            continue;
          }
          for (int innerCol = 0; innerCol < ${WPT}; innerCol++) {
            int outputCol = globalCol + innerCol * ${RTS};
            if (outputCol >= ${Math.ceil(this.outputShape[2] / 2)}) {
              continue;
            }
            ${applyActivationSnippet}
            imageStore(outputColor, ivec2(outputCol, outputRow),
                       result[innerRow][innerCol]);
          }
        }
      }
    `;
  }
}

export class MatMulPackedProgramCSV4 implements GPGPUProgram {
  variableNames = ['matrixA', 'matrixB'];
  usesPackedTextures = true;
  outputShape: number[];
  userCode: string;
  localGroupSize: number[];
  workPerThread: number[];

  constructor(
      aShape: [number, number, number], outputShape: [number, number, number],
      transposeA = false, transposeB = false, TS: number, TSK: number,
      WPT: number, addBias = false, activation: string = null) {
    this.outputShape = outputShape;

    const sharedDim = transposeA ? aShape[1] : aShape[2];
    const sharedDimensionPacked = Math.ceil(sharedDim / 2);

    const [TSM, TSN] = [TS, TS];
    const [WPTM, WPTN] = [WPT, WPT];
    const LPTA = TSK * WPTM * WPTN / TSN;
    // const LPTB = TSK * WPTM * WPTN / TSM;
    const [RTSM, RTSN] = [TSM / WPTM, TSN / WPTN];

    this.localGroupSize = [RTSN, RTSM];
    this.workPerThread = [WPTN, WPTM];

    const aSample = transposeA ? `tiledIndex * 2, (offsetM + row) * 2` :
                                 `(offsetM + row) * 2, tiledIndex * 2`;
    const bSample = transposeB ? `(offsetN + row) * 2, tiledIndex * 2` :
                                 `tiledIndex * 2, (offsetN + row) * 2`;
    const aSwizzle = transposeA ? ['a.xxyy', 'a.zzww'] : ['a.xxzz', 'a.yyww'];
    const bSwizzle = transposeB ? ['b.xzxz', 'b.ywyw'] : ['b.xyxy', 'b.zwzw'];

    let activationSnippet = '', applyActivationSnippet = '';
    if (activation) {
      activationSnippet = `vec4 activation(vec4 x) {
        ${activation}
      }`;
      applyActivationSnippet = `acc[wm][wn] = activation(acc[wm][wn]);`;
    }

    if (addBias) {
      console.error('bias is not supported');
    }

    this.userCode = `
      ${activationSnippet}

      shared vec4 Asub[${TSM}][${TSK}];
      shared vec4 Bsub[${TSK}][${TSN}];

      void main() {
        ivec3 rc = getOutputCoords();
        int tidm = int(gl_LocalInvocationID.y);
        int tidn = int(gl_LocalInvocationID.x);
        int offsetM = ${TSM} * int(gl_WorkGroupID.y);
        int offsetN = ${TSN} * int(gl_WorkGroupID.x);

        vec4 Breg[${WPTN}];
        vec4 acc[${WPTM}][${WPTN}];

        for (int wm = 0; wm < ${WPTM}; wm++) {
          for (int wn = 0; wn < ${WPTN}; wn++) {
            acc[wm][wn] = vec4(0);
          }
        }

        // Loop over all tiles
        int numTiles = ${Math.ceil(sharedDimensionPacked / TSK)};
        for (int t = 0; t < numTiles; t++) {
          // Load one tile of A and B into local memory
          for (int i = 0; i < ${LPTA}; i++) {
            int tid = tidm * ${RTSN} + tidn;
            int id = i * ${RTSN} * ${RTSM} + tid;
            int row = id / ${TSK};
            int col = imod(id, ${TSK});
            int tiledIndex = ${TSK} * t + col;
            Asub[row][col] = getMatrixA(rc.x, ${aSample});
            Bsub[col][row] = getMatrixB(rc.x, ${bSample});
          }

          memoryBarrierShared();
          barrier();

          // Loop over the values of a single tile
          int sizeTS = (t == (numTiles - 1) &&
                        ${sharedDimensionPacked % TSK} != 0) ?
                        ${sharedDimensionPacked % TSK} : ${TSK};
          for (int k = 0; k < sizeTS; k++) {
            for (int wn = 0; wn < ${WPTN}; wn++) {
              int col = tidn + wn * ${RTSN};
              Breg[wn] = Bsub[k][col];
            }

            for (int wm = 0; wm < ${WPTM}; wm++) {
              int row = tidm + wm * ${RTSM};
              vec4 a = Asub[row][k];
              for (int wn = 0; wn < ${WPTN}; wn++) {
                vec4 b = Breg[wn];
                acc[wm][wn] += (${aSwizzle[0]} * ${bSwizzle[0]}) +
                               (${aSwizzle[1]} * ${bSwizzle[1]});
              }
            }
          }

          // Synchronize before loading the next tile.
          barrier();
        }

        // Store the final result
        for (int wm = 0; wm < ${WPTM}; wm++) {
          int globalRow = offsetM + tidm + wm * ${RTSM};
          if (globalRow >= ${Math.ceil(this.outputShape[1] / 2)}) {
            continue;
          }
          for (int wn = 0; wn < ${WPTN}; wn++) {
            int globalCol = offsetN + tidn + wn * ${RTSN};
            if (globalCol >= ${Math.ceil(this.outputShape[2] / 2)}) {
              continue;
            }
            ${applyActivationSnippet}
            imageStore(outputColor, ivec2(globalCol, globalRow),
                       acc[wm][wn]);
          }
        }
      }
    `;
  }
}
