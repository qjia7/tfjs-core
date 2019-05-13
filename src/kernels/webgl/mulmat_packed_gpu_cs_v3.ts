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
