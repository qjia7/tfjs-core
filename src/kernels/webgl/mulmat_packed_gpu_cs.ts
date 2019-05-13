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

export class MatMulPackedProgramCS implements GPGPUProgram {
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
