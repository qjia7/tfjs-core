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

import {Conv2DInfo} from '../../ops/conv_util';
import {GPGPUProgram} from './gpgpu_math';

export class Conv2DProgramCS implements GPGPUProgram {
  variableNames = ['x', 'W'];
  outputShape: number[];
  userCode: string;
  localGroupSize = [16, 16];

  constructor(convInfo: Conv2DInfo) {
    this.outputShape = convInfo.outShape;
    const padTop = convInfo.padInfo.top;
    const padLeft = convInfo.padInfo.left;
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const dilationHeight = convInfo.dilationHeight;
    const dilationWidth = convInfo.dilationWidth;
    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;

    const inputDepthNearestVec4 = Math.floor(convInfo.inChannels / 4) * 4;
    const inputDepthVec4Remainder = convInfo.inChannels % 4;
    const BLOCK_SIZE = 16;
    this.userCode = `
      const ivec2 strides = ivec2(${strideHeight}, ${strideWidth});
      const ivec2 pads = ivec2(${padTop}, ${padLeft});
      shared float cache[${BLOCK_SIZE * strideHeight + 2}]
        [${(BLOCK_SIZE * strideHeight + 2) * convInfo.inChannels}];
      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d2 = coords[3];

        ivec2 xRCCorner = coords.yz * strides - pads;
        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        int row = 2 * int(gl_LocalInvocationID.x);
        int col = 2 * int(gl_LocalInvocationID.y);
        for (int i = 0; i < ${convInfo.inChannels}; i++)
        {
          cache[row][col * ${convInfo.inChannels} + i] =
             getX(batch, xRCorner, xCCorner, i);
          cache[row][(col + 1) * ${convInfo.inChannels} + i] =
             getX(batch, xRCorner, xCCorner + 1, i);
          cache[row + 1][col * ${convInfo.inChannels} + i] =
             getX(batch, xRCorner + 1, xCCorner, i);
          cache[row + 1][(col + 1) * ${convInfo.inChannels} + i] =
             getX(batch, xRCorner + 1, xCCorner + 1, i);
        }

        if (row == 0)
        {
          int sXR = xRCorner +  ${BLOCK_SIZE} * 2;
          for (int i = 0; i < ${convInfo.inChannels}; i++)
          {
            cache[${BLOCK_SIZE} * 2][col * ${convInfo.inChannels} + i] =
               getX(batch, sXR, xCCorner, i);
            cache[${BLOCK_SIZE} * 2][(col + 1) * ${convInfo.inChannels} + i] =
               getX(batch, sXR, xCCorner + 1, i);
            cache[${BLOCK_SIZE} * 2 + 1][col * ${convInfo.inChannels} + i] =
               getX(batch, sXR + 1, xCCorner, i);
            cache[${BLOCK_SIZE} * 2 + 1]
               [(col + 1) * ${convInfo.inChannels} + i] =
               getX(batch, sXR + 1, xCCorner + 1, i);
          }
        }
        if (col == 0)
        {
          int sXC = xCCorner + ${BLOCK_SIZE} * 2;
          for (int i = 0; i < ${convInfo.inChannels}; i++)
          {
            cache[row][(${BLOCK_SIZE} * 2) * ${convInfo.inChannels} + i] =
               getX(batch, xRCorner, sXC, i);
            cache[row][(${BLOCK_SIZE} * 2 + 1) * ${convInfo.inChannels} + i] =
               getX(batch, xRCorner, sXC + 1, i);
            cache[row + 1][(${BLOCK_SIZE} * 2) * ${convInfo.inChannels} + i] =
               getX(batch, xRCorner + 1, sXC, i);
            cache[row + 1]
               [(${BLOCK_SIZE} * 2 + 1) * ${convInfo.inChannels} + i] =
               getX(batch, xRCorner + 1, sXC + 1, i);
          }
        }

        if (row == 0 && col == 0)
        {
          int sXR = xRCorner + ${BLOCK_SIZE} * 2;
          int sXC = xCCorner + ${BLOCK_SIZE} * 2;
          for (int i = 0; i < ${convInfo.inChannels}; i++)
          {
            cache[${BLOCK_SIZE} * 2]
                [(${BLOCK_SIZE} * 2) * ${convInfo.inChannels} + i] =
              getX(batch, sXR, sXC, i);
            cache[${BLOCK_SIZE} * 2]
                [(${BLOCK_SIZE} * 2 + 1) * ${convInfo.inChannels} + i] =
              getX(batch, sXR, sXC + 1, i);
            cache[${BLOCK_SIZE} * 2 + 1]
                [(${BLOCK_SIZE} * 2) * ${convInfo.inChannels} + i] =
              getX(batch, sXR + 1, sXC, i);
            cache[${BLOCK_SIZE} * 2 + 1]
              [(${BLOCK_SIZE} * 2 + 1) * ${convInfo.inChannels} + i] =
              getX(batch, sXR + 1, sXC + 1, i);
          }
        }

        memoryBarrierShared();
        barrier();

        // Convolve x(?, ?, d1) with w(:, :, d1, d2) to get y(yR, yC, d2).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        for (int wR = 0; wR < ${filterHeight}; wR++) {
          int xR = xRCorner + wR * ${dilationHeight};
          int sR = row + wR;
          if (xR < 0 || xR >= ${convInfo.inHeight}) {
            continue;
          }

          for (int wC = 0; wC < ${filterWidth}; wC++) {
            int xC = xCCorner + wC * ${dilationWidth};
            int sC = (col + wC) * ${convInfo.inChannels};
            if (xC < 0 || xC >= ${convInfo.inWidth}) {
              continue;
            }

            for (int d1 = 0; d1 < ${inputDepthNearestVec4}; d1 += 4) {
              vec4 xValues = vec4(0, 0, 0, 0);
              if ((int(gl_WorkGroupID.x) + 1) * ${BLOCK_SIZE} >
                  ${convInfo.inHeight} ||
                  (int(gl_WorkGroupID.y) + 1) * ${BLOCK_SIZE} >
                  ${convInfo.inWidth})
              {
                xValues = vec4(
                  getX(batch, xR, xC, d1),
                  getX(batch, xR, xC, d1 + 1),
                  getX(batch, xR, xC, d1 + 2),
                  getX(batch, xR, xC, d1 + 3)
                );
              }
              else
              {
                xValues = vec4(
                  cache[sR][sC + d1],
                  cache[sR][sC + d1 + 1],
                  cache[sR][sC + d1 + 2],
                  cache[sR][sC + d1 + 3]
                );
              }
              vec4 wValues = vec4(
                getW(wR, wC, d1, d2),
                getW(wR, wC, d1 + 1, d2),
                getW(wR, wC, d1 + 2, d2),
                getW(wR, wC, d1 + 3, d2)
              );

              dotProd += dot(xValues, wValues);
            }

            if (${inputDepthVec4Remainder === 1}) {
              if ((int(gl_WorkGroupID.x) + 1) * ${BLOCK_SIZE} >
              ${convInfo.inHeight} ||
              (int(gl_WorkGroupID.y) + 1) * ${BLOCK_SIZE} >
              ${convInfo.inWidth})
              {
                dotProd +=
                  getX(batch, xR, xC, ${inputDepthNearestVec4}) *
                  getW(wR, wC, ${inputDepthNearestVec4}, d2);
              }
              else
              {
                dotProd +=
                  cache[sR][sC + ${inputDepthNearestVec4}] *
                  getW(wR, wC, ${inputDepthNearestVec4}, d2);
              }
            } else if (${inputDepthVec4Remainder === 2}) {
              vec2 xValues = vec2(0, 0);
              if ((int(gl_WorkGroupID.x) + 1) * ${BLOCK_SIZE} >
                  ${convInfo.inHeight} ||
                  (int(gl_WorkGroupID.y) + 1) * ${BLOCK_SIZE} >
                  ${convInfo.inWidth})
              {
                xValues = vec2(
                  getX(batch, xR, xC, ${inputDepthNearestVec4}),
                  getX(batch, xR, xC, ${inputDepthNearestVec4} + 1)
                );
              }
              else
              {
                xValues = vec2(
                  cache[sR][sC + ${inputDepthNearestVec4}],
                  cache[sR][sC + ${inputDepthNearestVec4} + 1]
                );
              }
              vec2 wValues = vec2(
                getW(wR, wC, ${inputDepthNearestVec4}, d2),
                getW(wR, wC, ${inputDepthNearestVec4} + 1, d2)
              );
              dotProd += dot(xValues, wValues);
            } else if (${inputDepthVec4Remainder === 3}) {
              vec3 xValues = vec3(0, 0, 0);
              if ((int(gl_WorkGroupID.x) + 1) * ${BLOCK_SIZE} >
                  ${convInfo.inHeight} ||
                  (int(gl_WorkGroupID.y) + 1) * ${BLOCK_SIZE} >
                  ${convInfo.inWidth})
              {
                xValues = vec3(
                  getX(batch, xR, xC, ${inputDepthNearestVec4}),
                  getX(batch, xR, xC, ${inputDepthNearestVec4} + 1),
                  getX(batch, xR, xC, ${inputDepthNearestVec4} + 2)
                );
              }
              else
              {
                xValues = vec3(
                  cache[sR][sC + ${inputDepthNearestVec4}],
                  cache[sR][sC + ${inputDepthNearestVec4} + 1],
                  cache[sR][sC + ${inputDepthNearestVec4} + 2]
                );
              }

              vec3 wValues = vec3(
                getW(wR, wC, ${inputDepthNearestVec4}, d2),
                getW(wR, wC, ${inputDepthNearestVec4} + 1, d2),
                getW(wR, wC, ${inputDepthNearestVec4} + 2, d2)
              );
              dotProd += dot(xValues, wValues);
            }
          }
        }
        setOutput(dotProd);
      }
    `;
  }
}
