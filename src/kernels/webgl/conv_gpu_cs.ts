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
  localGroupSize: [number, number];  // x, y

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

    this.localGroupSize = [8, 7];
    // outWidth should be divisible by localGroupSize[1]

    this.userCode = `
      const ivec2 strides = ivec2(${strideHeight}, ${strideWidth});
      const ivec2 pads = ivec2(${padTop}, ${padLeft});

      const int CACHE_H = ${filterHeight};
      const int CACHE_W = ${
        (this.localGroupSize[1] - 1) * strideWidth + filterWidth +
        (filterWidth - 1) * (dilationWidth - 1)};
      const int CACHE_C = ${convInfo.inChannels};
      const int CACHE_WC = CACHE_W * CACHE_C;
      const int CACHE_HWC = CACHE_H * CACHE_W * CACHE_C;
      // Combine CACHE_W and CACHE_C
      shared float cache[CACHE_H][CACHE_W * CACHE_C];

      void main() {
        ivec4 coords = getFirstThreadOutputCoords();
        int batch = coords[0];
        ivec2 cacheRCCorner = coords.yz * strides - pads;
        int cacheRCorner = cacheRCCorner.x;
        int cacheCCorner = cacheRCCorner.y;

        // Fill cache using all threads in a local group
        int index = int(gl_LocalInvocationIndex);
        while (index < CACHE_HWC) {
          int r = index / CACHE_WC;
          int cd = index - r * CACHE_WC;
          int c = cd / CACHE_C;
          int d = cd - c * CACHE_C;

          int xR = cacheRCorner + r * ${dilationHeight};
          int xC = cacheCCorner + c;
          if (xR >= 0 && xR < ${convInfo.inHeight} &&
              xC >= 0 && xC < ${convInfo.inWidth}) {
            cache[r][cd] = getX(batch, xR, xC, d);
          }

          index += ${this.localGroupSize[0] * this.localGroupSize[1]};
        }

        memoryBarrierShared();
        barrier();

        // Discard threads that are out of X bounds
        if (int(gl_GlobalInvocationID.x) >= ${convInfo.outChannels}) {
          return;
        }

        coords = getOutputCoords();
        int d2 = coords[3];
        ivec2 xRCCorner = coords.yz * strides - pads;
        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        // Convolve x(?, ?, d1) with w(:, :, d1, d2) to get y(yR, yC, d2).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        for (int wR = 0; wR < ${filterHeight}; wR++) {
          int xR = xRCorner + wR * ${dilationHeight};
          if (xR < 0 || xR >= ${convInfo.inHeight}) {
            continue;
          }
          int sR = wR;

          for (int wC = 0; wC < ${filterWidth}; wC++) {
            int xC = xCCorner + wC * ${dilationWidth};
            if (xC < 0 || xC >= ${convInfo.inWidth}) {
              continue;
            }
            int sC = (xC - cacheCCorner) * CACHE_C;

            for (int d1 = 0; d1 < ${inputDepthNearestVec4}; d1 += 4) {
              vec4 xValues = vec4(
                cache[sR][sC + d1],
                cache[sR][sC + d1 + 1],
                cache[sR][sC + d1 + 2],
                cache[sR][sC + d1 + 3]
              );
              vec4 wValues = vec4(
                getW(wR, wC, d1, d2),
                getW(wR, wC, d1 + 1, d2),
                getW(wR, wC, d1 + 2, d2),
                getW(wR, wC, d1 + 3, d2)
              );
              dotProd += dot(xValues, wValues);
            }

            if (${inputDepthVec4Remainder === 1}) {
              dotProd +=
                cache[sR][sC + ${inputDepthNearestVec4}] *
                getW(wR, wC, ${inputDepthNearestVec4}, d2);
            } else if (${inputDepthVec4Remainder === 2}) {
              vec2 xValues = vec2(
                cache[sR][sC + ${inputDepthNearestVec4}],
                cache[sR][sC + ${inputDepthNearestVec4} + 1]
              );
              vec2 wValues = vec2(
                getW(wR, wC, ${inputDepthNearestVec4}, d2),
                getW(wR, wC, ${inputDepthNearestVec4} + 1, d2)
              );
              dotProd += dot(xValues, wValues);
            } else if (${inputDepthVec4Remainder === 3}) {
              vec3 xValues = vec3(
                cache[sR][sC + ${inputDepthNearestVec4}],
                cache[sR][sC + ${inputDepthNearestVec4} + 1],
                cache[sR][sC + ${inputDepthNearestVec4} + 2]
              );
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
