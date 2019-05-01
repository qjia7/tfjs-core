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

export class DepthwiseConv2DProgramCS implements GPGPUProgram {
  variableNames = ['x', 'W'];
  outputShape: number[];
  userCode: string;
  localGroupSize: [number, number];

  constructor(convInfo: Conv2DInfo) {
    this.outputShape = convInfo.outShape;
    const xNumRows = convInfo.inHeight;
    const xNumCols = convInfo.inWidth;
    const padTop = convInfo.padInfo.top;
    const padLeft = convInfo.padInfo.left;
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const dilationHeight = convInfo.dilationHeight;
    const dilationWidth = convInfo.dilationWidth;
    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;
    const channelMul = convInfo.outChannels / convInfo.inChannels;

    this.localGroupSize = [8 * channelMul, 7];
    // outWidth should be divisible by localGroupSize[1]

    this.userCode = `
      const ivec2 strides = ivec2(${strideHeight}, ${strideWidth});
      const ivec2 pads = ivec2(${padTop}, ${padLeft});

      const int CACHE_H = ${filterHeight};
      const int CACHE_W = ${
        (this.localGroupSize[1] - 1) * strideWidth + filterWidth +
        (filterWidth - 1) * (dilationWidth - 1)};
      const int CACHE_C = ${this.localGroupSize[0] / channelMul};
      const int CACHE_WC = CACHE_W * CACHE_C;
      const int CACHE_HWC = CACHE_H * CACHE_W * CACHE_C;
      // Combine CACHE_W and CACHE_C
      shared float cache[CACHE_H][CACHE_W * CACHE_C];

      void main() {
        ivec4 coords = getFirstThreadOutputCoords();
        int batch = coords.x;
        ivec2 cacheRCCorner = coords.yz * strides - pads;
        int cacheRCorner = cacheRCCorner.x;
        int cacheCCorner = cacheRCCorner.y;
        int cacheDCorner = coords.w / ${channelMul};

        // Fill cache using all threads in a local group
        int index = int(gl_LocalInvocationIndex);
        while (index < CACHE_HWC) {
          int r = index / CACHE_WC;
          int cd = index - r * CACHE_WC;
          int c = cd / CACHE_C;
          int d = cd - c * CACHE_C;

          if ((cacheRCorner + r) >= 0 &&
              (cacheCCorner + c) >= 0 &&
              (cacheRCorner + r) < ${convInfo.inHeight} &&
              (cacheCCorner + c) < ${convInfo.inWidth} &&
              (cacheDCorner + d) < ${convInfo.inChannels}) {
            cache[r][cd] = getX(batch, cacheRCorner + r * ${dilationHeight},
                                cacheCCorner + c, cacheDCorner + d);
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
        ivec2 xRCCorner = coords.yz * strides - pads;
        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;
        int d2 = coords.w;
        int d1 = d2 / ${channelMul};
        int q = d2 - d1 * ${channelMul};

        // Convolve x(?, ?, d1) with w(:, :, d1, q) to get y(yR, yC, d2).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        // TODO: Flatten the two for loops and vec4 the operations.
        for (int wR = 0; wR < ${filterHeight}; wR++) {
          int xR = xRCorner + wR;
          if (xR < 0 || xR >= ${xNumRows}) {
            continue;
          }
          int sR = xR - cacheRCorner;

          for (int wC = 0; wC < ${filterWidth}; wC++) {
            int xC = xCCorner + wC * ${dilationWidth};
            if (xC < 0 || xC >= ${xNumCols}) {
              continue;
            }
            int sC = (xC - cacheCCorner) * CACHE_C;

            float xVal = cache[sR][sC + d1 - cacheDCorner];
            float wVal = getW(wR, wC, d1, q);
            dotProd += xVal * wVal;
          }
        }
        setOutput(dotProd);
      }
    `;
  }
}
