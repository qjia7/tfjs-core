/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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

import {ENV} from '../../environment';
import {getBroadcastDims} from '../../ops/broadcast_util';
import * as util from '../../util';
import {getGlslDifferences, GLSL} from './glsl_version';
import * as shader_util from './shader_compiler_util';

export type ShapeInfo = {
  logicalShape: number[],
  texShape: [number, number],
  isUniform: boolean,
  isPacked: boolean,
  flatOffset: number
};

export type InputInfo = {
  name: string,
  shapeInfo: ShapeInfo
};

export function makeShader(
    inputsInfo: InputInfo[], outputShape: ShapeInfo, userCode: string,
    usesPackedTextures: boolean): string {
  const prefixSnippets: string[] = [];
  inputsInfo.forEach(x => {
    const size = util.sizeFromShape(x.shapeInfo.logicalShape);

    // Snippet when we decided to upload the values as uniform.
    if (x.shapeInfo.isUniform) {
      prefixSnippets.push(
          `uniform float ${x.name}${size > 1 ? `[${size}]` : ''};`);
    } else {
      prefixSnippets.push(`uniform sampler2D ${x.name};`);
      prefixSnippets.push(`uniform int offset${x.name};`);
    }
  });
  const inputPrefixSnippet = prefixSnippets.join('\n');

  const inputSamplingSnippet =
      inputsInfo
          .map(x => getInputSamplingSnippet(x, outputShape, usesPackedTextures))
          .join('\n');
  const outTexShape = outputShape.texShape;
  const glsl = getGlslDifferences();
  const floatTextureSampleSnippet = getFloatTextureSampleSnippet(glsl);
  let outputSamplingSnippet: string;
  let floatTextureSetOutputSnippet: string;
  let shaderPrefix = getShaderPrefix(glsl);

  if (outputShape.isPacked) {
    outputSamplingSnippet =
        getPackedOutputSamplingSnippet(outputShape.logicalShape, outTexShape);
    floatTextureSetOutputSnippet = getFloatTextureSetRGBASnippet(glsl);
  } else {
    outputSamplingSnippet =
        getOutputSamplingSnippet(outputShape.logicalShape, outTexShape);
    floatTextureSetOutputSnippet = getFloatTextureSetRSnippet(glsl);
  }

  if (usesPackedTextures) {
    shaderPrefix += SHADER_PACKED_PREFIX;
  }

  const source = [
    shaderPrefix, floatTextureSampleSnippet, floatTextureSetOutputSnippet,
    inputPrefixSnippet, outputSamplingSnippet, inputSamplingSnippet, userCode
  ].join('\n');
  return source;
}

export function makeCSShader(
    inputsInfo: InputInfo[], outputShape: ShapeInfo, userCode: string,
    usesPackedTextures: boolean): string {
  const prefixSnippets: string[] = [];
  inputsInfo.forEach(x => {
    const size = util.sizeFromShape(x.shapeInfo.logicalShape);

    // Snippet when we decided to upload the values as uniform.
    if (x.shapeInfo.isUniform) {
      prefixSnippets.push(
          `uniform float ${x.name}${size > 1 ? `[${size}]` : ''};`);
    } else {
      prefixSnippets.push(`uniform sampler2D ${x.name};`);
      prefixSnippets.push(`uniform int offset${x.name};`);
    }
  });
  const inputPrefixSnippet = prefixSnippets.join('\n');

  const inputSamplingSnippet = inputsInfo
                                   .map(
                                       x => getInputSamplingSnippetCS(
                                           x, outputShape, usesPackedTextures))
                                   .join('\n');
  const outTexShape = outputShape.texShape;
  const glsl = getGlslDifferences();
  const floatTextureSampleSnippet = getFloatTextureSampleSnippetCS(glsl);
  let outputSamplingSnippet: string;
  let floatTextureSetOutputSnippet: string;
  let shaderPrefix: string;

  if (outputShape.isPacked) {
    shaderPrefix = getShaderPrefixCS(glsl, 'rgba32f');
    outputSamplingSnippet =
        getPackedOutputSamplingSnippetCS(outputShape.logicalShape, outTexShape);
    floatTextureSetOutputSnippet = getFloatTextureSetRGBASnippetCS(glsl);
  } else {
    shaderPrefix = getShaderPrefixCS(glsl, 'r32f');
    outputSamplingSnippet =
        getOutputSamplingSnippetCS(outputShape.logicalShape, outTexShape);
    floatTextureSetOutputSnippet = getFloatTextureSetRSnippetCS(glsl);
  }

  if (usesPackedTextures) {
    shaderPrefix += SHADER_PACKED_PREFIX;
  }

  const source = [
    shaderPrefix, floatTextureSampleSnippet, floatTextureSetOutputSnippet,
    inputPrefixSnippet, outputSamplingSnippet, inputSamplingSnippet, userCode
  ].join('\n');
  return source;
}

function getSamplerFromInInfo(inInfo: InputInfo): string {
  const shape = inInfo.shapeInfo.logicalShape;
  switch (shape.length) {
    case 0:
      return getSamplerScalar(inInfo);
    case 1:
      return getSampler1D(inInfo);
    case 2:
      return getSampler2D(inInfo);
    case 3:
      return getSampler3D(inInfo);
    case 4:
      return getSampler4D(inInfo);
    case 5:
      return getSampler5D(inInfo);
    case 6:
      return getSampler6D(inInfo);
    default:
      throw new Error(
          `${shape.length}-D input sampling` +
          ` is not yet supported`);
  }
}

function getSamplerFromInInfoCS(inInfo: InputInfo): string {
  const shape = inInfo.shapeInfo.logicalShape;
  switch (shape.length) {
    case 0:
      return getSamplerScalarCS(inInfo);
    case 1:
      return getSampler1DCS(inInfo);
    case 2:
      return getSampler2DCS(inInfo);
    case 3:
      return getSampler3DCS(inInfo);
    case 4:
      return getSampler4DCS(inInfo);
    default:
      throw new Error(
          `${shape.length}-D input sampling` +
          ` is not yet supported`);
  }
}

function getPackedSamplerFromInInfo(inInfo: InputInfo): string {
  const shape = inInfo.shapeInfo.logicalShape;
  switch (shape.length) {
    case 0:
      return getPackedSamplerScalar(inInfo);
    case 1:
      return getPackedSampler1D(inInfo);
    case 2:
      return getPackedSampler2D(inInfo);
    case 3:
      return getPackedSampler3D(inInfo);
    default:
      return getPackedSamplerND(inInfo);
  }
}

function getPackedSamplerFromInInfoCS(inInfo: InputInfo): string {
  const shape = inInfo.shapeInfo.logicalShape;
  switch (shape.length) {
    case 0:
      return getPackedSamplerScalarCS(inInfo);
    case 1:
      return getPackedSampler1DCS(inInfo);
    case 2:
      return getPackedSampler2DCS(inInfo);
    case 3:
      return getPackedSampler3DCS(inInfo);
    default:
      return getPackedSamplerNDCS(inInfo);
  }
}

function getInputSamplingSnippetCS(
    inInfo: InputInfo, outShapeInfo: ShapeInfo,
    usesPackedTextures = false): string {
  let res = '';
  if (usesPackedTextures) {
    res += getPackedSamplerFromInInfoCS(inInfo);
  } else {
    res += getSamplerFromInInfoCS(inInfo);
  }

  const inShape = inInfo.shapeInfo.logicalShape;
  const outShape = outShapeInfo.logicalShape;
  if (inShape.length <= outShape.length) {
    if (usesPackedTextures) {
      res += getPackedSamplerAtOutputCoords(inInfo, outShapeInfo);
    } else {
      res += getSamplerAtOutputCoordsCS(inInfo, outShapeInfo);
    }
  }
  return res;
}

function getInputSamplingSnippet(
    inInfo: InputInfo, outShapeInfo: ShapeInfo,
    usesPackedTextures = false): string {
  let res = '';
  if (usesPackedTextures) {
    res += getPackedSamplerFromInInfo(inInfo);
  } else {
    res += getSamplerFromInInfo(inInfo);
  }

  const inShape = inInfo.shapeInfo.logicalShape;
  const outShape = outShapeInfo.logicalShape;
  if (inShape.length <= outShape.length) {
    if (usesPackedTextures) {
      res += getPackedSamplerAtOutputCoords(inInfo, outShapeInfo);
    } else {
      res += getSamplerAtOutputCoords(inInfo, outShapeInfo);
    }
  }
  return res;
}

function getPackedOutputSamplingSnippet(
    outShape: number[], outTexShape: [number, number]): string {
  switch (outShape.length) {
    case 0:
      return getOutputScalarCoords();
    case 1:
      return getOutputPacked1DCoords(outShape as [number], outTexShape);
    case 2:
      return getOutputPacked2DCoords(outShape as [number, number], outTexShape);
    case 3:
      return getOutputPacked3DCoords(
          outShape as [number, number, number], outTexShape);
    default:
      return getOutputPackedNDCoords(outShape, outTexShape);
  }
}

function getPackedOutputSamplingSnippetCS(
    outShape: number[], outTexShape: [number, number]): string {
  switch (outShape.length) {
    case 0:
      return getOutputScalarCoords();
    case 1:
      return getOutputPacked1DCoordsCS(outShape as [number], outTexShape);
    case 2:
      return getOutputPacked2DCoordsCS(
          outShape as [number, number], outTexShape);
    case 3:
      return getOutputPacked3DCoordsCS(
          outShape as [number, number, number], outTexShape);
    default:
      return getOutputPackedNDCoordsCS(outShape, outTexShape);
  }
}

function getOutputSamplingSnippet(
    outShape: number[], outTexShape: [number, number]): string {
  switch (outShape.length) {
    case 0:
      return getOutputScalarCoords();
    case 1:
      return getOutput1DCoords(outShape as [number], outTexShape);
    case 2:
      return getOutput2DCoords(outShape as [number, number], outTexShape);
    case 3:
      return getOutput3DCoords(
          outShape as [number, number, number], outTexShape);
    case 4:
      return getOutput4DCoords(
          outShape as [number, number, number, number], outTexShape);
    case 5:
      return getOutput5DCoords(
          outShape as [number, number, number, number, number], outTexShape);
    case 6:
      return getOutput6DCoords(
          outShape as [number, number, number, number, number, number],
          outTexShape);
    default:
      throw new Error(
          `${outShape.length}-D output sampling is not yet supported`);
  }
}

function getOutputSamplingSnippetCS(
    outShape: number[], outTexShape: [number, number]): string {
  switch (outShape.length) {
    case 0:
      return getOutputScalarCoords();
    case 1:
      return getOutput1DCoordsCS(outShape as [number], outTexShape);
    case 2:
      return getOutput2DCoordsCS(outShape as [number, number], outTexShape);
    case 3:
      return getOutput3DCoordsCS(
          outShape as [number, number, number], outTexShape);
    case 4:
      return getOutput4DCoordsCS(
          outShape as [number, number, number, number], outTexShape);
    case 5:
      return getOutput5DCoords(
          outShape as [number, number, number, number, number], outTexShape);
    case 6:
      return getOutput6DCoords(
          outShape as [number, number, number, number, number, number],
          outTexShape);
    default:
      throw new Error(
          `${outShape.length}-D output sampling is not yet supported`);
  }
}

function getFloatTextureSampleSnippet(glsl: GLSL): string {
  return `
    float sampleTexture(sampler2D textureSampler, vec2 uv) {
      return ${glsl.texture2D}(textureSampler, uv).r;
    }
  `;
}

function getFloatTextureSampleSnippetCS(glsl: GLSL): string {
  return `
    float sampleTexture(sampler2D textureSampler, ivec2 uv) {
      return texelFetch(textureSampler, uv, 0).r;
    }
  `;
}

function getFloatTextureSetRSnippet(glsl: GLSL): string {
  return `
    void setOutput(float val) {
      ${glsl.output} = vec4(val, 0, 0, 0);
    }
  `;
}

function getFloatTextureSetRGBASnippet(glsl: GLSL): string {
  return `
    void setOutput(vec4 val) {
      ${glsl.output} = val;
    }
  `;
}

function getFloatTextureSetRSnippetCS(glsl: GLSL): string {
  return `
    void setOutput(float val) {
      imageStore(
          ${glsl.output}, ivec2(gl_GlobalInvocationID.xy), vec4(val, 0, 0, 0));
    }
  `;
}

function getFloatTextureSetRGBASnippetCS(glsl: GLSL): string {
  return `
    void setOutput(vec4 val) {
      imageStore(
        ${glsl.output}, ivec2(gl_GlobalInvocationID.xy), val);
    }
  `;
}

function getShaderPrefix(glsl: GLSL): string {
  let nanChecks = '';
  if (ENV.get('PROD')) {
    nanChecks = `
      bool isNaN(float val) {
        return false;
      }

      bool hasNaN(vec4 values) {
        return false;
      }
    `;
  } else {
    /**
     * Previous NaN check '(val < 0.0 || 0.0 < val || val == 0.0) ? false :
     * true' does not work on iOS 12
     */
    nanChecks = `
      bool isNaN(float val) {
        return (val < 1.0 || 0.0 < val || val == 0.0) ? false : true;
      }

      bvec4 isNaN(vec4 val) {
        return bvec4(
          isNaN(val.x),
          isNaN(val.y),
          isNaN(val.z),
          isNaN(val.w)
        );
      }

      bool hasNaN(vec4 values) {
        return any(bvec4(
          isNaN(values.x),
          isNaN(values.y),
          isNaN(values.z),
          isNaN(values.w)
        ));
      }
    `;
  }

  const SHADER_PREFIX = `${glsl.version}
    precision highp float;
    precision highp int;
    precision highp sampler2D;
    ${glsl.varyingFs} vec2 resultUV;
    ${glsl.defineOutput}
    const vec2 halfCR = vec2(0.5, 0.5);

    struct ivec5
    {
      int x;
      int y;
      int z;
      int w;
      int u;
    };

    struct ivec6
    {
      int x;
      int y;
      int z;
      int w;
      int u;
      int v;
    };

    ${nanChecks}

    float getNaN(vec4 values) {
      return dot(vec4(1), values);
    }

    ${glsl.defineRound}

    int imod(int x, int y) {
      return x - y * (x / y);
    }

    //Based on the work of Dave Hoskins
    //https://www.shadertoy.com/view/4djSRW
    #define HASHSCALE1 443.8975
    float random(float seed){
      vec2 p = resultUV * seed;
      vec3 p3  = fract(vec3(p.xyx) * HASHSCALE1);
      p3 += dot(p3, p3.yzx + 19.19);
      return fract((p3.x + p3.y) * p3.z);
    }

    ${SAMPLE_1D_SNIPPET}
    ${SAMPLE_2D_SNIPPET}
    ${SAMPLE_3D_SNIPPET}
  `;

  return SHADER_PREFIX;
}

const SAMPLE_1D_SNIPPET = `
vec2 uvFromFlat(int texNumR, int texNumC, int index) {
  int texR = index / texNumC;
  int texC = index - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
vec2 packedUVfrom1D(int texNumR, int texNumC, int index) {
  int texelIndex = index / 2;
  int texR = texelIndex / texNumC;
  int texC = texelIndex - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
`;

const SAMPLE_1D_SNIPPET_CS = `
ivec2 uvFromFlat(int texNumR, int texNumC, int index) {
  int texR = index / texNumC;
  int texC = index - texR * texNumC;
  return ivec2(texC, texR);
}
ivec2 packedUVfrom1D(int texNumR, int texNumC, int index) {
  int texelIndex = index / 2;
  int texR = texelIndex / texNumC;
  int texC = texelIndex - texR * texNumC;
  return ivec2(texC, texR);
}
`;

const SAMPLE_2D_SNIPPET = `
vec2 packedUVfrom2D(int texelsInLogicalRow, int texNumR,
  int texNumC, int row, int col) {
  int texelIndex = (row / 2) * texelsInLogicalRow + (col / 2);
  int texR = texelIndex / texNumC;
  int texC = texelIndex - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
`;

const SAMPLE_2D_SNIPPET_CS = `
ivec2 packedUVfrom2D(int texelsInLogicalRow, int texNumR,
  int texNumC, int row, int col) {
  int texelIndex = (row / 2) * texelsInLogicalRow + (col / 2);
  int texR = texelIndex / texNumC;
  int texC = texelIndex - texR * texNumC;
  return ivec2(texC, texR);
}
`;

const SAMPLE_3D_SNIPPET = `
vec2 packedUVfrom3D(int texNumR, int texNumC,
    int texelsInBatch, int texelsInLogicalRow, int b,
    int row, int col) {
  int index = b * texelsInBatch + (row / 2) * texelsInLogicalRow + (col / 2);
  int texR = index / texNumC;
  int texC = index - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
`;

const SAMPLE_3D_SNIPPET_CS = `
ivec2 packedUVfrom3D(int texNumR, int texNumC,
    int texelsInBatch, int texelsInLogicalRow, int b,
    int row, int col) {
  int index = b * texelsInBatch + (row / 2) * texelsInLogicalRow + (col / 2);
  int texR = index / texNumC;
  int texC = index - texR * texNumC;
  return ivec2(texC, texR);
}
`;

const SHADER_PACKED_PREFIX = `
  float getChannel(vec4 frag, vec2 innerDims) {
    vec2 modCoord = mod(innerDims, 2.);
    return modCoord.x == 0. ?
      (modCoord.y == 0. ? frag.r : frag.g) :
      (modCoord.y == 0. ? frag.b : frag.a);
  }
  float getChannel(vec4 frag, int dim) {
    float modCoord = mod(float(dim), 2.);
    return modCoord == 0. ? frag.r : frag.g;
  }
`;

function getShaderPrefixCS(glsl: GLSL, imageFormat: string): string {
  let nanChecks = '';
  if (ENV.get('PROD')) {
    nanChecks = `
      bool isNaN(float val) {
        return false;
      }

      bool hasNaN(vec4 values) {
        return false;
      }
    `;
  } else {
    /**
     * Previous NaN check '(val < 0.0 || 0.0 < val || val == 0.0) ? false :
     * true' does not work on iOS 12
     */
    nanChecks = `
      bool isNaN(float val) {
        return (val < 1.0 || 0.0 < val || val == 0.0) ? false : true;
      }

      bvec4 isNaN(vec4 val) {
        return bvec4(
          isNaN(val.x),
          isNaN(val.y),
          isNaN(val.z),
          isNaN(val.w)
        );
      }

      bool hasNaN(vec4 values) {
        return any(bvec4(
          isNaN(values.x),
          isNaN(values.y),
          isNaN(values.z),
          isNaN(values.w)
        ));
      }
    `;
  }

  const SHADER_PREFIX = `#version 310 es
    #define TILE_WIDTH 32
    #define TILE_HEIGHT 32
    precision highp float;
    precision highp int;
    layout(local_size_x=TILE_WIDTH, local_size_y=TILE_HEIGHT) in;
    precision highp sampler2D;
    layout(${imageFormat}, binding = 4) writeonly
       uniform highp image2D outputColor;
    const vec2 halfCR = vec2(0.5, 0.5);

    struct ivec5
    {
      int x;
      int y;
      int z;
      int w;
      int u;
    };

    struct ivec6
    {
      int x;
      int y;
      int z;
      int w;
      int u;
      int v;
    };

    ${nanChecks}

    float getNaN(vec4 values) {
      return dot(vec4(1), values);
    }

    ${glsl.defineRound}

    int imod(int x, int y) {
      return x - y * (x / y);
    }

    //Based on the work of Dave Hoskins
    //https://www.shadertoy.com/view/4djSRW
    #define HASHSCALE1 443.8975

    ${SAMPLE_1D_SNIPPET_CS}
    ${SAMPLE_2D_SNIPPET_CS}
    ${SAMPLE_3D_SNIPPET_CS}
  `;

  return SHADER_PREFIX;
}

function getOutputScalarCoords() {
  return `
    int getOutputCoords() {
      return 0;
    }
  `;
}

function getOutputPacked1DCoords(
    shape: [number], texShape: [number, number]): string {
  const packedTexShape =
      [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
  if (packedTexShape[0] === 1) {
    return `
      int getOutputCoords() {
        return 2 * int(resultUV.x * ${packedTexShape[1]}.0);
      }
    `;
  }

  if (packedTexShape[1] === 1) {
    return `
      int getOutputCoords() {
        return 2 * int(resultUV.y * ${packedTexShape[0]}.0);
      }
    `;
  }

  return `
    int getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${packedTexShape[0]}, ${packedTexShape[1]}));
      return resTexRC.x * ${packedTexShape[1]} + resTexRC.y;
    }
  `;
}

function getOutputPacked1DCoordsCS(
    shape: [number], texShape: [number, number]): string {
  const packedTexShape =
      [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
  if (packedTexShape[0] === 1) {
    return `
    int getOutputCoords() {
      return 2 * int(gl_GlobalInvocationID.x);
    }
  `;
  }

  if (packedTexShape[1] === 1) {
    return `
    int getOutputCoords() {
      return 2 * int(gl_GlobalInvocationID.y);
    }
  `;
  }

  return `
  int getOutputCoords() {
    return int(gl_GlobalInvocationID.y) * ${packedTexShape[1]} +
       int(gl_GlobalInvocationID.x);
  }
`;
}

function getOutput1DCoords(
    shape: [number], texShape: [number, number]): string {
  if (texShape[0] === 1) {
    return `
      int getOutputCoords() {
        return int(resultUV.x * ${texShape[1]}.0);
      }
    `;
  }
  if (texShape[1] === 1) {
    return `
      int getOutputCoords() {
        return int(resultUV.y * ${texShape[0]}.0);
      }
    `;
  }
  return `
    int getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${texShape[0]}, ${texShape[1]}));
      return resTexRC.x * ${texShape[1]} + resTexRC.y;
    }
  `;
}

function getOutput1DCoordsCS(
    shape: [number], texShape: [number, number]): string {
  if (texShape[0] === 1) {
    return `
      int getOutputCoords() {
        return int(gl_GlobalInvocationID.x);
      }
    `;
  }
  if (texShape[1] === 1) {
    return `
      int getOutputCoords() {
        return int(gl_GlobalInvocationID.y);
      }
    `;
  }
  return `
  int getOutputCoords() {
    return int(gl_GlobalInvocationID.y) * ${texShape[1]} +
        int(gl_GlobalInvocationID.x);
  }
`;
}

function getOutputPacked3DCoords(
    shape: [number, number, number], texShape: [number, number]): string {
  const packedTexShape =
      [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
  const texelsInLogicalRow = Math.ceil(shape[2] / 2);
  const texelsInBatch = texelsInLogicalRow * Math.ceil(shape[1] / 2);

  return `
    ivec3 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${packedTexShape[0]}, ${packedTexShape[1]}));
      int index = resTexRC.x * ${packedTexShape[1]} + resTexRC.y;

      int b = index / ${texelsInBatch};
      index -= b * ${texelsInBatch};

      int r = 2 * (index / ${texelsInLogicalRow});
      int c = imod(index, ${texelsInLogicalRow}) * 2;

      return ivec3(b, r, c);
    }
  `;
}

function getOutputPacked3DCoordsCS(
    shape: [number, number, number], texShape: [number, number]): string {
  const packedTexShape =
      [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
  const texelsInLogicalRow = Math.ceil(shape[2] / 2);
  const texelsInBatch = texelsInLogicalRow * Math.ceil(shape[1] / 2);

  return `
  ivec3 getOutputCoords() {
    ivec2 resTexRC = ivec2(gl_GlobalInvocationID.yx);
    int index = resTexRC.x * ${packedTexShape[1]} + resTexRC.y;

    int b = index / ${texelsInBatch};
    index -= b * ${texelsInBatch};

    int r = 2 * (index / ${texelsInLogicalRow});
    int c = imod(index, ${texelsInLogicalRow}) * 2;

    return ivec3(b, r, c);
  }
`;
}

function getOutput3DCoords(
    shape: [number, number, number], texShape: [number, number]): string {
  const coordsFromIndexSnippet =
      shader_util.getLogicalCoordinatesFromFlatIndex(['r', 'c', 'd'], shape);

  return `
    ivec3 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${texShape[0]}, ${texShape[1]}));
      int index = resTexRC.x * ${texShape[1]} + resTexRC.y;
      ${coordsFromIndexSnippet}
      return ivec3(r, c, d);
    }
  `;
}

function getOutput3DCoordsCS(
    shape: [number, number, number], texShape: [number, number]): string {
  const stride0 = shape[1] * shape[2];
  const stride1 = shape[2];
  return `
  ivec3 getOutputCoords() {
    int index = int(gl_GlobalInvocationID.y) * ${texShape[1]} +
        int(gl_GlobalInvocationID.x);
    int r = index / ${stride0};
    index -= r * ${stride0};
    int c = index / ${stride1};
    int d = index - c * ${stride1};
    return ivec3(r, c, d);
  }
`;
}

function getOutputPackedNDCoords(
    shape: number[], texShape: [number, number]): string {
  const packedTexShape =
      [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];

  const texelsInLogicalRow = Math.ceil(shape[shape.length - 1] / 2);
  const texelsInBatch =
      texelsInLogicalRow * Math.ceil(shape[shape.length - 2] / 2);
  let texelsInBatchN = texelsInBatch;
  let batches = ``;
  let coords = 'b, r, c';

  for (let b = 2; b < shape.length - 1; b++) {
    texelsInBatchN *= shape[shape.length - b - 1];
    batches = `
      int b${b} = index / ${texelsInBatchN};
      index -= b${b} * ${texelsInBatchN};
    ` + batches;
    coords = `b${b}, ` + coords;
  }

  return `
    ivec${shape.length} getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${packedTexShape[0]}, ${packedTexShape[1]}));
      int index = resTexRC.x * ${packedTexShape[1]} + resTexRC.y;

      ${batches}

      int b = index / ${texelsInBatch};
      index -= b * ${texelsInBatch};

      int r = 2 * (index / ${texelsInLogicalRow});
      int c = imod(index, ${texelsInLogicalRow}) * 2;

      return ivec${shape.length}(${coords});
    }
  `;
}

function getOutputPackedNDCoordsCS(
    shape: number[], texShape: [number, number]): string {
  const packedTexShape =
      [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];

  const texelsInLogicalRow = Math.ceil(shape[shape.length - 1] / 2);
  const texelsInBatch =
      texelsInLogicalRow * Math.ceil(shape[shape.length - 2] / 2);
  let texelsInBatchN = texelsInBatch;
  let batches = ``;
  let coords = 'b, r, c';

  for (let b = 2; b < shape.length - 1; b++) {
    texelsInBatchN *= shape[shape.length - b - 1];
    batches = `
    int b${b} = index / ${texelsInBatchN};
    index -= b${b} * ${texelsInBatchN};
  ` + batches;
    coords = `b${b}, ` + coords;
  }

  return `
  ivec${shape.length} getOutputCoords() {
    ivec2 resTexRC = ivec2(gl_GlobalInvocationID.yx);
    int index = resTexRC.x * ${packedTexShape[1]} + resTexRC.y;

    ${batches}

    int b = index / ${texelsInBatch};
    index -= b * ${texelsInBatch};

    int r = 2 * (index / ${texelsInLogicalRow});
    int c = imod(index, ${texelsInLogicalRow}) * 2;

    return ivec${shape.length}(${coords});
  }
`;
}

function getOutput4DCoords(
    shape: [number, number, number, number],
    texShape: [number, number]): string {
  const coordsFromIndexSnippet = shader_util.getLogicalCoordinatesFromFlatIndex(
      ['r', 'c', 'd', 'd2'], shape);

  return `
    ivec4 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
        vec2(${texShape[0]}, ${texShape[1]}));
      int index = resTexRC.x * ${texShape[1]} + resTexRC.y;
      ${coordsFromIndexSnippet}
      return ivec4(r, c, d, d2);
    }
  `;
}

function getOutput5DCoords(
    shape: [number, number, number, number, number],
    texShape: [number, number]): string {
  const coordsFromIndexSnippet = shader_util.getLogicalCoordinatesFromFlatIndex(
      ['r', 'c', 'd', 'd2', 'd3'], shape);

  return `
    ivec5 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx * vec2(${texShape[0]},
                             ${texShape[1]}));

      int index = resTexRC.x * ${texShape[1]} + resTexRC.y;

      ${coordsFromIndexSnippet}

      ivec5 outShape = ivec5(r, c, d, d2, d3);
      return outShape;
    }
  `;
}

function getOutput6DCoords(
    shape: [number, number, number, number, number, number],
    texShape: [number, number]): string {
  const coordsFromIndexSnippet = shader_util.getLogicalCoordinatesFromFlatIndex(
      ['r', 'c', 'd', 'd2', 'd3', 'd4'], shape);

  return `
    ivec6 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
        vec2(${texShape[0]}, ${texShape[1]}));
      int index = resTexRC.x * ${texShape[1]} + resTexRC.y;

      ${coordsFromIndexSnippet}

      ivec6 result = ivec6(r, c, d, d2, d3, d4);
      return result;
    }
  `;
}

function getOutputPacked2DCoords(
    shape: [number, number], texShape: [number, number]): string {
  const packedTexShape =
      [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
  if (util.arraysEqual(shape, texShape)) {
    return `
      ivec2 getOutputCoords() {
        return 2 * ivec2(resultUV.yx * vec2(${packedTexShape[0]}, ${
        packedTexShape[1]}));
      }
    `;
  }

  // texels needed to accommodate a logical row
  const texelsInLogicalRow = Math.ceil(shape[1] / 2);

  /**
   * getOutputCoords
   *
   * resTexRC: The rows and columns of the texels. If you move over one
   * texel to the right in the packed texture, you are moving over one column
   * (not two).
   *
   * index: The texel index
   */
  return `
    ivec2 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${packedTexShape[0]}, ${packedTexShape[1]}));

      int index = resTexRC.x * ${packedTexShape[1]} + resTexRC.y;
      int r = 2 * (index / ${texelsInLogicalRow});
      int c = imod(index, ${texelsInLogicalRow}) * 2;

      return ivec2(r, c);
    }
  `;
}

function getOutputPacked2DCoordsCS(
    shape: [number, number], texShape: [number, number]): string {
  const packedTexShape =
      [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
  if (util.arraysEqual(shape, texShape)) {
    return `
    ivec2 getOutputCoords() {
      return 2 * ivec2(gl_GlobalInvocationID.yx);
    }
  `;
  }

  // texels needed to accommodate a logical row
  const texelsInLogicalRow = Math.ceil(shape[1] / 2);

  /**
   * getOutputCoords
   *
   * resTexRC: The rows and columns of the texels. If you move over one
   * texel to the right in the packed texture, you are moving over one column
   * (not two).
   *
   * index: The texel index
   */
  return `
  ivec2 getOutputCoords() {
    ivec2 resTexRC = ivec2(gl_GlobalInvocationID.yx);

    int index = resTexRC.x * ${packedTexShape[1]} + resTexRC.y;
    int r = 2 * (index / ${texelsInLogicalRow});
    int c = imod(index, ${texelsInLogicalRow}) * 2;

    return ivec2(r, c);
  }
`;
}

function getOutput4DCoordsCS(
    shape: [number, number, number, number],
    texShape: [number, number]): string {
  const stride2 = shape[3];
  const stride1 = shape[2] * stride2;
  const stride0 = shape[1] * stride1;
  return `
  ivec4 getOutputCoords() {
    int index = int(gl_GlobalInvocationID.y) * ${texShape[1]} +
        int(gl_GlobalInvocationID.x);

    int r = index / ${stride0};
    index -= r * ${stride0};

    int c = index / ${stride1};
    index -= c * ${stride1};

    int d = index / ${stride2};
    int d2 = index - d * ${stride2};

    return ivec4(r, c, d, d2);
  }
`;
}

function getOutput2DCoords(
    shape: [number, number], texShape: [number, number]): string {
  if (util.arraysEqual(shape, texShape)) {
    return `
      ivec2 getOutputCoords() {
        return ivec2(resultUV.yx * vec2(${texShape[0]}, ${texShape[1]}));
      }
    `;
  }
  if (shape[1] === 1) {
    return `
      ivec2 getOutputCoords() {
        ivec2 resTexRC = ivec2(resultUV.yx *
                               vec2(${texShape[0]}, ${texShape[1]}));
        int index = resTexRC.x * ${texShape[1]} + resTexRC.y;
        return ivec2(index, 0);
      }
    `;
  }
  if (shape[0] === 1) {
    return `
      ivec2 getOutputCoords() {
        ivec2 resTexRC = ivec2(resultUV.yx *
                               vec2(${texShape[0]}, ${texShape[1]}));
        int index = resTexRC.x * ${texShape[1]} + resTexRC.y;
        return ivec2(0, index);
      }
    `;
  }
  return `
    ivec2 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${texShape[0]}, ${texShape[1]}));
      int index = resTexRC.x * ${texShape[1]} + resTexRC.y;
      int r = index / ${shape[1]};
      int c = index - r * ${shape[1]};
      return ivec2(r, c);
    }
  `;
}

function getOutput2DCoordsCS(
    shape: [number, number], texShape: [number, number]): string {
  if (util.arraysEqual(shape, texShape)) {
    return `
    ivec2 getOutputCoords() {
      return ivec2(gl_GlobalInvocationID.yx);
    }
  `;
  }
  if (shape[1] === 1) {
    return `
    ivec2 getOutputCoords() {
      int index = int(gl_GlobalInvocationID.y) * ${texShape[1]} +
          int(gl_GlobalInvocationID.x);
      return ivec2(index, 0);
    }
  `;
  }
  if (shape[0] === 1) {
    return `
    ivec2 getOutputCoords() {
      int index = int(gl_GlobalInvocationID.y) * ${texShape[1]} +
          int(gl_GlobalInvocationID.x);
      return ivec2(0, index);
    }
  `;
  }
  return `
  ivec2 getOutputCoords() {
    int index = int(gl_GlobalInvocationID.y) * ${texShape[1]} +
        int(gl_GlobalInvocationID.x);
    int r = index / ${shape[1]};
    int c = index - r * ${shape[1]};
    return ivec2(r, c);
  }
`;
}

function getFlatOffsetUniformName(texName: string): string {
  return `offset${texName}`;
}

function getPackedSamplerScalar(inputInfo: InputInfo): string {
  const texName = inputInfo.name;
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  const glsl = getGlslDifferences();
  return `
    vec4 ${funcName}() {
      return ${glsl.texture2D}(${texName}, halfCR);
    }
  `;
}

function getPackedSamplerScalarCS(inputInfo: InputInfo): string {
  const texName = inputInfo.name;
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  return `
    vec4 ${funcName}() {
      return texelFetch(${texName}, ivec2(0, 0), 0);
    }
  `;
}

function getSamplerScalar(inputInfo: InputInfo): string {
  const texName = inputInfo.name;
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  if (inputInfo.shapeInfo.isUniform) {
    return `float ${funcName}() {return ${texName};}`;
  }
  const [texNumR, texNumC] = inputInfo.shapeInfo.texShape;
  if (texNumR === 1 && texNumC === 1) {
    return `
      float ${funcName}() {
        return sampleTexture(${texName}, halfCR);
      }
    `;
  }

  const [tNumR, tNumC] = inputInfo.shapeInfo.texShape;
  const offset = getFlatOffsetUniformName(texName);
  return `
    float ${funcName}() {
      vec2 uv = uvFromFlat(${tNumR}, ${tNumC}, ${offset});
      return sampleTexture(${texName}, uv);
    }
  `;
}

function getSamplerScalarCS(inputInfo: InputInfo): string {
  const texName = inputInfo.name;
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  if (inputInfo.shapeInfo.isUniform) {
    return `float ${funcName}() {return ${texName};}`;
  }
  const [texNumR, texNumC] = inputInfo.shapeInfo.texShape;
  if (texNumR === 1 && texNumC === 1) {
    return `
      float ${funcName}() {
        return sampleTexture(${texName}, ivec2(0, 0));
      }
    `;
  }

  const [tNumR, tNumC] = inputInfo.shapeInfo.texShape;
  const offset = getFlatOffsetUniformName(texName);
  return `
    float ${funcName}() {
      ivec2 uv = uvFromFlat(${tNumR}, ${tNumC}, ${offset});
      return sampleTexture(${texName}, uv);
    }
  `;
}

function getPackedSampler1D(inputInfo: InputInfo): string {
  const texName = inputInfo.name;
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  const texShape = inputInfo.shapeInfo.texShape;
  const packedTexShape =
      [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
  const glsl = getGlslDifferences();

  return `
    vec4 ${funcName}(int index) {
      vec2 uv = packedUVfrom1D(
        ${packedTexShape[0]}, ${packedTexShape[1]}, index);
      return ${glsl.texture2D}(${texName}, uv);
    }
  `;
}

function getPackedSampler1DCS(inputInfo: InputInfo): string {
  const texName = inputInfo.name;
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  const texShape = inputInfo.shapeInfo.texShape;
  const packedTexShape =
      [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];

  return `
    vec4 ${funcName}(int index) {
      ivec2 uv = packedUVfrom1D(
        ${packedTexShape[0]}, ${packedTexShape[1]}, index);
      return texelFetch(${texName}, uv, 0);
    }
  `;
}

function getSampler1D(inputInfo: InputInfo): string {
  const texName = inputInfo.name;
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);

  if (inputInfo.shapeInfo.isUniform) {
    // Uniform arrays will be less than 65505 (no risk of float16 overflow).
    return `
      float ${funcName}(int index) {
        ${getUniformSampler(inputInfo)}
      }
    `;
  }

  const texShape = inputInfo.shapeInfo.texShape;
  const tNumR = texShape[0];
  const tNumC = texShape[1];

  if (tNumC === 1 && tNumR === 1) {
    return `
      float ${funcName}(int index) {
        return sampleTexture(${texName}, halfCR);
      }
    `;
  }
  const offset = getFlatOffsetUniformName(texName);
  if (tNumC === 1) {
    return `
      float ${funcName}(int index) {
        vec2 uv = vec2(0.5, (float(index + ${offset}) + 0.5) / ${tNumR}.0);
        return sampleTexture(${texName}, uv);
      }
    `;
  }
  if (tNumR === 1) {
    return `
      float ${funcName}(int index) {
        vec2 uv = vec2((float(index + ${offset}) + 0.5) / ${tNumC}.0, 0.5);
        return sampleTexture(${texName}, uv);
      }
    `;
  }
  return `
    float ${funcName}(int index) {
      vec2 uv = uvFromFlat(${tNumR}, ${tNumC}, index + ${offset});
      return sampleTexture(${texName}, uv);
    }
  `;
}

function getSampler1DCS(inputInfo: InputInfo): string {
  const texName = inputInfo.name;
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);

  if (inputInfo.shapeInfo.isUniform) {
    // Uniform arrays will be less than 65505 (no risk of float16 overflow).
    return `
      float ${funcName}(int index) {
        ${getUniformSampler(inputInfo)}
      }
    `;
  }

  const texShape = inputInfo.shapeInfo.texShape;
  const tNumR = texShape[0];
  const tNumC = texShape[1];

  if (tNumC === 1 && tNumR === 1) {
    return `
      float ${funcName}(int index) {
        return sampleTexture(${texName}, ivec2(0, 0));
      }
    `;
  }
  const offset = getFlatOffsetUniformName(texName);
  if (tNumC === 1) {
    return `
      float ${funcName}(int index) {
        ivec2 uv = ivec2(0, index + ${offset});
        return sampleTexture(${texName}, uv);
      }
    `;
  }
  if (tNumR === 1) {
    return `
      float ${funcName}(int index) {
        ivec2 uv = ivec2(index + ${offset}, 0);
        return sampleTexture(${texName}, uv);
      }
    `;
  }
  return `
    float ${funcName}(int index) {
      ivec2 uv = uvFromFlat(${tNumR}, ${tNumC}, index + ${offset});
      return sampleTexture(${texName}, uv);
    }
  `;
}

function getPackedSampler2D(inputInfo: InputInfo): string {
  const shape = inputInfo.shapeInfo.logicalShape;
  const texName = inputInfo.name;
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  const texShape = inputInfo.shapeInfo.texShape;

  const texNumR = texShape[0];
  const texNumC = texShape[1];
  const glsl = getGlslDifferences();
  if (texShape != null && util.arraysEqual(shape, texShape)) {
    return `
      vec4 ${funcName}(int row, int col) {
        vec2 uv = (vec2(col, row) + halfCR) / vec2(${texNumC}.0, ${texNumR}.0);

        return ${glsl.texture2D}(${texName}, uv);
      }
    `;
  }

  const packedTexShape =
      [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
  const valuesPerRow = Math.ceil(shape[1] / 2);

  return `
    vec4 ${funcName}(int row, int col) {
      vec2 uv = packedUVfrom2D(${valuesPerRow}, ${packedTexShape[0]}, ${
      packedTexShape[1]}, row, col);
      return ${glsl.texture2D}(${texName}, uv);
    }
  `;
}

function getPackedSampler2DCS(inputInfo: InputInfo): string {
  const shape = inputInfo.shapeInfo.logicalShape;
  const texName = inputInfo.name;
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  const texShape = inputInfo.shapeInfo.texShape;

  if (texShape != null && util.arraysEqual(shape, texShape)) {
    return `
      vec4 ${funcName}(int row, int col) {
        ivec2 uv = ivec2(col, row);
        return texelFetch(${texName}, uv, 0);
      }
    `;
  }

  const packedTexShape =
      [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
  const valuesPerRow = Math.ceil(shape[1] / 2);

  return `
    vec4 ${funcName}(int row, int col) {
      ivec2 uv = packedUVfrom2D(${valuesPerRow}, ${packedTexShape[0]}, ${
      packedTexShape[1]}, row, col);
      return texelFetch(${texName}, uv, 0);
    }
  `;
}

function getSampler2D(inputInfo: InputInfo): string {
  const shape = inputInfo.shapeInfo.logicalShape;
  const texName = inputInfo.name;
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  const texShape = inputInfo.shapeInfo.texShape;

  if (texShape != null && util.arraysEqual(shape, texShape)) {
    const texNumR = texShape[0];
    const texNumC = texShape[1];
    return `
    float ${funcName}(int row, int col) {
      vec2 uv = (vec2(col, row) + halfCR) / vec2(${texNumC}.0, ${texNumR}.0);
      return sampleTexture(${texName}, uv);
    }
  `;
  }

  const {newShape, keptDims} = util.squeezeShape(shape);
  const squeezedShape = newShape;
  if (squeezedShape.length < shape.length) {
    const newInputInfo = squeezeInputInfo(inputInfo, squeezedShape);
    const params = ['row', 'col'];
    return `
      ${getSamplerFromInInfo(newInputInfo)}
      float ${funcName}(int row, int col) {
        return ${funcName}(${getSqueezedParams(params, keptDims)});
      }
    `;
  }

  if (inputInfo.shapeInfo.isUniform) {
    // Uniform arrays will be less than 65505 (no risk of float16 overflow).
    return `
      float ${funcName}(int row, int col) {
        int index = round(dot(vec2(row, col), vec2(${shape[1]}, 1)));
        ${getUniformSampler(inputInfo)}
      }
    `;
  }

  const texNumR = texShape[0];
  const texNumC = texShape[1];
  const offset = getFlatOffsetUniformName(texName);
  if (texNumC === 1) {
    // index is used directly as physical (no risk of float16 overflow).
    return `
    float ${funcName}(int row, int col) {
      float index = dot(vec3(row, col, ${offset}), vec3(${shape[1]}, 1, 1));
      vec2 uv = vec2(0.5, (index + 0.5) / ${texNumR}.0);
      return sampleTexture(${texName}, uv);
    }
  `;
  }
  if (texNumR === 1) {
    // index is used directly as physical (no risk of float16 overflow).
    return `
    float ${funcName}(int row, int col) {
      float index = dot(vec3(row, col, ${offset}), vec3(${shape[1]}, 1, 1));
      vec2 uv = vec2((index + 0.5) / ${texNumC}.0, 0.5);
      return sampleTexture(${texName}, uv);
    }
  `;
  }

  return `
  float ${funcName}(int row, int col) {
    // Explicitly use integer operations as dot() only works on floats.
    int index = row * ${shape[1]} + col + ${offset};
    vec2 uv = uvFromFlat(${texNumR}, ${texNumC}, index);
    return sampleTexture(${texName}, uv);
  }
`;
}

function getSampler2DCS(inputInfo: InputInfo): string {
  const shape = inputInfo.shapeInfo.logicalShape;
  const texShape = inputInfo.shapeInfo.texShape;
  const texName = inputInfo.name;
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  const texNumR = texShape[0];
  const texNumC = texShape[1];
  if (texShape != null && util.arraysEqual(shape, texShape)) {
    return `
    float ${funcName}(int row, int col) {
      ivec2 uv = ivec2(col, row);
      return sampleTexture(${texName}, uv);
    }
  `;
  }
  const {newShape, keptDims} = util.squeezeShape(shape);
  const squeezedShape = newShape;
  if (squeezedShape.length < shape.length) {
    const newInputInfo = squeezeInputInfo(inputInfo, squeezedShape);
    const params = ['row', 'col'];
    return `
      ${getSamplerFromInInfoCS(newInputInfo)}
      float ${funcName}(int row, int col) {
        return ${funcName}(${getSqueezedParams(params, keptDims)});
      }
    `;
  }

  if (inputInfo.shapeInfo.isUniform) {
    // Uniform arrays will be less than 65505 (no risk of float16 overflow).
    return `
      float ${funcName}(int row, int col) {
        int index = round(dot(vec2(row, col), vec2(${shape[1]}, 1)));
        ${getUniformSampler(inputInfo)}
      }
    `;
  }
  const offset = getFlatOffsetUniformName(texName);
  if (texNumC === 1) {
    return `
    float ${funcName}(int row, int col) {
      int index = row * ${shape[1]} + col + ${offset};
      ivec2 uv = ivec2(0, index);
      return sampleTexture(${texName}, uv);
    }
  `;
  }
  if (texNumR === 1) {
    return `
    float ${funcName}(int row, int col) {
      int index = row * ${shape[1]} + col + ${offset};
      ivec2 uv = ivec2(index, 0);
      return sampleTexture(${texName}, uv);
    }
  `;
  }
  return `
  float ${funcName}(int row, int col) {
    // Explicitly use integer operations as dot() only works on floats.
    int index = row * ${shape[1]} + col + ${offset};
    ivec2 uv = uvFromFlat(${texNumR}, ${texNumC}, index);
    return sampleTexture(${texName}, uv);
  }
`;
}

function getPackedSampler3D(inputInfo: InputInfo): string {
  const shape = inputInfo.shapeInfo.logicalShape;
  const texName = inputInfo.name;
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  const texShape = inputInfo.shapeInfo.texShape;
  const packedTexShape =
      [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];

  if (shape[0] === 1) {
    const squeezedShape = shape.slice(1);
    const keptDims = [1, 2];
    const newInputInfo = squeezeInputInfo(inputInfo, squeezedShape);
    const params = ['b', 'row', 'col'];
    return `
        ${getPackedSamplerFromInInfo(newInputInfo)}
        vec4 ${funcName}(int b, int row, int col) {
          return ${funcName}(${getSqueezedParams(params, keptDims)});
        }
      `;
  }

  const texNumR = packedTexShape[0];
  const texNumC = packedTexShape[1];

  const valuesPerRow = Math.ceil(shape[2] / 2);
  const texelsInBatch = valuesPerRow * Math.ceil(shape[1] / 2);
  const glsl = getGlslDifferences();

  return `
    vec4 ${funcName}(int b, int row, int col) {
      vec2 uv = packedUVfrom3D(
        ${texNumR}, ${texNumC}, ${texelsInBatch}, ${valuesPerRow}, b, row, col);
      return ${glsl.texture2D}(${texName}, uv);
    }
  `;
}

function getPackedSampler3DCS(inputInfo: InputInfo): string {
  const shape = inputInfo.shapeInfo.logicalShape;
  const texName = inputInfo.name;
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  const texShape = inputInfo.shapeInfo.texShape;
  const packedTexShape =
      [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];

  if (shape[0] === 1) {
    const squeezedShape = shape.slice(1);
    const keptDims = [1, 2];
    const newInputInfo = squeezeInputInfo(inputInfo, squeezedShape);
    const params = ['b', 'row', 'col'];
    return `
        ${getPackedSamplerFromInInfoCS(newInputInfo)}
        vec4 ${funcName}(int b, int row, int col) {
          return ${funcName}(${getSqueezedParams(params, keptDims)});
        }
      `;
  }

  const texNumR = packedTexShape[0];
  const texNumC = packedTexShape[1];

  const valuesPerRow = Math.ceil(shape[2] / 2);
  const texelsInBatch = valuesPerRow * Math.ceil(shape[1] / 2);

  return `
    vec4 ${funcName}(int b, int row, int col) {
      ivec2 uv = packedUVfrom3D(
        ${texNumR}, ${texNumC}, ${texelsInBatch}, ${valuesPerRow}, b, row, col);
      return texelFetch(${texName}, uv, 0);
    }
  `;
}

function getSampler3D(inputInfo: InputInfo): string {
  const shape = inputInfo.shapeInfo.logicalShape;
  const texName = inputInfo.name;
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  const stride0 = shape[1] * shape[2];
  const stride1 = shape[2];

  const {newShape, keptDims} = util.squeezeShape(shape);
  const squeezedShape = newShape;
  if (squeezedShape.length < shape.length) {
    const newInputInfo = squeezeInputInfo(inputInfo, squeezedShape);
    const params = ['row', 'col', 'depth'];
    return `
        ${getSamplerFromInInfo(newInputInfo)}
        float ${funcName}(int row, int col, int depth) {
          return ${funcName}(${getSqueezedParams(params, keptDims)});
        }
      `;
  }

  if (inputInfo.shapeInfo.isUniform) {
    // Uniform arrays will be less than 65505 (no risk of float16 overflow).
    return `
      float ${funcName}(int row, int col, int depth) {
        int index = round(dot(vec3(row, col, depth),
                          vec3(${stride0}, ${stride1}, 1)));
        ${getUniformSampler(inputInfo)}
      }
    `;
  }

  const texShape = inputInfo.shapeInfo.texShape;
  const texNumR = texShape[0];
  const texNumC = texShape[1];
  const flatOffset = inputInfo.shapeInfo.flatOffset;
  if (texNumC === stride0 && flatOffset == null) {
    // texC is used directly as physical (no risk of float16 overflow).
    return `
        float ${funcName}(int row, int col, int depth) {
          float texR = float(row);
          float texC = dot(vec2(col, depth), vec2(${stride1}, 1));
          vec2 uv = (vec2(texC, texR) + halfCR) /
                     vec2(${texNumC}.0, ${texNumR}.0);
          return sampleTexture(${texName}, uv);
        }
      `;
  }

  if (texNumC === stride1 && flatOffset == null) {
    // texR is used directly as physical (no risk of float16 overflow).
    return `
    float ${funcName}(int row, int col, int depth) {
      float texR = dot(vec2(row, col), vec2(${shape[1]}, 1));
      float texC = float(depth);
      vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${texNumC}.0, ${texNumR}.0);
      return sampleTexture(${texName}, uv);
    }
  `;
  }

  const offset = getFlatOffsetUniformName(texName);
  return `
      float ${funcName}(int row, int col, int depth) {
        // Explicitly use integer operations as dot() only works on floats.
        int index = row * ${stride0} + col * ${stride1} + depth + ${offset};
        vec2 uv = uvFromFlat(${texNumR}, ${texNumC}, index);
        return sampleTexture(${texName}, uv);
      }
  `;
}

function getSampler3DCS(inputInfo: InputInfo): string {
  const texShape = inputInfo.shapeInfo.texShape;
  const shape = inputInfo.shapeInfo.logicalShape;
  const texName = inputInfo.name;
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  const texNumR = texShape[0];
  const texNumC = texShape[1];
  const stride0 = shape[1] * shape[2];
  const stride1 = shape[2];

  const {newShape, keptDims} = util.squeezeShape(shape);
  const squeezedShape = newShape;
  if (squeezedShape.length < shape.length) {
    const newInputInfo = squeezeInputInfo(inputInfo, squeezedShape);
    const params = ['row', 'col', 'depth'];
    return `
        ${getSamplerFromInInfoCS(newInputInfo)}
        float ${funcName}(int row, int col, int depth) {
          return ${funcName}(${getSqueezedParams(params, keptDims)});
        }
      `;
  }

  if (texNumC === stride0) {
    return `
        float ${funcName}(int row, int col, int depth) {
          int texR = row;
          int texC = col * ${stride1} + depth;
          ivec2 uv = ivec2(texC, texR);
          return sampleTexture(${texName}, uv);
        }
      `;
  }

  if (texNumC === stride1) {
    return `
    float ${funcName}(int row, int col, int depth) {
      int texR = row * ${shape[1]} + col;
      int texC = depth;
      ivec2 uv = ivec2(texC, texR);
      return sampleTexture(${texName}, uv);
    }
  `;
  }

  return `
      float ${funcName}(int row, int col, int depth) {
        ivec2 uv = UVfrom3D(
            ${texNumR}, ${texNumC}, ${stride0}, ${stride1}, row, col, depth);
        return sampleTexture(${texName}, uv);
      }
  `;
}

function getPackedSamplerND(inputInfo: InputInfo): string {
  const shape = inputInfo.shapeInfo.logicalShape;
  const rank = shape.length;
  const texName = inputInfo.name;
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  const texShape = inputInfo.shapeInfo.texShape;
  const packedTexShape =
      [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
  const texNumR = packedTexShape[0];
  const texNumC = packedTexShape[1];

  const valuesPerRow = Math.ceil(shape[rank - 1] / 2);
  let texelsInBatch = valuesPerRow * Math.ceil(shape[rank - 2] / 2);
  let params = `int b, int row, int col`;
  let index = `b * ${texelsInBatch} + (row / 2) * ${valuesPerRow} + (col / 2)`;
  for (let b = 2; b < rank - 1; b++) {
    params = `int b${b}, ` + params;
    texelsInBatch *= shape[rank - b - 1];
    index = `b${b} * ${texelsInBatch} + ` + index;
  }
  const glsl = getGlslDifferences();
  return `
    vec4 ${funcName}(${params}) {
      int index = ${index};
      int texR = index / ${texNumC};
      int texC = index - texR * ${texNumC};
      vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${texNumC}, ${texNumR});
      return ${glsl.texture2D}(${texName}, uv);
    }
  `;
}

function getPackedSamplerNDCS(inputInfo: InputInfo): string {
  const shape = inputInfo.shapeInfo.logicalShape;
  const rank = shape.length;
  const texName = inputInfo.name;
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  const texShape = inputInfo.shapeInfo.texShape;
  const packedTexShape =
      [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
  const texNumC = packedTexShape[1];

  const valuesPerRow = Math.ceil(shape[rank - 1] / 2);
  let texelsInBatch = valuesPerRow * Math.ceil(shape[rank - 2] / 2);
  let params = `int b, int row, int col`;
  let index = `b * ${texelsInBatch} + (row / 2) * ${valuesPerRow} + (col / 2)`;
  for (let b = 2; b < rank - 1; b++) {
    params = `int b${b}, ` + params;
    texelsInBatch *= shape[rank - b - 1];
    index = `b${b} * ${texelsInBatch} + ` + index;
  }
  return `
    vec4 ${funcName}(${params}) {
      int index = ${index};
      int texR = index / ${texNumC};
      int texC = index - texR * ${texNumC};
      ivec2 uv = ivec2(texC, texR);
      return texelFetch(${texName}, uv, 0);
    }
  `;
}

function getSampler4D(inputInfo: InputInfo): string {
  const shape = inputInfo.shapeInfo.logicalShape;
  const texName = inputInfo.name;
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  const stride2 = shape[3];
  const stride1 = shape[2] * stride2;
  const stride0 = shape[1] * stride1;

  const {newShape, keptDims} = util.squeezeShape(shape);
  if (newShape.length < shape.length) {
    const newInputInfo = squeezeInputInfo(inputInfo, newShape);
    const params = ['row', 'col', 'depth', 'depth2'];
    return `
      ${getSamplerFromInInfo(newInputInfo)}
      float ${funcName}(int row, int col, int depth, int depth2) {
        return ${funcName}(${getSqueezedParams(params, keptDims)});
      }
    `;
  }

  if (inputInfo.shapeInfo.isUniform) {
    // Uniform arrays will be less than 65505 (no risk of float16 overflow).
    return `
      float ${funcName}(int row, int col, int depth, int depth2) {
        int index = round(dot(vec4(row, col, depth, depth2),
                          vec4(${stride0}, ${stride1}, ${stride2}, 1)));
        ${getUniformSampler(inputInfo)}
      }
    `;
  }

  const flatOffset = inputInfo.shapeInfo.flatOffset;
  const texShape = inputInfo.shapeInfo.texShape;
  const texNumR = texShape[0];
  const texNumC = texShape[1];

  if (texNumC === stride0 && flatOffset == null) {
    // texC is used directly as physical (no risk of float16 overflow).
    return `
      float ${funcName}(int row, int col, int depth, int depth2) {
        float texR = float(row);
        float texC =
            dot(vec3(col, depth, depth2),
                vec3(${stride1}, ${stride2}, 1));
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${texNumC}.0, ${texNumR}.0);
        return sampleTexture(${texName}, uv);
      }
    `;
  }
  if (texNumC === stride2 && flatOffset == null) {
    // texR is used directly as physical (no risk of float16 overflow).
    return `
      float ${funcName}(int row, int col, int depth, int depth2) {
        float texR = dot(vec3(row, col, depth),
                         vec3(${shape[1] * shape[2]}, ${shape[2]}, 1));
        float texC = float(depth2);
        vec2 uv = (vec2(texC, texR) + halfCR) /
                  vec2(${texNumC}.0, ${texNumR}.0);
        return sampleTexture(${texName}, uv);
      }
    `;
  }

  const offset = getFlatOffsetUniformName(texName);
  return `
    float ${funcName}(int row, int col, int depth, int depth2) {
      // Explicitly use integer operations as dot() only works on floats.
      int index = row * ${stride0} + col * ${stride1} +
          depth * ${stride2} + depth2;
      vec2 uv = uvFromFlat(${texNumR}, ${texNumC}, index + ${offset});
      return sampleTexture(${texName}, uv);
    }
  `;
}

function getSampler4DCS(inputInfo: InputInfo): string {
  const shape = inputInfo.shapeInfo.logicalShape;
  const texShape = inputInfo.shapeInfo.texShape;
  const texName = inputInfo.name;
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  const texNumR = texShape[0];
  const texNumC = texShape[1];
  const stride2 = shape[3];
  const stride1 = shape[2] * stride2;
  const stride0 = shape[1] * stride1;
  const {newShape, keptDims} = util.squeezeShape(shape);
  if (newShape.length < shape.length) {
    const newInputInfo = squeezeInputInfo(inputInfo, newShape);
    const params = ['row', 'col', 'depth', 'depth2'];
    return `
      ${getSamplerFromInInfoCS(newInputInfo)}
      float ${funcName}(int row, int col, int depth, int depth2) {
        return ${funcName}(${getSqueezedParams(params, keptDims)});
      }
    `;
  }
  if (texNumC === stride0) {
    return `
      float ${funcName}(int row, int col, int depth, int depth2) {
        int texR = row;
        int texC = col * ${stride1} + depth * ${stride2} + depth2;
        ivec2 uv = ivec2(texC, texR);
        return sampleTexture(${texName}, uv);
      }
    `;
  }
  if (texNumC === stride2) {
    return `
      float ${funcName}(int row, int col, int depth, int depth2) {
        int texR = row * ${shape[1] * shape[2]} + col * ${shape[2]} + depth;
        int texC = depth2;
        ivec2 uv = ivec2(texC, texR);
        return sampleTexture(${texName}, uv);
      }
    `;
  }
  return `
    float ${funcName}(int row, int col, int depth, int depth2) {
      ivec2 uv = UVfrom4D(${texNumR}, ${texNumC}, ${stride0}, ${stride1},
          ${stride2}, row, col, depth, depth2);
      return sampleTexture(${texName}, uv);
    }
  `;
}

function getSampler5D(inputInfo: InputInfo): string {
  const shape = inputInfo.shapeInfo.logicalShape;
  const texName = inputInfo.name;
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  const stride3 = shape[4];
  const stride2 = shape[3] * stride3;
  const stride1 = shape[2] * stride2;
  const stride0 = shape[1] * stride1;

  const {newShape, keptDims} = util.squeezeShape(shape);
  if (newShape.length < shape.length) {
    const newInputInfo = squeezeInputInfo(inputInfo, newShape);
    const params = ['row', 'col', 'depth', 'depth2', 'depth3'];
    return `
      ${getSamplerFromInInfo(newInputInfo)}
      float ${funcName}(int row, int col, int depth, int depth2, int depth3) {
        return ${funcName}(${getSqueezedParams(params, keptDims)});
      }
    `;
  }

  if (inputInfo.shapeInfo.isUniform) {
    // Uniform arrays will be less than 65505 (no risk of float16 overflow).
    return `
      float ${funcName}(int row, int col, int depth, int depth2, int depth3) {
        float index = dot(
          vec4(row, col, depth, depth2),
          vec4(${stride0}, ${stride1}, ${stride2}, ${stride3})) +
          depth3;
        ${getUniformSampler(inputInfo)}
      }
    `;
  }

  const flatOffset = inputInfo.shapeInfo.flatOffset;
  const texShape = inputInfo.shapeInfo.texShape;
  const texNumR = texShape[0];
  const texNumC = texShape[1];

  if (texNumC === stride0 && flatOffset == null) {
    // texC is used directly as physical (no risk of float16 overflow).
    return `
      float ${funcName}(int row, int col, int depth, int depth2, int depth3) {
        int texR = row;
        float texC = dot(vec4(col, depth, depth2, depth3),
                         vec4(${stride1}, ${stride2}, ${stride3}, 1));
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${texNumC}.0, ${texNumR}.0);
        return sampleTexture(${texName}, uv);
      }
    `;
  }

  if (texNumC === stride3 && flatOffset == null) {
    // texR is used directly as physical (no risk of float16 overflow).
    return `
      float ${funcName}(int row, int col, int depth, int depth2, int depth3) {
        float texR = dot(
          vec4(row, col, depth, depth2),
          vec4(${shape[1] * shape[2] * shape[3]},
               ${shape[2] * shape[3]}, ${shape[3]}, 1));
        int texC = depth3;
        vec2 uv = (vec2(texC, texR) + halfCR) /
                  vec2(${texNumC}.0, ${texNumR}.0);
        return sampleTexture(${texName}, uv);
      }
    `;
  }

  const offset = getFlatOffsetUniformName(texName);
  return `
    float ${funcName}(int row, int col, int depth, int depth2, int depth3) {
      // Explicitly use integer operations as dot() only works on floats.
      int index = row * ${stride0} + col * ${stride1} + depth * ${stride2} +
          depth2 * ${stride3} + depth3 + ${offset};
      vec2 uv = uvFromFlat(${texNumR}, ${texNumC}, index);
      return sampleTexture(${texName}, uv);
    }
  `;
}

function getSampler6D(inputInfo: InputInfo): string {
  const shape = inputInfo.shapeInfo.logicalShape;
  const texName = inputInfo.name;
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);

  const {newShape, keptDims} = util.squeezeShape(shape);
  if (newShape.length < shape.length) {
    const newInputInfo = squeezeInputInfo(inputInfo, newShape);
    const params = ['row', 'col', 'depth', 'depth2', 'depth3', 'depth4'];
    return `
      ${getSamplerFromInInfo(newInputInfo)}
      float ${funcName}(int row, int col, int depth,
                    int depth2, int depth3, int depth4) {
        return ${funcName}(${getSqueezedParams(params, keptDims)});
      }
    `;
  }

  const stride4 = shape[5];
  const stride3 = shape[4] * stride4;
  const stride2 = shape[3] * stride3;
  const stride1 = shape[2] * stride2;
  const stride0 = shape[1] * stride1;

  if (inputInfo.shapeInfo.isUniform) {
    // Uniform arrays will be less than 65505 (no risk of float16 overflow).
    return `
      float ${funcName}(int row, int col, int depth,
                  int depth2, int depth3, int depth4) {
        int index = round(dot(
          vec4(row, col, depth, depth2),
          vec4(${stride0}, ${stride1}, ${stride2}, ${stride3})) +
          dot(
            vec2(depth3, depth4),
            vec2(${stride4}, 1)));
        ${getUniformSampler(inputInfo)}
      }
    `;
  }

  const flatOffset = inputInfo.shapeInfo.flatOffset;
  const texShape = inputInfo.shapeInfo.texShape;
  const texNumR = texShape[0];
  const texNumC = texShape[1];
  if (texNumC === stride0 && flatOffset == null) {
    // texC is used directly as physical (no risk of float16 overflow).
    return `
      float ${funcName}(int row, int col, int depth,
                    int depth2, int depth3, int depth4) {
        int texR = row;
        float texC = dot(vec4(col, depth, depth2, depth3),
          vec4(${stride1}, ${stride2}, ${stride3}, ${stride4})) +
               float(depth4);
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${texNumC}.0, ${texNumR}.0);
        return sampleTexture(${texName}, uv);
      }
    `;
  }
  if (texNumC === stride4 && flatOffset == null) {
    // texR is used directly as physical (no risk of float16 overflow).
    return `
      float ${funcName}(int row, int col, int depth,
                    int depth2, int depth3, int depth4) {
        float texR = dot(vec4(row, col, depth, depth2),
          vec4(${shape[1] * shape[2] * shape[3] * shape[4]},
               ${shape[2] * shape[3] * shape[4]},
               ${shape[3] * shape[4]},
               ${shape[4]})) + float(depth3);
        int texC = depth4;
        vec2 uv = (vec2(texC, texR) + halfCR) /
                  vec2(${texNumC}.0, ${texNumR}.0);
        return sampleTexture(${texName}, uv);
      }
    `;
  }
  const offset = getFlatOffsetUniformName(texName);
  return `
    float ${funcName}(int row, int col, int depth,
                  int depth2, int depth3, int depth4) {
      // Explicitly use integer operations as dot() only works on floats.
      int index = row * ${stride0} + col * ${stride1} + depth * ${stride2} +
          depth2 * ${stride3} + depth3 * ${stride4} + depth4 + ${offset};
      vec2 uv = uvFromFlat(${texNumR}, ${texNumC}, index);
      return sampleTexture(${texName}, uv);
    }
  `;
}
/*
function getSamplerFlatCS(inputInfo: InputInfo): string {
  const texName = inputInfo.name;
  const texShape = inputInfo.shapeInfo.texShape;
  const funcName =
      'get' + texName.charAt(0).toUpperCase() + texName.slice(1) + 'Flat';
  const tNumR = texShape[0];
  const tNumC = texShape[1];
  if (tNumC === 1 && tNumR === 1) {
    return `
      float ${funcName}(int index) {
        return sampleTexture(${texName}, ivec2(0, 0));
      }
    `;
  }
  if (tNumC === 1) {
    return `
      float ${funcName}(int index) {
        ivec2 uv = ivec2(0, index);
        return sampleTexture(${texName}, uv);
      }
    `;
  }
  if (tNumR === 1) {
    return `
      float ${funcName}(int index) {
        ivec2 uv = ivec2(index, 0);
        return sampleTexture(${texName}, uv);
      }
    `;
  }
  return `
    float ${funcName}(int index) {
      ivec2 uv = UVfrom1D(${tNumR}, ${tNumC}, index);
      return sampleTexture(${texName}, uv);
    }
  `;
}
*/
function getUniformSampler(inputInfo: InputInfo): string {
  const texName = inputInfo.name;
  const inSize = util.sizeFromShape(inputInfo.shapeInfo.logicalShape);

  if (inSize === 1) {
    return `return ${texName};`;
  }
  return `
    for (int i = 0; i < ${inSize}; i++) {
      if (i == index) {
        return ${texName}[i];
      }
    }
  `;
}

function getPackedSamplerAtOutputCoords(
    inputInfo: InputInfo, outShapeInfo: ShapeInfo) {
  const texName = inputInfo.name;
  const texFuncSnippet = texName.charAt(0).toUpperCase() + texName.slice(1);
  const funcName = 'get' + texFuncSnippet + 'AtOutCoords';
  const inRank = inputInfo.shapeInfo.logicalShape.length;
  const outRank = outShapeInfo.logicalShape.length;

  const broadcastDims = getBroadcastDims(
      inputInfo.shapeInfo.logicalShape, outShapeInfo.logicalShape);

  const type = getCoordsDataType(outRank);
  const rankDiff = outRank - inRank;
  let coordsSnippet: string;
  const fields = ['x', 'y', 'z', 'w', 'u', 'v'];

  if (inRank === 0) {
    coordsSnippet = '';
  } else if (outRank < 2 && broadcastDims.length >= 1) {
    coordsSnippet = 'coords = 0;';
  } else {
    coordsSnippet =
        broadcastDims.map(d => `coords.${fields[d + rankDiff]} = 0;`)
            .join('\n');
  }
  let unpackedCoordsSnippet = '';
  if (outRank < 2 && inRank > 0) {
    unpackedCoordsSnippet = 'coords';
  } else {
    unpackedCoordsSnippet = inputInfo.shapeInfo.logicalShape
                                .map((s, i) => `coords.${fields[i + rankDiff]}`)
                                .join(', ');
  }

  let output = `return outputValue;`;

  if (inRank === 1 && outRank > 1) {
    output = `
      return vec4(outputValue.xy, outputValue.xy);
    `;
  } else if (inRank === 0 && outRank > 0) {
    if (outRank === 1) {
      output = `
        return vec4(outputValue.x, outputValue.x, 0., 0.);
      `;
    } else {
      output = `
        return vec4(outputValue.x);
      `;
    }
  } else if (broadcastDims.length) {
    const rows = inRank - 2;
    const cols = inRank - 1;

    if (broadcastDims.indexOf(rows) > -1 && broadcastDims.indexOf(cols) > -1) {
      output = `return vec4(outputValue.x);`;
    } else if (broadcastDims.indexOf(rows) > -1) {
      output = `return vec4(outputValue.x, outputValue.y, ` +
          `outputValue.x, outputValue.y);`;
    } else if (broadcastDims.indexOf(cols) > -1) {
      output = `return vec4(outputValue.xx, outputValue.zz);`;
    }
  }

  return `
    vec4 ${funcName}() {
      ${type} coords = getOutputCoords();
      ${coordsSnippet}
      vec4 outputValue = get${texFuncSnippet}(${unpackedCoordsSnippet});
      ${output}
    }
  `;
}

function getSamplerAtOutputCoords(
    inputInfo: InputInfo, outShapeInfo: ShapeInfo) {
  const texName = inputInfo.name;
  const texFuncSnippet = texName.charAt(0).toUpperCase() + texName.slice(1);
  const funcName = 'get' + texFuncSnippet + 'AtOutCoords';
  const outTexShape = outShapeInfo.texShape;
  const inTexShape = inputInfo.shapeInfo.texShape;
  const inRank = inputInfo.shapeInfo.logicalShape.length;
  const outRank = outShapeInfo.logicalShape.length;

  if (!inputInfo.shapeInfo.isUniform && inRank === outRank &&
      inputInfo.shapeInfo.flatOffset == null &&
      util.arraysEqual(inTexShape, outTexShape)) {
    return `
      float ${funcName}() {
        return sampleTexture(${texName}, resultUV);
      }
    `;
  }

  const type = getCoordsDataType(outRank);
  const broadcastDims = getBroadcastDims(
      inputInfo.shapeInfo.logicalShape, outShapeInfo.logicalShape);
  const rankDiff = outRank - inRank;
  let coordsSnippet: string;
  const fields = ['x', 'y', 'z', 'w', 'u', 'v'];

  if (inRank === 0) {
    coordsSnippet = '';
  } else if (outRank < 2 && broadcastDims.length >= 1) {
    coordsSnippet = 'coords = 0;';
  } else {
    coordsSnippet =
        broadcastDims.map(d => `coords.${fields[d + rankDiff]} = 0;`)
            .join('\n');
  }
  let unpackedCoordsSnippet = '';
  if (outRank < 2 && inRank > 0) {
    unpackedCoordsSnippet = 'coords';
  } else {
    unpackedCoordsSnippet = inputInfo.shapeInfo.logicalShape
                                .map((s, i) => `coords.${fields[i + rankDiff]}`)
                                .join(', ');
  }

  return `
    float ${funcName}() {
      ${type} coords = getOutputCoords();
      ${coordsSnippet}
      return get${texFuncSnippet}(${unpackedCoordsSnippet});
    }
  `;
}

function getSamplerAtOutputCoordsCS(
    inputInfo: InputInfo, outShapeInfo: ShapeInfo) {
  const texName = inputInfo.name;
  const texFuncSnippet = texName.charAt(0).toUpperCase() + texName.slice(1);
  const funcName = 'get' + texFuncSnippet + 'AtOutCoords';
  const outTexShape = outShapeInfo.texShape;
  const inTexShape = inputInfo.shapeInfo.texShape;
  const inRank = inputInfo.shapeInfo.logicalShape.length;
  const outRank = outShapeInfo.logicalShape.length;

  if (!inputInfo.shapeInfo.isUniform && inRank === outRank &&
      inputInfo.shapeInfo.flatOffset == null &&
      util.arraysEqual(inTexShape, outTexShape)) {
    return `
      float ${funcName}() {
        return sampleTexture(${texName}, ivec2(gl_GlobalInvocationID.xy));
      }
    `;
  }

  return `
    float ${funcName}() {
      int index = int(gl_GlobalInvocationID.y) * ${outTexShape[1]} +
                  int(gl_GlobalInvocationID.x);
      int texR = index / ${inTexShape[1]};
      int texC = index - texR * ${inTexShape[1]};
      ivec2 uv = ivec2(texC, texR);

      return sampleTexture(${texName}, uv);
    }
  `;
}

export function getCoordsDataType(rank: number): string {
  if (rank <= 1) {
    return 'int';
  } else if (rank === 2) {
    return 'ivec2';
  } else if (rank === 3) {
    return 'ivec3';
  } else if (rank === 4) {
    return 'ivec4';
  } else if (rank === 5) {
    return 'ivec5';
  } else if (rank === 6) {
    return 'ivec6';
  } else {
    throw Error(`GPU for rank ${rank} is not yet supported`);
  }
}

/** Returns a new input info (a copy) that has a squeezed logical shape. */
function squeezeInputInfo(
    inInfo: InputInfo, squeezedShape: number[]): InputInfo {
  // Deep copy.
  const newInputInfo: InputInfo = JSON.parse(JSON.stringify(inInfo));
  newInputInfo.shapeInfo.logicalShape = squeezedShape;
  return newInputInfo;
}

function getSqueezedParams(params: string[], keptDims: number[]): string {
  return keptDims.map(d => params[d]).join(', ');
}
