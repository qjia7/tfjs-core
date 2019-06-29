async function conv2dBenchmark1(LS_X, LS_Y) {
  console.log(`LS_X = ${LS_X}, LS_Y = ${LS_Y}`);

  tf.ENV.set('WEBGL_PACK', false);
  tf.ENV.set('WEBGL_CONV_LS_X', LS_X);
  tf.ENV.set('WEBGL_CONV_LS_Y', LS_Y);

  const times = [];
  const x = tf.randomNormal([1, 100, 100, 16]);  // [N, Hi, Wi, Ci]
  const w = tf.randomNormal([5, 5, 16, 32]);     // [Hk, Wk, Ci, Co]
  const s = 1;                                   // stride
  const p = 'same';                              // padding
  let y = tf.conv2d(x, w, s, p);
  await y.data();

  console.log('start...');

  for (let i = 0; i < 100; i++) {
    const start = performance.now();
    y = tf.conv2d(x, w, s, p);
    await y.data();
    times.push(performance.now() - start);
  }

  const avgTime = times.reduce((a, b) => a + b, 0) / times.length;

  console.log(`Average time = ${avgTime.toFixed(2)} ms`);
  console.log('-------------------------');

  document.getElementById('conv2dOutput1').innerText =
      `Average time = ${avgTime.toFixed(2)} ms`;
  showNumTensors();
}

async function conv2dBenchmark2(LS_X, LS_Y) {
  console.log(`LS_X = ${LS_X}, LS_Y = ${LS_Y}`);

  tf.ENV.set('WEBGL_PACK', false);
  tf.ENV.set('WEBGL_CONV_LS_X', LS_X);
  tf.ENV.set('WEBGL_CONV_LS_Y', LS_Y);

  const times = [];
  const x = tf.randomNormal([1, 14, 14, 256]);  // [N, Hi, Wi, Ci]
  const w = tf.randomNormal([3, 3, 256, 256]);  // [Hk, Wk, Ci, Co]
  const s = 1;                                  // stride
  const p = 'same';                             // padding
  let y = tf.conv2d(x, w, s, p);
  await y.data();

  console.log('start...');

  for (let i = 0; i < 100; i++) {
    const start = performance.now();
    y = tf.conv2d(x, w, s, p);
    await y.data();
    times.push(performance.now() - start);
  }

  const avgTime = times.reduce((a, b) => a + b, 0) / times.length;

  console.log(`Average time = ${avgTime.toFixed(2)} ms`);
  console.log('-------------------------');

  document.getElementById('conv2dOutput2').innerText =
      `Average time = ${avgTime.toFixed(2)} ms`;
  showNumTensors();
}
