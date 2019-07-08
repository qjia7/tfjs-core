async function matMulBenchmark(N, version, LS, TS, WPT) {
  switch (version) {
    case 0:
      console.log(`N = ${N}, version = 0, LS = ${LS}`);
      break;
    case 1:
      console.log(`N = ${N}, version = 1, TS = ${TS}`);
      break;
    case 2:
    case 3:
      console.log(`N = ${N}, version = ${version}, TS = ${TS}, WPT = ${WPT}`);
  }

  tf.ENV.set('WEBGL_MATMUL_VERSION', version);
  tf.ENV.set('WEBGL_MATMUL_V0_LS', LS);
  tf.ENV.set('WEBGL_MATMUL_TS', TS);
  tf.ENV.set('WEBGL_MATMUL_WPT', WPT);

  const times = [];
  const A = tf.randomNormal([N, N]);
  const B = tf.randomNormal([N, N]);
  let C = tf.matMul(A, B);
  await C.data();

  console.log('start...');

  for (let i = 0; i < 100; i++) {
    const start = performance.now();
    C = tf.matMul(A, B);
    await C.data();
    times.push(performance.now() - start);
  }

  const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
  const GFLOPS = N * N * N / avgTime / 1000000;

  console.log(
      `Average time = ${avgTime.toFixed(2)} ms, GFLOPS = ${GFLOPS.toFixed(2)}`);
  console.log('---------------------------------------');

  document.getElementById('matmulOutput').innerText =
      `Average time = ${avgTime.toFixed(2)} ms, GFLOPS = ${GFLOPS.toFixed(2)}`;
  showNumTensors();
}

async function matMulBenchmarkV4(N, TSMN, TSK, WPTMN) {
  console.log(`N = ${N}, version = 4, TSM/N = ${TSMN}, TSK = ${TSK}, WPTM/N = ${
      WPTMN}`);

  tf.ENV.set('WEBGL_MATMUL_VERSION', 4);
  tf.ENV.set('WEBGL_MATMUL_TS', TSMN);
  tf.ENV.set('WEBGL_MATMUL_TSK', TSK);
  tf.ENV.set('WEBGL_MATMUL_WPT', WPTMN);

  const times = [];
  const A = tf.randomNormal([N, N]);
  const B = tf.randomNormal([N, N]);
  let C = tf.matMul(A, B);
  await C.data();

  console.log('start...');

  for (let i = 0; i < 100; i++) {
    const start = performance.now();
    C = tf.matMul(A, B);
    await C.data();
    times.push(performance.now() - start);
  }

  const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
  const GFLOPS = N * N * N / avgTime / 1000000;

  console.log(
      `Average time = ${avgTime.toFixed(2)} ms, GFLOPS = ${GFLOPS.toFixed(2)}`);
  console.log('---------------------------------------');

  document.getElementById('matmulOutputV4').innerText =
      `Average time = ${avgTime.toFixed(2)} ms, GFLOPS = ${GFLOPS.toFixed(2)}`;
  showNumTensors();
}
