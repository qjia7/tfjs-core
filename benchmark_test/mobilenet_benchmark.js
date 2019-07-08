var mobilenet;

async function mobilenetBenchmark(version, LS, TS, WPT) {
  switch (version) {
    case 0:
      console.log(`version = 0, LS = ${LS}`);
      break;
    case 1:
      console.log(`version = 1, TS = ${TS}`);
      break;
    case 2:
    case 3:
      console.log(`version = ${version}, TS = ${TS}, WPT = ${WPT}`);
  }

  tf.ENV.set('WEBGL_MATMUL_VERSION', version);
  tf.ENV.set('WEBGL_MATMUL_V0_LS', LS);
  tf.ENV.set('WEBGL_MATMUL_TS', TS);
  tf.ENV.set('WEBGL_MATMUL_WPT', WPT);

  const MOBILENET_MODEL_PATH =
      // tslint:disable-next-line:max-line-length
      'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json';
  if (mobilenet === undefined) {
    console.log('loading...');
    mobilenet = await tf.loadLayersModel(MOBILENET_MODEL_PATH);
  }

  const times = [];
  const input = tf.randomNormal([1, 224, 224, 3]);
  let output = mobilenet.predict(input);
  await output.data();

  console.log('start...');

  for (let i = 0; i < 100; i++) {
    const start = performance.now();
    output = mobilenet.predict(input);
    await output.data();
    times.push(performance.now() - start);
  }

  const avgTime = times.reduce((a, b) => a + b, 0) / times.length;

  console.log(`Average time = ${avgTime.toFixed(2)} ms`);
  console.log('-------------------------');

  document.getElementById('mobilenetOutput').innerText =
      `Average time = ${avgTime.toFixed(2)} ms`;
  showNumTensors();
}
