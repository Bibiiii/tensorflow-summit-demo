import regeneratorRuntime from 'regenerator-runtime';
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import iris from '../data/iris.json';
import irisTesting from '../data/iris-testing.json';

window.isModelLoading = false;
window.isModelComplete = false;

// set up data
function getData() {
  const inputs = tf.tensor2d(
    iris.map((item) => [
      item.sepal_length,
      item.sepal_width,
      item.petal_length,
      item.petal_width,
    ])
  );

  const labels = tf.tensor2d(
    iris.map((item) => [
      item.species === 'setosa' ? 1 : 0,
      item.species === 'virginica' ? 1 : 0,
      item.species === 'versicolor' ? 1 : 0,
    ])
  );

  return {
    inputs,
    labels,
  };
}

function getTestingData(species) {
  const testData = irisTesting.filter((i) => i.species === species);

  const testingData = tf.tensor2d(
    testData.map((item) => [
      item.sepal_length,
      item.sepal_width,
      item.petal_length,
      item.petal_width,
    ])
  );

  return testingData;
}

function createModel() {
  // build neural network
  const model = tf.sequential();
  // add an input layer
  model.add(
    tf.layers.dense({
      inputShape: [4],
      activation: 'sigmoid',
      units: 5,
    })
  );
  // add hidden layer
  model.add(
    tf.layers.dense({
      inputShape: [5],
      activation: 'sigmoid',
      units: 3,
    })
  );
  // add output payer
  model.add(
    tf.layers.dense({
      activation: 'sigmoid',
      units: 3,
    })
  );

  return model;
}

// train network
async function trainModel(model, inputs, labels) {
  model.compile({
    loss: tf.losses.meanSquaredError,
    optimizer: tf.train.adam(0.06),
  });

  return await model.fit(inputs, labels, {
    epochs: 100,
    callbacks: tfvis.show.fitCallbacks(
      { name: 'Training Performance' },
      ['loss', 'mse'],
      { heights: 200, callbacks: ['onEpochEnd'] }
    ),
  });
}

async function testModel(model, testingData) {
  return await model.predict(testingData).data();
}

async function onClick(e, model) {
  const modelPrediction = document.getElementById('model_prediction');
  if (!window.isModelLoading) {
    window.isModelLoading = true;
    modelPrediction.innerHTML = 'Results loading';

    const species = e.target.id;
    const testingData = getTestingData(species);

    const results = await testModel(model, testingData);

    function parseNum(num) {
      return Math.round((num + Number.EPSILON) * 100) / 100;
    }

    window.isModelLoading = false;
    console.log(results);
    modelPrediction.innerHTML = `
      <ul>
        <li>Setosa: ${parseNum(results[0])}</li>
        <li>Virginica: ${parseNum(results[1])}</li>
        <li>Versicolor: ${parseNum(results[2])}</li>
      </ul>`;
  }
}

async function run() {
  const buttons = document.getElementsByTagName('button');
  const modelPrediction = document.getElementById('model_prediction');
  window.isModelLoading = true;
  modelPrediction.innerHTML = 'Model loading';
  const data = getData();
  const model = createModel();

  const { inputs, labels } = data;

  tfvis.show.modelSummary({ name: 'Model Summary' }, model);
  await trainModel(model, inputs, labels);

  window.isModelLoading = false;
  window.isModelComplete = true;
  modelPrediction.innerHTML = 'Model ready';

  for (let button of buttons) {
    button.addEventListener('click', function (e) {
      onClick(e, model);
    });
    button.disabled = false;
  }

  // console.log('done training');
  // const results = await testModel(model, testingData);
  // console.log(await results);
}

document.addEventListener('DOMContentLoaded', run);
