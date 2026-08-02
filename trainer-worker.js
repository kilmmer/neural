class Matrix {
  constructor(rows, cols) {
    this.rows = rows;
    this.cols = cols;
    this.data = Array.from({ length: rows }, () => Array(cols).fill(0));
  }

  static fromArray(values) {
    const matrix = new Matrix(values.length, 1);
    matrix.data.forEach((row, index) => { row[0] = values[index]; });
    return matrix;
  }

  add(value) {
    this.data = this.data.map((row, r) => row.map((cell, c) => (
      cell + (value instanceof Matrix ? value.data[r][c] : value)
    )));
    return this;
  }

  multiply(value) {
    this.data = this.data.map((row, r) => row.map((cell, c) => (
      cell * (value instanceof Matrix ? value.data[r][c] : value)
    )));
    return this;
  }

  map(fn) {
    this.data = this.data.map((row, r) => row.map((cell, c) => fn(cell, r, c)));
    return this;
  }

  toArray() { return this.data.flat(); }

  static multiply(a, b) {
    if (a.cols !== b.rows) throw new Error('Dimensões de matriz incompatíveis.');
    return new Matrix(a.rows, b.cols).map((_, row, col) => (
      a.data[row].reduce((sum, value, index) => sum + value * b.data[index][col], 0)
    ));
  }

  static map(matrix, fn) {
    return new Matrix(matrix.rows, matrix.cols).map((_, row, col) => fn(matrix.data[row][col], row, col));
  }

  static transpose(matrix) {
    return new Matrix(matrix.cols, matrix.rows).map((_, row, col) => matrix.data[col][row]);
  }

  static subtract(a, b) {
    return new Matrix(a.rows, a.cols).map((_, row, col) => a.data[row][col] - b.data[row][col]);
  }
}

class WorkerNetwork {
  constructor(state) {
    this.layerSizes = [...state.layerSizes];
    this.learningRate = state.learningRate;
    this.weights = state.weights.map(data => {
      const matrix = new Matrix(data.length, data[0].length);
      matrix.data = data.map(row => [...row]);
      return matrix;
    });
    this.biases = state.biases.map(data => {
      const matrix = new Matrix(data.length, 1);
      matrix.data = data.map(row => [...row]);
      return matrix;
    });
  }

  sigmoid(value) {
    return value >= 0
      ? 1 / (1 + Math.exp(-value))
      : Math.exp(value) / (1 + Math.exp(value));
  }

  dsigmoid(value) { return value * (1 - value); }

  softmax(matrix) {
    const maximum = Math.max(...matrix.toArray());
    const exponentials = Matrix.map(matrix, value => Math.exp(value - maximum));
    const total = exponentials.toArray().reduce((sum, value) => sum + value, 0);
    return exponentials.multiply(1 / total);
  }

  forward(inputArray) {
    const activations = [Matrix.fromArray(inputArray)];
    for (let index = 0; index < this.weights.length; index++) {
      const values = Matrix.multiply(this.weights[index], activations[index]).add(this.biases[index]);
      activations.push(index === this.weights.length - 1
        ? this.softmax(values)
        : values.map(this.sigmoid.bind(this)));
    }
    return { activations, output: activations.at(-1) };
  }

  train(inputArray, targetArray) {
    const { activations, output } = this.forward(inputArray);
    let errors = Matrix.subtract(Matrix.fromArray(targetArray), output);
    for (let index = this.weights.length - 1; index >= 0; index--) {
      const previousErrors = index > 0
        ? Matrix.multiply(Matrix.transpose(this.weights[index]), errors)
        : null;
      const deltas = index === this.weights.length - 1
        ? errors.multiply(this.learningRate)
        : Matrix.map(activations[index + 1], this.dsigmoid)
          .multiply(errors)
          .multiply(this.learningRate);
      this.weights[index].add(Matrix.multiply(deltas, Matrix.transpose(activations[index])));
      this.biases[index].add(deltas);
      errors = previousErrors;
    }
  }

  serialize() {
    return {
      layerSizes: [...this.layerSizes],
      learningRate: this.learningRate,
      weights: this.weights.map(matrix => matrix.data),
      biases: this.biases.map(matrix => matrix.data),
    };
  }
}

let trainingData = [];

self.addEventListener('message', event => {
  const message = event.data;
  if (message.type === 'init') {
    trainingData = message.dataset;
    self.postMessage({ type: 'ready', workerId: message.workerId });
    return;
  }

  if (message.type !== 'train') return;
  try {
    const network = new WorkerNetwork(message.network);
    const sampleIndexes = Array.isArray(message.sampleIndexes) ? message.sampleIndexes : [message.sampleIndex];
    const input = Array(network.layerSizes[0]).fill(0);
    const target = Array(network.layerSizes.at(-1)).fill(0);
    let previousInput = -1;
    let previousTarget = -1;
    let lossTotal = 0;
    let predictionIndex = 0;
    let confidence = 0;

    sampleIndexes.forEach((sampleIndex, position) => {
      const sample = trainingData[sampleIndex];
      if (previousInput >= 0) input[previousInput] = 0;
      if (previousTarget >= 0) target[previousTarget] = 0;
      input[sample.inputIndex] = 1;
      target[sample.targetIndex] = 1;
      previousInput = sample.inputIndex;
      previousTarget = sample.targetIndex;
      const before = network.forward(input).output.toArray();
      lossTotal -= Math.log(Math.max(before[sample.targetIndex], 1e-12));
      if (position === 0) {
        predictionIndex = before.reduce((best, value, index) => value > before[best] ? index : best, 0);
        confidence = before[predictionIndex];
      }
      network.train(input, target);
    });

    self.postMessage({
      type: 'trained',
      roundId: message.roundId,
      workerId: message.workerId,
      sampleIndex: sampleIndexes[0],
      predictionIndex,
      confidence,
      meanLoss: lossTotal / sampleIndexes.length,
      trainedSteps: sampleIndexes.length,
      network: network.serialize(),
    });
  } catch (error) {
    self.postMessage({
      type: 'error',
      roundId: message.roundId,
      workerId: message.workerId,
      message: error instanceof Error ? error.message : String(error),
    });
  }
});
