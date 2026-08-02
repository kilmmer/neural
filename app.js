class Matrix {
  constructor(rows, cols) {
    this.rows = rows;
    this.cols = cols;
    this.data = Array.from({ length: rows }, () => Array(cols).fill(0));
  }

  resize(rows, cols) {
    const next = new Matrix(rows, cols);
    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        next.data[row][col] = row < this.rows && col < this.cols
          ? this.data[row][col]
          : (Math.random() * 2 - 1) * 0.15;
      }
    }
    this.rows = rows;
    this.cols = cols;
    this.data = next.data;
  }

  static fromArray(values) {
    const result = new Matrix(values.length, 1);
    result.data.forEach((row, index) => { row[0] = values[index]; });
    return result;
  }

  toArray() { return this.data.flat(); }

  randomize(scale = 1) {
    this.data = this.data.map(row => row.map(() => (Math.random() * 2 - 1) * scale));
    return this;
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

  static multiply(a, b) {
    if (a.cols !== b.rows) throw new Error('Dimensões de matriz incompatíveis.');
    return new Matrix(a.rows, b.cols).map((_, row, col) => (
      a.data[row].reduce((sum, value, index) => sum + value * b.data[index][col], 0)
    ));
  }

  static map(matrix, fn) {
    return new Matrix(matrix.rows, matrix.cols).map((_, row, col) => (
      fn(matrix.data[row][col], row, col)
    ));
  }

  static transpose(matrix) {
    return new Matrix(matrix.cols, matrix.rows).map((_, row, col) => matrix.data[col][row]);
  }

  static subtract(a, b) {
    return new Matrix(a.rows, a.cols).map((_, row, col) => a.data[row][col] - b.data[row][col]);
  }
}

class NeuralNetwork {
  constructor(layerSizes) {
    this.layerSizes = [...layerSizes];
    this.learningRate = 0.1;
    this.weights = [];
    this.biases = [];
    for (let index = 1; index < layerSizes.length; index++) {
      this.weights.push(new Matrix(layerSizes[index], layerSizes[index - 1]).randomize());
      this.biases.push(new Matrix(layerSizes[index], 1).randomize());
    }
  }

  get inputNodes() { return this.layerSizes[0]; }
  get outputNodes() { return this.layerSizes.at(-1); }
  get hiddenLayerCount() { return Math.max(0, this.layerSizes.length - 2); }
  get lastHiddenIndex() { return this.layerSizes.length - 2; }
  get hiddenNodes() { return this.layerSizes.slice(1, -1).reduce((sum, size) => sum + size, 0); }

  canAddNeuron(maximum) {
    return this.layerSizes[this.lastHiddenIndex] < maximum;
  }

  addNeuronToLayer(layerIndex) {
    const incomingIndex = layerIndex - 1;
    const previousSize = this.layerSizes[layerIndex];
    const outgoing = this.weights[incomingIndex + 1];
    const donorIndex = Array.from({ length: previousSize }, (_, index) => index).reduce((best, index) => {
      const strength = outgoing.data.reduce((sum, row) => sum + Math.abs(row[index]), 0);
      const bestStrength = outgoing.data.reduce((sum, row) => sum + Math.abs(row[best]), 0);
      return strength > bestStrength ? index : best;
    }, 0);

    this.layerSizes[layerIndex]++;
    this.weights[incomingIndex].resize(this.layerSizes[layerIndex], this.layerSizes[layerIndex - 1]);
    this.biases[incomingIndex].resize(this.layerSizes[layerIndex], 1);
    this.weights[incomingIndex + 1].resize(this.layerSizes[layerIndex + 1], this.layerSizes[layerIndex]);

    const newIndex = this.layerSizes[layerIndex] - 1;
    this.weights[incomingIndex].data[newIndex] = this.weights[incomingIndex].data[donorIndex]
      .map(weight => weight + (Math.random() * 2 - 1) * 0.008);
    this.biases[incomingIndex].data[newIndex][0] = this.biases[incomingIndex].data[donorIndex][0]
      + (Math.random() * 2 - 1) * 0.004;

    this.weights[incomingIndex + 1].data.forEach(row => {
      const sharedWeight = row[donorIndex] / 2;
      row[donorIndex] = sharedWeight;
      row[newIndex] = sharedWeight;
    });

    return { donorIndex, newIndex };
  }

  addHiddenLayer(neuronCount) {
    const outputBias = this.biases.at(-1);
    const previousSize = this.layerSizes.at(-2);
    const insertedSize = Math.max(neuronCount, previousSize);
    const previousOutputWeights = this.weights.at(-1);
    if (insertedSize !== previousSize) previousOutputWeights.resize(this.outputNodes, insertedSize);
    const transition = new Matrix(insertedSize, previousSize);
    const transitionBias = new Matrix(insertedSize, 1);

    // sigmoid(5.5x - 2.75) aproxima a identidade em [0, 1], faixa das ativações ocultas.
    // Pequeno ruído quebra a simetria sem apagar a representação aprendida.
    transition.data = transition.data.map((row, target) => row.map((_, source) => (
      target === source ? 5.5 + (Math.random() * 2 - 1) * 0.015 : (Math.random() * 2 - 1) * 0.004
    )));
    transitionBias.data = transitionBias.data.map(() => [-2.75 + (Math.random() * 2 - 1) * 0.008]);

    this.layerSizes.splice(-1, 0, insertedSize);
    this.weights = [
      ...this.weights.slice(0, -1),
      transition,
      previousOutputWeights,
    ];
    this.biases = [
      ...this.biases.slice(0, -1),
      transitionBias,
      outputBias,
    ];
    return insertedSize;
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
      const isOutput = index === this.weights.length - 1;
      activations.push(isOutput ? this.softmax(values) : values.map(this.sigmoid.bind(this)));
    }
    return { activations, output: activations.at(-1) };
  }

  feedForward(inputArray) {
    const { activations, output } = this.forward(inputArray);
    return {
      activations: activations.map(activation => activation.toArray()),
      output: output.toArray(),
    };
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
}

class AttentionNetwork {
  constructor(vocabSize, contextSize = 3, embeddingSize = 10) {
    this.vocabSize = vocabSize;
    this.contextSize = contextSize;
    this.embeddingSize = embeddingSize;
    this.learningRate = 0.035;
    const randomRow = scale => Array.from({ length: embeddingSize }, () => (Math.random() * 2 - 1) * scale);
    this.embeddings = Array.from({ length: vocabSize }, () => randomRow(0.18));
    this.outputWeights = Array.from({ length: vocabSize }, () => randomRow(0.12));
    this.outputBiases = Array(vocabSize).fill(0);
    this.query = randomRow(0.18);
  }

  forward(contextIndices) {
    const indices = contextIndices.slice(-this.contextSize);
    const scores = indices.map(index => this.embeddings[index].reduce((sum, value, dim) => sum + value * this.query[dim], 0));
    const maximum = Math.max(...scores);
    const scoreExps = scores.map(score => Math.exp(score - maximum));
    const scoreTotal = scoreExps.reduce((sum, value) => sum + value, 0) || 1;
    const attention = scoreExps.map(value => value / scoreTotal);
    const context = Array(this.embeddingSize).fill(0);
    indices.forEach((wordIndex, position) => this.embeddings[wordIndex].forEach((value, dim) => { context[dim] += value * attention[position]; }));
    const logits = this.outputWeights.map((row, word) => row.reduce((sum, value, dim) => sum + value * context[dim], this.outputBiases[word]));
    const maxLogit = Math.max(...logits);
    const exps = logits.map(value => Math.exp(value - maxLogit));
    const total = exps.reduce((sum, value) => sum + value, 0) || 1;
    return { output: exps.map(value => value / total), attention, context, indices };
  }

  train(contextIndices, targetIndex) {
    const result = this.forward(contextIndices);
    const contextGradient = Array(this.embeddingSize).fill(0);
    for (let word = 0; word < this.vocabSize; word++) {
      const error = (word === targetIndex ? 1 : 0) - result.output[word];
      for (let dim = 0; dim < this.embeddingSize; dim++) {
        contextGradient[dim] += error * this.outputWeights[word][dim];
        this.outputWeights[word][dim] += this.learningRate * error * result.context[dim];
      }
      this.outputBiases[word] += this.learningRate * error;
    }
    result.indices.forEach((wordIndex, position) => {
      for (let dim = 0; dim < this.embeddingSize; dim++) {
        this.embeddings[wordIndex][dim] += this.learningRate * contextGradient[dim] * result.attention[position];
        const distinctiveness = this.embeddings[wordIndex][dim] - result.context[dim];
        this.query[dim] += this.learningRate * 0.08 * result.attention[position] * contextGradient[dim] * distinctiveness;
      }
    });
    return result;
  }

  serialize() {
    return { vocabSize: this.vocabSize, contextSize: this.contextSize, embeddingSize: this.embeddingSize, learningRate: this.learningRate, embeddings: this.embeddings, outputWeights: this.outputWeights, outputBiases: this.outputBiases, query: this.query };
  }

  static restore(state) {
    const model = new AttentionNetwork(state.vocabSize, state.contextSize, state.embeddingSize);
    Object.assign(model, state);
    return model;
  }
}

const MAX_CORPUS_TOKENS = 5000;

class TextProcessor {
  constructor(text) {
    const parsedWords = text.toLocaleLowerCase('pt-BR').match(/[\p{L}\p{N}]+(?:['’-][\p{L}\p{N}]+)*/gu) ?? [];
    this.wasTruncated = parsedWords.length > MAX_CORPUS_TOKENS;
    this.words = parsedWords.slice(0, MAX_CORPUS_TOKENS);
    this.vocab = [...new Set(this.words)];
    this.vocabSize = this.vocab.length;
    this.wordIndex = new Map(this.vocab.map((word, index) => [word, index]));
  }

  wordToVector(word) {
    return this.indexToVector(this.wordIndex.get(word) ?? -1);
  }

  indexToVector(index) {
    const vector = Array(this.vocabSize).fill(0);
    if (index >= 0) vector[index] = 1;
    return vector;
  }

  vectorToWord(vector) {
    const index = vector.reduce((best, value, current) => value > vector[best] ? current : best, 0);
    return { word: this.vocab[index], confidence: vector[index], index };
  }

  generateData(contextSize = 3) {
    return this.words.slice(0, -1).map((word, index) => ({
      inputIndex: this.wordIndex.get(word),
      targetIndex: this.wordIndex.get(this.words[index + 1]),
      inputWord: word,
      targetWord: this.words[index + 1],
      contextIndices: this.words.slice(Math.max(0, index - contextSize + 1), index + 1).map(item => this.wordIndex.get(item)),
      contextWords: this.words.slice(Math.max(0, index - contextSize + 1), index + 1),
    }));
  }
}

const elements = {
  corpus: document.querySelector('#corpusText'),
  restart: document.querySelector('#restartButton'),
  pause: document.querySelector('#pauseButton'),
  step: document.querySelector('#stepButton'),
  speed: document.querySelector('#speedSelect'),
  execution: document.querySelector('#executionMode'),
  contextSize: document.querySelector('#contextSize'),
  workerStatus: document.querySelector('#workerStatus'),
  generate: document.querySelector('#generateButton'),
  theme: document.querySelector('#themeToggle'),
  seed: document.querySelector('#seedWord'),
  status: document.querySelector('#statusMessage'),
  epoch: document.querySelector('#epochDisplay'),
  loss: document.querySelector('#lossDisplay'),
  neurons: document.querySelector('#neuronCount'),
  layers: document.querySelector('#layerCount'),
  workers: document.querySelector('#workerCount'),
  splitCount: document.querySelector('#splitCount'),
  stagnation: document.querySelector('#stagnationDisplay'),
  bar: document.querySelector('#patienceBar'),
  growthRule: document.querySelector('#growthRule'),
  generated: document.querySelector('#generatedText'),
  generatorModel: document.querySelector('#generatorModel'),
  temperature: document.querySelector('#temperatureSelect'),
  save: document.querySelector('#saveButton'),
  load: document.querySelector('#loadButton'),
  export: document.querySelector('#exportButton'),
  import: document.querySelector('#importButton'),
  importFile: document.querySelector('#importFile'),
  mlpTrainLoss: document.querySelector('#mlpTrainLoss'),
  mlpTestLoss: document.querySelector('#mlpTestLoss'),
  attentionTrainLoss: document.querySelector('#attentionTrainLoss'),
  attentionTestLoss: document.querySelector('#attentionTestLoss'),
  historyCanvas: document.querySelector('#historyCanvas'),
  growthLog: document.querySelector('#growthLog'),
  currentInput: document.querySelector('#currentInput'),
  currentTarget: document.querySelector('#currentTarget'),
  currentPrediction: document.querySelector('#currentPrediction'),
  currentConfidence: document.querySelector('#currentConfidence'),
  detailTitle: document.querySelector('#neuronDetailsTitle'),
  detailContent: document.querySelector('#neuronDetailsContent'),
  teacherTitle: document.querySelector('#teacherTitle'),
  teacherText: document.querySelector('#teacherText'),
  canvas: document.querySelector('#networkCanvas'),
};

const ctx = elements.canvas.getContext('2d');
const config = {
  trainingInterval: 400,
  evaluationEvery: 8,
  patienceLimit: 8,
  improvementDelta: 0.002,
  targetLoss: 0.08,
  maxNeuronsPerLayer: 8,
  maxHiddenLayers: 3,
  newLayerSize: 4,
  maxVisualNodesPerLayer: 56,
  drawInterval: 33,
};

let processor;
let network;
let attentionNetwork;
let dataset = [];
let testDataset = [];
let activeCorpus = '';
let epoch = 0;
let bestLoss = Infinity;
let displayedLoss = null;
let stagnation = 0;
let paused = true;
let visualInput = null;
let lastTrainingAt = 0;
let formationStartedAt = 0;
let growthAnimation = null;
let neuronHitAreas = [];
let selectedNeuron = null;
let selectedNeuronEpoch = -1;
let workerPool = [];
let requestedWorkerCount = 2;
let workerRoundId = 0;
let parallelBusy = false;
let stepsSinceEvaluation = 0;
let renderDirty = true;
let lastDrawAt = 0;
let cachedColors = {};
let animationUntil = 0;
let attentionLoss = null;
let mlpTestLoss = null;
let attentionTestLoss = null;
let mlpEvaluatedTrainLoss = null;
let attentionEvaluatedTrainLoss = null;
let lossHistory = [];
let growthHistory = [];
let lastGrowthEpoch = 0;
const STORAGE_KEY = 'neural-lab-model-v2';
const SESSION_KEY = 'neural-lab-live-session-v2';

function cssColor(name) {
  if (!cachedColors[name]) cachedColors[name] = getComputedStyle(document.body).getPropertyValue(name).trim();
  return cachedColors[name];
}

function markRenderDirty() { renderDirty = true; }

function setStatus(message = '') {
  elements.status.textContent = message;
}

function setLesson(title, text) {
  elements.teacherTitle.textContent = title;
  elements.teacherText.textContent = text;
}

function updateTrainingButton() {
  elements.pause.textContent = paused ? 'Iniciar' : 'Pausar';
  elements.pause.setAttribute('aria-pressed', String(paused));
  elements.pause.setAttribute('aria-label', paused ? 'Iniciar treinamento' : 'Pausar treinamento');
}

function applyTheme(theme) {
  const isDark = theme === 'dark';
  document.body.dataset.theme = theme;
  elements.theme.setAttribute('aria-pressed', String(isDark));
  elements.theme.setAttribute('aria-label', isDark ? 'Ativar modo claro' : 'Ativar modo escuro');
  elements.theme.lastElementChild.textContent = isDark ? 'Tema claro' : 'Tema escuro';
  cachedColors = {};
  markRenderDirty();
}

function getInitialTheme() {
  try {
    const saved = localStorage.getItem('neural-theme');
    if (saved === 'light' || saved === 'dark') return saved;
  } catch { /* Preferência ficará apenas na sessão. */ }
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function resetLearningPanel() {
  elements.currentInput.textContent = '—';
  elements.currentTarget.textContent = '—';
  elements.currentPrediction.textContent = '—';
  elements.currentConfidence.textContent = '—';
  setLesson('Laboratório pronto', 'Insira um corpus e use “Avançar 1 passo” para observar o aprendizado.');
}

function resetNeuronDetails() {
  selectedNeuron = null;
  selectedNeuronEpoch = -1;
  elements.detailTitle.textContent = 'Selecione um neurônio';
  elements.detailContent.textContent = 'Clique em um neurônio para ver ativação, bias e conexões.';
}

function updateGrowthExplanation() {
  if (!network) {
    elements.growthRule.textContent = 'A rede cresce após 8 avaliações sem melhora.';
    return;
  }
  if (displayedLoss !== null && displayedLoss <= config.targetLoss) {
    elements.growthRule.textContent = 'Loss abaixo do alvo: crescimento não é necessário.';
    return;
  }
  const adaptationRemaining = Math.max(0, Math.max(128, dataset.length * 2) - (epoch - lastGrowthEpoch));
  if (adaptationRemaining > 0) {
    elements.growthRule.textContent = `Adaptação estrutural: mais ${adaptationRemaining} exemplos antes de avaliar crescimento.`;
    return;
  }
  const remaining = Math.max(0, config.patienceLimit - stagnation);
  elements.growthRule.textContent = `${remaining} avaliações sem melhora até o próximo crescimento.`;
}

function updateMetrics() {
  elements.epoch.textContent = epoch.toLocaleString('pt-BR');
  elements.loss.textContent = displayedLoss === null ? '—' : displayedLoss.toFixed(4);
  elements.neurons.textContent = network?.hiddenNodes ?? '—';
  elements.layers.textContent = network?.hiddenLayerCount ?? '—';
  elements.workers.textContent = workerPool.length;
  elements.splitCount.textContent = dataset.length ? `${dataset.length} / ${testDataset.length}` : '—';
  elements.mlpTrainLoss.textContent = mlpEvaluatedTrainLoss === null ? '—' : mlpEvaluatedTrainLoss.toFixed(3);
  elements.mlpTestLoss.textContent = mlpTestLoss === null ? '—' : mlpTestLoss.toFixed(3);
  elements.attentionTrainLoss.textContent = attentionEvaluatedTrainLoss === null ? '—' : attentionEvaluatedTrainLoss.toFixed(3);
  elements.attentionTestLoss.textContent = attentionTestLoss === null ? '—' : attentionTestLoss.toFixed(3);
  elements.stagnation.textContent = `${stagnation} / ${config.patienceLimit}`;
  elements.bar.style.width = `${Math.min(100, stagnation / config.patienceLimit * 100)}%`;
  updateGrowthExplanation();
}

function restoreMlp(state) {
  const model = new NeuralNetwork(state.layerSizes);
  model.learningRate = state.learningRate;
  model.weights.forEach((matrix, index) => { matrix.data = state.weights[index].map(row => [...row]); });
  model.biases.forEach((matrix, index) => { matrix.data = state.biases[index].map(row => [...row]); });
  return model;
}

function serializeNetwork() {
  return {
    layerSizes: [...network.layerSizes],
    learningRate: network.learningRate,
    weights: network.weights.map(matrix => matrix.data),
    biases: network.biases.map(matrix => matrix.data),
  };
}

function applyAveragedNetworks(states) {
  network.weights.forEach((matrix, matrixIndex) => {
    matrix.data = matrix.data.map((row, rowIndex) => row.map((_, colIndex) => (
      states.reduce((sum, state) => sum + state.weights[matrixIndex][rowIndex][colIndex], 0)
      / states.length
    )));
  });
  network.biases.forEach((matrix, matrixIndex) => {
    matrix.data = matrix.data.map((row, rowIndex) => row.map((_, colIndex) => (
      states.reduce((sum, state) => sum + state.biases[matrixIndex][rowIndex][colIndex], 0)
      / states.length
    )));
  });
}

function stopWorkers() {
  workerRoundId++;
  workerPool.forEach(worker => worker.terminate());
  workerPool = [];
  parallelBusy = false;
  updateMetrics();
}

function fallbackToLocal(message) {
  stopWorkers();
  requestedWorkerCount = 1;
  elements.execution.value = '1';
  elements.step.textContent = 'Avançar 1 passo';
  elements.workerStatus.textContent = message;
}

function ensureWorkerPool() {
  if (requestedWorkerCount === 1) return true;
  if (workerPool.length === requestedWorkerCount) return true;
  stopWorkers();

  if (typeof Worker === 'undefined') {
    fallbackToLocal('Web Workers não estão disponíveis; usando a thread principal.');
    return false;
  }

  try {
    workerPool = Array.from({ length: requestedWorkerCount }, (_, workerId) => {
      const worker = new Worker('trainer-worker.js');
      worker.postMessage({ type: 'init', workerId, dataset });
      return worker;
    });
    elements.workerStatus.textContent = `${requestedWorkerCount} workers treinam em paralelo; a interface permanece na thread principal.`;
    updateMetrics();
    return true;
  } catch {
    fallbackToLocal('Não foi possível iniciar workers; use um servidor HTTP local.');
    return false;
  }
}

function requestWorkerTraining(worker, workerId, roundId, sampleIndex, state) {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => finish(new Error('Tempo limite do worker excedido.')), 8000);
    const onMessage = event => {
      const message = event.data;
      if (message.roundId !== roundId || message.workerId !== workerId) return;
      if (message.type === 'error') finish(new Error(message.message));
      else if (message.type === 'trained') finish(null, message);
    };
    const onError = event => finish(new Error(event.message || 'Falha no worker.'));
    const finish = (error, result) => {
      clearTimeout(timeout);
      worker.removeEventListener('message', onMessage);
      worker.removeEventListener('error', onError);
      if (error) reject(error); else resolve(result);
    };
    worker.addEventListener('message', onMessage);
    worker.addEventListener('error', onError);
    worker.postMessage({ type: 'train', workerId, roundId, sampleIndex, network: state });
  });
}

function configureExecutionMode() {
  const selectedCount = Number(elements.execution.value);
  requestedWorkerCount = [1, 2, 4].includes(selectedCount) ? selectedCount : 1;
  stopWorkers();
  if (requestedWorkerCount === 1) {
    elements.step.textContent = 'Avançar 1 passo';
    elements.workerStatus.textContent = 'Treinamento e interface compartilham a thread principal.';
  } else {
    elements.step.textContent = 'Avançar 1 rodada';
    elements.workerStatus.textContent = `${requestedWorkerCount} workers serão iniciados junto com a rede.`;
  }
  setLesson('Modo de execução alterado', requestedWorkerCount === 1
    ? 'No modo local, treinamento e desenho dividem a mesma thread.'
    : `No modo paralelo, ${requestedWorkerCount} cópias treinam simultaneamente e seus pesos são combinados pela média.`);
}

function prepareCorpus() {
  activeCorpus = elements.corpus.value;
  processor = new TextProcessor(activeCorpus);
  const allData = processor.generateData(Number(elements.contextSize.value));
  dataset = allData.filter((_, index) => index % 5 !== 4);
  testDataset = allData.filter((_, index) => index % 5 === 4);
  if (!testDataset.length && dataset.length > 2) testDataset.push(dataset.pop());
  network = undefined;
  attentionNetwork = undefined;
  epoch = 0;
  bestLoss = Infinity;
  displayedLoss = null;
  stagnation = 0;
  attentionLoss = null;
  mlpTestLoss = null;
  attentionTestLoss = null;
  mlpEvaluatedTrainLoss = null;
  attentionEvaluatedTrainLoss = null;
  lossHistory = [];
  growthHistory = [];
  lastGrowthEpoch = 0;
  stepsSinceEvaluation = 0;
  paused = true;
  visualInput = null;
  lastTrainingAt = 0;
  growthAnimation = null;
  animationUntil = 0;
  neuronHitAreas = [];
  markRenderDirty();
  stopWorkers();
  elements.growthLog.replaceChildren();
  elements.generated.textContent = 'Insira um corpus e inicie o treinamento.';
  resetLearningPanel();
  resetNeuronDetails();
  updateTrainingButton();
  updateMetrics();

  if (dataset.length === 0) {
    setStatus('Insira um corpus com ao menos duas palavras para iniciar.');
    return false;
  }
  const truncationNotice = processor.wasTruncated ? ` Limite de ${MAX_CORPUS_TOKENS} tokens aplicado para proteger o navegador.` : '';
  setStatus(`Corpus preparado: ${dataset.length} exemplos de treino, ${testDataset.length} de teste e ${processor.vocabSize} palavras únicas.${truncationNotice}`);
  setLesson('Corpus separado sem vazamento', `80% dos exemplos treinam os modelos; 20% ficam reservados para medir generalização usando até ${elements.contextSize.value} palavras de contexto.`);
  return true;
}

function updateLossEstimate(sampleLoss) {
  displayedLoss = displayedLoss === null
    ? sampleLoss
    : displayedLoss * 0.9 + sampleLoss * 0.1;
}

function initializeNetwork() {
  if (dataset.length === 0) return false;
  const initialHidden = Math.min(5, processor.vocabSize);
  network = new NeuralNetwork([processor.vocabSize, initialHidden, processor.vocabSize]);
  attentionNetwork = new AttentionNetwork(processor.vocabSize, Number(elements.contextSize.value));
  visualInput = processor.indexToVector(dataset[0].inputIndex);
  formationStartedAt = performance.now();
  animationUntil = formationStartedAt + 12000;
  markRenderDirty();
  const initialOutput = network.feedForward(visualInput).output;
  displayedLoss = -Math.log(Math.max(initialOutput[dataset[0].targetIndex], 1e-12));
  bestLoss = displayedLoss;
  updateMetrics();
  setLesson('Rede criada', `Arquitetura inicial: ${processor.vocabSize} entradas → ${initialHidden} neurônios ocultos → ${processor.vocabSize} saídas.`);
  return true;
}

function syncCurrentCorpus() {
  if (elements.corpus.value !== activeCorpus) return prepareCorpus();
  return dataset.length > 0;
}

function addGrowthEvent(text) {
  const event = document.createElement('div');
  event.textContent = text;
  elements.growthLog.append(event);
  if (elements.growthLog.children.length > 4) elements.growthLog.firstElementChild.remove();
}

function updateGrowth(loss) {
  const adaptationSteps = Math.max(128, dataset.length * 2);
  if (epoch - lastGrowthEpoch < adaptationSteps) {
    bestLoss = Math.min(bestLoss, loss);
    stagnation = 0;
    return;
  }

  if (loss < bestLoss - config.improvementDelta) {
    bestLoss = loss;
    stagnation = 0;
    return;
  }

  if (loss <= config.targetLoss) {
    stagnation = 0;
    return;
  }

  stagnation++;
  if (stagnation < config.patienceLimit) return;

  if (network.canAddNeuron(config.maxNeuronsPerLayer)) {
    const layerIndex = network.lastHiddenIndex;
    const { donorIndex, newIndex: neuronIndex } = network.addNeuronToLayer(layerIndex);
    growthAnimation = { type: 'neuron', layerIndex, neuronIndex, startedAt: performance.now() };
    animationUntil = performance.now() + 1200;
    const message = `Loss estagnada: o neurônio ${donorIndex + 1} foi dividido com o novo neurônio ${neuronIndex + 1}; entradas herdadas e pesos de saída redistribuídos.`;
    addGrowthEvent(`+ neurônio ${neuronIndex + 1} · conexões rebalanceadas`);
    growthHistory.push({ epoch, type: 'neuron', layer: layerIndex });
    lastGrowthEpoch = epoch;
    setLesson('A rede ganhou capacidade', message);
  } else if (network.hiddenLayerCount < config.maxHiddenLayers) {
    const insertedSize = network.addHiddenLayer(config.newLayerSize);
    const layerIndex = network.lastHiddenIndex;
    growthAnimation = { type: 'layer', layerIndex, startedAt: performance.now() };
    animationUntil = performance.now() + insertedSize * 110 + 1000;
    const message = `A camada anterior atingiu ${config.maxNeuronsPerLayer} neurônios; nasceu a camada oculta ${layerIndex} com ${insertedSize} neurônios e uma transformação próxima da identidade.`;
    addGrowthEvent(`+ camada ${layerIndex} · ${insertedSize} neurônios preservados`);
    growthHistory.push({ epoch, type: 'layer', layer: layerIndex });
    lastGrowthEpoch = epoch;
    setLesson('Nova camada criada', message);
  } else {
    stagnation = config.patienceLimit;
    elements.growthRule.textContent = 'Limite didático de arquitetura atingido.';
    return;
  }

  stagnation = 0;
  bestLoss = loss;
}

function crossEntropy(model, samples, kind, limit = 32) {
  if (!model || !samples.length) return null;
  const stride = Math.max(1, Math.floor(samples.length / limit));
  let total = 0;
  let count = 0;
  for (let index = 0; index < samples.length && count < limit; index += stride) {
    const sample = samples[index];
    const output = kind === 'attention'
      ? model.forward(sample.contextIndices).output
      : model.feedForward(processor.indexToVector(sample.inputIndex)).output;
    total -= Math.log(Math.max(output[sample.targetIndex], 1e-12));
    count++;
  }
  return count ? total / count : null;
}

function recordHistory(evaluated = false) {
  if (!network || !attentionNetwork || !evaluated) return;
  attentionEvaluatedTrainLoss = crossEntropy(attentionNetwork, dataset, 'attention');
  mlpTestLoss = crossEntropy(network, testDataset, 'mlp');
  attentionTestLoss = crossEntropy(attentionNetwork, testDataset, 'attention');
  lossHistory.push({ epoch, mlpTrain: mlpEvaluatedTrainLoss, mlpTest: mlpTestLoss, attentionTrain: attentionEvaluatedTrainLoss, attentionTest: attentionTestLoss });
  if (lossHistory.length > 160) lossHistory.shift();
  drawHistory();
}

function drawHistory() {
  const canvas = elements.historyCanvas;
  const rect = canvas.getBoundingClientRect();
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.max(1, Math.floor(rect.width * ratio));
  canvas.height = Math.max(1, Math.floor(rect.height * ratio));
  const chart = canvas.getContext('2d');
  chart.setTransform(ratio, 0, 0, ratio, 0, 0);
  chart.clearRect(0, 0, rect.width, rect.height);
  if (lossHistory.length < 2) return;
  const values = lossHistory.flatMap(point => [point.mlpTest, point.attentionTest]).filter(Number.isFinite);
  const maximum = Math.max(1, ...values);
  const drawLine = (key, color) => {
    chart.beginPath();
    lossHistory.forEach((point, index) => {
      const x = index / Math.max(1, lossHistory.length - 1) * rect.width;
      const y = rect.height - Math.min(maximum, point[key] ?? maximum) / maximum * (rect.height - 5) - 2;
      if (index === 0) chart.moveTo(x, y); else chart.lineTo(x, y);
    });
    chart.strokeStyle = color;
    chart.lineWidth = 1.6;
    chart.stroke();
  };
  drawLine('mlpTest', cssColor('--accent'));
  drawLine('attentionTest', cssColor('--positive'));
  const firstEpoch = lossHistory[0].epoch;
  const lastEpoch = lossHistory.at(-1).epoch;
  growthHistory.filter(item => item.epoch >= firstEpoch).forEach(item => {
    const x = (item.epoch - firstEpoch) / Math.max(1, lastEpoch - firstEpoch) * rect.width;
    chart.fillStyle = cssColor('--warning');
    chart.fillRect(x, 0, 1, rect.height);
  });
}

function updateLearningPanel(sample, prediction) {
  elements.currentInput.textContent = sample.contextWords.join(' · ');
  elements.currentTarget.textContent = sample.targetWord;
  elements.currentPrediction.textContent = prediction.word;
  elements.currentConfidence.textContent = `${(prediction.confidence * 100).toFixed(1)}%`;
  markRenderDirty();

  if (prediction.word === sample.targetWord) {
    setLesson('A rede acertou antes do ajuste', `Para o contexto “${sample.contextWords.join(' ')}”, a maior saída já era “${sample.targetWord}”. O treino reforçou esse caminho.`);
  } else {
    setLesson('Backpropagation em ação', `A rede previu “${prediction.word}”, mas o alvo era “${sample.targetWord}”. Os pesos foram ajustados de trás para frente.`);
  }
}

function finishTrainingSteps(completedSteps) {
  epoch += completedSteps;
  stepsSinceEvaluation += completedSteps;
  let evaluated = false;
  if (stepsSinceEvaluation >= config.evaluationEvery) {
    stepsSinceEvaluation %= config.evaluationEvery;
    mlpEvaluatedTrainLoss = crossEntropy(network, dataset, 'mlp');
    updateGrowth(mlpEvaluatedTrainLoss);
    evaluated = true;
  }
  recordHistory(evaluated);
  updateMetrics();
}

function performTrainingStep() {
  if (!network) return;
  const sample = dataset[Math.floor(Math.random() * dataset.length)];
  const input = processor.indexToVector(sample.inputIndex);
  const target = processor.indexToVector(sample.targetIndex);
  visualInput = input;
  const beforeTraining = network.feedForward(input).output;
  const prediction = processor.vectorToWord(beforeTraining);
  updateLossEstimate(-Math.log(Math.max(beforeTraining[sample.targetIndex], 1e-12)));
  network.train(input, target);
  const attentionBefore = attentionNetwork.train(sample.contextIndices, sample.targetIndex).output;
  const sampleAttentionLoss = -Math.log(Math.max(attentionBefore[sample.targetIndex], 1e-12));
  attentionLoss = attentionLoss === null ? sampleAttentionLoss : attentionLoss * 0.9 + sampleAttentionLoss * 0.1;
  updateLearningPanel(sample, prediction);
  finishTrainingSteps(1);
}

async function performParallelRound() {
  if (!network || parallelBusy) return false;
  if (!ensureWorkerPool()) {
    performTrainingStep();
    return true;
  }

  parallelBusy = true;
  const roundId = ++workerRoundId;
  const state = serializeNetwork();
  const sampleIndexes = workerPool.map((_, workerId) => (epoch + workerId) % dataset.length);

  try {
    const results = await Promise.all(workerPool.map((worker, workerId) => (
      requestWorkerTraining(worker, workerId, roundId, sampleIndexes[workerId], state)
    )));
    if (roundId !== workerRoundId) return false;
    applyAveragedNetworks(results.map(result => result.network));
    const representative = results[0];
    const sample = dataset[representative.sampleIndex];
    visualInput = processor.indexToVector(sample.inputIndex);
    const roundLoss = results.reduce((sum, result) => (
      sum - Math.log(Math.max(result.targetProbability, 1e-12))
    ), 0) / results.length;
    updateLossEstimate(roundLoss);
    results.forEach(result => {
      const attentionSample = dataset[result.sampleIndex];
      const output = attentionNetwork.train(attentionSample.contextIndices, attentionSample.targetIndex).output;
      const loss = -Math.log(Math.max(output[attentionSample.targetIndex], 1e-12));
      attentionLoss = attentionLoss === null ? loss : attentionLoss * 0.9 + loss * 0.1;
    });
    updateLearningPanel(sample, {
      word: processor.vocab[representative.predictionIndex],
      confidence: representative.confidence,
      index: representative.predictionIndex,
    });
    setLesson('Rodada paralela concluída', `${results.length} workers processaram exemplos simultaneamente. Os pesos resultantes foram combinados pela média.`);
    finishTrainingSteps(results.length);
    return true;
  } catch (error) {
    if (roundId !== workerRoundId) return false;
    fallbackToLocal(`Falha no processamento paralelo: ${error.message}. Usando modo local.`);
    performTrainingStep();
    return true;
  } finally {
    if (roundId === workerRoundId) parallelBusy = false;
  }
}

function ensureNetwork() {
  if (!syncCurrentCorpus()) return false;
  return network ? true : initializeNetwork();
}

function startTraining() {
  if (!ensureNetwork()) return;
  paused = false;
  lastTrainingAt = performance.now();
  updateTrainingButton();
  setStatus('Treinamento em execução. Pause para inspecionar com calma.');
}

function pauseTraining() {
  paused = true;
  updateTrainingButton();
  setStatus('Treinamento pausado. Você pode avançar um passo por vez.');
}

async function stepTraining() {
  if (!ensureNetwork()) return;
  paused = true;
  updateTrainingButton();
  if (requestedWorkerCount > 1) await performParallelRound();
  else performTrainingStep();
  setStatus(requestedWorkerCount > 1 ? `Rodada paralela concluída; ${epoch} exemplos processados.` : `Passo ${epoch} concluído.`);
}

function trainFrame(now) {
  if (!network || paused || now - lastTrainingAt < config.trainingInterval) return;
  lastTrainingAt = now;
  if (requestedWorkerCount > 1) void performParallelRound();
  else performTrainingStep();
}

function sampleNextWord(probabilities, recentIndices, temperature = 0.9, topK = 6) {
  const recentWindow = recentIndices.slice(-6);
  const immediatePrevious = recentWindow.at(-1);
  const ranked = probabilities.map((probability, index) => {
    const occurrences = recentWindow.reduce((total, value) => total + Number(value === index), 0);
    let score = Math.pow(Math.max(probability, 1e-12), 1 / temperature);
    score /= Math.pow(1.8, occurrences);
    if (index === immediatePrevious) score /= 4;
    return { index, probability, score };
  }).sort((a, b) => b.score - a.score).slice(0, Math.min(topK, probabilities.length));

  const total = ranked.reduce((sum, item) => sum + item.score, 0);
  let draw = Math.random() * total;
  for (const item of ranked) {
    draw -= item.score;
    if (draw <= 0) return item;
  }
  return ranked.at(-1);
}

function generateStory() {
  if (!network || !processor) {
    setStatus('Inicie a rede antes de gerar uma sequência.');
    return;
  }
  const seedWords = elements.seed.value.toLocaleLowerCase('pt-BR').match(/[\p{L}\p{N}]+(?:['’-][\p{L}\p{N}]+)*/gu) ?? [];
  if (!seedWords.length || seedWords.some(word => !processor.vocab.includes(word))) {
    setStatus('Todas as palavras do contexto inicial precisam existir no corpus.');
    return;
  }

  elements.generated.replaceChildren();
  const generated = [...seedWords];
  seedWords.forEach(seed => {
    const word = document.createElement('span');
    word.className = 'generated-word seed';
    word.textContent = `${seed} `;
    elements.generated.append(word);
  });
  for (let index = 0; index < 15; index++) {
    const output = elements.generatorModel.value === 'attention'
      ? attentionNetwork.forward(generated.slice(-attentionNetwork.contextSize).map(word => processor.wordIndex.get(word))).output
      : network.feedForward(processor.wordToVector(current)).output;
    const recentIndices = generated.map(word => processor.wordIndex.get(word));
    const sampled = sampleNextWord(output, recentIndices, Number(elements.temperature.value));
    const prediction = { word: processor.vocab[sampled.index], confidence: sampled.probability, index: sampled.index };
    const word = document.createElement('span');
    word.className = `generated-word ${prediction.confidence > 0.8 ? 'high' : 'low'}`;
    word.textContent = `${prediction.word} `;
    elements.generated.append(word);
    generated.push(prediction.word);
  }
  elements.generated.append('…');
  visualInput = processor.wordToVector(seedWords.at(-1));
  markRenderDirty();
  setStatus('Sequência gerada com top-k, temperatura e penalidade de repetição.');
  setLesson('Geração probabilística', 'Em vez de escolher sempre a maior saída, o laboratório sorteia entre alternativas plausíveis e reduz a chance de repetir palavras recentes.');
}

function createSnapshot() {
  if (!network || !attentionNetwork || !processor) throw new Error('Crie e treine a rede antes de salvar.');
  return {
    format: 'neural-lab-v2',
    exportedAt: new Date().toISOString(),
    corpus: activeCorpus,
    contextSize: Number(elements.contextSize.value),
    vocabulary: processor.vocab,
    split: { strategy: '80/20 determinístico; cada quinto exemplo vai para teste', train: dataset.length, test: testDataset.length },
    metrics: { epoch, bestLoss, displayedLoss, attentionLoss, mlpTestLoss, attentionTestLoss, mlpEvaluatedTrainLoss, attentionEvaluatedTrainLoss, stagnation, lastGrowthEpoch },
    history: lossHistory,
    growthHistory,
    mlp: serializeNetwork(),
    attention: attentionNetwork.serialize(),
  };
}

function applySnapshot(snapshot, options = {}) {
  if (snapshot?.format !== 'neural-lab-v2' || !snapshot.corpus || !snapshot.mlp || !snapshot.attention) throw new Error('Arquivo incompatível ou incompleto.');
  stopWorkers();
  elements.corpus.value = snapshot.corpus;
  elements.contextSize.value = String(snapshot.contextSize || 3);
  if (!prepareCorpus()) throw new Error('O corpus salvo não produz exemplos suficientes.');
  network = restoreMlp(snapshot.mlp);
  attentionNetwork = AttentionNetwork.restore(snapshot.attention);
  ({ epoch = 0, bestLoss = Infinity, displayedLoss = null, attentionLoss = null, mlpTestLoss = null, attentionTestLoss = null, mlpEvaluatedTrainLoss = null, attentionEvaluatedTrainLoss = null, stagnation = 0, lastGrowthEpoch = 0 } = snapshot.metrics || {});
  lossHistory = Array.isArray(snapshot.history) ? snapshot.history : [];
  growthHistory = Array.isArray(snapshot.growthHistory) ? snapshot.growthHistory : [];
  visualInput = processor.indexToVector(dataset[0].inputIndex);
  paused = true;
  formationStartedAt = 0;
  updateTrainingButton();
  updateMetrics();
  drawHistory();
  markRenderDirty();
  setStatus(`Modelo carregado na época ${epoch.toLocaleString('pt-BR')}. O treinamento permanece pausado.`);
  setLesson('Estado restaurado', 'Corpus, vocabulário, duas arquiteturas, pesos, embeddings, métricas e histórico voltaram ao ponto salvo.');

  if (options.resume) {
    paused = false;
    lastTrainingAt = performance.now();
    updateTrainingButton();
    setStatus(`Sessão recuperada após o refresh na etapa ${epoch.toLocaleString('pt-BR')}; treinamento retomado.`);
    setLesson('Live Server detectado', 'A página recarregou, mas o último checkpoint foi restaurado porque o treinamento já estava em execução.');
  }
}

function saveSessionCheckpoint() {
  try {
    if (!network || !attentionNetwork) {
      localStorage.removeItem(SESSION_KEY);
      return false;
    }
    localStorage.setItem(SESSION_KEY, JSON.stringify({
      snapshot: createSnapshot(),
      wasRunning: !paused,
      executionMode: requestedWorkerCount,
      trainingInterval: config.trainingInterval,
    }));
    return true;
  } catch {
    return false;
  }
}

function restoreSessionCheckpoint() {
  try {
    const saved = localStorage.getItem(SESSION_KEY);
    if (!saved) return false;
    const checkpoint = JSON.parse(saved);
    if (!checkpoint.snapshot) return false;
    if ([1, 2, 4].includes(checkpoint.executionMode)) {
      elements.execution.value = String(checkpoint.executionMode);
      configureExecutionMode();
    }
    if ([150, 400, 700].includes(checkpoint.trainingInterval)) {
      config.trainingInterval = checkpoint.trainingInterval;
      elements.speed.value = String(checkpoint.trainingInterval);
    }
    applySnapshot(checkpoint.snapshot, { resume: checkpoint.wasRunning === true });
    return true;
  } catch {
    try { localStorage.removeItem(SESSION_KEY); } catch { /* Armazenamento indisponível. */ }
    return false;
  }
}

function saveModel() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(createSnapshot()));
    setStatus('Rede salva neste navegador.');
  } catch (error) { setStatus(`Não foi possível salvar: ${error.message}`); }
}

function loadModel() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (!saved) throw new Error('nenhuma rede salva neste navegador');
    applySnapshot(JSON.parse(saved));
  } catch (error) { setStatus(`Não foi possível carregar: ${error.message}.`); }
}

function exportModel() {
  try {
    const blob = new Blob([JSON.stringify(createSnapshot(), null, 2)], { type: 'application/json' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `rede-neural-epoca-${epoch}.json`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 0);
    setStatus('Modelo e parâmetros exportados em JSON.');
  } catch (error) { setStatus(`Não foi possível exportar: ${error.message}`); }
}

async function importModel(file) {
  try { applySnapshot(JSON.parse(await file.text())); }
  catch (error) { setStatus(`Não foi possível importar: ${error.message}`); }
  finally { elements.importFile.value = ''; }
}

function updateNeuronDetails() {
  if (!selectedNeuron || !network) return;
  const input = visualInput || processor.indexToVector(dataset[0].inputIndex);
  const result = network.feedForward(input);
  const { layer, layerIndex, index } = selectedNeuron;
  let title;
  let content;

  if (layer === 'input') {
    const weights = network.weights[0].data.map(row => row[index]);
    const average = weights.reduce((sum, value) => sum + Math.abs(value), 0) / weights.length;
    title = `Entrada · “${processor.vocab[index]}”`;
    content = `Ativação ${input[index].toFixed(2)} · ${weights.length} saídas · |peso| médio ${average.toFixed(3)}`;
  } else if (layer === 'hidden') {
    const incoming = network.weights[layerIndex - 1].data[index];
    const outgoing = network.weights[layerIndex].data.map(row => row[index]);
    const allWeights = [...incoming, ...outgoing];
    const average = allWeights.reduce((sum, value) => sum + Math.abs(value), 0) / allWeights.length;
    title = `Oculta ${layerIndex} · neurônio ${index + 1}`;
    content = `Ativação ${result.activations[layerIndex][index].toFixed(3)} · bias ${network.biases[layerIndex - 1].data[index][0].toFixed(3)} · ${allWeights.length} conexões · |peso| médio ${average.toFixed(3)}`;
  } else {
    const incoming = network.weights.at(-1).data[index];
    const average = incoming.reduce((sum, value) => sum + Math.abs(value), 0) / incoming.length;
    title = `Saída · “${processor.vocab[index]}”`;
    content = `Probabilidade ${(result.output[index] * 100).toFixed(1)}% · bias ${network.biases.at(-1).data[index][0].toFixed(3)} · ${incoming.length} entradas · |peso| médio ${average.toFixed(3)}`;
  }

  elements.detailTitle.textContent = title;
  elements.detailContent.textContent = content;
  selectedNeuronEpoch = epoch;
}

function resizeCanvas() {
  const rect = elements.canvas.getBoundingClientRect();
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  elements.canvas.width = Math.max(1, Math.floor(rect.width * ratio));
  elements.canvas.height = Math.max(1, Math.floor(rect.height * ratio));
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  markRenderDirty();
}

function drawEmptyState(width, height) {
  ctx.fillStyle = cssColor('--muted');
  ctx.textAlign = 'center';
  ctx.font = '700 14px system-ui';
  ctx.fillText('A rede ainda não existe', width / 2, height / 2 - 8);
  ctx.font = '12px system-ui';
  ctx.fillText('Insira um corpus e clique em Iniciar ou Avançar 1 passo', width / 2, height / 2 + 16);
}

function getDisplayIndices(size, layerIndex, values) {
  const limit = config.maxVisualNodesPerLayer;
  if (size <= limit) return Array.from({ length: size }, (_, index) => index);
  const indices = Array.from({ length: limit }, (_, index) => Math.round(index * (size - 1) / (limit - 1)));
  const strongest = values.reduce((best, value, index) => value > values[best] ? index : best, 0);
  const important = [strongest];
  if (selectedNeuron?.layerIndex === layerIndex) important.push(selectedNeuron.index);
  if (growthAnimation?.layerIndex === layerIndex && growthAnimation.neuronIndex !== undefined) important.push(growthAnimation.neuronIndex);
  important.forEach((index, offset) => { indices[indices.length - 1 - offset] = index; });
  return [...new Set(indices)].sort((a, b) => a - b);
}

function formationVisibility(now, displayCounts) {
  const elapsed = now - formationStartedAt;
  const starts = [0];
  for (let layer = 1; layer < displayCounts.length; layer++) {
    const previousDelay = layer === 1 ? 72 : 105;
    starts[layer] = starts[layer - 1] + displayCounts[layer - 1] * previousDelay + 180;
  }
  const visible = displayCounts.map((size, layer) => {
    const delay = layer === 0 || layer === displayCounts.length - 1 ? 72 : 130;
    return Math.min(size, Math.max(0, Math.floor((elapsed - starts[layer]) / delay) + 1));
  });
  return { elapsed, starts, visible };
}

function growthProgress(now, layerIndex, neuronIndex) {
  if (!growthAnimation || growthAnimation.layerIndex !== layerIndex) return 1;
  if (growthAnimation.type === 'neuron' && growthAnimation.neuronIndex !== neuronIndex) return 1;
  const delay = growthAnimation.type === 'layer' ? neuronIndex * 110 : 0;
  return Math.max(0, Math.min(1, (now - growthAnimation.startedAt - delay) / 700));
}

function drawNetwork(now) {
  const rect = elements.canvas.getBoundingClientRect();
  const width = rect.width;
  const height = rect.height;
  ctx.clearRect(0, 0, width, height);

  if (!network || dataset.length === 0) {
    neuronHitAreas = [];
    drawEmptyState(width, height);
    return;
  }

  const input = visualInput || processor.indexToVector(dataset[0].inputIndex);
  const result = network.feedForward(input);
  const layerCount = network.layerSizes.length;
  const displayIndices = network.layerSizes.map((size, layer) => (
    getDisplayIndices(size, layer, layer === 0 ? input : result.activations[layer])
  ));
  const xPositions = Array.from({ length: layerCount }, (_, layer) => (
    58 + (width - 116) * layer / (layerCount - 1)
  ));
  const nodeY = (total, index) => {
    const available = Math.max(40, height - 82);
    const spacing = Math.min(available / Math.max(total, 1), 32);
    return 48 + (available - spacing * total) / 2 + index * spacing + spacing / 2;
  };
  const { elapsed, starts, visible } = formationVisibility(now, displayIndices.map(indices => indices.length));

  const drawSynapse = (x1, y1, x2, y2, weight, opacity) => {
    if (Math.abs(weight) < 0.06 || opacity <= 0) return;
    const alpha = Math.min(0.66, Math.abs(weight) * 0.36) * opacity;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.lineWidth = Math.min(Math.abs(weight) * 1.2, 2.2);
    ctx.strokeStyle = weight > 0
      ? `rgba(15, 159, 110, ${alpha})`
      : `rgba(216, 83, 79, ${alpha})`;
    ctx.stroke();
  };

  for (let layer = 0; layer < layerCount - 1; layer++) {
    const edgeStart = Math.max(
      starts[layer] + displayIndices[layer].length * 72,
      starts[layer + 1] + displayIndices[layer + 1].length * 72,
    ) + 120;
    const formationOpacity = Math.max(0, Math.min(1, (elapsed - edgeStart) / 750));
    for (let targetPosition = 0; targetPosition < visible[layer + 1]; targetPosition++) {
      const target = displayIndices[layer + 1][targetPosition];
      for (let sourcePosition = 0; sourcePosition < visible[layer]; sourcePosition++) {
        const source = displayIndices[layer][sourcePosition];
        const targetGrowth = growthProgress(now, layer + 1, target);
        const sourceGrowth = growthProgress(now, layer, source);
        drawSynapse(
          xPositions[layer],
          nodeY(displayIndices[layer].length, sourcePosition),
          xPositions[layer + 1],
          nodeY(displayIndices[layer + 1].length, targetPosition),
          network.weights[layer].data[target][source],
          formationOpacity * Math.min(targetGrowth, sourceGrowth),
        );
      }
    }
  }

  neuronHitAreas = [];
  for (let layer = 0; layer < layerCount; layer++) {
    const isInput = layer === 0;
    const isOutput = layer === layerCount - 1;
    const values = isInput ? input : result.activations[layer];
    const label = isInput ? 'Entrada' : isOutput ? 'Saída' : `Oculta ${layer}`;
    ctx.fillStyle = cssColor('--muted');
    ctx.textAlign = 'center';
    ctx.font = '700 10px system-ui';
    ctx.fillText(`${label} · ${network.layerSizes[layer]}`, xPositions[layer], 22);

    for (let position = 0; position < visible[layer]; position++) {
      const index = displayIndices[layer][position];
      const value = values[index] ?? 0;
      const progress = growthProgress(now, layer, index);
      const radius = (isInput || isOutput ? 3.2 : 4.3) * (0.25 + progress * 0.75);
      const y = nodeY(displayIndices[layer].length, position);
      ctx.beginPath();
      ctx.arc(xPositions[layer], y, radius, 0, Math.PI * 2);
      ctx.fillStyle = !isInput && !isOutput
        ? progress < 1 ? cssColor('--positive') : cssColor('--accent')
        : `rgb(${Math.round(242 - value * 170)}, ${Math.round(244 - value * 110)}, ${Math.round(246 - value * 80)})`;
      ctx.fill();
      neuronHitAreas.push({
        layer: isInput ? 'input' : isOutput ? 'output' : 'hidden',
        layerIndex: layer,
        index,
        x: xPositions[layer],
        y,
      });
    }
  }

  const selected = neuronHitAreas.find(area => (
    area.layer === selectedNeuron?.layer
    && area.layerIndex === selectedNeuron?.layerIndex
    && area.index === selectedNeuron?.index
  ));
  if (selected) {
    ctx.beginPath();
    ctx.arc(selected.x, selected.y, 10, 0, Math.PI * 2);
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = cssColor('--warning');
    ctx.stroke();
  }

  if (growthAnimation) {
    const layerDuration = growthAnimation.type === 'layer'
      ? network.layerSizes[growthAnimation.layerIndex] * 110 + 700
      : 700;
    if (now - growthAnimation.startedAt > layerDuration) growthAnimation = null;
  }
  if (selectedNeuron && selectedNeuronEpoch !== epoch) updateNeuronDetails();
}

function selectNeuron(event) {
  const rect = elements.canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const hit = neuronHitAreas.find(area => Math.hypot(area.x - x, area.y - y) <= 14);
  if (!hit) return;
  selectedNeuron = { layer: hit.layer, layerIndex: hit.layerIndex, index: hit.index };
  selectedNeuronEpoch = -1;
  updateNeuronDetails();
  markRenderDirty();
}

function animate(now) {
  trainFrame(now);
  const animationActive = now < animationUntil || Boolean(growthAnimation);
  if ((renderDirty || animationActive) && now - lastDrawAt >= config.drawInterval) {
    drawNetwork(now);
    renderDirty = false;
    lastDrawAt = now;
  }
  requestAnimationFrame(animate);
}

elements.restart.addEventListener('click', prepareCorpus);
elements.pause.addEventListener('click', () => { if (paused) startTraining(); else pauseTraining(); });
elements.step.addEventListener('click', stepTraining);
elements.speed.addEventListener('change', () => {
  config.trainingInterval = Number(elements.speed.value);
  setStatus(`Velocidade alterada para ${elements.speed.selectedOptions[0].text.toLocaleLowerCase('pt-BR')}.`);
});
elements.execution.addEventListener('change', configureExecutionMode);
elements.contextSize.addEventListener('change', prepareCorpus);
elements.generate.addEventListener('click', generateStory);
elements.save.addEventListener('click', saveModel);
elements.load.addEventListener('click', loadModel);
elements.export.addEventListener('click', exportModel);
elements.import.addEventListener('click', () => elements.importFile.click());
elements.importFile.addEventListener('change', () => { if (elements.importFile.files[0]) void importModel(elements.importFile.files[0]); });
elements.canvas.addEventListener('click', selectNeuron);
elements.theme.addEventListener('click', () => {
  const theme = document.body.dataset.theme === 'dark' ? 'light' : 'dark';
  applyTheme(theme);
  try { localStorage.setItem('neural-theme', theme); } catch { /* Preferência ficará apenas na sessão. */ }
});
window.addEventListener('resize', () => { resizeCanvas(); drawHistory(); });
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'hidden') saveSessionCheckpoint();
});
window.addEventListener('beforeunload', () => {
  saveSessionCheckpoint();
  stopWorkers();
});

applyTheme(getInitialTheme());
resizeCanvas();
configureExecutionMode();
prepareCorpus();
restoreSessionCheckpoint();
requestAnimationFrame(animate);
