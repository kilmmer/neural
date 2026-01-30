Perfeito. Li o HTML inteiro e vou te devolver um README claro, técnico e honesto, explicando o que é, como funciona, por que funciona e quais são os limites desse experimento.

⸻

🧠 Rede Neural Evolutiva (Auto-Growth)

Este projeto é um experimento educacional e visual que demonstra uma rede neural artificial capaz de crescer dinamicamente quando entra em estagnação durante o treinamento.

Ele não usa frameworks, não usa bibliotecas externas e roda 100% no navegador, com foco em aprendizado, visualização e conceitos fundamentais de IA.

⸻

🎯 Objetivo do Projeto

Demonstrar, de forma visual e interativa:
	•	Como funciona uma rede neural feedforward clássica
	•	Como ocorre o treinamento via backpropagation
	•	O que é estagnação de aprendizado
	•	Como uma rede pode aumentar sua capacidade estrutural (mais neurônios) quando não evolui

⚠️ Importante: isso não é um LLM, nem um modelo de linguagem moderno. É um modelo didático, propositalmente simples.

⸻

🧩 Visão Geral da Arquitetura

Entrada (One-Hot Words)
        ↓
Camada Oculta (dinâmica, cresce sozinha)
        ↓
Saída (Próxima Palavra)

	•	Entrada: vetor one-hot da palavra atual
	•	Saída: vetor one-hot da palavra seguinte
	•	Aprendizado: previsão da próxima palavra (modelo bigrama)

⸻

🖥️ Interface

Coluna Esquerda — Controle e Diagnóstico

1️⃣ Cérebro Dinâmico
	•	Campo para inserir o corpus de texto
	•	Botão para reiniciar a rede do zero
	•	Métricas em tempo real:
	•	Epochs
	•	Erro atual (loss)
	•	Número de neurônios ocultos
	•	Nível de frustração (estagnação)

2️⃣ Testar Inteligência
	•	Palavra inicial (seed)
	•	Geração de texto palavra por palavra
	•	Cor indica confiança da predição

⸻

Coluna Direita — Visualização Neural
	•	Neurônios:
	•	Entrada (esquerda)
	•	Ocultos (centro, em destaque)
	•	Saída (direita)
	•	Sinapses:
	•	Verde → reforço (peso positivo)
	•	Vermelho → inibição (peso negativo)
	•	Espessura e opacidade representam intensidade do peso

⸻

🧠 Como a Rede Funciona (Passo a Passo)

1️⃣ Processamento de Texto

Classe TextProcessor:
	•	Normaliza o texto
	•	Divide em palavras
	•	Cria vocabulário único
	•	Converte palavras em vetores one-hot
	•	Gera pares (palavra_atual → próxima_palavra)

⸻

2️⃣ Rede Neural

Classe NeuralNetwork:
	•	Feedforward clássico:
	•	Sigmoid em todas as camadas
	•	Backpropagation manual
	•	Taxa de aprendizado fixa (0.1)
	•	Sem regularização
	•	Sem normalização

⸻

3️⃣ Crescimento Dinâmico (Auto-Growth)

Este é o núcleo do experimento.

Conceito:
Se a rede para de melhorar, ela ganha mais capacidade.

Implementação:
	•	Monitora o erro (loss)
	•	Se o erro não melhora após N iterações:
	•	Um novo neurônio oculto é adicionado
	•	As matrizes de pesos são redimensionadas preservando memória
	•	Novas sinapses começam com pesos pequenos aleatórios

nn.addHiddenNeuron();

Isso simula:
	•	Plasticidade neural
	•	Aumento de complexidade sob demanda
	•	Evita começar com um modelo grande demais

⸻

📈 Loop de Treinamento
	•	Executa continuamente via requestAnimationFrame
	•	Treina em mini-batches aleatórios
	•	Atualiza:
	•	Visualização
	•	Métricas
	•	Lógica de crescimento
	•	O treinamento nunca para, a rede está sempre aprendendo

⸻

✨ Geração de Texto
	•	Começa com uma palavra conhecida
	•	Cada palavra prevista vira a próxima entrada
	•	Mostra visualmente:
	•	Confiança alta → verde
	•	Confiança média → amarelo

⚠️ Limitação esperada:
	•	Pode entrar em loops
	•	Pode repetir padrões
	•	Não entende semântica

Isso é esperado e correto para este tipo de modelo.

⸻

🧪 O Que Este Projeto NÃO É

Sejamos rigorosos:

❌ Não é um LLM
❌ Não entende contexto longo
❌ Não usa embeddings
❌ Não usa atenção
❌ Não generaliza bem

Ele é intencionalmente simples.

⸻

✅ O Que Este Projeto É

✔ Um laboratório visual de redes neurais
✔ Um exemplo real de backprop do zero
✔ Uma prova de conceito de crescimento estrutural
✔ Excelente para estudo, ensino e experimentação
✔ Ótimo ponto de partida para evoluir para:
	•	Embeddings
	•	N-grams
	•	RNNs
	•	Attention
	•	Neuroevolução real

⸻

🚀 Ideias de Evolução (Críticas Construtivas)

Um cético bem informado diria:
	•	Crescer neurônios não garante melhor generalização
	•	Pode levar a overfitting
	•	Falta critério estatístico robusto para crescimento

Possíveis melhorias:
	•	Critério baseado em média móvel do loss
	•	Limite máximo de neurônios
	•	Penalidade por complexidade
	•	Dropout visual
	•	Evoluir para embeddings contínuos
	•	Substituir sigmoid por ReLU / Softmax

⸻

📦 Como Usar
	1.	Salve o arquivo como index.html
	2.	Abra no navegador
	3.	Observe a rede aprender
	4.	Altere o corpus
	5.	Veja o cérebro crescer

  Perfeito. Abaixo está o README em inglês, mantendo rigor técnico, clareza e sem exageros conceituais.

⸻

[EN]

⸻

🧠 Evolutionary Neural Network (Auto-Growth)

This project is an educational and visual experiment that demonstrates a neural network capable of dynamically growing its hidden layer when learning stagnates.

It uses pure JavaScript, no external libraries, and runs entirely in the browser, with a strong focus on core AI fundamentals, transparency, and visualization.

⸻

🎯 Project Goal

To demonstrate, in a clear and interactive way:
	•	How a classic feedforward neural network works
	•	How backpropagation updates weights
	•	What training stagnation looks like
	•	How a model can increase its structural capacity when it stops improving

⚠️ Important: This is not a Large Language Model. It is a deliberately simple, didactic neural network.

⸻

🧩 Architecture Overview

Input (One-Hot Words)
        ↓
Hidden Layer (dynamic, self-growing)
        ↓
Output (Next Word)

	•	Input: one-hot vector representing the current word
	•	Output: one-hot vector representing the next word
	•	Task: next-word prediction (bigram-style learning)

⸻

🖥️ Interface Overview

Left Column — Control & Diagnostics

1️⃣ Dynamic Brain
	•	Text area for the training corpus
	•	Button to reset the network from scratch
	•	Real-time metrics:
	•	Epoch count
	•	Current loss
	•	Number of hidden neurons
	•	Frustration level (stagnation indicator)

2️⃣ Intelligence Test
	•	Seed word input
	•	Word-by-word text generation
	•	Color-coded confidence display

⸻

Right Column — Neural Visualization
	•	Neurons:
	•	Input layer (left)
	•	Hidden layer (center, highlighted)
	•	Output layer (right)
	•	Synapses:
	•	Green → positive reinforcement
	•	Red → inhibition
	•	Line thickness and opacity represent weight magnitude

⸻

🧠 How the System Works (Step by Step)

1️⃣ Text Processing

Handled by the TextProcessor class:
	•	Normalizes text
	•	Tokenizes words
	•	Builds a unique vocabulary
	•	Converts words to one-hot vectors
	•	Generates training pairs (current_word → next_word)

⸻

2️⃣ Neural Network

Implemented in the NeuralNetwork class:
	•	Standard feedforward architecture
	•	Sigmoid activation functions
	•	Manual backpropagation
	•	Fixed learning rate (0.1)
	•	No regularization
	•	No normalization

Everything is explicit and readable.

⸻

3️⃣ Dynamic Growth (Auto-Growth)

This is the core concept of the experiment.

Idea:
When the network stops improving, it adds capacity.

Mechanism:
	•	Tracks the current loss
	•	If loss does not significantly improve for a fixed number of iterations:
	•	A new hidden neuron is added
	•	Weight matrices are resized while preserving learned values
	•	New synapses start with small random weights

nn.addHiddenNeuron();

This simulates:
	•	Neural plasticity
	•	Capacity growth on demand
	•	Avoiding overparameterization at initialization

⸻

📈 Training Loop
	•	Runs continuously using requestAnimationFrame
	•	Trains on random mini-batches
	•	Updates:
	•	Network visualization
	•	Metrics
	•	Growth logic
	•	Training never stops — the system is always adapting

⸻

✨ Text Generation
	•	Starts from a known seed word
	•	Each predicted word becomes the next input
	•	Confidence visualization:
	•	High confidence → green
	•	Medium confidence → yellow

⚠️ Expected limitations:
	•	Can loop
	•	Can repeat patterns
	•	No semantic understanding

This behavior is correct and expected for this architecture.

⸻

🧪 What This Project Is NOT

Let’s be precise:

❌ Not a Large Language Model
❌ No long-term context
❌ No embeddings
❌ No attention mechanism
❌ No strong generalization

These are intentional constraints.

⸻

✅ What This Project IS

✔ A visual neural network laboratory
✔ A from-scratch backpropagation example
✔ A proof of concept for structural growth
✔ Excellent for learning and teaching
✔ A solid foundation for future experiments such as:
	•	Embeddings
	•	N-grams
	•	RNNs
	•	Attention mechanisms
	•	Neuroevolution

⸻

🚀 Critical Notes & Possible Improvements

A well-informed skeptic would say:
	•	Adding neurons does not guarantee better generalization
	•	Risk of overfitting exists
	•	Growth criterion is heuristic, not statistical

Possible enhancements:
	•	Moving-average loss evaluation
	•	Maximum neuron cap
	•	Complexity penalty
	•	Dropout (even visually)
	•	Replace sigmoid with ReLU / Softmax
	•	Transition to continuous embeddings

⸻

📦 How to Run
	1.	Save the file as index.html
	2.	Open it in any modern browser
	3.	Observe the network learn and grow
	4.	Modify the corpus
	5.	Watch the brain evolve

⸻
