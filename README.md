# 🧠 Rede Neural Evolutiva (Auto-Growth)
### 🧠 Evolutionary Neural Network (Auto-Growth)

> 🇧🇷 Versão em Português  
> 🇺🇸 English version below

---

## 🇧🇷 Português (pt-BR)

### 📌 Visão Geral

Este projeto é um **experimento educacional e visual** que demonstra uma **rede neural artificial capaz de crescer dinamicamente** quando o aprendizado entra em estagnação.

Ele é implementado em **JavaScript puro**, **sem bibliotecas externas**, e roda **100% no navegador**, com foco em **fundamentos de redes neurais, transparência e visualização**.

> ⚠️ Importante: este projeto **não é um LLM**. É apenas uma rede neural **simples e didática**, construída para estudo.

---

### 🎯 Objetivo

Demonstrar de forma clara e interativa:

- Funcionamento de uma **rede neural feedforward**
- Treinamento via **backpropagation**
- Identificação de **estagnação de aprendizado**
- Crescimento estrutural automático da rede (**auto-growth**) de neurônios e camadas

---

### 🧩 Arquitetura

Entrada (One-Hot Words)
↓
Camadas Ocultas (dinâmicas)
↓
Saída (Próxima Palavra)

- **Entrada**: vetor one-hot da palavra atual  
- **Saída**: vetor one-hot da próxima palavra  
- **Modelo**: previsão da próxima palavra (bigrama)

---

### 🖥️ Interface

#### Coluna Esquerda — Controle e Diagnóstico
- Inserção do **corpus de texto**
- Reinício completo da rede
- Métricas em tempo real:
  - Epochs
  - Erro atual (loss)
  - Quantidade de neurônios e camadas ocultas
  - Indicador de estagnação (frustração)

#### Coluna Direita — Visualização Neural
- Neurônios de entrada, ocultos e saída
- Sinapses coloridas:
  - Verde → reforço
  - Vermelho → inibição
- Intensidade visual representa o peso da conexão

---

### 🧠 Funcionamento Interno

#### 1️⃣ Processamento de Texto
Classe `TextProcessor`:
- Normalização do texto
- Tokenização
- Criação de vocabulário
- Vetorização one-hot
- Geração de pares `(palavra atual → próxima palavra)`

#### 2️⃣ Rede Neural
Classe `NeuralNetwork`:
- Feedforward clássico
- Sigmoide na camada oculta e softmax na saída
- Backpropagation manual
- Taxa de aprendizado fixa
- Sem regularização ou normalização

#### 3️⃣ Crescimento Dinâmico (Auto-Growth)
- O erro é monitorado continuamente
- Se não houver melhora significativa:
  - Um novo neurônio oculto é adicionado
- Pesos existentes são preservados
- Novas sinapses iniciam com valores pequenos aleatórios

---

### 📈 Loop de Treinamento
- Execução contínua com `requestAnimationFrame`
- Treinamento em mini-batches aleatórios
- Atualização em tempo real da visualização e métricas
- O treinamento **nunca para**

---

### ✨ Geração de Texto
- Inicia a partir de uma palavra conhecida
- Cada palavra prevista vira a próxima entrada
- Confiança visual:
  - Verde → alta
  - Amarelo → média

Limitações esperadas:
- Repetição de padrões
- Loops
- Ausência de semântica

---

### 🧪 O Que Este Projeto NÃO É
- ❌ Não é um LLM
- ❌ Não usa embeddings
- ❌ Não possui atenção
- ❌ Não mantém contexto longo
- ❌ Não generaliza semanticamente

---

### ✅ O Que Este Projeto É
- ✔ Laboratório visual de redes neurais
- ✔ Backpropagation do zero
- ✔ Prova de conceito de crescimento estrutural
- ✔ Ideal para estudo e ensino

---

### 🚀 Possíveis Evoluções
- Critério estatístico mais robusto para crescimento
- Limite máximo de neurônios
- Penalização por complexidade
- Dropout
- Embeddings contínuos
- RNNs ou mecanismos de atenção

---

### ▶️ Como Executar
1. Mantenha `index.html`, `styles.css` e `app.js` na mesma pasta
2. Abra `index.html` em um navegador moderno (ou sirva a pasta com um servidor HTTP estático)
3. Observe a rede aprender e crescer; use **Pausar** para inspecionar o estado
4. Modifique o corpus, mantendo pelo menos duas palavras, e reinicie a rede

---

## 🇺🇸 English

### 📌 Overview

This project is an **educational and visual experiment** that demonstrates a **neural network capable of dynamically growing** when learning stagnates.

It is built using **pure JavaScript**, **no external libraries**, and runs **entirely in the browser**, focusing on **neural network fundamentals, transparency, and visualization**.

> ⚠️ Important: this is **not a Large Language Model**. It is a **deliberately simple, didactic neural network**.

---

### 🎯 Purpose

To demonstrate:

- How a **feedforward neural network** works
- **Backpropagation** training
- Learning **stagnation detection**
- Automatic structural growth (**auto-growth**)

---

### 🧩 Architecture

Input (One-Hot Words)
↓
Hidden Layer (Dynamic)
↓
Output (Next Word)

---

### 🖥️ Interface
- Left panel: control, corpus input, diagnostics
- Right panel: real-time neural visualization
- Color-coded synapses indicate reinforcement or inhibition

---

### 🧠 Internal Design
- One-hot text encoding
- Manual matrix math
- Sigmoid hidden activations and softmax output
- Dynamic hidden-neuron and hidden-layer expansion when loss plateaus

---

### 📈 Training Loop
- Continuous training via `requestAnimationFrame`
- Random mini-batches
- Real-time metrics and visualization
- Training never stops

---

### ✨ Text Generation
- Seed-based word generation
- Confidence-based coloring
- Expected pattern repetition and loops

---

### ❌ What This Is NOT
- Not an LLM
- No embeddings
- No attention
- No long-term context
- No semantic reasoning

---

### ✅ What This IS
- A neural network learning lab
- A from-scratch backprop example
- A structural growth proof of concept
- Excellent for learning and experimentation

---

### ▶️ How to Run
1. Keep `index.html`, `styles.css`, and `app.js` in the same directory
2. Open `index.html` in a modern browser (or serve the directory through a static HTTP server)
3. Watch the network learn and evolve; use **Pause** to inspect its state

---

### 🚀 Next Steps
- Statistical growth criteria
- Regularization techniques
- Embeddings
- RNNs or attention-based models

---

📌 **License**: Free for educational and experimental use.
