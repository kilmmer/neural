# 🧠 Rede Neural Evolutiva (Auto-Growth)
### 🧠 Evolutionary Neural Network (Auto-Growth)

> 🇧🇷 Versão em Português  
> 🇺🇸 English version below

---

## 🇧🇷 Português (pt-BR)

### 📌 Visão Geral

Este projeto é um **experimento educacional e visual** que demonstra uma **rede neural artificial capaz de crescer dinamicamente** quando o aprendizado entra em estagnação.

O laboratório compara duas arquiteturas: a **MLP evolutiva**, que recebe a última palavra, e um **Tiny Transformer causal**, com dicionário, embeddings de tokens e posições, atenção multi-head, projeções Q/K/V, residual, normalização e feed-forward. O corpus é separado de forma determinística em **80% para treino e 20% para teste**.

Ele é implementado em **JavaScript puro**, **sem bibliotecas externas**, e roda **100% no navegador**, com foco em **fundamentos de redes neurais, transparência e visualização**.

> ⚠️ Importante: este projeto **não é um LLM**. É apenas uma rede neural **simples e didática**, construída para estudo.

---

### 🎯 Objetivo

Demonstrar de forma clara e interativa:

- Funcionamento de uma **rede neural feedforward**
- Treinamento via **backpropagation**
- Identificação de **estagnação de aprendizado**
- Crescimento estrutural automático da rede (**auto-growth**) de neurônios e camadas
- Comparação de loss de treino/teste entre MLP e Tiny Transformer
- Histórico visual de loss e eventos de crescimento
- Persistência no navegador e importação/exportação completa em JSON
- KV Cache por cabeça durante a geração do Tiny Transformer
- Worker dedicado e contador de passos separado para o Tiny Transformer
- Larguras adaptativas e diferentes por camada na MLP, calculadas a partir do dicionário
- Tokenizer BPE treinado no corpus, com vocabulário 256/512/768 e tokens especiais
- Early stopping independente por loss de teste, com restauração automática dos melhores parâmetros

O JSON inclui corpus, dicionário, pesos, biases, embeddings, posições, matrizes Q/K/V, feed-forward, métricas e histórico. Ele usa o formato educacional deste projeto e não é diretamente um arquivo TensorFlow ou PyTorch.

---

### 🧩 Arquitetura

Modelo A: palavra one-hot → camadas ocultas dinâmicas → próxima palavra

Modelo B: contexto → embeddings + posições → atenção multi-head → residual + normalização → feed-forward → próxima palavra

- **Entrada**: vetor one-hot da palavra atual  
- **Saída**: vetor one-hot da próxima palavra  
- **Modelos**: previsão da próxima palavra por bigrama (MLP) e por contexto de 2–12 palavras (Tiny Transformer)

---

### 🖥️ Interface

#### Coluna Esquerda — Controle e Diagnóstico
- Inserção do **corpus de texto**
- Reinício completo da rede
- Treinamento contínuo ou avanço de **um passo por vez**
- Controle de velocidade para observação didática
- Execução local ou paralela com **2 ou 4 Web Workers**
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
- Rótulos dinâmicos para todas as camadas
- Inspeção de neurônios por clique
- Painel que mostra entrada, alvo, previsão e confiança a cada passo

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
- Execução cadenciada com `requestAnimationFrame`
- Treinamento por amostras aleatórias, com modo passo a passo
- Atualização em tempo real da visualização e métricas
- Pausa e retomada controladas pelo usuário
- No modo paralelo, cada worker treina uma cópia e os parâmetros são agregados pela média
- O dataset armazena índices compactos em vez de milhares de vetores one-hot duplicados
- A loss usa média móvel das amostras já treinadas, sem varrer o corpus na thread principal
- A visualização limita camadas grandes a 56 pontos e redesenha apenas quando necessário
- Corpus limitado a 5.000 tokens para manter memória e resposta previsíveis

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
- ❌ Não é um Transformer de produção ou um LLM de larga escala
- ❌ Não mantém contexto além da janela configurada
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
- Mais blocos Transformer e contexto maior

---

### ▶️ Como Executar
1. Mantenha `index.html`, `styles.css` e `app.js` na mesma pasta
2. Sirva a pasta com um servidor HTTP estático (necessário para o modo paralelo com Web Workers)
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
- Not a production-scale Transformer or LLM
- No context beyond the configured window
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
2. Serve the directory through a static HTTP server (required for parallel Web Worker mode)
3. Watch the network learn and evolve; use **Pause** to inspect its state

---

### 🚀 Next Steps
- Statistical growth criteria
- Regularization techniques
- More Transformer blocks and a larger context window

---

📌 **License**: Free for educational and experimental use.
