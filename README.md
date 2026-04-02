# 📊 NLP Homework 2: Statistical Language Models & POS Tagging

![Stats](https://img.shields.io/badge/Method-Probabilistic_Modeling-cyan)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![NLTK](https://img.shields.io/badge/Library-NLTK/Regex-red)

This repository contains the implementation for Homework 2 in the **Natural Language Processing** course. The project focuses on the statistical foundations of language, specifically how to predict the next word in a sequence and how to tag words with their grammatical roles.

---

## 📋 Project Overview

In this assignment, we explore the probabilistic nature of text. We build models that "learn" the structure of a language by calculating the likelihood of word sequences.

### Key Concepts & Techniques:
* **N-gram Language Models:** Building Unigram, Bigram, and Trigram models to estimate sentence probabilities.
* **Smoothing Techniques:** Implementing **Laplace (Add-one)** or **Kneser-Ney** smoothing to handle the "Zero Probability" problem for unseen words.
* **POS Tagging (Part-of-Speech):** Assigning grammatical tags (Noun, Verb, Adj) to words in a sentence.
* **Hidden Markov Models (HMM):** Utilizing transition and emission probabilities for sequence labeling.
* **Viterbi Algorithm:** Implementing the dynamic programming approach to find the most likely sequence of tags.

---

## 📂 Repository Structure

* `ngram_model.py` - Core logic for calculating N-gram probabilities and generating text.
* `smoothing.py` - Implementation of various smoothing algorithms.
* `pos_tagger.py` - HMM-based tagger using the Viterbi algorithm.
* `eval_metrics.py` - Tools for calculating **Perplexity** to evaluate the language model.

---

## 🚀 How to Run

Clone the repository and run the scripts to see the language model in action:

```bash
# 1. Clone the repository
git clone [https://github.com/AhmadEgbaria1/NLP-homework2.git](https://github.com/AhmadEgbaria1/NLP-homework2.git)
cd NLP-homework2

# 2. Run the models
# This will train the N-gram model and evaluate it using Perplexity
python ngram_model.py

# To run the POS Tagger:
python pos_tagger.py
