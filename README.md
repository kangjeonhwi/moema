# Llama-MoE Upcycling Project

## Overview

This repository is a research and engineering project to **convert Llama models into Mixture-of-Experts (MoE) models** based on [Taishi-N324/Drop-Upcycling][1], leveraging the architecture and components of Hugging Face's `transformers` library, especially `modeling_llama` and `modeling_mixtral`.  
The goal is to enable Llama-based models to benefit from sparse MoE architectures, advanced routing strategies, and partial upcycling/re-initialization techniques.

## Motivation

- **Drop-Upcycling** ([ICLR'25], [Taishi-N324/Drop-Upcycling][1]) introduced a method for training sparse MoE models with partial re-initialization, improving efficiency and performance.
- This project adapts and extends those ideas to the Hugging Face transformers ecosystem, focusing on Llama and Mistral model implementations.

## Key Features

- **Llama → MoE Conversion:**  
  Turn dense Llama models into sparse MoE models by replacing the feedforward layers with expert blocks, inspired by Mixtral and Drop-Upcycling.
- **Integration with Hugging Face Transformers:**  
  Built on top of `modeling_llama` and `modeling_mistral` for maximum compatibility and flexibility.
- **Upcycling & Routing Methods:**  
  The codebase will be extended to support various routing algorithms and upcycling (partial re-initialization) strategies as described in the Drop-Upcycling paper.
- **Research-Friendly:**  
  Modular, extensible design for experimenting with new MoE routing, initialization, and training methods.

## Planned Extensions

- Additional routing strategies (beyond top-k, e.g., learned routing, noisy gating)
- More upcycling and re-initialization methods
- Evaluation scripts and benchmarks for MoE-converted Llama models

## Reference

This project is **heavily based on**:

- [Taishi-N324/Drop-Upcycling][1] ([ICLR'25] Drop-Upcycling: Training Sparse Mixture of Experts with Partial Re-initialization)
- Hugging Face `transformers` (especially `modeling_llama.py` and `modeling_mistral.py`)

Please see the original Drop-Upcycling [repository][1] and [paper](https://openreview.net/forum?id=gx1wHnf5Vp) for more details.

## License

This project follows the **Apache-2.0 License** as in the original [Drop-Upcycling][1] repository.
