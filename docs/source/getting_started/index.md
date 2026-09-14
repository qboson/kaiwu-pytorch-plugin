<!-- ```YAML
title: Kaiwu PyTorch Plugin Getting Started
slug: kpp-getting-started
sidebar_position: 0
layout: home
hide: false
keywords:
  - KPP
  - Guide
``` -->


# Getting Started

> **Kaiwu-PyTorch-Plugin (KPP)** is a PyTorch plugin for training and evaluating quantum-native energy models and enhanced AI models on **Special Purpose Quantum Computers (SPQC)**.
> 
> It offloads Boltzmann sampling to the **Coherent Ising Machine (CIM)** through the Kaiwu SDK, while keeping all other computations, parameter updates, autograd, data loading, in the standard PyTorch workflow.

This guide walks you through the essentials in four parts:

```{toctree}
:maxdepth: 1
:hidden:

background
introduction
installation
quickstart
```

## Prerequisite

Before proceeding, ensure you have basic knowledge of energy-based models and Boltzmann machines.  
Please read the [Prerequisites](background.md), it covers the essentials of RBM and energy functions.  
For a deeper dive into the statistical physics and neural network foundations, see the [Theoretical Foundations](../theoretical-foundations/index.md).

## Overview

Core concepts, design goals, features, and typical usage workflow.  
Read the [Overview](introduction.md) to learn:

- What is KPP? Quantum-Classical Hybrid Programming Suite
- System Architecture & Component Interaction
- Core Concepts: Hybrid Workflow, Energy Model Hierarchy, Hardware Abstraction
- Typical Usage Workflow

## Installation Guide

System requirements, local (conda/pip) and Docker setup, Kaiwu SDK configuration, and installation verification.  
See the [Installation Guide](installation.md) for:

- System Requirements
- Option 1: Local Setup
- Option 2: Docker Setup
- Kaiwu SDK Configuration & License
- Installation Verification

## Quick Start

RBM and BM training examples, sampler switching (SA / CIM), and next learning paths.  
Jump to the [Quick Start](quickstart.md) page to explore:

- RBM Training Walkthrough
- Code Mapping
- Switching Samplers: SA (classical) vs CIM (quantum)
- Next Learning Paths & Resources

---

# Tutorials

The [Tutorials](tutorials/index.md) section is designed for users who have already installed KPP and read the [Quick Start](quickstart.md) guide. It provides hands-on, end-to-end examples of KPP in action.

## Recommended Learning Paths

**Beginner Path**, from the sampling bottleneck to your first application:

1. Complete the [Quick Start](quickstart.md) to understand the basic API and training loop
2. Study [Why Quantum? Revisiting the Sampling Bottleneck](tutorials/quantum_sampling_bottleneck.md)
3. Explore [Simulated Annealing for Ising Models](tutorials/simulated_annealing.md) to establish the sampling baseline
4. Learn [Integrating Quantum Samplers into PyTorch](tutorials/quantum_sampling_pytorch.md) to integrate the CIM sampler
5. Apply [RBM Classification](tutorials/rbm_classification.md) for your first end-to-end application

**Advanced Path**, deeper architectures and generative models:

1. [DBN Classification](tutorials/dbn_classification.md): extend RBM into deep hierarchical features
2. [BM Generation](tutorials/bm_generation.md): generative modeling with a fully connected Boltzmann Machine
3. [Q-VAE (MNIST)](tutorials/qvae_mnist.md): quantum-enhanced variational autoencoder for image generation and representation learning

**Complete Path**: work through all tutorials in order for comprehensive mastery of KPP. See the [Tutorials index](tutorials/index.md) for the full syllabus, including upcoming tutorials (Q-VAE for single-cell transcriptomics, Q-Diffusion for protein sequence generation).
