---
title: '1.4 The Renormalization Group: From Microscopic Spins to Macroscopic Features'
slug: kpp-theoretical-foundations-stat-renormalization
sidebar_position: 3
hide: false
hide_child: false
---


# 1.4 The Renormalization Group: From Microscopic Spins to Macroscopic Features

> So far we have seen how the Ising model describes a system of interacting spins and how the Boltzmann distribution governs its equilibrium behaviour. But what happens when we **zoom out, i.e.** when we stop looking at every individual spin and instead focus on larger, coarse-grained blocks? This question is answered by the **Renormalization Group (RG)**, a powerful framework from statistical physics that explains why the laws of nature appear simple at different scales and why seemingly different systems can behave identically near a critical point. Remarkably, the RG also provides a deep physical analogy for the hierarchical feature learning performed by deep neural networks.

## The Core Idea: Coarse-Graining
Imagine a two-dimensional Ising model on a square lattice. Instead of observing every spin, we partition the lattice into blocks of size $2 \times 2$, and replace each block with a single effective spin. This is called **coarse-graining**. The new effective spins interact with each other according to new, renormalized couplings $J'$ and fields $h'$. The key insight is that the rules of the universe – the Hamiltonian – change when you change your observational scale.

The process of coarse-graining is repeated iteratively:

- Start with spins at the finest scale.
- Apply a coarse-graining transformation (e.g., block averaging or majority rule).
- Obtain a new, coarser Hamiltonian with new parameters.
- Repeat.

This defines a **flow** in the space of Hamiltonians.

## Fixed Points and Universality
As the RG flow is iterated, the parameters $(J, h, \ldots)$ move through the parameter space. The system may eventually reach a **fixed point**– a point where further coarse-graining leaves the Hamiltonian unchanged. Fixed points correspond to **critical points** of the system, where the correlation length diverges and the system becomes scale-invariant.

The profound consequence is **universality**: many different microscopic systems (different metals, different liquids) flow to the same fixed point. This is why we can compute properties of a liquid without knowing every detail of the molecules – the relevant physics at the macroscopic scale is determined by the fixed point, not by the microscopic details.

## The Renormalization Group and Deep Learning
The RG provides a striking physical analogy for the **hierarchical feature learning** performed by stacked RBMs (Deep Belief Networks, see Section [4.3 Beyond Single Layers: Stacking for Deep Learning](kpp-theoretical-foundations-bm-deep.md)).

| Renormalization Group | Deep Belief Network (Stacked RBM) |
| --- | --- |
| Coarse-grain spin blocks | Pooling / abstracting features |
| New effective couplings | Higher-layer weights |
| Fixed point (scale invariance) | Top-level, most abstract representation |
| Universality (same fixed point from many microstates) | Generalisation: same high-level features emerge from many training examples |

In a DBN, each new hidden layer can be seen as a **coarse-graining operation** on the distribution of features from the layer below. The first hidden layer captures local edges and corners; the second layer combines these into mid-level shapes; the third layer forms high-level object representations. This process is directly analogous to an RG flow, where the “microscopic” data distribution is progressively simplified into a “macroscopic” abstract representation.

Just as the RG reveals that all critical points belong to a finite number of universality classes, deep learning shows that a well-trained network automatically discovers the universal structure of the data – the features that generalise across examples.

## Statistical Physics Foundations: Extensive and Intensive Variables
To fully appreciate the RG analogy, we must distinguish between **extensive** and **intensive** variables in thermodynamics:

- **Extensive variables** (e.g., volume $V$, energy $E$, entropy $S$, number of particles $N$) scale with the size of the system.
- **Intensive variables** (e.g., temperature $T$, pressure $p$, chemical potential $\mu$) are independent of system size.

In the RG flow, the coarse-graining operation reduces the effective number of degrees of freedom—the "system size" decreases. Intensive variables like temperature and pressure remain invariant under scaling, while extensive variables like energy and entropy scale linearly with the number of effective spins.

This distinction is mirrored in deep learning: high-level representations are **intensive**—they capture invariant features of the data (e.g., "catness") that are independent of the input's spatial resolution or pixel count—while lower-level features are **extensive**, requiring more units to represent fine-grained details.

## The Connection to the Sampling Bottleneck
The RG also offers a physical perspective on the sampling problem introduced in later sections. Near a critical point, the correlation length diverges, and the system becomes difficult to simulate – Markov chains mix extremely slowly (critical slowing down). This is precisely the difficulty encountered when sampling from Boltzmann machines that are near their “critical” state, i.e., when the model has learned a rich, multimodal distribution. Understanding this slowdown is central to appreciating why alternative approaches – such as quantum sampling – are needed.

## Summary
- **Coarse-graining** transforms a system at one scale to an equivalent system at a larger scale.
- **Renormalization Group** is the study of how the Hamiltonian changes under coarse-graining, leading to **fixed points** and **universality**.
- The RG provides a deep physical analogy for **hierarchical feature learning** in deep neural networks: each layer coarse-grains the features from the layer below.
- Critical slowing down in physical systems mirrors the **sampling bottleneck** in energy-based models, motivating the search for more efficient sampling methods.
