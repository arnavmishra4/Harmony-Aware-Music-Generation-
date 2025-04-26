# Advanced Harmony-Aware Music Generation Model

## Overview

This project implements an advanced music generation model capable of creating complex and coherent musical pieces. It utilizes a two-stage process involving a main Transformer model equipped with Harmonic Relative Positional Encoding (H-RPE) and Rotary Positional Embeddings (RoPE) to generate a musical backbone, followed by a TransformerXL-based Melody Model, also using RoPE, to generate intricate melodies conditioned on the backbone.

## Table of Contents

1.  [Dataset](#dataset)
2.  [Model Architecture](#model-architecture)
    * [Harmonic Relative Positional Encoding (H-RPE)](#harmonic-relative-positional-encoding-h-rpe)
    * [Main Transformer Model](#main-transformer-model)
        * [Rotary Positional Embedding (RoPE) in Main Model](#rotary-positional-embedding-rope-in-main-model)
        * [Output Format (Main Model)](#output-format-main-model)
    * [Melody Model](#melody-model)
        * [Input & Training Data (Melody Model)](#input--training-data-melody-model)
        * [TransformerXL Architecture](#transformerxl-architecture)
        * [Rotary Positional Embedding (RoPE) in Melody Model](#rotary-positional-embedding-rope-in-melody-model)
        * [Final Output Format](#final-output-format)
3.  [Evaluation Metrics](#evaluation-metrics)
4.  [Setup & Usage](#setup--usage) (Optional: Add if needed)
5.  [Results](#results)

---

## Dataset

Our model is trained on a dataset derived from `[Specify source, e.g., Lakh MIDI Dataset, specific curated collection of MIDI files, symbolic music notation database]`.

**Preprocessing:**
The raw data `[e.g., MIDI files]` is preprocessed into a structured format suitable for transformer models. This involves:
* `[Describe step 1, e.g., Quantizing note timings to a specific grid]`
* `[Describe step 2, e.g., Extracting chord progressions, tempo changes, time signatures]`
* `[Describe step 3, e.g., Tokenizing musical events like note onset, pitch, duration, velocity, rests, chords, tempo markers]`
* `[Describe step 4, e.g., Separating melody tracks from accompaniment/chords for the two-stage training]`
* `[Any other relevant preprocessing steps]`

**Dataset Structure:**
The processed data used for the **Main Transformer Model** consists of sequences representing the musical backbone. Each element in the sequence might correspond to a time step or an event, and the columns/features include:

* `column_name_1`: `[Description, e.g., Token ID for chord symbol]`
* `column_name_2`: `[Description, e.g., Token ID for rhythmic pattern]`
* `column_name_3`: `[Description, e.g., Tempo information token]`
* `column_name_4`: `[Description, e.g., Time signature token]`
* `column_name_5`: `[Description, e.g., Bar/Beat position marker]`
* `[Add other relevant columns/features]`

*(See the [Melody Model](#input--training-data-melody-model) section for its specific dataset structure).*

---

## Model Architecture

Our music generation process involves two main stages, leveraging advanced Transformer architectures and specialized positional encodings.

### Harmonic Relative Positional Encoding (H-RPE)

**Purpose:**
Standard positional encodings in Transformers capture sequential order but often fail to represent the crucial *harmonic* relationships between musical elements (like notes or chords) that are fundamental to music theory and perception. H-RPE is designed to explicitly encode these harmonic relationships, informing the model about concepts like consonance, dissonance, chord functions, and key contexts relative to other elements in the sequence.

**How it Works:**
H-RPE calculates positional information based not just on sequential distance but also on the harmonic distance or relationship between musical events.
* `[Explain the core mechanism: e.g., Does it use tonal pitch class distances? Interval analysis? Chord progression rules? Key signature context?]`
* `[Describe how the harmonic context is determined: e.g., Is it based on a look-back window? Pre-computed harmonic analysis of the sequence? Combination with absolute position?]`
* `[Explain how this harmonic information is mathematically encoded and integrated into the model: e.g., Added to input embeddings? Incorporated into the attention mechanism like RoPE?]`
* `[Mention any specific theoretical basis, e.g., Based on spiral array model, Tonnetz, etc., if applicable]`

H-RPE allows the attention mechanism to prioritize or weight interactions between harmonically related elements, leading to more musically coherent and structured outputs from the main transformer. It is primarily used within the [Main Transformer Model](#main-transformer-model).

### Main Transformer Model

**Purpose:**
This model forms the first stage of our generation process. Its goal is to learn the high-level structure and harmonic/rhythmic content of the music. It generates a representation of the musical backbone (e.g., chord progressions, rhythmic patterns, structural markers).

**Architecture:**
We use a standard Transformer architecture consisting of:
* Input Embedding Layer: Converts input tokens into dense vectors.
* Positional Encoding: Incorporates both standard sequential positional information and our custom [H-RPE](#harmonic-relative-positional-encoding-h-rpe).
* Multi-Layer Transformer Encoder/Decoder: Comprises multiple blocks, each containing:
    * Multi-Head Self-Attention (MHSA): Allows the model to weigh the importance of different tokens in the input sequence. This is where [RoPE](#rotary-positional-embedding-rope-in-main-model) is applied.
    * Feed-Forward Neural Network (FFN): Applied independently to each position.
    * Layer Normalization and Residual Connections.
* Output Layer: Maps the transformer's output back to token probabilities (e.g., predicting the next chord or rhythmic token).

**Input:** Preprocessed sequences representing the musical backbone, as described in the [Dataset](#dataset) section, combined with H-RPE and RoPE.

#### Rotary Positional Embedding (RoPE) in Main Model

**Purpose:**
RoPE is employed within the self-attention mechanism of the main Transformer model to encode relative positional information more effectively than absolute or learned positional embeddings, especially for sequential data like music.

**How it Works:**
Instead of adding positional information to the embeddings, RoPE *rotates* the query and key vectors in the self-attention mechanism based on their absolute positions. The key insight is that the dot product attention between two vectors (queries and keys) then naturally depends only on their *relative* positions and the content of the vectors themselves.
* It applies position-dependent rotations to the query ($q$) and key ($k$) vectors using rotation matrices derived from sinusoidal functions of the position indices ($m$ and $n$).
* The formulation ensures that the inner product $\langle \text{Rot}(q, m), \text{Rot}(k, n) \rangle$ is a function of the embeddings and the relative distance ($m-n$).
* This provides the attention mechanism with built-in relative position awareness without modifying the input embeddings directly.

#### Output Format (Main Model)

The output of the main Transformer model is a sequence of tokens representing the generated musical backbone.

**Example Structure:**
