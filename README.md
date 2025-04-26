# Advanced Harmony-Aware Music Generation Model

## Table of Contents

1.  [Dataset](#dataset)
2.  [Model Architecture](#model-architecture)
    * [Harmonic Relative Positional Encoding (H-RPE)](#harmonic-relative-positional-encoding-h-rpe)
    * [Main Transformer Model (Harmony/Backbone)](#main-transformer-model-harmonybackbone)
        * [Rotary Positional Embedding (RoPE) in Main Model](#rotary-positional-embedding-rope-in-main-model)
        * [Output Format (Main Model)](#output-format-main-model)
    * [Melody Model](#melody-model)
        * [Input & Training Data (Melody Model)](#input--training-data-melody-model)
        * [TransformerXL Architecture](#transformerxl-architecture)
        * [Rotary Positional Embedding (RoPE) in Melody Model](#rotary-positional-embedding-rope-in-melody-model)
        * [Final Output Format (Melody)](#final-output-format-melody)
3.  [Evaluation Metrics](#evaluation-metrics)
4.  [Setup & Usage](#setup--usage)
5.  [Results](#results)

---

## Dataset

Our model is trained on a dataset derived from `[Specify source, e.g., Lakh MIDI Dataset, specific curated collection of MIDI files, symbolic music notation database]`.

**Preprocessing:**
The raw data `[e.g., MIDI files]` is preprocessed into a structured format suitable for transformer models. This involves:
* `[Describe step 1, e.g., Quantizing note timings to a specific grid]`
* `[Describe step 2, e.g., Extracting chord progressions, tempo changes, time signatures]`
* `[Describe step 3, e.g., Tokenizing musical events like note onset, pitch, duration, velocity, rests, chords, tempo markers into numerical IDs]`
* `[Describe step 4, e.g., Separating melody tracks from accompaniment/chords for the two-stage training]`
* `[Describe step 5, e.g., Computing harmonic features like root pitch class (root_pc), chord quality (quality_code), and chord function (function_code) relative to a key or local context]`
* `[Any other relevant preprocessing steps]`

**Dataset Structure (Main Model - Harmony/Backbone):**
The processed data used for the **Main Transformer Model** consists of sequences representing the musical backbone (e.g., chord progressions). Each data point typically contains an input sequence and the target element to predict. Features include:

* `input_ids`: Sequence of token IDs representing the preceding chord events.
* `target_id`: The token ID of the next chord event to be predicted.
* `root_pc`: Sequence of root pitch classes (0-11) corresponding to the `input_ids`.
* `quality_code`: Sequence of numerical codes representing chord quality (e.g., major, minor, dominant 7th) corresponding to the `input_ids`.
* `function_code`: Sequence of numerical codes representing the harmonic function (e.g., Tonic, Dominant, Subdominant) corresponding to the `input_ids`.

**Example (Main Model Data Point):**
```json
{
  "input_ids": [30219, 23481, 30219, 38716, 33697, 38716, 30219, 32211, 30219, 23481, 30219, 38716, 33697, 38716, 30219, 38716, 32211, 38716, 33697, 38716, 36453, 38716, 33697, 38716, 32211, 38716, 30219, 30219, 23481, 40340, 38716, 30219],
  "target_id": 23481,
  "root_pc": [4, 3, 4, 11, 8, 11, 4, 6, 4, 3, 4, 11, 8, 11, 4, 11, 6, 11, 8, 11, 9, 11, 8, 11, 6, 11, 4, 4, 3, 1, 11, 4],
  "quality_code": [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
  "function_code": [0, 0, 0, 9, 4, 9, 0, 3, 0, 0, 0, 9, 4, 9, 0, 9, 3, 9, 4, 9, 2, 9, 4, 9, 3, 9, 0, 0, 0, 5, 9, 0]
}
```
(See the Melody Model section for its specific dataset structure).

Model Architecture
Our music generation process involves two main stages, leveraging advanced Transformer architectures and specialized positional encodings.

Harmonic Relative Positional Encoding (H-RPE)
Purpose:
Standard positional encodings in Transformers capture sequential order but often fail to represent the crucial harmonic relationships between musical elements (like notes or chords) that are fundamental to music theory and perception. H-RPE is designed to explicitly encode these harmonic relationships, informing the model about concepts like consonance, dissonance, chord functions, and key contexts relative to other elements in the sequence.

How it Works:
H-RPE calculates positional information based not just on sequential distance but also on the harmonic distance or relationship between musical events.

[Explain the core mechanism: e.g., How are the root_pc, quality_code, function_code used? Does it calculate harmonic distance using interval cycles, tonal pitch space models (like Tonnetz or spiral array), or learned harmonic embeddings?]
[Describe how the harmonic context is determined: e.g., Is it based on comparing the current element to all previous elements within the attention window? Does it involve pre-computed key analysis?]
[Explain how this harmonic information is mathematically encoded and integrated into the model: e.g., Is it combined with RoPE? Is it a separate embedding added to the input? Does it modify the attention scores directly?]
[Mention any specific theoretical basis, e.g., Based on music theory principles of voice leading, harmonic function, etc., if applicable]
H-RPE allows the attention mechanism to prioritize or weight interactions between harmonically related elements, leading to more musically coherent and structured outputs from the main transformer. It is primarily used within the Main Transformer Model.

Main Transformer Model (Harmony/Backbone)
Purpose:
This model forms the first stage of our generation process. Its goal is to learn the high-level structure and harmonic/rhythmic content of the music. It generates a representation of the musical backbone (e.g., chord progressions with associated harmonic features).

Architecture:
We use a standard Transformer architecture consisting of:

Input Embedding Layer: Converts input chord token IDs (input_ids) into dense vectors. Additional embeddings might be used for root_pc, quality_code, function_code.
Positional Encoding: Incorporates positional information using RoPE and potentially harmonic relational information via H-RPE.
Multi-Layer Transformer Encoder/Decoder: [Specify if Encoder, Decoder, or Encoder-Decoder] blocks, each containing:
Multi-Head Self-Attention (MHSA): Allows the model to weigh the importance of different tokens in the input sequence. RoPE is applied here. H-RPE might influence attention scores or embeddings.
Feed-Forward Neural Network (FFN): Applied independently to each position.
Layer Normalization and Residual Connections.
Output Layer: Maps the transformer's output representation to probabilities over the vocabulary of chord tokens to predict the target_id.
Input: Data points containing sequences of chord input_ids and associated harmonic features (root_pc, quality_code, function_code), as described in the Dataset section.

Rotary Positional Embedding (RoPE) in Main Model
Purpose:
RoPE is employed within the self-attention mechanism of the main Transformer model to encode relative positional information more effectively than absolute or learned positional embeddings, especially for sequential data like music.

How it Works:
Instead of adding positional information to the embeddings, RoPE rotates the query and key vectors in the self-attention mechanism based on their absolute positions. The key insight is that the dot product attention between two vectors (queries and keys) then naturally depends only on their relative positions and the content of the vectors themselves.

It applies position-dependent rotations to the query ($q$) and key ($k$) vectors using rotation matrices derived from sinusoidal functions of the position indices ($m$ and $n$).
The formulation ensures that the inner product $\langle \text{Rot}(q, m), \text{Rot}(k, n) \rangle$ is a function of the embeddings and the relative distance ($m-n$).
This provides the attention mechanism with built-in relative position awareness without modifying the input embeddings directly.
Output Format (Main Model)
The output of the main Transformer model during inference is a generated sequence of backbone elements, including predicted chord token IDs and potentially their associated harmonic features.

Example Generated Sequence (Conceptual):

JSON

[
  {"predicted_id": 30219, "gen_root_pc": 4, "gen_quality": 8, "gen_function": 0},
  {"predicted_id": 23481, "gen_root_pc": 3, "gen_quality": 8, "gen_function": 0},
  {"predicted_id": 30219, "gen_root_pc": 4, "gen_quality": 8, "gen_function": 0},
  {"predicted_id": 38716, "gen_root_pc": 11, "gen_quality": 8, "gen_function": 9},
  // ... and so on
]
(Note: Adjust this example based on what your model actually outputs during generation. It might just be the predicted_id sequence, which is then used by the melody model).

Melody Model
Purpose:
This second-stage model takes the generated musical backbone (chord progression) from the main model as conditioning information and generates a coherent and expressive melody line that fits the underlying harmonic structure.

Input & Training Data (Melody Model)
Input:
The input to the Melody Model consists of the target melody sequence (event_ids) and conditioning information derived from the musical backbone (either from the dataset during training or generated by the Main Model during inference).

metadata: Contains global information like key, time signature, and tempo.
conditioning_*: Sequences representing the chord backbone (token IDs, root pitch class, quality, function) aligned with the melody events. These provide the harmonic context.
event_ids: The sequence of melody event tokens (e.g., note onsets, pitches, durations, rests) that the model learns to predict.
Training Dataset Structure:
The Melody Model is trained on data points pairing a melody sequence with its corresponding harmonic context and metadata.

Example (Melody Model Data Point):

JSON

{
  "metadata": {
    "key_signature_root": 4,
    "key_signature_mode": "major",
    "time_signature_num": 3,
    "time_signature_den": 4,
    "tempo_bpm": 120
  },
  "conditioning_chord_ids": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  "conditioning_root_pc": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  "conditioning_quality_code": [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
  "conditioning_function_code": [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
  "event_ids": [210, 226, 154, 178, 66, 190, 154, 178, 66, 190, 154, 178, 66, 190, 154, 178, 66, 210, 210, 210, 210, 253, 147, 178, 59, 190, 147, 178, 59, 190, 147, 178, 59, 190, 147, 178, 59]
}
TransformerXL Architecture
Purpose:
Generating long, coherent melodies requires capturing dependencies over extended time spans. Standard Transformers have a fixed context window, limiting their ability to model long-range dependencies. TransformerXL is used for the Melody Model to overcome this limitation.

How it Works:
TransformerXL introduces two key concepts:

Segment-Level Recurrence: The input sequence (melody event_ids and conditioning info) is processed in segments (chunks). The hidden states computed for a previous segment are cached and reused as context when processing the current segment. This creates a recurrence relation between segments, allowing information to flow across segment boundaries and effectively extending the context length far beyond the fixed window size.
Relative Positional Encoding Scheme: Since the same absolute positional encoding would be reused across different segments (causing ambiguity), TransformerXL employs a more sophisticated relative positional encoding scheme. This ensures that the positional information remains consistent and meaningful even when processing different segments with reused hidden states. (Note: This relative PE scheme works alongside or is potentially augmented/replaced by RoPE in our specific implementation).
By using TransformerXL, the Melody Model can maintain musical coherence and reference thematic material or harmonic context from much earlier parts of the generated piece.

Rotary Positional Embedding (RoPE) in Melody Model
Purpose:
Similar to its use in the main model, RoPE is employed within the TransformerXL's attention mechanism in the Melody Model to effectively encode relative positions within the processing segments.

How it Works:
RoPE provides relative positional information within each segment being processed by the TransformerXL. Its ability to encode relative positions naturally complements TransformerXL's segment-level recurrence and relative attention mechanism.

It rotates the query and key vectors based on their positions within the current segment and its cached context.
This helps the model understand the relative timing and placement of melody notes (event_ids) in relation to each other and to the conditioning_* information, enabling better melodic phrasing and harmonic alignment even across long sequences handled by the recurrence mechanism.
Final Output Format (Melody)
The final output of the Melody Model during inference is a sequence of generated melody event_ids.

Example Generated Sequence (Conceptual):

JSON

{
  "generated_event_ids": [210, 226, 154, 178, 66, 190, 154, 178, 66, 190, 154, 178, 66, 190, 147, 178, 59, 210, 210, 253, 147, 178, 59, ...]
}
(Note: This sequence of IDs can then be decoded back into musical events like notes and rests and potentially synthesized into MIDI or audio).

Evaluation Metrics
The performance of the models is evaluated using the following metrics:

SSMD (Statistical Self-Similarity Matrix Distance): [Briefly explain what it measures in your context, e.g., Captures structural similarity and repetition patterns compared to real music. Lower is generally better.]
GS (Groove Similarity): [Briefly explain what it measures in your context, e.g., Assesses rhythmic accuracy and microtiming ('groove') compared to a reference dataset, often using velocity and timing information.]
Training Accuracy: [Briefly explain what it measures, e.g., Next-token prediction accuracy on the training set for the main/melody model. Specify if top-1 or other.]
Validation Accuracy: [Briefly explain what it measures, e.g., Next-token prediction accuracy on a held-out validation set for the main/melody model. Specify if top-1 or other.]
[Add other relevant metrics, e.g., Perplexity, Loss, BLEU score if applicable, specific music theory metrics]
Setup & Usage
Bash

# Clone the repository
# git clone [Your Repo URL]
# cd [Your Repo Directory]

# Install dependencies
# pip install -r requirements.txt

# --- Training ---
# (Add commands or instructions for training the main model)
# python train_main_model.py --data [path] --config [path] ...

# (Add commands or instructions for training the melody model)
# python train_melody_model.py --data [path] --config [path] ...

# --- Generation ---
# (Add commands or instructions for running inference)
# python generate.py \
#   --main_model_checkpoint [path] \
#   --melody_model_checkpoint [path] \
#   --output_path [output_directory] \
#   --num_pieces [number] \
#   [Other generation parameters like temperature, top-k, etc.]

# (Add instructions on how to interpret or use the output files)

(Note: Fill this section with actual commands and instructions specific to your project setup.)

Results
SSMD Score: [Insert SSMD Score]
GS Score: [Insert GS Score]
Training Accuracy (Main Model): [Insert Training Accuracy / Final Loss]
Validation Accuracy (Main Model): [Insert Validation Accuracy / Final Loss]
Training Accuracy (Melody Model): [Insert Training Accuracy / Final Loss]
Validation Accuracy (Melody Model): [Insert Validation Accuracy / Final Loss]
[Add results for any other metrics]
