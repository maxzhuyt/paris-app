## Model Configuration and Determinism

The PARIS application implements two report-generation pipelines. To promote reproducibility, all models were configured with low-randomness settings, though none of these API-based models guarantee fully deterministic outputs.

**Pipeline 1 (Transcript-based)** works in two stages. Audio is first extracted from the bodycam video and transcribed using OpenAI's Whisper-1 speech recognition model with its temperature set to 0, which biases the model toward selecting the most likely token at each step. The transcript is then passed to DeepSeek-Reasoner (R1), a reasoning-focused language model, also with temperature set to 0. In both API calls, top_p is fixed at 1.0 (no nucleus sampling truncation), so that temperature alone controls the degree of randomness.

**Pipeline 2 (Video-based)** skips transcription entirely. The full video is uploaded to Google's Gemini 2.5 Pro, which processes both the visual and audio content natively. Temperature is set to 0 and top_p is fixed at 1.0, consistent with Pipeline 1.

Both pipelines use the same system prompt, which instructs the model to write a police incident report in third person using professional language, include only facts supported by the input, and insert bracketed placeholders (e.g., `[INSERT: missing detail]`) rather than guessing when information is missing.

**Limitations on determinism.** Setting temperature to 0 does not make these models deterministic. Cloud-hosted large language models are served across distributed hardware, and identical inputs can produce different outputs across runs due to factors such as floating-point arithmetic differences, GPU non-determinism, request batching, and silent model version updates by the provider. These are inherent properties of current commercial AI APIs and cannot be fully controlled by the researcher. As a result, repeated runs on the same input may yield stylistically or substantively different reports, and our results should be interpreted with this variability in mind.
