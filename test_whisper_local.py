#!/usr/bin/env python3
"""Quick determinism test for local whisper-large-v3 inference.

Runs each cached WAV through transcription twice and asserts outputs are identical.
"""

import os
import sys
from pathlib import Path

# Must be set before any CUDA context
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline as hf_pipeline

torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

WAV_FILES = list(Path("video_cache").glob("*.wav"))
MODEL_ID = "openai/whisper-large-v3"
DEVICE = "cuda:0"
DTYPE = torch.float16

def load_model():
    print(f"Loading {MODEL_ID} onto {DEVICE} ...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        dtype=DTYPE,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        attn_implementation="eager",
    )
    model.to(DEVICE)
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    pipe = hf_pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=DTYPE,
        device=DEVICE,
    )
    print("Model loaded.\n")
    return pipe

def transcribe(pipe, wav_path: Path) -> str:
    with torch.no_grad():
        result = pipe(
            str(wav_path),
            generate_kwargs={
                "task": "transcribe",
                "language": "english",
                "do_sample": False,
                "num_beams": 1,
            },
            return_timestamps=False,
        )
    return result["text"].strip()

def main():
    pipe = load_model()
    all_passed = True

    for wav in sorted(WAV_FILES):
        print(f"--- {wav.name} ---")
        t1 = transcribe(pipe, wav)
        t2 = transcribe(pipe, wav)
        match = (t1 == t2)
        status = "PASS (deterministic)" if match else "FAIL (non-deterministic!)"
        print(f"Run 1: {t1[:120]}{'...' if len(t1) > 120 else ''}")
        print(f"Run 2: {t2[:120]}{'...' if len(t2) > 120 else ''}")
        print(f"Result: {status}\n")
        if not match:
            all_passed = False

    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
