# Enterprise Incident Triage (Hybrid RAG + QLoRA)

I built this project to explore how to turn messy, unstructured server logs into clean, predictable JSON reports that automated systems can actually read. 

When an outage happens, standard LLMs tend to be overly conversational. They give general advice instead of machine-readable data. In this project, I combined hybrid search (to retrieve the right company runbook policies) with 4-bit fine-tuning (to force the model to output strict JSON).

---

## What I Built

The pipeline has two main stages:

1. **Retrieval (Hybrid RAG):** 
   - I used `ChromaDB` with `all-MiniLM-L6-v2` for semantic search, and paired it with `BM25Okapi` to catch exact error codes (like `AUTH_INVALID_CREDENTIALS`) that vector search often misses.
   - The results are combined using Reciprocal Rank Fusion (RRF) and reranked using a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) to surface the exact runbook section needed.

2. **Fine-Tuning (QLoRA):**
   - I took `Qwen/Qwen2.5-0.5B-Instruct` and quantized it to 4-bit using `BitsAndBytes` so it could train easily on a single free T4 GPU in Google Colab.
   - Using LoRA (rank 16, alpha 32), I only trained 2.16M parameters (about 0.44% of the total model).
   - This taught the model to stop chatting and strictly generate a 4-key JSON schema (`incident_type`, `severity`, `root_cause`, and `remediation`).

---

## What I Learned & Measured

- **Dense search alone wasn't enough:** When testing exact technical keywords, pure vector search struggled. Adding BM25 brought the retrieval hit rate from 60% up to 80% and fixed the keyword mismatch issue.
- **Base model vs Fine-tuned:** Before fine-tuning, the base model failed schema compliance completely (0%). After 12 epochs of training (loss dropped from 3.00 to 0.96), the fine-tuned adapter achieved 100% valid JSON adherence on unseen logs.
- **Parameter efficiency:** Achieving strict formatting didn't require fine-tuning the whole model or using a massive LLM—a 0.5B parameter model with a small adapter handled the task well.

---

## Example Run

**Incoming Error Log:**
CRITICAL 2026-08-28 17:15:00.320 [envoy-proxy-worker-2] CircuitBreaker: Target service 'inventory-catalog' error rate spiked to 64% over rolling window. State changed to OPEN.

**Pipeline Output:**
{
  "incident_type": "CIRCUIT_BREAKER",
  "severity": "CRITICAL",
  "root_cause": "Target service 'inventory-catalog' error rate exceeded 64% over rolling window, triggering circuit breaker policy.",
  "remediation": "Circuit breaker triggered, initiating recovery path."
}

---

## Project Layout

- `notebooks/01_hybrid_rag_evaluation.ipynb` — Document chunking, building the BM25 and ChromaDB indexes, and evaluating retrieval metrics.
- `notebooks/02_peft_lora_finetuning.ipynb` — Formatting training data, 4-bit QLoRA training loop, JSON validation, and the final end-to-end pipeline.
- `artifacts/incident_triage_lora_adapter/` — The saved adapter weights and tokenizer files.

## Tools Used
Python, PyTorch, Hugging Face (Transformers, PEFT, TRL), ChromaDB, Rank-BM25, Sentence-Transformers, BitsAndBytes.
