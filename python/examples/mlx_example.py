#!/usr/bin/env python3
"""
Example: KV cache compaction with mlx-lm on Apple Silicon.

Demonstrates:
  1. Loading a model with mlx-lm
  2. Prefilling a long context
  3. Compacting the KV cache (keeping 50% of tokens)
  4. Generating with the compacted cache
  5. Comparing quality vs full-cache generation

Requirements:
  pip install mlx mlx-lm numpy
  cd python && pip install -e .

Usage:
  python examples/mlx_example.py --model mlx-community/Qwen3-8B-4bit
  python examples/mlx_example.py --model mlx-community/Llama-3.2-3B-Instruct-4bit --ratio 0.3

  # With speculative decoding (compact + fast generation):
  python examples/mlx_example.py --model mlx-community/Qwen3-8B-4bit \
    --draft mlx-community/Qwen3-0.6B-4bit --ratio 0.5
"""

import argparse
import sys
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="KV cache compaction with MLX")
    parser.add_argument("--model", "-m", default="mlx-community/Qwen3-8B-4bit",
                        help="MLX model name or path")
    parser.add_argument("--ratio", "-r", type=float, default=0.5,
                        help="Target keep ratio (0.0-1.0)")
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="Max tokens to generate")
    parser.add_argument("--prompt", "-p", default=None,
                        help="Custom prompt (default: built-in long context)")
    parser.add_argument("--draft", "-d", default=None,
                        help="Draft model for speculative decoding (e.g. mlx-community/Qwen3-0.6B-4bit)")
    parser.add_argument("--num-draft", type=int, default=3,
                        help="Draft tokens per speculative step (default: 3)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    try:
        import mlx.core as mx
        from mlx_lm import load, generate
        from mlx_lm.models.cache import make_prompt_cache
    except ImportError:
        print("Error: mlx-lm not installed. Run: pip install mlx mlx-lm")
        sys.exit(1)

    from kv_compact.mlx import compact_cache

    # Load model
    print(f"Loading {args.model}...")
    model, tokenizer = load(args.model)

    # Build prompt
    if args.prompt:
        prompt = args.prompt
    else:
        # A moderately long context to demonstrate compaction
        prompt = (
            "You are a helpful assistant. Here is a research paper summary:\n\n"
            "Title: Fast KV Compaction via Attention Matching\n\n"
            "Abstract: We present a method for compressing the key-value (KV) cache "
            "in transformer language models by up to 50x with minimal quality loss. "
            "Our approach selects the most important cached positions based on "
            "attention patterns and refits the value vectors using least-squares "
            "optimization. Unlike eviction-based methods that simply drop tokens, "
            "our compaction preserves the information content of removed positions "
            "by redistributing it across the remaining cache entries. "
            "The method requires no model retraining and can be applied to any "
            "pre-trained transformer at inference time. We demonstrate that "
            "compacted caches maintain >99.9% cosine similarity with full-cache "
            "attention outputs across a range of models and tasks.\n\n"
            "Section 3.1 - Key Selection: We score each cached position by its "
            "maximum attention weight across a set of reference queries. Positions "
            "with higher attention scores are more important and are kept in the "
            "compacted cache. The reference queries can be the actual recent queries "
            "or, more efficiently, a proxy derived from the key vectors themselves.\n\n"
            "Section 3.2 - Mass Matching (Beta): An optional NNLS solve that "
            "computes per-key importance weights to preserve the attention "
            "distribution's partition function. In practice, this step can be "
            "skipped as the value refitting alone achieves equal quality.\n\n"
            "Section 3.3 - Value Refitting: The core quality-preserving step. "
            "We solve a least-squares problem to find new value vectors for the "
            "kept positions that minimize the reconstruction error of the full "
            "attention output. This is what distinguishes compaction from simple "
            "eviction.\n\n"
            "Based on this paper, what are the key advantages of KV cache "
            "compaction over simple token eviction?"
        )

    tokens = tokenizer.encode(prompt)
    print(f"Prompt: {len(tokens)} tokens")

    # === Generate with full cache (baseline) ===
    print("\n--- Full cache generation ---")
    t0 = time.perf_counter()
    full_output = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens)
    full_time = time.perf_counter() - t0
    print(f"Time: {full_time:.1f}s")
    print(f"Output: {full_output[:200]}...")

    # === Generate with compacted cache ===
    print(f"\n--- Compacted cache generation (keep {args.ratio:.0%}) ---")

    # Create cache and prefill
    cache = make_prompt_cache(model)
    input_ids = mx.array(tokenizer.encode(prompt))[None]
    model(input_ids, cache=cache)
    mx.eval([c.state for c in cache if hasattr(c, "state")])

    # Compact
    stats = compact_cache(
        model, cache,
        target_ratio=args.ratio,
        verbose=args.verbose,
    )
    print(
        f"Compacted: {stats.original_len} -> {stats.compacted_len} tokens "
        f"({stats.n_layers_compacted} layers in {stats.elapsed_ms:.0f}ms)"
    )
    print(f"Avg cosine similarity: {stats.avg_cosine_sim:.4f}")
    print(f"Avg MSE: {stats.avg_mse:.6f}")

    # Generate from compacted cache
    t0 = time.perf_counter()
    compact_output = generate(
        model, tokenizer,
        prompt=prompt,
        max_tokens=args.max_tokens,
        cache=cache,
    )
    compact_time = time.perf_counter() - t0
    print(f"Time: {compact_time:.1f}s")
    print(f"Output: {compact_output[:200]}...")

    # === Compare ===
    print("\n--- Comparison ---")
    print(f"Cache: {stats.original_len} -> {stats.compacted_len} tokens "
          f"({stats.compacted_len/stats.original_len:.1%} of original)")
    print(f"Cosine sim: {stats.avg_cosine_sim:.4f}")

    # Token overlap between outputs
    full_tokens = tokenizer.encode(full_output)
    compact_tokens = tokenizer.encode(compact_output)
    min_len = min(len(full_tokens), len(compact_tokens))
    if min_len > 0:
        matches = sum(1 for a, b in zip(full_tokens[:min_len], compact_tokens[:min_len]) if a == b)
        print(f"Token agreement (first {min_len}): {matches}/{min_len} ({matches/min_len:.1%})")

    # === Speculative decoding with compacted cache ===
    if args.draft:
        from kv_compact.mlx import compact_and_generate_speculative

        print(f"\n--- Compacted + speculative decoding (draft: {args.draft}) ---")
        print(f"Loading draft model {args.draft}...")
        draft_model, _ = load(args.draft)

        t0 = time.perf_counter()
        spec_output = compact_and_generate_speculative(
            model, draft_model, tokenizer, prompt,
            max_tokens=args.max_tokens,
            target_ratio=args.ratio,
            num_draft_tokens=args.num_draft,
            verbose=args.verbose,
        )
        spec_time = time.perf_counter() - t0
        print(f"Time: {spec_time:.1f}s")
        print(f"Output: {spec_output[:200]}...")

        print(f"\n--- Timing comparison ---")
        print(f"  Full cache:                  {full_time:.1f}s")
        print(f"  Compacted:                   {compact_time:.1f}s")
        print(f"  Compacted + speculative:     {spec_time:.1f}s")


if __name__ == "__main__":
    main()
