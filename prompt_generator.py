from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

from .utils import write_json


DOMAINS = {
    "coding": [
        "Write and explain a Python function that debounces async API calls.",
        "Find the bug in this pseudocode scheduler and provide a corrected version.",
        "Design a CUDA kernel strategy for reducing a matrix by rows.",
    ],
    "reasoning": [
        "A train leaves at noon and another leaves later; solve step by step with constraints.",
        "Compare two conflicting witness statements and infer the most likely timeline.",
        "Rank these options under uncertainty and justify the tradeoffs.",
    ],
    "multilingual": [
        "Translate, summarize, and critique this idea in Spanish, Japanese, and Arabic.",
        "Explain the same technical concept in French for a child and in German for an engineer.",
        "Detect language switches and preserve names while translating.",
    ],
    "math": [
        "Prove a small theorem about modular arithmetic and then compute examples.",
        "Solve a probability problem involving conditional events and hidden variables.",
        "Derive the gradient of a two-layer network with layer normalization.",
    ],
    "long_context": [
        "Remember these constraints: alpha=7, beta is twice alpha, avoid using the word 'simple'. Now solve a planning task.",
        "Given a long meeting transcript outline, extract decisions, risks, owners, and deadlines.",
        "Track five fictional variables across ten numbered paragraphs and answer consistency questions.",
    ],
    "roleplay": [
        "Act as a skeptical database reliability engineer reviewing a risky migration.",
        "Roleplay a patient tutor helping a confused student debug a proof.",
        "Respond as a legal editor improving clarity without changing meaning.",
    ],
    "instruction": [
        "Follow exactly: return JSON only with fields answer, assumptions, and checks.",
        "Use three bullet points, no more than eight words each, and include one equation.",
        "Refuse unsafe steps, then provide a benign alternative workflow.",
    ],
    "edge_cases": [
        "Handle empty input, duplicate IDs, invalid UTF-8, and timezone ambiguity.",
        "Explain how the answer changes if all constraints are contradictory.",
        "Produce a robust parser plan for nested quotes and escaped delimiters.",
    ],
    "memory": [
        "Memorize this mapping for the next answer: red=13, cedar=blue, north=triangle.",
        "Use earlier facts in this prompt to answer the final question without restating all facts.",
        "Track a sequence of operations on a stack and report the final state.",
    ],
    "logic": [
        "Solve a Knights and Knaves puzzle with three people and one unreliable narrator.",
        "Convert natural language constraints into symbolic logic and find a satisfying assignment.",
        "Identify whether the argument is valid, sound, both, or neither.",
    ],
    "creative": [
        "Write a compact science fiction scene that embeds a valid sorting algorithm metaphor.",
        "Create a poem with strict syllable counts and an internal contradiction resolved at the end.",
        "Invent a product pitch for a tool that helps dreams compile into code.",
    ],
    "symbolic": [
        "Simplify a symbolic expression and explain each rewrite rule.",
        "Manipulate a context-free grammar and show whether a target string is accepted.",
        "Evaluate a lambda calculus expression using normal-order reduction.",
    ],
}


@dataclass
class PromptRecord:
    id: int
    domain: str
    prompt: str


def generate_prompts(count: int = 1000, seed: int = 17) -> list[PromptRecord]:
    rng = random.Random(seed)
    domains = list(DOMAINS)
    records: list[PromptRecord] = []
    for idx in range(count):
        domain = domains[idx % len(domains)]
        base = rng.choice(DOMAINS[domain])
        modifiers = [
            "Be precise and expose hidden assumptions.",
            "Include adversarial edge cases.",
            "Prefer stepwise reasoning with compact final output.",
            "Use at least two different representation styles.",
            "Make the task difficult enough to activate specialized capabilities.",
        ]
        prompt = f"{base}\n\n{rng.choice(modifiers)}\nPrompt id: {idx}."
        records.append(PromptRecord(idx, domain, prompt))
    rng.shuffle(records)
    return records


def save_prompts(path: str | Path, count: int = 1000, seed: int = 17) -> None:
    write_json(path, [record.__dict__ for record in generate_prompts(count, seed)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="prompts/generated_prompts.json")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    save_prompts(args.out, args.count, args.seed)
    print(f"wrote {args.count} prompts to {args.out}")


if __name__ == "__main__":
    main()
