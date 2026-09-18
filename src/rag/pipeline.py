"""RAG query pipeline: retrieve context -> build augmented prompt -> generate."""

from dataclasses import dataclass, field

from src.config import BenchmarkConfig, DEFAULT_CRITERIA, LOGS_DIR
from src.rag.index import (
    CorpusIndex,
    RetrievalResult,
    get_global_index,
)


@dataclass
class RagAnswer:
    """A RAG query answer with evidence."""

    question: str
    answer: str
    retrieved_context: list[dict] = field(default_factory=list)
    model: str = ""


def _format_context(results: list[RetrievalResult]) -> tuple[str, list[dict]]:
    """Format retrieved chunks into a context block and structured evidence."""
    ctx_lines = ["Relevant knowledge base excerpts:"]
    evidence = []
    for r in results:
        ctx_lines.append(f"\n[{r.rank + 1}] ({r.chunk.source}) {r.chunk.text}")
        evidence.append({
            "rank": r.rank + 1,
            "title": r.chunk.title,
            "source": r.chunk.source,
            "score": round(r.score, 3),
            "snippet": r.chunk.text[:200],
        })
    return "\n".join(ctx_lines), evidence


def rag_query(
    question: str,
    model: str | None = None,
    max_new_tokens: int = 120,
    index: CorpusIndex | None = None,
) -> RagAnswer:
    """
    Answer a question using retrieved knowledge + a local model.

    If no model is provided, returns retrieval-based evidence with a
    template answer (offline mode) so RAG can be tested without a GPU.
    """
    # Retrieve context
    if index is None:
        index = get_global_index()
    results = index.search(question, top_k=3)
    ctx_block, evidence = _format_context(results)

    answer = ""
    # Generative mode: only when a local model is explicitly requested. This
    # keeps the default path lightweight and offline (pure retrieval).
    if model:
        augmented_prompt = (
            f"You are a technical program manager assistant for AI inference.\n"
            f"Answer the question using ONLY the context below. "
            f"Cite the source titles.\n\n"
            f"{ctx_block}\n\n"
            f"Question: {question}\n\n"
            f"Concise answer:"
        )
        try:
            answer = _generate(augmented_prompt, model, max_new_tokens)
        except Exception:
            answer = ""

    if not answer:
        answer = _fallback_answer(question, results)

    return RagAnswer(
        question=question,
        answer=answer,
        retrieved_context=evidence,
        model=model or "retrieval-only",
    )


def _generate(prompt: str, model_name: str, max_new_tokens: int) -> str:
    """Generate an answer with a local causal LM."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    gen_model = AutoModelForCausalLM.from_pretrained(model_name)
    gen_model.eval()
    if torch.cuda.is_available():
        gen_model.to("cuda")

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    for k, v in inputs.items():
        inputs[k] = v.to(gen_model.device)

    with torch.no_grad():
        out = gen_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _fallback_answer(question: str, results: list[RetrievalResult]) -> str:
    """Compose an evidence-grounded answer without a generative model."""
    if not results:
        return "No relevant knowledge found. Consider adding documentation."
    top = results[0]
    lines = [f"Based on '{top.chunk.title}':", "", top.chunk.text]
    return " ".join(lines)[:1200]


def rag_cli(question: str, model: str | None = None):
    """CLI-friendly RAG run."""
    answer = rag_query(question, model=model)
    print(f"\nQ: {answer.question}\n")
    print(f"Context evidence ({len(answer.retrieved_context)} chunks):")
    for ev in answer.retrieved_context:
        print(f"  [{ev['rank']}] {ev['title']}  (score {ev['score']})\n      {ev['snippet']}")
    print(f"\nA: {answer.answer}\n")
    return answer