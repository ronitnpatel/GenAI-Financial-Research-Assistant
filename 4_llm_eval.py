"""
Custom RAG Evaluation using Groq as the evaluator
Metrics:
  - faithfulness      : Does the answer match the retrieved context? (0-1)
  - answer_relevancy  : Is the answer relevant to the question? (0-1)
  - correctness       : Does the answer match the ground truth? (0-1)
"""

import json
import pandas as pd
from groq import Groq
from rag_pipeline import RAGPipeline

GROQ_API_KEY = "" 
GROQ_MODEL   = "llama-3.3-70b-versatile"

# Ground Truth evaluation set

EVAL_SET = [
    {
        "question": "What major acquisition did JPMorgan complete in 2023?",
        "ground_truth": "JPMorgan acquired First Republic Bank in 2023 after it was seized by regulators.",
    },
    {
        "question": "How does JPMorgan describe its liquidity risk management approach?",
        "ground_truth": (
            "JPMorgan maintains a liquidity reserve and monitors the liquidity coverage "
            "ratio (LCR) to manage short-term liquidity risk."
        ),
    },
    {
        "question": "What are the main business segments JPMorgan operates in?",
        "ground_truth": (
            "JPMorgan Chase operates through four main segments: Consumer and Community Banking, "
            "Commercial Banking, Corporate and Investment Bank, and Asset and Wealth Management."
        ),
    },
    {
        "question": "How does JPMorgan describe its approach to credit risk management?",
        "ground_truth": (
            "JPMorgan manages credit risk through underwriting standards, credit limits, "
            "portfolio diversification, and ongoing monitoring of counterparty exposure."
        ),
    },
    {
        "question": "How does JPMorgan describe the competitive environment it operates in?",
        "ground_truth": (
            "JPMorgan faces competition from other large banks, regional banks, fintech companies, "
            "and non-bank financial institutions across all its business segments."
        ),
    },
]

# LLM Eval

class LLMEval:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)

    def _ask(self, prompt):
        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()

    # Extract a 0-1 score from LLM response
    def _parse_score(self, text):
        try:
            # Look for JSON score field first
            data = json.loads(text)
            return float(data.get("score", 0))
        except Exception:
            # If not, find a number in the text
            import re
            matches = re.findall(r"0?\.\d+|[01]\.0*|\b[01]\b", text)
            return float(matches[0]) if matches else 0.0

    # Score how faithful the answer is to the retrieved context
    def faithfulness(self, answer, context):
        prompt = f"""You are evaluating a RAG system. Score how faithful the answer 
            is to the provided context. A faithful answer only uses information from the context.

            Context:
            {context[:2000]}

            Answer:
            {answer}

            Respond with ONLY a JSON object like: {{"score": 0.85, "reason": "brief reason"}}
            Score must be between 0 and 1."""
        result = self._ask(prompt)
        return self._parse_score(result)

    # Score how relevant the answer is to the question
    def answer_relevancy(self, question, answer):
        prompt = f"""You are evaluating a RAG system. Score how relevant and complete 
            the answer is for the given question.

            Question:
            {question}

            Answer:
            {answer}

            Respond with ONLY a JSON object like: {{"score": 0.85, "reason": "brief reason"}}
            Score must be between 0 and 1."""
        result = self._ask(prompt)
        return self._parse_score(result)

    # Score how correct the answer is compared to ground truth
    def correctness(self, answer, ground_truth):
        prompt = f"""You are evaluating a RAG system. Score how factually correct 
            the answer is compared to the ground truth.

            Ground Truth:
            {ground_truth}

            Answer:
            {answer}

            Respond with ONLY a JSON object like: {{"score": 0.85, "reason": "brief reason"}}
            Score must be between 0 and 1."""
        result = self._ask(prompt)
        return self._parse_score(result)


# Main

def main():
    print("-" * 60)
    print("  RAG Evaluation (LLM-Eval)")
    print("-" * 60)

    rag = RAGPipeline()
    eval = LLMEval()

    rows = []
    print(f"\nEvaluating {len(EVAL_SET)} questions...\n")

    for i, item in enumerate(EVAL_SET, 1):
        q  = item["question"]
        gt = item["ground_truth"]

        print(f"[{i}/{len(EVAL_SET)}] {q}")

        # Get RAG answer
        result  = rag.query(q, top_k=5)
        answer = result["answer"].split("Sources Used:")[0].strip()
        context = result["context"]

        print(f"  Answer: {answer[:100]}...")

        # Score with LLM eval
        faith   = eval.faithfulness(answer, context)
        relev   = eval.answer_relevancy(q, answer)
        correct = eval.correctness(answer, gt)

        print(f"  Faithfulness={faith:.2f}  Relevancy={relev:.2f}  Correctness={correct:.2f}\n")

        rows.append({
            "question":         q,
            "ground_truth":     gt,
            "answer":           answer,
            "faithfulness":     faith,
            "answer_relevancy": relev,
            "correctness":      correct,
        })

    # Build results dataframe
    df = pd.DataFrame(rows)

    print("\n=== Summary Results ===")
    print(df[["question", "faithfulness", "answer_relevancy", "correctness"]].to_string(index=False))

    print("\n=== Aggregate Scores ===")
    for col in ["faithfulness", "answer_relevancy", "correctness"]:
        print(f"  {col:25s}: {df[col].mean():.3f}")

    # Save to CSV
    out = "llm_evaluation_results.csv"
    df.to_csv(out, index=False)
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()