"""
RAG Evaluation using DeepEval framework
Measures:
  - Faithfulness        : Does the answer stick to the retrieved context?
  - Answer Relevancy    : Is the answer relevant to the question?
  - Contextual Precision: Are the retrieved chunks ranked well?
  - Contextual Recall   : Does the context cover the ground truth?
  - Hallucination       : Does the answer contain made-up information?
"""

import os
from deepeval import evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    HallucinationMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM
from groq import Groq
import pandas as pd
from rag_pipeline import RAGPipeline


GROQ_API_KEY = ""  
GROQ_MODEL   = "llama-3.3-70b-versatile"  

# Groq wrapper for DeepEval 
# Wraps Groq API so DeepEval can use it 
class GroqDeepEvalModel(DeepEvalBaseLLM):

    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)

    def load_model(self):
        return self.client

    def generate(self, prompt):
        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()

    async def a_generate(self, prompt):
        return self.generate(prompt)

    def get_model_name(self):
        return GROQ_MODEL


# Evaluation set
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


# Build test cases
# Run RAG on each question and wrap results in DeepEval LLMTestCase
def build_test_cases(rag, eval_set):
    test_cases = []
    print(f"Running RAG on {len(eval_set)} questions...\n")
    for i, item in enumerate(eval_set, 1):
        q  = item["question"]
        gt = item["ground_truth"]

        print(f"  [{i}/{len(eval_set)}] {q}")
        result = rag.query(q, top_k=5)

        answer = result["answer"].split("Sources Used:")[0].strip()
        contexts = [c["text"] for c in result["source_chunks"]]

        print(f"  Answer: {answer[:100]}...\n")

        test_cases.append(
            LLMTestCase(
                input=q,
                actual_output=answer,
                expected_output=gt,
                retrieval_context=contexts,
                context=contexts,
            )
        )

    return test_cases


# Main
def main():
    print("-" * 60)
    print("  RAG Evaluation with DeepEval + Groq")
    print("-" * 60)

    # Load RAG pipeline
    rag = RAGPipeline()

    # Build test cases
    eval_subset = EVAL_SET[:1]
    test_cases = build_test_cases(rag, eval_subset)

    # Set up Groq as the evaluator 
    groq_model = GroqDeepEvalModel()

    # Define metrics
    metrics = [
        FaithfulnessMetric(
            threshold=0.5,
            model=groq_model,
            include_reason=False,
        ),
        AnswerRelevancyMetric(
            threshold=0.5,
            model=groq_model,
            include_reason=False,
        ),
        ContextualPrecisionMetric(
            threshold=0.5,
            model=groq_model,
            include_reason=False,
        ),
        ContextualRecallMetric(
            threshold=0.5,
            model=groq_model,
            include_reason=False,
        ),
        HallucinationMetric(
            threshold=0.5,
            model=groq_model,
            include_reason=False,
        ),
    ]

    # Run evaluation
    print("Running DeepEval metrics...\n")
    results = evaluate(test_cases, metrics)

    # Collect and display results
    rows = []
    for tc, result in zip(test_cases, results.test_results):
        row = {"question": tc.input, "answer": tc.actual_output}
        for metric_data in result.metrics_data:
            row[metric_data.name] = round(metric_data.score, 3) if metric_data.score is not None else None
        rows.append(row)

    df = pd.DataFrame(rows)

    score_cols = [col for col in [
        "Faithfulness",
        "Answer Relevancy",
        "Contextual Precision",
        "Contextual Recall",
        "Hallucination",
    ] if col in df.columns]

    print("\n=== Summarize Scores ===")
    print(df[["question"] + score_cols].to_string(index=False))

    print("\n=== Aggregate Scores ===")
    for col in score_cols:
        mean_val = df[col].mean()
        print(f"  {col:25s}: {mean_val:.3f}")

    out = "deepeval_results.csv"
    df.to_csv(out, index=False)
    print(f"\nFull results saved to: {out}")


if __name__ == "__main__":
    main()