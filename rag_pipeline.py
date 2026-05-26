"""
Create RAG pipeline — retriever + generator end-to-end.
Run this for interactive terminal Q&A
"""

from retriever import Retriever
from generator import Generator


class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.generator = Generator()

    def query(self, question, top_k: int = 5, verbose: bool = False):

        # Retrieve top-k relevant chunks from ChromaDB
        chunks = self.retriever.retrieve(question, top_k=top_k)

        # Build context string
        context = self.retriever.format_context(chunks)

        # Generate answer with local LLM
        answer = self.generator.generate(question, context)

        result = {
            "question": question,
            "answer": answer,
            "source_chunks": chunks,
            "context": context,
        }

        if verbose:
            bar = "=" * 60
            print(f"\n{bar}")
            print(f"Q: {question}")
            print(f"\nTop {top_k} retrieved chunks:")
            for i, c in enumerate(chunks, 1):
                print(
                    f"  [{i}] year={c['metadata']['year']}  "
                    f"score={c['score']}  file={c['metadata']['filename']}"
                )
                print(f"      {c['text'][:140].strip()}...")
            print(f"\nA: {answer}")
            print(bar)

        return result


# Demo CLI
SAMPLE_QUESTIONS = [
    "How did JPMorgan characterise its exposure to interest rate risk?",
    "What major acquisition did JPMorgan complete in 2023?",
    "How does JPMorgan describe its liquidity risk management approach?",
    "What are the main business segments JPMorgan operates in?",
    "How does JPMorgan describe its approach to credit risk management?",
    "How does JPMorgan describe the competitive environment it operates in?",
    "What is JPMorgan's dividend policy?",
]


def main():
    print("=" * 60)
    print("  JPMorgan Chase 10-K RAG System")
    print("=" * 60)
    print("\nSample questions you can ask:")
    for i, q in enumerate(SAMPLE_QUESTIONS, 1):
        print(f"  {i}. {q}")
    print("\nType a question and press Enter. Type 'quit' to exit.\n")

    rag = RAGPipeline()

    while True:
        try:
            question = input("Ask > ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break
        rag.query(question, top_k=8, verbose=True)


if __name__ == "__main__":
    main()
