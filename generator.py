"""
Use Groq's free cloud API and Meta's LLama Models
to answer questions given retrieved context
"""

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

GROQ_API_KEY = ""   
GROQ_MODEL   = "llama-3.3-70b-versatile"  

PROMPT_TEMPLATE = """You are a financial analyst assistant specialising in JPMorgan Chase bank filings.
Use ONLY the context below to answer the question.
If the answer is not present in the context, say exactly:
"I could not find this information in the provided 10-K filings."

Context from JPMorgan Chase 10-K filings:
{context}

Question: {question}

Instructions:
- Answer in clear, concise prose
- You MUST cite your sources inline using the format (Source X, Year: YYYY) after every fact
- If multiple sources support a fact, cite all of them e.g. (Source 1, Year: 2022; Source 2, Year: 2023)
- End your answer with a "Sources Used:" section listing which sources you cited

Answer:"""


class Generator:
    def __init__(self):
        print(f"[Generator] Connecting to Groq model: {GROQ_MODEL}")
        self.llm = ChatGroq(
            model=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            temperature=0,
        )
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=PROMPT_TEMPLATE,
        )
        self.chain = prompt | self.llm

    # Generate an answer given a question and retrieved context
    def generate(self, question, context):
        return self.chain.invoke(
            {"context": context, "question": question}
        ).content.strip()