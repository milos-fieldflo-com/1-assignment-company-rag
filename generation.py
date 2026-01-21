import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from retrieval import retrieve_documents

load_dotenv()

# Prompt optimized for a Help Desk context
RAG_PROMPT = """You are a FieldFlo Support Assistant. Answer the question based ONLY on the context provided.

Context:
{context}

Question: {question}

If the context doesn't answer the question, say so. Do not make up features.
Answer:"""

def format_docs(docs):
    return "\n\n".join([f"Source: {d.metadata['title']}\nContent: {d.page_content}" for d in docs])

def generate_answer(question: str, topics: list = None):
    """
    End-to-end RAG pipeline: Filter -> Retrieve -> Generate
    """
    # 1. Retrieval
    retrieved_docs = retrieve_documents(question, topics=topics)
    
    if not retrieved_docs:
        return "I couldn't find any information in the help center matching your request."

    # 2. Context Formatting
    context_text = format_docs(retrieved_docs)
    
    # 3. Generation
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({
        "context": context_text,
        "question": question
    })
    
    return {
        "answer": response,
        "sources": [d.metadata['title'] for d in retrieved_docs]
    }

if __name__ == "__main__":
    # Test
    q = "How do I create a proposal?"
    print(f"Question: {q}")
    # We apply the 'CRM' filter because Proposals are part of CRM in FieldFlo
    result = generate_answer(q, topics=["CRM"])
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")