import fitz
import os
import numpy as np
import json
from openai import OpenAI

import dotenv
from openai.types import CreateEmbeddingResponse
from openai.types.chat import ChatCompletion

dotenv.load_dotenv('../.env')

embedding_model = "text-embedding-3-small"
openai_model = "gpt-4o-mini-2024-07-18"

def extract_text_from_pdf(pdf_path):
    mypdf = fitz.open(pdf_path)
    all_text = ""
    for page_num in range(mypdf.page_count):
        page = mypdf[page_num]
        text = page.get_text("text")
        all_text += text
    return all_text


def chunk_text(text, n, overlap):
    chunks = []
    for i in range(0, len(text), n - overlap):
        chunks.append(text[i:i + n])
    return chunks


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def semantic_search(query, text_chunks, embeddings, k=5):
    response = client.embeddings.create(model=embedding_model, input=query)
    query_embedding = response.data[0].embedding
    similarity_scores = []

    for i, chunk_embedding in enumerate(embeddings):
        similarity_score = cosine_similarity(np.array(query_embedding), np.array(chunk_embedding.embedding))
        similarity_scores.append((i, similarity_score))

    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    top_indices = [index for index, _ in similarity_scores[:k]]
    return [text_chunks[index] for index in top_indices]


client = OpenAI()

pdf_path = "../data/AI_Information.pdf"

extracted_text: str = extract_text_from_pdf(pdf_path)
text_chunks: list[str] = chunk_text(extracted_text, 1000, 200)

pdf_embedded: CreateEmbeddingResponse = client.embeddings.create(model=embedding_model, input=text_chunks)

data: dict[str, str] = {'has_answer': True,
                        'ideal_answer': 'Explainable AI (XAI) aims to make AI systems more transparent and understandable, '
                                        'providing insights into how they make decisions. It\'s considered important for building trust, '
                                        'accountability, and ensuring fairness in AI systems.',
                        'question': 'What is \'Explainable AI\' and why is it considered important?',
                        'reasoning': 'The document directly defines and explains the importance of XAI.',
                        'reference': 'Chapter 5: The Future of Artificial Intelligence - Explainable AI (XAI); Chapter 19: AI and Ethics'}

top_chunks: list[str] = semantic_search(data['question'], text_chunks, pdf_embedded.data, k=2)

user_prompt: str = "\n".join(
    [f"Context {i + 1}:\n{chunk}\n=====================================\n" for i, chunk in enumerate(top_chunks)])
user_prompt = f"{user_prompt}\nQuestion: {data['question']}"

ai_response: ChatCompletion = client.chat.completions.create(
    model=openai_model,
    temperature=0,
    messages=[
        {"role": "system",
         "content": "You are an AI assistant that strictly answers based on the given context. If the answer cannot be derived directly from the provided context, respond with: 'I do not have enough information to answer that.'"},
        {"role": "user", "content": user_prompt}
    ]
)
evaluate_system_prompt: str = ("You are an intelligent evaluation system tasked with assessing the AI assistant's responses. "
                               "If the AI assistant's response is very close to the true response, assign a score of 1. "
                               "If the response is incorrect or unsatisfactory in relation to the true response,"
                               " assign a score of 0. If the response is partially aligned with the true response, assign a score of 0.5.")
evaluation_prompt: str = (f"User Query: {data['question']}\nAI Response:\n{ai_response.choices[0].message.content}\n"
                          f"{evaluate_system_prompt}")

evaluation_response: ChatCompletion = client.chat.completions.create(
    model=openai_model,
    temperature=0,
    messages=[
        {"role": "system", "content": evaluate_system_prompt},
        {"role": "user", "content": evaluation_prompt}
    ]
)

print(evaluation_response.choices[0].message.content)
