import fitz
import os
import numpy as np
import json
from openai import OpenAI

import dotenv;
from openai.types import CreateEmbeddingResponse
from openai.types.chat import ChatCompletion

dotenv.load_dotenv('../.env')


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


def create_embeddings(text, model="BAAI/bge-en-icl"):
    response = client.embeddings.create(
        model=model,
        input=text
    )

    return response


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def semantic_search(query, text_chunks, embeddings, k=5):
    query_embedding = create_embeddings(query).data[0].embedding
    similarity_scores = []

    for i, chunk_embedding in enumerate(embeddings):
        similarity_score = cosine_similarity(np.array(query_embedding), np.array(chunk_embedding.embedding))
        similarity_scores.append((i, similarity_score))

    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    top_indices = [index for index, _ in similarity_scores[:k]]
    return [text_chunks[index] for index in top_indices]


def generate_response(system_prompt_, user_message, model="meta-llama/Llama-3.2-3B-Instruct"):
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt_},
            {"role": "user", "content": user_message}
        ]
    )
    return response


client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.getenv("NEBIUS_API_KEY")
)

knowledge_base_chunked: CreateEmbeddingResponse = create_embeddings(chunk_text(extract_text_from_pdf(
    "../data/AI_Information.pdf"), 1000, 200))

user_query = "What is 'Explainable AI' and why is it considered important?"
ideal_answer = (
    "Explainable AI (XAI) aims to make AI systems more transparent and understandable, providing insights into how they make decisions. "
    "It's considered important for building trust, accountability, and ensuring fairness in AI systems.")

"""
Explainable AI (XAI) is a set of techniques that aim to make AI decisions more understandable, 
enabling users to assess their fairness and accuracy.
 It is considered important because it helps build trust in AI systems by providing insights into their decision-making processes, 
 making it easier for users to evaluate the reliability and fairness of AI-driven outcomes."""

top_chunks_for_user_query_from_knowledge: list[str] = semantic_search(user_query,
                                                                      chunk_text(extract_text_from_pdf(
                                                                          "../data/AI_Information.pdf"), 1000, 200),
                                                                      knowledge_base_chunked.data, k=2)

ai_response: ChatCompletion = generate_response(
    "You are an AI assistant that strictly answers based on the given context. "
    "If the answer cannot be derived directly from the provided context, respond with: "
    "'I do not have enough information to answer that.'",
    (
        f"{"\n".join([f"Context {i + 1}:\n{chunk}\n=====================================\n" for i, chunk in enumerate(top_chunks_for_user_query_from_knowledge)])}\n"
        f"Question: {user_query}"))

evaluation_response: ChatCompletion = generate_response(
    system_prompt_="You are an intelligent evaluation system tasked with assessing the AI assistant's responses. "
                   "If the AI assistant's response is very close to the true response, assign a score of 1. "
                   "If the response is incorrect or unsatisfactory in relation to the true response, assign a score of 0. "
                   "If the response is partially aligned with the true response, assign a score of 0.5.",
    user_message=f"User Query: {user_query}\n"
                 f"AI Response:\n"
                 f"{ai_response.choices[0].message.content}\n"
                 f"True Response: {ideal_answer}\n"
                 f"You are an intelligent evaluation system tasked with assessing the AI assistant's responses. "
                 "If the AI assistant's response is very close to the true response, assign a score of 1. "
                 "If the response is incorrect or unsatisfactory in relation to the true response, assign a score of 0. "
                 "If the response is partially aligned with the true response, assign a score of 0.5.")

print(evaluation_response.choices[0].message.content)
