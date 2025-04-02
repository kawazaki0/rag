import gradio as gr
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from openai import OpenAI


load_dotenv('../../.env')

client = OpenAI()

system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use three sentence maximum and keep the answer concise. "
    "Context: {context}"
)

# from langchain_huggingface import HuggingFaceEmbeddings
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

docs = ["Kot siedzi na drzewie.", "Pies biega po łące.", "Samochód jedzie szybko."]
vectorstore = InMemoryVectorStore(embeddings)

vectorstore.add_documents([Document(d) for d in docs])


def chatbot_response(question, history):
    print(question, history)
    top_chunk = vectorstore.similarity_search_with_score(question, k=1)
    print(top_chunk[0][0].page_content)
    response = client.responses.create(
        model="gpt-3.5-turbo",
        input=[
            {"role": "system", "content": system_prompt.format(context=top_chunk[0][0].page_content), },
            {"role": "user", "content": f"history: {history[-5:]} ====, message: {question}", }
        ]
    )
    return response.output_text


chatbot = gr.ChatInterface(fn=chatbot_response, title="Chatbot")

if __name__ == "__main__":
    chatbot.launch()
