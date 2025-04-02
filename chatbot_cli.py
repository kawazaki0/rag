from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

load_dotenv('../.env')

client = OpenAI()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = InMemoryVectorStore(embeddings)

if __name__ == "__main__":
    docs = ["Kot siedzi na drzewie.", "Pies biega po łące.", "Samochód jedzie szybko."]
    vectorstore.add_documents([Document(d) for d in docs])
    while True:
        question = input("Q: ")
        top_chunk = vectorstore.similarity_search_with_score(question, k=1)
        response = client.responses.create(
            model="gpt-3.5-turbo",
            input=[
                {"role": "system",
                 "content": (
                     f"Use the given context to answer the question. "
                     f"If you don't know the answer, say you don't know. "
                     f"Use three sentence maximum and keep the answer concise. "
                     f"Context: {top_chunk[0][0].page_content}"
                 )},
                {"role": "user", "content": f"question: {question}", }
            ]
        )
        print('A:', response.output_text)
