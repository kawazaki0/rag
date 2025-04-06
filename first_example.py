from dotenv import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_core.documents import Document

# Założenia:
# 1. `vectorstore`: Obiekt Chroma (lub innej bazy) ZAŁADOWANY danymi po chunkingu i embeddingu.
# 2. `llm`: Zainicjalizowany model, np. ChatOpenAI(model="gpt-3.5-turbo").
# 3. `embeddings`: Model embeddingów użyty do `vectorstore`, np. OpenAIEmbeddings().

load_dotenv('../.env')

# Definicja Retrievera (interfejs do bazy wektorowej)
llm = ChatOpenAI(model="gpt-3.5-turbo")
vectorstore = InMemoryVectorStore(OpenAIEmbeddings(model="text-embedding-3-small"))
vectorstore.add_documents([Document(d) for d in ["Kot siedzi na drzewie.", "Pies biega po łące.", "Samochód jedzie szybko."]])

retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Pobierz Top 3 fragmenty

# Definicja szablonu Promptu
template = """Odpowiedz na pytanie bazując TYLKO na poniższym kontekście:

{context}

Pytanie: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Funkcja formatująca pobrane dokumenty (obiekty LangChain Document) w jeden string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Definicja Łańcucha RAG używając LangChain Expression Language (LCEL)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()} # Pobierz i sformatuj kontekst, przekaż pytanie
    | prompt           # Wypełnij szablon promptu
    | llm              # Wyślij do LLM
    | StrOutputParser() # Wyciągnij odpowiedź jako string
)

# Wywołanie
question = "Jakie są główne cechy produktu X?"
response = rag_chain.invoke(question)
print(response)