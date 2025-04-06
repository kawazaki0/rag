---
width: 1920
height: 1200
transition: slide
css:
 - chapter4/styles.css
---

## RAG dla Python Developerów: Od Konceptu do Kodu 🐍

**Czyli jak sprawić, by LLM rozmawiał z Twoimi danymi (i nie halucynował!)**

---

## Po co nam RAG? (Problem do rozwiązania)

*   **LLM-y nie znają Twoich prywatnych danych:** Były trenowane na ogromnych, ale *ogólnych* i często *nieaktualnych* zbiorach danych (np. wiedza kończy się w 2023).
*   **Halucynacje:** LLM-y potrafią "wymyślać" odpowiedzi, gdy nie znają faktów (lub gdy fakty są sprzeczne z ich treningiem).
*   **Brak źródeł:** Standardowe LLM-y rzadko podają, skąd wzięły informacje, co utrudnia weryfikację.

**Rozwiązanie: Retrieval-Augmented Generation (RAG)**
Pozwalamy LLM-owi korzystać z "otwartej książki" - Twojej własnej, aktualnej bazy wiedzy - podczas odpowiadania na pytania. Łączymy moc wyszukiwania informacji z mocą generowania języka.

---

## RAG: Trzy Kluczowe Kroki - Koncept

Pomyśl o RAG jak o super-inteligentnym asystencie badawczym dla AI:

1.  **(R)etrieve - Znajdź Informacje:**
    *   **Cel:** Gdy przychodzi pytanie, znajdujemy *najbardziej trafne* fragmenty wiedzy w Twojej "bibliotece" (np. dokumentach firmowych, bazie danych, PDF-ach).
    *   **Jak?** Nie tylko po słowach kluczowych, ale głównie **po znaczeniu** (wyszukiwanie semantyczne). System rozumie, że "urlop zimowy" i "wyjazd na narty" to podobne tematy.

2.  **(A)ugment - Przygotuj "Ściągawkę":**
    *   **Cel:** Znalezione informacje formatujemy i łączymy z oryginalnym pytaniem, tworząc dla LLM-a idealną "ściągawkę" (kontekst).
    *   **Przykład:** Zamiast tylko pytać LLM "Jaka jest cena produktu X?", dajemy mu: `Kontekst: [Fragment cennika: Produkt X kosztuje 100 PLN netto...] Pytanie: Jaka jest cena produktu X?`

3.  **(G)enerate - Wygeneruj Odpowiedź:**
    *   **Cel:** LLM (np. GPT-4, Gemini) działa jak 'autor' - na podstawie dostarczonej "ściągawki" generuje spójną i trafną odpowiedź, opartą na *dostarczonych* faktach.
    *   **Kontrola:** Możemy wpływać na styl odpowiedzi (np. bardziej faktograficzny vs kreatywny - `temperature`) i jej długość (`max_tokens`).

---

## Poziom 2: Klocki RAG - Komponenty i Narzędzia Python!

Co technicznie kryje się za każdym krokiem?

*   **Retrieve (Vector DB & Embeddings):**
    *   **Co:** Wyszukiwanie semantyczne w bazie wiedzy.
    *   **Embeddings (Reprezentacja Znaczenia):** Modele zamieniające tekst na wektory (listy liczb).
        *   Narzędzia: `openai` (API, np. `text-embedding-3-small`), `sentence-transformers` (modele lokalne/open-source np. `all-MiniLM-L6-v2`).
      *   **Vector DB (Przechowywanie i Wyszukiwanie Wektorów):** Bazy zoptymalizowane pod kątem szybkiego wyszukiwania podobnych wektorów (najbliższych sąsiadów KNN/ANN).
          *   Narzędzia: `chromadb`, `faiss-cpu`/`faiss-gpu` (biblioteki), Pinecone (`pinecone-client`), Weaviate, Qdrant (bazy/usługi).
    *   *[Ikony: Vector DBs (Chroma, FAISS, Pinecone), SentenceTransformers]*

*   **Augment (Prompt Engineering):**
    *   **Co:** Tworzenie finalnego promptu dla LLM.
    *   **Techniki:**
        *   Proste łączenie (Stuffing): `f"Kontekst: {context}\n\nPytanie: {query}"`. Dobre dla krótkich kontekstów.
        *   Zaawansowane: Map-Reduce, Refine (obsługa wielu/długich dokumentów, często wspierane przez frameworki jak LangChain).

*   **Generate (LLM):**
    *   **Co:** Silnik językowy generujący odpowiedzi.
    *   **Narzędzia Python:** `openai` (GPT-4/3.5), `google-generativeai` (Gemini), `transformers` (modele open-source jak Llama/Mistral), `ollama`, `vllm`.
    *   *[Ikony: Python, OpenAI, Google Cloud, Hugging Face]*

---

## Przygotowanie Bazy Wiedzy - Porządkowanie "Biblioteki" AI

Zanim RAG zadziała, musimy przygotować dane. To jak katalogowanie biblioteki:

1.  **Load:** Wczytaj dane z różnych źródeł (PDF, WWW, DB, TXT...).
    *   *Narzędzia:* Loadery w LangChain/LlamaIndex, `requests`, `beautifulsoup4`, `pypdf`, `sqlalchemy` etc.
2.  **Chunking (Dzielenie na Fragmenty):** Podziel długie dokumenty na mniejsze, logiczne części (np. akapity, sekcje o stałej długości). Kluczowe, by LLM dostał precyzyjny kontekst.
    *   *Narzędzia:* Splittery w LangChain/LlamaIndex (np. `RecursiveCharacterTextSplitter`), własne skrypty.
3.  **Embeddings (Tworzenie Wektorów):** Dla każdego fragmentu wygeneruj wektor reprezentujący jego znaczenie.
    *   *Narzędzia:* Patrz komponent 'Retrieve'.
4.  **Index (Indeksowanie):** Zapisz fragmenty tekstu wraz z ich wektorami w bazie wektorowej, tworząc indeks umożliwiający szybkie wyszukiwanie.
    *   *Narzędzia:* Patrz komponent 'Retrieve' (Vector DB).

*   **Wyzwanie:** Dobór strategii chunkingu i modelu embeddingów ma duży wpływ na jakość RAG!

---

## Architektura RAG: Przepływ Danych

Oto jak informacje przepływają w typowym systemie RAG:

```
+-------------------+      +----------------------+      +-------------------------+      +----------------+      +--------+
| Zapytanie         | ---> | Retrieve             | ---> | Augment                 | ---> | Generate (LLM) | ---> | Odp.   |
| (np. "Co to X?")  |      | (Szukaj w bazie      |      | (Prompt = Kontekst +    |      | (np. GPT-4,    |      |        |
|                   |      | wektorowej; chunking |      |         Zapytanie)      |      |  Gemini)       |      |        |
+-------------------+      |  & embedding         |      +-------------------------+      +----------------+      +--------+
                           |  na etapie prep.)    |       |
                           +----------------------+       | Znalezione fragmenty
                             ^                            v
                             |----------------------------| (Kontekst)
                             |
                       +----------------------+
                       | Baza Wiedzy          |
                       | (Wektorowa Baza:     |
                       |  Fragmenty + Wektory)|
                       +----------------------+
```

---

## Poziom 3: Szybka Implementacja z LangChain 🦜🔗

LangChain upraszcza "sklejanie" komponentów RAG. Zobaczmy kompletny łańcuch:

```python
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_core.documents import Document

# Założenia:
# 1. `vectorstore`: Obiekt VectorStore załadowany danymi po chunkingu i embeddingu.
# 2. `embeddings`: Model embeddingów użyty do `vectorstore`, np. OpenAIEmbeddings().
vectorstore = InMemoryVectorStore(OpenAIEmbeddings(model="text-embedding-3-small"))
vectorstore.add_documents([Document(d) for d in ["Kot siedzi na drzewie.", "Pies biega po łące.", "Samochód jedzie szybko."]])
# 3. `llm`: Zainicjalizowany model, np. ChatOpenAI(model="gpt-3.5-turbo").
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Definicja Retrievera (interfejs do bazy wektorowej)
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
```
*   `RunnablePassthrough()`: Przekazuje wejście (tutaj: pytanie) bez zmian.
*   `|` (pipe): Łączy komponenty w sekwencyjny pipeline.

---

## Poziom 3: LangChain - Przygotowanie Bazy Wektorowej (Krok 1: Retrieve Setup)

Jak stworzyć `vectorstore` użyty w poprzednim slajdzie?

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document # Lub użyj DocumentLoaderów

# 0. Załaduj dane (np. z pliku TXT, PDF etc. używając DocumentLoaderów LangChain)
# Załóżmy, że masz listę stringów `texts` lub obiektów `Document`
texts = ["Jabłka są okrągłe...", "Gruszki są słodkie...", "...więcej o Pythonie..."]
docs = [Document(page_content=t) for t in texts] # Przykładowe tworzenie obiektów Document

# 1. Chunking: Podziel dokumenty na mniejsze fragmenty
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

# 2. Wybierz model embeddingów
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 3. Stwórz Vector Store (np. Chroma) i zaindeksuj fragmenty
#    (Obliczy embeddingi dla `splits` i zapisze w bazie)
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_db" # Opcjonalnie: zapisz na dysku
)

# Teraz `vectorstore` jest gotowy do użycia w `rag_chain`
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Test retrievera
retrieved_docs = retriever.invoke("Opowiedz mi o owocach")
print(f"Pobrano {len(retrieved_docs)} fragmentów.")
# Można sprawdzić zawartość retrieved_docs[0].page_content
```

---

## Poziom 4: Zajrzyjmy pod Maskę - Implementacja 'Raw' w Pythonie 🛠️

A jak zrobić RAG bez LangChain? Używając podstawowych bibliotek:

**Krok 1: Retrieve (Embeddings + Wyszukiwanie Wektorowe z NumPy)**

```python
import openai
import numpy as np

# Konfiguracja API (załóżmy ustawione zmienne środowiskowe)
# import os
# openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Funkcja do generowania Embeddingów ---
def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   try:
       response = openai.embeddings.create(input=[text], model=model)
       return response.data[0].embedding # Zwraca listę float
   except Exception as e:
       print(f"Błąd embeddingu dla: {text[:50]}... Error: {e}")
       return None

# --- Przykładowe Dane (po chunkingu!) ---
# W praktyce te fragmenty pochodzą z etapu przygotowania bazy wiedzy
document_chunks = [
    "Jabłka są zazwyczaj okrągłe i rosną na drzewach jabłoni.",
    "Gruszki mają charakterystyczny kształt, często wydłużony u dołu.",
    "Python jest językiem programowania interpretowanym, wysokiego poziomu.",
    "Czerwone jabłka odmiany 'Ligol' są znane ze swojej chrupkości."
]

# --- Wygeneruj i przechowaj embeddingi (w pamięci dla przykładu) ---
# W realnym systemie: użyj FAISS, ChromaDB API, Pinecone API itp.
docs_data = []
for chunk in document_chunks:
    embedding_list = get_embedding(chunk)
    if embedding_list:
        docs_data.append({
            "text": chunk,
            # Konwertuj na numpy array float32 dla wydajności obliczeń
            "embedding": np.array(embedding_list, dtype=np.float32)
        })

# --- Wyszukiwanie Podobieństwa Kosinusowego ---
def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0: return 0.0
    return dot_product / (norm_v1 * norm_v2)

# --- Znajdź Top-K najbardziej podobnych fragmentów ---
user_query = "Opowiedz mi o kształcie owoców"
query_embedding_list = get_embedding(user_query)

if query_embedding_list:
    query_vector = np.array(query_embedding_list, dtype=np.float32)

    # Oblicz podobieństwo do wszystkich fragmentów w bazie
    scores = []
    for item in docs_data:
        sim = cosine_similarity(query_vector, item["embedding"])
        scores.append({"text": item["text"], "score": sim})

    # Sortuj i wybierz top K
    top_k = 2
    results = sorted(scores, key=lambda x: x["score"], reverse=True)[:top_k]

    # Stwórz kontekst z najlepszych wyników
    context = "\n\n".join([res["text"] for res in results])

    print(f"--- Pobrany Kontekst (Top {top_k}) ---")
    print(context)
else:
    print("Nie udało się uzyskać embeddingu dla zapytania.")
    context = "" # Ustaw pusty kontekst w razie błędu
```

---

## Poziom 4: Implementacja 'Raw' - Kroki 2 & 3 (Augment + Generate)

Mając `context` z poprzedniego kroku, ręcznie tworzymy prompt i wywołujemy API LLM:

```python
import openai # Upewnij się, że skonfigurowane

# --- Krok 2: Augment (Ręczne Tworzenie Promptu) ---
# Używamy `context` i `user_query` z poprzednich kroków

if context: # Sprawdź, czy kontekst został znaleziony
    final_prompt = f"""Na podstawie poniższego kontekstu odpowiedz na pytanie. Odpowiadaj tylko informacjami zawartymi w kontekście.

Kontekst:
---
{context}
---

Pytanie: {user_query}

Odpowiedź:"""
else: # Fallback, jeśli nie znaleziono kontekstu
    final_prompt = f"Odpowiedz na pytanie: {user_query}"
    print("INFO: Nie znaleziono relevantnego kontekstu, wysyłam samo pytanie do LLM.")


print("\n--- Finalny Prompt dla LLM ---")
print(final_prompt)


# --- Krok 3: Generate (Bezpośrednie Wywołanie API LLM) ---
def call_llm_api(prompt_to_send):
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo", # Lub inny model
            messages=[
                {"role": "system", "content": "Jesteś pomocnym asystentem. Odpowiadaj na pytania użytkownika. Jeśli dostarczono kontekst, opieraj odpowiedź głównie na nim."},
                {"role": "user", "content": prompt_to_send}
            ],
            temperature=0.1 # Niska temperatura dla bardziej faktograficznych odpowiedzi
        )
        answer = response.choices[0].message.content
        return answer
    except Exception as e:
        print(f"Błąd podczas wywołania API LLM: {e}")
        return "Przepraszam, wystąpił błąd."

# Wywołanie LLM
final_answer = call_llm_api(final_prompt)

print("\n--- Ostateczna Odpowiedź LLM ---")
print(final_answer)
```

---

## Praktyczny Aspekt: Co gdy RAG nie wie?

Co jeśli zapytasz chatbota RAG o pogodę, a jego baza wiedzy zawiera tylko dane o produktach firmy?

*   **Dobra praktyka:** System powinien to rozpoznać i odpowiedzieć grzecznie, zamiast halucynować.
    *   ✅ "Przepraszam, nie mam informacji o pogodzie. Mogę jednak odpowiedzieć na pytania dotyczące naszych produktów X, Y, Z."
    *   ❌ Unikamy: "Według moich danych, pogoda jest słoneczna." (gdy nie ma danych) lub "Błąd systemu."
*   **Jak to zaimplementować?**
    1.  **Sprawdź wynik Retrieve:** Jeśli `Retrieve` nie zwróciło żadnych sensownych fragmentów (niski `score` podobieństwa), to sygnał, że pytanie jest poza zakresem bazy wiedzy.
    2.  **Dodatkowa klasyfikacja:** Można użyć prostego LLM call lub klasyfikatora do oceny, czy pytanie dotyczy domeny bazy wiedzy.
    3.  **Fallback Prompt:** Jeśli pytanie jest poza zakresem, użyj predefiniowanej odpowiedzi lub innego promptu dla LLM, który instruuje go, jak grzecznie odmówić odpowiedzi.
*   **Ciągłe ulepszanie:** Loguj pytania, na które RAG nie znalazł odpowiedzi - to cenne źródło informacji, jak rozszerzyć bazę wiedzy!

---

## Podsumowanie

*   **Wiarygodność:** LLM opiera odpowiedzi na *konkretnych, dostarczonych* danych, a nie tylko na swojej wewnętrznej (czasem błędnej lub nieaktualnej) wiedzy.
*   **Aktualność i Specjalizacja:** Pozwala LLM "nauczyć się" Twojej firmowej, dziedzinowej lub po prostu najnowszej wiedzy bez kosztownego re-treningu.
*   **Redukcja Halucynacji:** Znacząco ogranicza tendencję LLM do wymyślania faktów.
*   **Możliwość Weryfikacji:** Można (i warto!) pokazywać użytkownikowi źródła (pobrane fragmenty), na podstawie których powstała odpowiedź.
*   **Tworzenie Użytecznych Aplikacji:** Umożliwia budowę chatbotów, systemów Q&A, asystentów, które faktycznie rozwiązują problemy w oparciu o specyficzne dane.

**Kluczowe Narzędzia Python:** `openai`, `google-generativeai`, `LangChain`, `LlamaIndex`, `sentence-transformers`, `chromadb`, `faiss`, `pinecone-client`, `numpy`, `requests`...

---

## The end!