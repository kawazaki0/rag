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

Jak sprawić, żeby LLM poznał Twój problem. 

Przykłądowy problem. Mamy firmę, w niej są HR, które mają własne procedury i reguły. Jak zautomatyzować proces odpowiedzi na pytania dotyczące tych procedur. 

Odpowiedzią jest RAG, bo dostarczam wiedzę uszytą na moją potrzebę, buduję kontekst i generuje odpowiedź w języku naturalnym odpowiednio sformatowaną.

Zagrożenia:
*   **LLM-y nie znają Twoich prywatnych danych:** Były trenowane na ogromnych, ale *ogólnych* i często *nieaktualnych* zbiorach danych (np. wiedza kończy się w 2023).
*   **Halucynacje:** LLM-y potrafią "wymyślać" odpowiedzi, gdy nie znają faktów (lub gdy fakty są sprzeczne z ich treningiem).
*   **Brak źródeł:** Standardowe LLM-y rzadko podają, skąd wzięły informacje, co utrudnia weryfikację. (a co z perplexity)

**Rozwiązanie: Retrieval-Augmented Generation (RAG)**
Pozwalamy LLM-owi korzystać z "otwartej książki" - Twojej własnej, aktualnej bazy wiedzy - podczas odpowiadania na pytania. Łączymy moc wyszukiwania informacji z mocą generowania języka.

---

## Poziom - koncept = RAG: Trzy Kluczowe Kroki

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

diagram architektury

---
Tak wygląda najprostrszy RAG. najpierw raw (Bez langchaina).

kod w pythonie

---

## Poziom - implementacja: Retrieve

Co technicznie kryje się za każdym krokiem?

*   **Retrieve (Vector DB & Embeddings):**
    *   **Co:** Wyszukiwanie semantyczne w bazie wiedzy.
    *   **Embeddings (Reprezentacja Znaczenia):** Modele zamieniające tekst na wektory (listy liczb).
        *   Narzędzia: `openai` (API, np. `text-embedding-3-small`), `sentence-transformers` (modele lokalne/open-source np. `all-MiniLM-L6-v2`).
      *   **Vector DB (Przechowywanie i Wyszukiwanie Wektorów):** Bazy zoptymalizowane pod kątem szybkiego wyszukiwania podobnych wektorów (najbliższych sąsiadów KNN/ANN).
          *   Narzędzia: `chromadb`, `faiss-cpu`/`faiss-gpu` (biblioteki), Pinecone (`pinecone-client`), Weaviate, Qdrant (bazy/usługi).
    *   *[Ikony: Vector DBs (Chroma, FAISS, Pinecone), SentenceTransformers]*

Kod w pythonie

---

## Poziom - implementacja: Augument

*   **Augment (Prompt Engineering):**
    *   **Co:** Tworzenie finalnego promptu dla LLM.
    *   **Techniki:**
        *   Proste łączenie (Stuffing): `f"Kontekst: {context}\n\nPytanie: {query}"`. Dobre dla krótkich kontekstów.
        *   Zaawansowane: Map-Reduce, Refine (obsługa wielu/długich dokumentów, często wspierane przez frameworki jak LangChain).

kod w pythonie

---
## Poziom - implementacja: Generate

*   **Generate (LLM):**
    *   **Co:** Silnik językowy generujący odpowiedzi.
    *   **Narzędzia Python:** `openai` (GPT-4/3.5), `google-generativeai` (Gemini), `transformers` (modele open-source jak Llama/Mistral), `ollama`, `vllm`.
    *   *[Ikony: Python, OpenAI, Google Cloud, Hugging Face]*

kod w pythonie

---

Ponownie najprostrzy RAG, ale langchain.

---
aws, google cloud?

streamlit?

---
tank you

notes: 

Najlepsze praktyki:
- langchain
- langfuse jako observability i prompt manager
- endpointy. id konwersacji. 
