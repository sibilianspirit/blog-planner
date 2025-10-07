import streamlit as st
import pandas as pd
import openai
from sentence_transformers import util
import torch
import io
import re

# Ustawienia strony Streamlit
st.set_page_config(page_title="Planer Treści SEO", layout="wide")

# --- Funkcje pomocnicze ---

@st.cache_data(show_spinner=False)
def get_openai_embeddings(_texts, api_key):
    """Generuje wektory (embeddings) za pomocą API OpenAI, czyszcząc dane wejściowe."""
    client = openai.OpenAI(api_key=api_key)
    clean_texts = [str(text).strip() if pd.notna(text) and str(text).strip() else " " for text in _texts]

    try:
        response = client.embeddings.create(
            input=clean_texts,
            model="text-embedding-3-large"  # ZMIANA MODELU EMBEDDINGOWEGO
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        st.error(f"Błąd podczas generowania wektorów OpenAI: {e}")
        return None

def find_first_competitor_url(row):
    """Znajduje pierwszy dostępny URL konkurenta w danym wierszu."""
    for col in row.index:
        if isinstance(col, str) and col.endswith(': URL') and pd.notna(row[col]):
            return row[col]
    return "Brak"

def generate_titles(api_key, keyword, volume, competitor_url):
    """Generuje tytuły za pomocą API OpenAI (GPT-4 Turbo)."""
    try:
        client = openai.OpenAI(api_key=api_key)
        prompt = f"""
        Jesteś ekspertem SEO i copywriterem specjalizującym się w tworzeniu angażujących tytułów na polskojęzyczne blogi.

        Przeanalizuj poniższe dane:
        - Słowo kluczowe do użycia: "{keyword}"
        - Miesięczny wolumen wyszukiwania: {volume}
        - Artykuł konkurencji: {competitor_url}

        Twoje zadanie:
        Zaproponuj 3 unikalne tytuły artykułów blogowych.

        Zasady, których musisz bezwzględnie przestrzegać:
        1. Każdy tytuł musi zawierać dokładną frazę kluczową: "{keyword}".
        2. Tytuły muszą mieć charakter informacyjny lub poradnikowy (np. "Jak...", "Co to jest...", "Przewodnik po...").
        3. Stosuj polskie zasady pisowni – tylko pierwsza litera w tytule wielka (reszta małymi, chyba że to nazwa własna).
        4. Zamiast dwukropka używaj myślnika (np. "Tytuł – podtykuł").
        5. Zwróć odpowiedź wyłącznie w formie listy numerowanej (1. Tytuł, 2. Tytuł, 3. Tytuł), bez żadnych dodatkowych wstępów ani wyjaśnień.
        """
        
        response = client.chat.completions.create(
            model="gpt-4-turbo", # ZMIANA MODELU CHATU NA NAJLEPSZY DOSTĘPNY
            messages=[
                {"role": "system", "content": "Jesteś ekspertem SEO i copywriterem."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
            # Usunięto parametr max_tokens dla maksymalnej kompatybilności
        )
        
        content = response.choices[0].message.content
        titles = re.findall(r'\d+\.\s*(.*)', content)
        while len(titles) < 3:
            titles.append("---")
        return titles[:3]

    except Exception as e:
        st.error(f"Błąd podczas generowania tytułów dla '{keyword}': {e}")
        return ["Błąd API", "Błąd API", "Błąd API"]

# --- Interfejs Użytkownika (UI) ---
st.title("🚀 Planer Treści SEO [Wersja Hybrydowa v4 - PRO]")
st.markdown("Najnowsza wersja z precyzyjnym mapowaniem rankingów i zaawansowanymi modelami AI od OpenAI.")

col1, col2 = st.columns(2)

with col1:
    st.header("1. Konfiguracja")
    num_to_generate = st.number_input("Liczba nowych artykułów do wygenerowania", min_value=1, value=20, help="Określ, dla ilu najważniejszych słów kluczowych chcesz wygenerować propozycje tytułów.")
    similarity_threshold = st.slider("Próg podobieństwa dla optymalizacji", min_value=0.7, max_value=1.0, value=0.8, step=0.01, help="Próg, powyżej którego artykuł zostanie uznany za 'Do optymalizacji' (domyślnie: 0.80)")

with col2:
    st.header("2. Wgraj pliki CSV")
    content_gap_file = st.file_uploader("1. Wgraj plik CSV z analizą Content Gap", type="csv")
    my_articles_file = st.file_uploader("2. Wgraj plik CSV z listą swoich artykułów", type="csv")
    ranking_file = st.file_uploader("3. Wgraj plik CSV z aktualnym rankingiem", type="csv")

# --- Logika Aplikacji ---
if st.button("Uruchom Analizę Hybrydową", type="primary"):
    openai_api_key = st.secrets.get("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("Klucz API OpenAI nie został znaleziony w sekretach Streamlit!")
    elif not all([content_gap_file, my_articles_file, ranking_file]):
        st.warning("Upewnij się, że wgrałeś wszystkie trzy pliki CSV.")
    else:
        with st.spinner("Przeprowadzam analizę... To może potrwać kilka minut."):
            try:
                # Wczytywanie i rygorystyczne czyszczenie danych
                df_gap = pd.read_csv(content_gap_file).dropna(subset=['Keyword']).astype({'Keyword': str})
                df_articles = pd.read_csv(my_articles_file)
                df_ranking = pd.read_csv(ranking_file)
                
                df_articles.rename(columns={'Address': 'URL', 'Title 1': 'Title'}, inplace=True)
                df_articles.dropna(subset=['Title', 'URL'], inplace=True)
                df_articles = df_articles[df_articles['Title'].str.strip() != '']
                df_articles = df_articles[~df_articles['Title'].str.contains("Bot Verification|Strona|Kategoria", na=False, case=False)]
                df_ranking.dropna(subset=['Słowo kluczowe', 'Adres URL'], inplace=True)
                
                st.info(f"Wczytano {len(df_gap)} słów kluczowych, {len(df_articles)} artykułów i {len(df_ranking)} rankingowych słów kluczowych.")
            except Exception as e:
                st.error(f"Błąd podczas wczytywania plików CSV: {e}")
                st.stop()

            # Krok 1: Mapowanie na podstawie rankingu
            ranking_keywords = set(df_ranking['Słowo kluczowe'].str.lower())
            ranking_map = df_ranking.set_index(df_ranking['Słowo kluczowe'].str.lower())['Adres URL'].to_dict()
            
            results = []
            keywords_for_semantic_check = []
            
            for _, row in df_gap.iterrows():
                keyword_lower = str(row['Keyword']).lower()
                if keyword_lower in ranking_keywords:
                    results.append({'Słowo kluczowe': row['Keyword'], 'Wolumen': row.get('Volume', 0), 'Status': 'Już istnieje', 'Akcja / Dopasowany URL': ranking_map[keyword_lower], 'Podobieństwo': 1.00})
                else:
                    keywords_for_semantic_check.append(row.to_dict())

            st.info(f"{len(results)} słów zmapowano na podstawie rankingu. Pozostało {len(keywords_for_semantic_check)} do analizy semantycznej.")

            # Krok 2: Analiza semantyczna dla pozostałych
            if keywords_for_semantic_check:
                df_semantic = pd.DataFrame(keywords_for_semantic_check)
                st.info("Generowanie wektorów przez API OpenAI (text-embedding-3-large)...")
                
                corpus_embeddings_openai = get_openai_embeddings(df_articles['Title'].tolist(), openai_api_key)
                query_embeddings_openai = get_openai_embeddings(df_semantic['Keyword'].tolist(), openai_api_key)

                if corpus_embeddings_openai and query_embeddings_openai:
                    corpus_tensor = torch.tensor(corpus_embeddings_openai)
                    query_tensor = torch.tensor(query_embeddings_openai)
                    hits = util.semantic_search(query_tensor, corpus_tensor, top_k=1)
                    
                    # --- POPRAWIONA I BEZPIECZNA PĘTLA ---
                    semantic_records = df_semantic.to_dict('records')
                    for row_dict, hit_list in zip(semantic_records, hits):
                        if hit_list:
                            best_hit = hit_list[0]
                            if best_hit['score'] > similarity_threshold:
                                status = 'Do optymalizacji'
                                url = df_articles.iloc[best_hit['corpus_id']]['URL']
                                score = round(best_hit['score'], 2)
                            else:
                                status = 'Nowy temat'
                                url = 'Stwórz nowy artykuł'
                                score = round(best_hit['score'], 2)
                        else:
                            status = 'Nowy temat'
                            url = 'Stwórz nowy artykuł'
                            score = 0.0

                        results.append({'Słowo kluczowe': row_dict['Keyword'], 'Wolumen': row_dict.get('Volume', 0), 'Status': status, 'Akcja / Dopasowany URL': url, 'Podobieństwo': score})
            
            df_results = pd.DataFrame(results)
            
            # Krok 3: Generowanie tytułów
            df_new_topics = df_results[df_results['Status'] == 'Nowy temat'].copy()
            if not df_new_topics.empty:
                df_to_process = df_new_topics.sort_values(by='Wolumen', ascending=False).head(num_to_generate)
                st.info(f"Generuję propozycje tytułów dla {len(df_to_process)} najważniejszych nowych tematów...")
                
                original_data = df_gap.set_index('Keyword')
                urls_to_add = original_data.apply(find_first_competitor_url, axis=1)
                df_to_process = df_to_process.merge(urls_to_add.rename('Competitor URL'), left_on='Słowo kluczowe', right_index=True, how='left')

                progress_bar = st.progress(0, text="Generowanie tytułów (GPT-4)...")
                
                generated_titles_list = []
                for i, row in enumerate(df_to_process.itertuples()):
                    titles = generate_titles(openai_api_key, row._1, row.Wolumen, row._6)
                    generated_titles_list.append(titles)
                    progress_bar.progress((i + 1) / len(df_to_process), text=f"Generowanie tytułów ({i+1}/{len(df_to_process)})")

                df_titles = pd.DataFrame(generated_titles_list, columns=['Propozycja tematu 1', 'Propozycja tematu 2', 'Propozycja tematu 3'], index=df_to_process.index)
                df_results = df_results.merge(df_titles, how='left', left_index=True, right_index=True)

            df_results.fillna('-', inplace=True)
            st.success("Analiza zakończona!")
            st.header("Wyniki Analizy i Plan Treści")
            
            cols_order = ['Słowo kluczowe', 'Wolumen', 'Status', 'Akcja / Dopasowany URL', 'Propozycja tematu 1', 'Propozycja tematu 2', 'Propozycja tematu 3', 'Podobieństwo']
            existing_cols = [col for col in cols_order if col in df_results.columns]
            df_results_sorted = df_results.sort_values(by=['Status', 'Wolumen'], ascending=[True, False])
            
            st.dataframe(df_results_sorted[existing_cols])
            
            csv_buffer = io.StringIO()
            df_results_sorted[existing_cols].to_csv(csv_buffer, index=False, encoding='utf-8')
            csv_bytes = csv_buffer.getvalue().encode('utf-8-sig')

            st.download_button("Pobierz gotowy plan treści jako CSV", csv_bytes, "plan_tresci.csv", "text/csv")
