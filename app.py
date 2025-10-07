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

# Dodajemy cache'owanie, aby nie generować tych samych wektorów wielokrotnie, oszczędzając API
@st.cache_data(show_spinner=False)
def get_openai_embeddings(_texts, api_key):
    """Generuje wektory (embeddings) za pomocą API OpenAI, czyszcząc dane wejściowe."""
    client = openai.OpenAI(api_key=api_key)
    
    # --- KLUCZOWA POPRAWKA: Zabezpieczenie przed pustymi danymi ---
    # Model OpenAI nie akceptuje pustych stringów ani NaN. Zamieniamy je na spację,
    # aby zachować spójność indeksów listy, co jest krytyczne dla dalszego działania.
    clean_texts = [str(text).strip() if pd.notna(text) and str(text).strip() else " " for text in _texts]

    try:
        response = client.embeddings.create(
            input=clean_texts,
            model="text-embedding-3-large"
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        st.error(f"Błąd podczas generowania wektorów OpenAI: {e}. Sprawdź, czy Twoje pliki CSV nie zawierają pustych wierszy w kolumnach 'Keyword' lub 'Title'.")
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
        4. Zamiast dwukropka używaj myślnika (np. "Tytuł – podtytuł").
        5. Zwróć odpowiedź wyłącznie w formie listy numerowanej (1. Tytuł, 2. Tytuł, 3. Tytuł), bez żadnych dodatkowych wstępów ani wyjaśnień.
        """
        
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": "Jesteś ekspertem SEO i copywriterem."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
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

st.title("🚀 Planer Treści SEO [Wersja Hybrydowa v2]")
st.markdown("Ulepszona wersja aplikacji, która wykorzystuje Twoje aktualne rankingi oraz zaawansowane modele OpenAI do precyzyjnego planowania treści.")

col1, col2 = st.columns(2)

with col1:
    st.header("1. Konfiguracja")
    num_to_generate = st.number_input(
        "Liczba nowych artykułów do wygenerowania", min_value=1, value=20,
        help="Określ, dla ilu najważniejszych słów kluczowych chcesz wygenerować propozycje tytułów."
    )
    similarity_threshold = st.slider(
        "Próg podobieństwa dla optymalizacji", min_value=0.7, max_value=1.0, value=0.8, step=0.01,
        help="Próg, powyżej którego artykuł zostanie uznany za 'Do optymalizacji' (domyślnie: 0.80)"
    )

with col2:
    st.header("2. Wgraj pliki CSV")
    content_gap_file = st.file_uploader("1. Wgraj plik CSV z analizą Content Gap", type="csv")
    my_articles_file = st.file_uploader("2. Wgraj plik CSV z listą swoich artykułów", type="csv")
    ranking_file = st.file_uploader("3. Wgraj plik CSV z aktualnym rankingiem", type="csv")

# --- Logika Aplikacji ---

if st.button("Uruchom Analizę Hybrydową", type="primary"):
    openai_api_key = st.secrets.get("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("Klucz API OpenAI nie został znaleziony w sekretach Streamlit! Upewnij się, że dodałeś go w ustawieniach aplikacji.")
    elif not all([content_gap_file, my_articles_file, ranking_file]):
        st.warning("Upewnij się, że wgrałeś wszystkie trzy pliki CSV.")
    else:
        with st.spinner("Przeprowadzam analizę... To może potrwać kilka minut."):
            try:
                df_gap = pd.read_csv(content_gap_file)
                df_articles = pd.read_csv(my_articles_file)
                df_ranking = pd.read_csv(ranking_file)
                
                # Rygorystyczne czyszczenie danych
                df_articles.rename(columns={'Address': 'URL', 'Title 1': 'Title'}, inplace=True)
                df_articles.dropna(subset=['Title', 'URL'], inplace=True)
                df_articles = df_articles[df_articles['Title'].str.strip() != '']
                df_articles = df_articles[~df_articles['Title'].str.contains("Bot Verification|Strona|Kategoria", na=False, case=False)]

                df_ranking.dropna(subset=['Słowo kluczowe', 'Adres URL'], inplace=True)
                df_ranking = df_ranking[df_ranking['Słowo kluczowe'].str.strip() != '']
                
                df_gap.dropna(subset=['Keyword'], inplace=True)
                df_gap = df_gap[df_gap['Keyword'].str.strip() != '']

                st.info(f"Znaleziono {len(df_gap)} słów kluczowych w content gap, {len(df_articles)} artykułów i {len(df_ranking)} rankingowych słów kluczowych.")
            except Exception as e:
                st.error(f"Błąd podczas wczytywania plików CSV: {e}")
                st.stop()

            # Krok 1: Twarde mapowanie na podstawie rankingu
            ranking_keywords = set(df_ranking['Słowo kluczowe'].str.lower())
            ranking_map = df_ranking.set_index(df_ranking['Słowo kluczowe'].str.lower())['Adres URL'].to_dict()
            
            results = []
            keywords_for_semantic_check = []
            
            for _, row in df_gap.iterrows():
                keyword_lower = str(row['Keyword']).lower()
                if keyword_lower in ranking_keywords:
                    results.append({
                        'Słowo kluczowe': row['Keyword'], 'Wolumen': row.get('Volume', 0), 
                        'Status': 'Już istnieje', 'Akcja / Dopasowany URL': ranking_map[keyword_lower], 'Podobieństwo': 1.00
                    })
                else:
                    keywords_for_semantic_check.append(row.to_dict())

            st.info(f"{len(results)} słów kluczowych zostało zmapowanych na podstawie rankingu. Pozostało {len(keywords_for_semantic_check)} do analizy semantycznej.")

            # Krok 2: Analiza semantyczna dla pozostałych
            if keywords_for_semantic_check:
                df_semantic = pd.DataFrame(keywords_for_semantic_check)
                
                article_titles = df_articles['Title'].tolist()
                semantic_keywords = df_semantic['Keyword'].tolist()

                st.info("Generowanie wektorów (embeddings) przez API OpenAI...")
                corpus_embeddings_openai = get_openai_embeddings(article_titles, openai_api_key)
                query_embeddings_openai = get_openai_embeddings(semantic_keywords, openai_api_key)

                if corpus_embeddings_openai is not None and query_embeddings_openai is not None:
                    corpus_tensor = torch.tensor(corpus_embeddings_openai)
                    query_tensor = torch.tensor(query_embeddings_openai)
                    
                    hits = util.semantic_search(query_tensor, corpus_tensor, top_k=1)
                    
                    for i, row_dict in enumerate(df_semantic.to_dict('records')):
                        best_hit = hits[i][0]
                        if best_hit['score'] > similarity_threshold:
                            matched_article_url = df_articles.iloc[best_hit['corpus_id']]['URL']
                            results.append({
                                'Słowo kluczowe': row_dict['Keyword'], 'Wolumen': row_dict.get('Volume', 0),
                                'Status': 'Do optymalizacji', 'Akcja / Dopasowany URL': matched_article_url, 'Podobieństwo': round(best_hit['score'], 2)
                            })
                        else:
                            results.append({
                                'Słowo kluczowe': row_dict['Keyword'], 'Wolumen': row_dict.get('Volume', 0),
                                'Status': 'Nowy temat', 'Akcja / Dopasowany URL': 'Stwórz nowy artykuł', 'Podobieństwo': round(best_hit['score'], 2)
                            })

            df_results = pd.DataFrame(results)
            
            # Krok 3: Generowanie tytułów dla "Nowych tematów"
            df_new_topics = df_results[df_results['Status'] == 'Nowy temat'].copy()
            
            if not df_new_topics.empty:
                df_new_topics_sorted = df_new_topics.sort_values(by='Wolumen', ascending=False)
                df_to_process = df_new_topics_sorted.head(num_to_generate)
                
                st.info(f"Znaleziono {len(df_new_topics)} nowych tematów. Generuję propozycje dla {len(df_to_process)} najważniejszych...")
                
                original_data_for_processing = df_gap[df_gap['Keyword'].isin(df_to_process['Słowo kluczowe'])].set_index('Keyword')
                competitor_urls = original_data_for_processing.apply(find_first_competitor_url, axis=1).reindex(df_to_process['Słowo kluczowe']).values
                
                df_to_process['Competitor URL'] = competitor_urls

                progress_bar = st.progress(0, text="Generowanie tytułów (GPT-4)...")
                total_to_process = len(df_to_process)
                
                generated_titles_list = []
                keywords_for_titles = []
                for i, (_, row) in enumerate(df_to_process.iterrows()):
                    titles = generate_titles(openai_api_key, row['Słowo kluczowe'], row['Wolumen'], row.get('Competitor URL', 'Brak'))
                    generated_titles_list.append(titles)
                    keywords_for_titles.append(row['Słowo kluczowe'])
                    progress_bar.progress((i + 1) / total_to_process, text=f"Generowanie tytułów (GPT-4)... ({i+1}/{total_to_process})")

                df_titles = pd.DataFrame({
                    'Słowo kluczowe': keywords_for_titles,
                    'Propozycja tematu 1': [titles[0] for titles in generated_titles_list],
                    'Propozycja tematu 2': [titles[1] for titles in generated_titles_list],
                    'Propozycja tematu 3': [titles[2] for titles in generated_titles_list]
                })

                df_results = pd.merge(df_results, df_titles, on='Słowo kluczowe', how='left')

            df_results.fillna('-', inplace=True)
            st.success("Analiza zakończona!")

            st.header("Wyniki Analizy i Plan Treści")
            
            df_results_sorted = df_results.sort_values(by=['Status', 'Wolumen'], ascending=[True, False])
            
            cols_order = ['Słowo kluczowe', 'Wolumen', 'Status', 'Akcja / Dopasowany URL', 'Propozycja tematu 1', 'Propozycja tematu 2', 'Propozycja tematu 3', 'Podobieństwo']
            existing_cols = [col for col in cols_order if col in df_results_sorted.columns]
            st.dataframe(df_results_sorted[existing_cols])
            
            csv_buffer = io.StringIO()
            df_results_sorted[existing_cols].to_csv(csv_buffer, index=False, encoding='utf-8')
            csv_bytes = csv_buffer.getvalue().encode('utf-8-sig')

            st.download_button(
                label="Pobierz gotowy plan treści jako CSV",
                data=csv_bytes,
                file_name="plan_tresci.csv",
                mime="text/csv",
            )
