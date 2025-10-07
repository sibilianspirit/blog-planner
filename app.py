import streamlit as st
import pandas as pd
import openai
from sentence_transformers import util
import torch
import io
import re
import time
import math

# Ustawienia strony Streamlit
st.set_page_config(page_title="Planer Treści SEO", layout="wide")

# --- Funkcje pomocnicze ---

@st.cache_data(show_spinner=False)
def get_openai_embeddings(_texts, api_key, batch_size=256, _progress_bar_placeholder=None):
    """
    Generuje wektory (embeddings) za pomocą API OpenAI, używając batchingu
    dla obsługi dużych list i uniknięcia błędów API.
    """
    client = openai.OpenAI(api_key=api_key)
    all_embeddings = []
    
    clean_texts = [str(text).strip() if pd.notna(text) and str(text).strip() else " " for text in _texts]

    num_batches = math.ceil(len(clean_texts) / batch_size)
    
    if _progress_bar_placeholder:
        progress_bar = _progress_bar_placeholder.progress(0, text=f"Generowanie wektorów... (Batch 0/{num_batches})")
    
    for i in range(num_batches):
        start_index = i * batch_size
        end_index = start_index + batch_size
        batch = clean_texts[start_index:end_index]
        
        try:
            response = client.embeddings.create(input=batch, model="text-embedding-3-large")
            all_embeddings.extend([item.embedding for item in response.data])
            
            if _progress_bar_placeholder:
                progress_bar.progress((i + 1) / num_batches, text=f"Generowanie wektorów... (Batch {i+1}/{num_batches})")
            
            time.sleep(1) # Krótka przerwa, aby nie przeciążać API

        except Exception as e:
            st.error(f"Błąd podczas przetwarzania batcha nr {i+1}: {e}")
            return None
            
    return all_embeddings

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
        Twoje zadanie: Zaproponuj 3 unikalne tytuły artykułów blogowych.
        Zasady:
        1. Każdy tytuł musi zawierać dokładną frazę kluczową: "{keyword}".
        2. Tytuły muszą mieć charakter informacyjny lub poradnikowy (np. "Jak...", "Co to jest...").
        3. Stosuj polskie zasady pisowni – tylko pierwsza litera w tytule wielka.
        4. Zamiast dwukropka używaj myślnika.
        5. Zwróć odpowiedź wyłącznie w formie listy numerowanej.
        """
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": "Jesteś ekspertem SEO i copywriterem."}, {"role": "user", "content": prompt}],
            temperature=0.7, max_tokens=200
        )
        content = response.choices[0].message.content
        titles = re.findall(r'\d+\.\s*(.*)', content)
        while len(titles) < 3: titles.append("---")
        return titles[:3]
    except Exception as e:
        st.error(f"Błąd podczas generowania tytułów dla '{keyword}': {e}")
        return ["Błąd API", "Błąd API", "Błąd API"]

# --- Interfejs Użytkownika (UI) ---
st.title("🚀 Planer Treści SEO [Wersja Hybrydowa v6 - PRO]")
st.markdown("Najnowsza wersja z precyzyjnym mapowaniem rankingów i zaawansowanymi modelami AI od OpenAI.")

col1, col2 = st.columns(2)
with col1:
    st.header("1. Konfiguracja")
    num_to_generate = st.number_input("Liczba nowych artykułów do wygenerowania", min_value=1, value=20)
    similarity_threshold = st.slider("Próg podobieństwa dla optymalizacji", min_value=0.7, max_value=1.0, value=0.8, step=0.01)
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
        with st.spinner("Przeprowadzam analizę..."):
            try:
                df_gap = pd.read_csv(content_gap_file).dropna(subset=['Keyword']).astype({'Keyword': str})
                df_articles = pd.read_csv(my_articles_file)
                df_ranking = pd.read_csv(ranking_file)
                
                df_articles.rename(columns={'Address': 'URL', 'Title 1': 'Title'}, inplace=True)
                df_articles.dropna(subset=['Title', 'URL'], inplace=True)
                df_articles = df_articles[df_articles['Title'].str.strip() != ''].reset_index(drop=True)
                df_articles = df_articles[~df_articles['Title'].str.contains("Bot Verification|Strona|Kategoria", na=False, case=False)].reset_index(drop=True)
                df_ranking.dropna(subset=['Słowo kluczowe', 'Adres URL'], inplace=True)
                
                st.info(f"Wczytano {len(df_gap)} słów kluczowych, {len(df_articles)} artykułów i {len(df_ranking)} rankingowych słów kluczowych.")
            except Exception as e:
                st.error(f"Błąd podczas wczytywania plików CSV: {e}")
                st.stop()

            ranking_map = {str(k).lower(): v for k, v in zip(df_ranking['Słowo kluczowe'], df_ranking['Adres URL'])}
            results = []
            keywords_for_semantic_check = []
            
            for _, row in df_gap.iterrows():
                keyword_lower = str(row['Keyword']).lower()
                if keyword_lower in ranking_map:
                    results.append({'Słowo kluczowe': row['Keyword'], 'Wolumen': row.get('Volume', 0), 'Status': 'Już istnieje', 'Akcja / Dopasowany URL': ranking_map[keyword_lower], 'Podobieństwo': 1.00})
                else:
                    keywords_for_semantic_check.append(row.to_dict())
            
            st.info(f"{len(results)} słów zmapowano na podstawie rankingu. Pozostało {len(keywords_for_semantic_check)} do analizy semantycznej.")
            
            if keywords_for_semantic_check:
                df_semantic = pd.DataFrame(keywords_for_semantic_check)
                
                embedding_progress_placeholder = st.empty()
                corpus_embeddings = get_openai_embeddings(df_articles['Title'].tolist(), openai_api_key, _progress_bar_placeholder=embedding_progress_placeholder)
                query_embeddings = get_openai_embeddings(df_semantic['Keyword'].tolist(), openai_api_key, _progress_bar_placeholder=embedding_progress_placeholder)
                embedding_progress_placeholder.empty()

                if corpus_embeddings and query_embeddings:
                    cosine_scores = util.cos_sim(torch.tensor(query_embeddings), torch.tensor(corpus_embeddings))
                    
                    semantic_records = df_semantic.to_dict('records')
                    for row_dict, scores in zip(semantic_records, cosine_scores):
                        top_result = torch.topk(scores, k=1)
                        score, corpus_idx = top_result.values[0].item(), top_result.indices[0].item()

                        if score > similarity_threshold:
                            status, url = 'Do optymalizacji', df_articles.iloc[corpus_idx]['URL']
                        else:
                            status, url = 'Nowy temat', 'Stwórz nowy artykuł'
                        results.append({'Słowo kluczowe': row_dict['Keyword'], 'Wolumen': row_dict.get('Volume', 0), 'Status': status, 'Akcja / Dopasowany URL': url, 'Podobieństwo': round(score, 2)})
            
            df_results = pd.DataFrame(results)
            df_new_topics = df_results[df_results['Status'] == 'Nowy temat'].copy()
            
            if not df_new_topics.empty:
                df_to_process = df_new_topics.sort_values(by='Wolumen', ascending=False).head(num_to_generate)
                st.info(f"Generuję propozycje tytułów dla {len(df_to_process)} najważniejszych nowych tematów...")
                
                df_gap_indexed = df_gap.set_index('Keyword')
                df_to_process['Competitor URL'] = df_to_process['Słowo kluczowe'].map(df_gap_indexed.apply(find_first_competitor_url, axis=1))

                progress_bar = st.progress(0, text="Generowanie tytułów (GPT-4)...")
                
                generated_titles_data = []
                for i, row in enumerate(df_to_process.itertuples()):
                    titles = generate_titles(openai_api_key, row._1, row.Wolumen, getattr(row, 'Competitor URL', 'Brak'))
                    generated_titles_data.append({'Słowo kluczowe': row._1, 'Propozycja tematu 1': titles[0], 'Propozycja tematu 2': titles[1], 'Propozycja tematu 3': titles[2]})
                    progress_bar.progress((i + 1) / len(df_to_process), text=f"Generowanie tytułów ({i+1}/{len(df_to_process)})")

                if generated_titles_data:
                    df_titles = pd.DataFrame(generated_titles_data)
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

            st.download_button("Pobierz gotowy plan treści jako CSV", csv_bytes, "plan_tresci.csv", "text/csv")
