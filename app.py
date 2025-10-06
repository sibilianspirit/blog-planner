import streamlit as st
import pandas as pd
import openai
from sentence_transformers import SentenceTransformer, util
import torch
import io
import re

# Ustawienia strony Streamlit
st.set_page_config(page_title="Planer Treści SEO", layout="wide")

# --- Funkcje pomocnicze ---

@st.cache_resource
def load_model():
    """Ładuje i cachuje model SentenceTransformer."""
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def find_first_competitor_url(row):
    """Znajduje pierwszy dostępny URL konkurenta w danym wierszu."""
    for col in row.index:
        if col.endswith(': URL') and pd.notna(row[col]):
            return row[col]
    return "Brak"

def generate_titles(api_key, keyword, volume, competitor_url):
    """Generuje tytuły za pomocą API OpenAI."""
    try:
        client = openai.OpenAI(api_key=api_key)
        
        prompt = f"""
        Jesteś światowej klasy strategiem contentu SEO i copywriterem. Twoim zadaniem jest tworzenie propozycji tytułów artykułów blogowych, które mają ogromną szansę na zdobycie wysokich pozycji w Google i przyciągnięcie uwagi czytelnika.

        Przeanalizuj poniższe dane:
        - Słowo kluczowe do targetowania: "{keyword}"
        - Miesięczny wolumen wyszukiwania: {volume}
        - Przykładowy artykuł konkurencji, który już jest w TOP10: {competitor_url}

        Twoje zadanie:
        Zaproponuj 3 unikalne, angażujące i zoptymalizowane pod SEO tytuły artykułów blogowych. Skup się na intencji użytkownika, która jest głównie informacyjna. Tytuły powinny być w formie listy numerowanej (1. Tytuł 1, 2. Tytuł 2, 3. Tytuł 3). Bądź kreatywny i unikaj prostego powtarzania słowa kluczowego.
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Jesteś ekspertem SEO i copywriterem."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150
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

st.title("🚀 Planer Treści SEO oparty o Analizę Content Gap")
st.markdown("Aplikacja do automatycznego planowania treści blogowych. Wgraj plik z analizą luki w treści oraz listę swoich artykułów, a aplikacja zidentyfikuje istniejące treści i wygeneruje propozycje nowych.")

# Kolumny dla lepszego układu
col1, col2 = st.columns(2)

with col1:
    st.header("1. Konfiguracja")
    num_to_generate = st.number_input(
        "Liczba nowych artykułów do wygenerowania", 
        min_value=1, value=20,
        help="Określ, dla ilu najważniejszych słów kluczowych (z największym wolumenem) chcesz wygenerować propozycje tytułów."
    )
    similarity_threshold = st.slider(
        "Próg podobieństwa semantycznego", 
        min_value=0.5, max_value=1.0, value=0.75, step=0.01,
        help="Jak bardzo podobne musi być słowo kluczowe do tytułu artykułu, aby uznać temat za pokryty? (domyślnie: 0.75)"
    )

with col2:
    st.header("2. Wgraj pliki CSV")
    content_gap_file = st.file_uploader("Wgraj plik CSV z analizą Content Gap", type="csv")
    my_articles_file = st.file_uploader("Wgraj plik CSV z listą swoich artykułów", type="csv")

# --- Logika Aplikacji ---

if st.button("Uruchom Analizę", type="primary"):
    openai_api_key = st.secrets.get("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("Klucz API OpenAI nie został znaleziony w sekretach Streamlit! Upewnij się, że dodałeś go w ustawieniach aplikacji.")
    elif content_gap_file is None or my_articles_file is None:
        st.warning("Upewnij się, że wgrałeś oba pliki CSV.")
    else:
        with st.spinner("Przeprowadzam analizę... To może potrwać kilka minut."):
            try:
                df_gap = pd.read_csv(content_gap_file)
                df_articles = pd.read_csv(my_articles_file)
                
                df_articles.rename(columns={'Address': 'URL', 'Title 1': 'Title'}, inplace=True)
                df_articles = df_articles[~df_articles['Title'].str.contains("Bot Verification|Strona|Kategoria", na=False, case=False)]
                df_articles.dropna(subset=['Title', 'URL'], inplace=True)
                
                st.info(f"Znaleziono {len(df_gap)} słów kluczowych w analizie content gap.")
                st.info(f"Znaleziono {len(df_articles)} unikalnych artykułów do analizy.")
            except Exception as e:
                st.error(f"Błąd podczas wczytywania plików CSV: {e}")
                st.stop()

            model = load_model()
            
            corpus_embeddings = model.encode(df_articles['Title'].tolist(), convert_to_tensor=True, show_progress_bar=True)
            query_embeddings = model.encode(df_gap['Keyword'].tolist(), convert_to_tensor=True, show_progress_bar=True)
            
            hits = util.semantic_search(query_embeddings, corpus_embeddings, top_k=1)
            
            results = []
            for i, row in df_gap.iterrows():
                keyword = row['Keyword']
                volume = row['Volume']
                best_hit = hits[i][0]
                
                if best_hit['score'] > similarity_threshold:
                    matched_article_url = df_articles.iloc[best_hit['corpus_id']]['URL']
                    results.append({'Słowo kluczowe': keyword, 'Wolumen': volume, 'Status': 'Już istnieje', 'Akcja / Dopasowany URL': matched_article_url, 'Podobieństwo': round(best_hit['score'], 2)})
                else:
                    results.append({'Słowo kluczowe': keyword, 'Wolumen': volume, 'Status': 'Nowy temat', 'Akcja / Dopasowany URL': 'Stwórz nowy artykuł', 'Podobieństwo': round(best_hit['score'], 2)})
            
            df_results = pd.DataFrame(results)
            
            df_new_topics = df_results[df_results['Status'] == 'Nowy temat'].copy()
            
            if not df_new_topics.empty:
                df_new_topics_sorted = df_new_topics.sort_values(by='Wolumen', ascending=False)
                df_to_process = df_new_topics_sorted.head(num_to_generate)
                
                st.info(f"Znaleziono {len(df_new_topics)} nowych tematów. Rozpoczynam generowanie propozycji dla {len(df_to_process)} najważniejszych z nich...")
                
                original_data_for_processing = df_gap[df_gap['Keyword'].isin(df_to_process['Słowo kluczowe'])]
                df_to_process['Competitor URL'] = original_data_for_processing.apply(find_first_competitor_url, axis=1).values
                
                progress_bar = st.progress(0, text="Generowanie tytułów...")
                total_to_process = len(df_to_process)
                
                generated_titles = []
                for i, (_, row) in enumerate(df_to_process.iterrows()):
                    titles = generate_titles(openai_api_key, row['Słowo kluczowe'], row['Wolumen'], row['Competitor URL'])
                    generated_titles.append(titles)
                    progress_bar.progress((i + 1) / total_to_process, text=f"Generowanie tytułów... ({i+1}/{total_to_process})")

                df_titles = pd.DataFrame(generated_titles, columns=['Propozycja tematu 1', 'Propozycja tematu 2', 'Propozycja tematu 3'], index=df_to_process.index)
                df_results = df_results.join(df_titles)
            
            df_results.fillna('-', inplace=True)
            st.success("Analiza zakończona!")

            st.header("Wyniki Analizy i Plan Treści")
            
            df_results_sorted = df_results.sort_values(by=['Status', 'Wolumen'], ascending=[True, False])
            st.dataframe(df_results_sorted)
            
            csv_buffer = io.StringIO()
            df_results_sorted.to_csv(csv_buffer, index=False, encoding='utf-8')
            csv_bytes = csv_buffer.getvalue().encode('utf-8')

            st.download_button(
                label="Pobierz gotowy plan treści jako CSV",
                data=csv_bytes,
                file_name="plan_tresci.csv",
                mime="text/csv",
            )
