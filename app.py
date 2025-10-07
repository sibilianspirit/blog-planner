import streamlit as st
import pandas as pd
import openai
from sentence_transformers import util
import torch
import io
import re
import time
import math
from sklearn.cluster import AgglomerativeClustering
import numpy as np

# Ustawienia strony Streamlit
st.set_page_config(page_title="Planer Treści SEO", layout="wide")

# --- Funkcje pomocnicze ---

@st.cache_data(show_spinner=False)
def get_openai_embeddings(texts_tuple, api_key, batch_size=256):
    """
    Generuje wektory (embeddings) za pomocą API OpenAI, używając batchingu
    dla obsługi dużych list i uniknięcia błędów API.
    """
    texts = list(texts_tuple)
    client = openai.OpenAI(api_key=api_key)
    all_embeddings = []
    
    clean_texts = [str(text).strip() if pd.notna(text) and str(text).strip() else " " for text in texts]

    num_batches = math.ceil(len(clean_texts) / batch_size)
    
    for i in range(num_batches):
        start_index = i * batch_size
        end_index = start_index + batch_size
        batch = clean_texts[start_index:end_index]
        
        try:
            response = client.embeddings.create(input=batch, model="text-embedding-3-large")
            all_embeddings.extend([item.embedding for item in response.data])
            time.sleep(1)

        except Exception as e:
            st.error(f"Błąd podczas przetwarzania batcha nr {i+1}: {e}")
            return None
            
    return all_embeddings

def cluster_keywords(keywords_df, embeddings, cluster_threshold=0.85):
    """
    Klasteryzuje frazy kluczowe na podstawie podobieństwa semantycznego.
    Zwraca DataFrame z przypisanymi klastrami i HEAD keywords.
    """
    if len(keywords_df) == 0:
        return keywords_df
    
    # Upewniamy się, że kolumna Wolumen istnieje
    if 'Volume' not in keywords_df.columns and 'Wolumen' not in keywords_df.columns:
        keywords_df['Wolumen'] = 0
    elif 'Volume' in keywords_df.columns and 'Wolumen' not in keywords_df.columns:
        keywords_df['Wolumen'] = keywords_df['Volume']
    
    # Konwersja do numpy array
    embeddings_array = np.array(embeddings)
    
    # Obliczanie macierzy podobieństwa
    similarity_matrix = util.cos_sim(embeddings_array, embeddings_array).numpy()
    
    # Konwersja podobieństwa na odległość
    distance_matrix = 1 - similarity_matrix
    
    # Klasteryzacja hierarchiczna
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1 - cluster_threshold,
        metric='precomputed',
        linkage='average'
    )
    
    cluster_labels = clustering.fit_predict(distance_matrix)
    
    # Dodanie informacji o klastrach do DataFrame
    keywords_df['Klaster_ID'] = cluster_labels
    
    # Wyznaczenie HEAD keyword dla każdego klastra (najwyższy wolumen)
    def get_head_keyword(group):
        if 'Wolumen' in group.columns and len(group) > 0:
            head_idx = group['Wolumen'].idxmax()
            keyword_col = 'Keyword' if 'Keyword' in group.columns else 'Słowo kluczowe'
            group['HEAD_Keyword'] = group.loc[head_idx, keyword_col]
            group['Typ_w_klastrze'] = 'RELATED'
            group.loc[head_idx, 'Typ_w_klastrze'] = 'HEAD'
            group['Liczba_fraz_w_klastrze'] = len(group)
        return group
    
    keywords_df = keywords_df.groupby('Klaster_ID', group_keys=False).apply(get_head_keyword)
    
    return keywords_df

def detect_search_intent(keyword):
    """
    Wykrywa intencję wyszukiwania na podstawie wzorców w frazie kluczowej.
    """
    keyword_lower = keyword.lower()
    
    # Wzorce informacyjne
    info_patterns = ['jak', 'co to', 'dlaczego', 'kiedy', 'gdzie', 'czy', 'jakie', 'różnica']
    # Wzorce transakcyjne
    trans_patterns = ['kup', 'cena', 'sklep', 'oferta', 'promocja', 'tani', 'najleps']
    # Wzorce nawigacyjne
    nav_patterns = ['logowanie', 'kontakt', 'strona', 'oficjalna']
    
    if any(pattern in keyword_lower for pattern in trans_patterns):
        return 'Transakcyjna'
    elif any(pattern in keyword_lower for pattern in nav_patterns):
        return 'Nawigacyjna'
    elif any(pattern in keyword_lower for pattern in info_patterns):
        return 'Informacyjna'
    else:
        return 'Mieszana'

def calculate_priority_score(row):
    """
    Oblicza scoring priorytetu dla danej frazy kluczowej.
    Wzór: (Wolumen × 0.5) + (Pozycja_odwrócona × 0.3) + (Podobieństwo × 0.2)
    """
    volume_score = min(row.get('Wolumen', 0) / 1000, 100)  # Normalizacja do 100
    
    # Scoring pozycji (im gorsza pozycja, tym większy potencjał)
    position = row.get('Aktualna_pozycja', 0)
    if position == 0:  # Nie rankuje
        position_score = 100
    elif position > 20:
        position_score = 80
    elif position > 10:
        position_score = 60
    elif position > 5:
        position_score = 40
    else:
        position_score = 20
    
    # Scoring podobieństwa (niższe = nowy temat = wyższy priorytet)
    similarity_score = (1 - row.get('Podobieństwo', 0)) * 100
    
    priority = (volume_score * 0.5) + (position_score * 0.3) + (similarity_score * 0.2)
    return round(priority, 2)

def find_first_competitor_url(row):
    """Znajduje pierwszy dostępny URL konkurenta w danym wierszu."""
    for col in row.index:
        if isinstance(col, str) and col.endswith(': URL') and pd.notna(row[col]):
            return row[col]
    return "Brak"

def generate_titles(api_key, keyword, volume, competitor_url, related_keywords="", max_retries=3):
    """Generuje tytuły za pomocą API OpenAI (GPT-4o)."""
    client = openai.OpenAI(api_key=api_key)
    
    related_info = f"\n- Powiązane frazy do uwzględnienia: {related_keywords}" if related_keywords else ""
    
    prompt = f"""
Jesteś ekspertem SEO i copywriterem specjalizującym się w tworzeniu angażujących tytułów na polskojęzyczne blogi.
Przeanalizuj poniższe dane:
- Główne słowo kluczowe: "{keyword}"
- Miesięczny wolumen wyszukiwania: {volume}{related_info}
- Artykuł konkurencji: {competitor_url}

Twoje zadanie: Zaproponuj 3 unikalne tytuły artykułów blogowych.
Zasady:
1. Główny tytuł musi zawierać dokładną frazę kluczową: "{keyword}".
2. Jeśli są powiązane frazy, włącz je naturalnie w treść tytułów.
3. Tytuły muszą mieć charakter informacyjny lub poradnikowy (np. "Jak...", "Co to jest...").
4. Stosuj polskie zasady pisowni – tylko pierwsza litera w tytule wielka.
5. Zamiast dwukropka używaj myślnika.
6. Zwróć odpowiedź wyłącznie w formie listy numerowanej.
"""
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Jesteś ekspertem SEO i copywriterem."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=400
            )
            content = response.choices[0].message.content
            titles = re.findall(r'\d+\.\s*(.*)', content)
            
            while len(titles) < 3:
                titles.append("---")
            
            time.sleep(0.5)
            return titles[:3]
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                continue
            else:
                st.warning(f"Błąd podczas generowania tytułów dla '{keyword}': {e}")
                return ["Błąd API", "Błąd API", "Błąd API"]

def validate_api_key(api_key):
    """Waliduje klucz API OpenAI."""
    try:
        client = openai.OpenAI(api_key=api_key)
        client.models.list()
        return True
    except Exception as e:
        st.error(f"Nieprawidłowy klucz API OpenAI: {e}")
        return False

# --- Interfejs Użytkownika (UI) ---
st.title("🚀 Planer Treści SEO [Wersja Hybrydowa v8 - ULTIMATE]")
st.markdown("✨ Wersja z klasteryzacją, analizą intencji, scoring'iem priorytetów i rozszerzonym statusem rankingu.")

col1, col2 = st.columns(2)
with col1:
    st.header("1. Konfiguracja")
    num_to_generate = st.number_input("Liczba nowych artykułów do wygenerowania", min_value=1, value=20)
    similarity_threshold = st.slider("Próg podobieństwa dla optymalizacji", min_value=0.7, max_value=1.0, value=0.8, step=0.01)
    cluster_threshold = st.slider("Próg podobieństwa dla klasteryzacji fraz", min_value=0.75, max_value=0.95, value=0.85, step=0.01, 
                                  help="Frazy powyżej tego progu zostaną zgrupowane w jeden klaster")
    enable_clustering = st.checkbox("Włącz klasteryzację fraz kluczowych", value=True)
    
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
        st.stop()
    
    if not validate_api_key(openai_api_key):
        st.stop()
    
    if not all([content_gap_file, my_articles_file, ranking_file]):
        st.warning("Upewnij się, że wgrałeś wszystkie trzy pliki CSV.")
        st.stop()
    
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
            
            # Sprawdzanie czy ranking ma kolumnę pozycji
            position_col = None
            for col in df_ranking.columns:
                if 'pozycj' in col.lower() or 'position' in col.lower():
                    position_col = col
                    break
            
            st.info(f"Wczytano {len(df_gap)} słów kluczowych, {len(df_articles)} artykułów i {len(df_ranking)} rankingowych słów kluczowych.")
        except Exception as e:
            st.error(f"Błąd podczas wczytywania plików CSV: {e}")
            st.stop()

        # Tworzenie mapy rankingu z pozycjami
        ranking_map = {}
        for _, row in df_ranking.iterrows():
            keyword = str(row['Słowo kluczowe']).lower()
            url = row['Adres URL']
            position = row[position_col] if position_col and pd.notna(row.get(position_col)) else 0
            ranking_map[keyword] = {'url': url, 'position': position}
        
        results = []
        keywords_for_semantic_check = []
        
        # Pierwsza faza: mapowanie na podstawie rankingu
        for _, row in df_gap.iterrows():
            keyword_lower = str(row['Keyword']).lower()
            if keyword_lower in ranking_map:
                rank_data = ranking_map[keyword_lower]
                position = rank_data['position']
                
                # Rozszerzony status z pozycją
                if position == 0 or pd.isna(position):
                    status = 'Do optymalizacji (Nie rankuje)'
                elif position <= 3:
                    status = f'Do optymalizacji (Poz. {int(position)} → TOP1)'
                elif position <= 10:
                    status = f'Do optymalizacji (Poz. {int(position)} → TOP3)'
                elif position <= 20:
                    status = f'Do optymalizacji (Poz. {int(position)} → TOP10)'
                else:
                    status = f'Do optymalizacji (Poz. {int(position)} → TOP20)'
                
                results.append({
                    'Słowo kluczowe': row['Keyword'],
                    'Wolumen': row.get('Volume', 0),
                    'Status': status,
                    'Akcja / Dopasowany URL': rank_data['url'],
                    'Podobieństwo': 1.00,
                    'Aktualna_pozycja': position
                })
            else:
                keywords_for_semantic_check.append(row.to_dict())
        
        st.info(f"{len(results)} słów zmapowano na podstawie rankingu. Pozostało {len(keywords_for_semantic_check)} do analizy semantycznej.")
        
        if keywords_for_semantic_check:
            df_semantic = pd.DataFrame(keywords_for_semantic_check)
            
            # Upewniamy się, że kolumna Wolumen istnieje
            if 'Volume' in df_semantic.columns and 'Wolumen' not in df_semantic.columns:
                df_semantic['Wolumen'] = df_semantic['Volume']
            elif 'Wolumen' not in df_semantic.columns:
                df_semantic['Wolumen'] = 0
            
            # Dodajemy kolumnę Słowo kluczowe jeśli jest Keyword
            if 'Keyword' in df_semantic.columns and 'Słowo kluczowe' not in df_semantic.columns:
                df_semantic['Słowo kluczowe'] = df_semantic['Keyword']
            
            st.info("🔄 Generowanie embeddingów dla artykułów...")
            
            # Generowanie embeddingów
            corpus_embeddings = get_openai_embeddings(
                tuple(df_articles['Title'].tolist()),
                openai_api_key
            )
            
            st.info("🔄 Generowanie embeddingów dla słów kluczowych...")
            
            query_embeddings = get_openai_embeddings(
                tuple(df_semantic['Keyword' if 'Keyword' in df_semantic.columns else 'Słowo kluczowe'].tolist()),
                openai_api_key
            )

            if not corpus_embeddings or not query_embeddings:
                st.error("Nie udało się wygenerować wektorów. Sprawdź klucz API i spróbuj ponownie.")
                st.stop()
            
            # Klasteryzacja fraz (jeśli włączona)
            if enable_clustering:
                st.info("🔄 Klasteryzuję frazy kluczowe...")
                df_semantic = cluster_keywords(df_semantic, query_embeddings, cluster_threshold)
            
            # Analiza semantyczna
            cosine_scores = util.cos_sim(torch.tensor(query_embeddings), torch.tensor(corpus_embeddings))
            
            semantic_records = df_semantic.to_dict('records')
            for row_dict, scores in zip(semantic_records, cosine_scores):
                top_result = torch.topk(scores, k=1)
                score, corpus_idx = top_result.values[0].item(), top_result.indices[0].item()
                closest_article_url = df_articles.iloc[corpus_idx]['URL']
                closest_article_title = df_articles.iloc[corpus_idx]['Title']
                
                # Pobieramy słowo kluczowe z odpowiedniej kolumny
                keyword = row_dict.get('Keyword', row_dict.get('Słowo kluczowe', ''))
                volume = row_dict.get('Volume', row_dict.get('Wolumen', 0))

                if score > similarity_threshold:
                    status = f'Do optymalizacji (Podobne do: {closest_article_title[:50]}...)'
                    url = closest_article_url
                else:
                    status = 'Nowy temat'
                    url = 'Stwórz nowy artykuł'
                
                # Wykrywanie intencji
                intent = detect_search_intent(keyword)
                
                result = {
                    'Słowo kluczowe': keyword,
                    'Wolumen': volume,
                    'Status': status,
                    'Akcja / Dopasowany URL': url,
                    'Najbliższy_artykuł': closest_article_url,
                    'Najbliższy_tytuł': closest_article_title,
                    'Podobieństwo': round(score, 2),
                    'Intencja': intent,
                    'Aktualna_pozycja': 0
                }
                
                # Dodanie informacji o klastrze
                if enable_clustering:
                    result['Klaster_ID'] = row_dict.get('Klaster_ID', 0)
                    result['HEAD_Keyword'] = row_dict.get('HEAD_Keyword', row_dict['Keyword'])
                    result['Typ_w_klastrze'] = row_dict.get('Typ_w_klastrze', 'HEAD')
                    result['Liczba_fraz_w_klastrze'] = row_dict.get('Liczba_fraz_w_klastrze', 1)
                
                results.append(result)
        
        df_results = pd.DataFrame(results)
        
        # Obliczanie scoring priorytetu
        df_results['Priorytet_Score'] = df_results.apply(calculate_priority_score, axis=1)
        
        # Generowanie tytułów tylko dla HEAD keywords w klastrach
        if enable_clustering:
            df_new_topics = df_results[
                (df_results['Status'] == 'Nowy temat') & 
                (df_results['Typ_w_klastrze'] == 'HEAD')
            ].copy()
        else:
            df_new_topics = df_results[df_results['Status'] == 'Nowy temat'].copy()
        
        if not df_new_topics.empty:
            df_to_process = df_new_topics.sort_values(by='Priorytet_Score', ascending=False).head(num_to_generate)
            st.info(f"Generuję propozycje tytułów dla {len(df_to_process)} najważniejszych nowych tematów...")
            
            df_gap_indexed = df_gap.set_index('Keyword')
            df_to_process['Competitor URL'] = df_to_process['Słowo kluczowe'].map(
                df_gap_indexed.apply(find_first_competitor_url, axis=1)
            )

            progress_bar = st.progress(0, text="Generowanie tytułów (GPT-4o)...")
            
            generated_titles_data = []
            for i, (idx, row) in enumerate(df_to_process.iterrows()):
                # Zbieranie powiązanych fraz z klastra
                related_keywords = ""
                if enable_clustering and row.get('Liczba_fraz_w_klastrze', 1) > 1:
                    related = df_results[
                        (df_results['Klaster_ID'] == row['Klaster_ID']) & 
                        (df_results['Typ_w_klastrze'] == 'RELATED')
                    ]['Słowo kluczowe'].tolist()
                    related_keywords = ", ".join(related[:5])  # Max 5 powiązanych fraz
                
                titles = generate_titles(
                    openai_api_key,
                    row['Słowo kluczowe'],
                    row['Wolumen'],
                    row.get('Competitor URL', 'Brak'),
                    related_keywords
                )
                generated_titles_data.append({
                    'Słowo kluczowe': row['Słowo kluczowe'],
                    'Propozycja_tematu_1': titles[0],
                    'Propozycja_tematu_2': titles[1],
                    'Propozycja_tematu_3': titles[2]
                })
                progress_bar.progress((i + 1) / len(df_to_process), text=f"Generowanie tytułów ({i+1}/{len(df_to_process)})")

            if generated_titles_data:
                df_titles = pd.DataFrame(generated_titles_data)
                df_results = pd.merge(df_results, df_titles, on='Słowo kluczowe', how='left')

        df_results.fillna('-', inplace=True)
        st.success("✅ Analiza zakończona!")
        
        # Statystyki
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Nowe tematy", len(df_results[df_results['Status'] == 'Nowy temat']))
        with col2:
            st.metric("Do optymalizacji", len(df_results[df_results['Status'].str.contains('optymalizacji', na=False)]))
        with col3:
            if enable_clustering:
                st.metric("Liczba klastrów", df_results['Klaster_ID'].nunique())
        with col4:
            st.metric("Średni priorytet", f"{df_results['Priorytet_Score'].mean():.1f}")
        
        st.header("📊 Wyniki Analizy i Plan Treści")
        
        # Sortowanie wyników
        df_results_sorted = df_results.sort_values(by=['Priorytet_Score', 'Wolumen'], ascending=[False, False])
        
        # Definicja kolejności kolumn
        cols_order = [
            'Słowo kluczowe', 'Wolumen', 'Priorytet_Score', 'Status', 
            'Akcja / Dopasowany URL', 'Najbliższy_artykuł', 'Podobieństwo',
            'Intencja', 'Aktualna_pozycja'
        ]
        
        if enable_clustering:
            cols_order.extend(['Klaster_ID', 'HEAD_Keyword', 'Typ_w_klastrze', 'Liczba_fraz_w_klastrze'])
        
        cols_order.extend(['Propozycja_tematu_1', 'Propozycja_tematu_2', 'Propozycja_tematu_3'])
        
        existing_cols = [col for col in cols_order if col in df_results_sorted.columns]
        
        st.dataframe(df_results_sorted[existing_cols], use_container_width=True)
        
        # Export do CSV
        csv_buffer = io.StringIO()
        df_results_sorted[existing_cols].to_csv(csv_buffer, index=False, encoding='utf-8')
        csv_bytes = csv_buffer.getvalue().encode('utf-8-sig')

        st.download_button(
            "📥 Pobierz gotowy plan treści jako CSV", 
            csv_bytes, 
            "plan_tresci_ultimate.csv", 
            "text/csv",
            type="primary"
        )
