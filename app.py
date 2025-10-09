import streamlit as st
import pandas as pd
import openai
from sentence_transformers import util
import torch
import io
import re
import time
import math
import hdbscan
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

def cluster_keywords_hdbscan(keywords_df, embeddings, min_cluster_size=2, min_samples=1):
    """
    Klasteryzuje frazy kluczowe używając algorytmu HDBSCAN.
    HDBSCAN automatycznie znajduje optymalną liczbę klastrów i wykrywa outliery.
    
    Args:
        keywords_df: DataFrame ze słowami kluczowymi
        embeddings: Lista embeddingów
        min_cluster_size: Minimalna wielkość klastra (domyślnie 2)
        min_samples: Minimalna liczba próbek w sąsiedztwie (domyślnie 1)
    
    Returns:
        DataFrame z przypisanymi klastrami i HEAD keywords
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
    
    # HDBSCAN klasteryzacja
    # metric='euclidean' działa dobrze z embeddingami znormalizowanymi
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom',  # Excess of Mass - bardziej konserwatywne
        prediction_data=True
    )
    
    cluster_labels = clusterer.fit_predict(embeddings_array)
    
    # Dodanie informacji o klastrach do DataFrame
    keywords_df['Klaster_ID'] = cluster_labels
    
    # -1 oznacza outlier (frazy, które nie pasują do żadnego klastra)
    # Przypisujemy im unikalne ID klastra
    outlier_mask = keywords_df['Klaster_ID'] == -1
    if outlier_mask.any():
        max_cluster_id = keywords_df['Klaster_ID'].max()
        unique_outlier_ids = range(max_cluster_id + 1, max_cluster_id + 1 + outlier_mask.sum())
        keywords_df.loc[outlier_mask, 'Klaster_ID'] = list(unique_outlier_ids)
        keywords_df.loc[outlier_mask, 'Jest_Outlier'] = True
    
    keywords_df['Jest_Outlier'] = keywords_df.get('Jest_Outlier', False)
    
    # Dodanie informacji o sile przynależności do klastra
    # Outliers mają probability = 0
    probabilities = clusterer.probabilities_ if hasattr(clusterer, 'probabilities_') else np.ones(len(keywords_df))
    keywords_df['Cluster_Probability'] = probabilities
    
    # Wyznaczenie HEAD keyword dla każdego klastra
    def get_head_keyword(group):
        if len(group) == 0:
            return group
            
        keyword_col = 'Keyword' if 'Keyword' in group.columns else 'Słowo kluczowe'
        
        # Dla outlierów (pojedyncze frazy) - są same HEAD
        if len(group) == 1:
            group['HEAD_Keyword'] = group[keyword_col].iloc[0]
            group['Typ_w_klastrze'] = 'HEAD'
            group['Liczba_fraz_w_klastrze'] = 1
            return group
        
        # Dla normalnych klastrów - wybieramy HEAD na podstawie wolumenu
        if 'Wolumen' in group.columns:
            # Sortujemy po wolumenie * probability (preferujemy frazy z wysoką pewnością)
            group['_score'] = group['Wolumen'] * (group['Cluster_Probability'] + 0.1)
            head_idx = group['_score'].idxmax()
            group.drop('_score', axis=1, inplace=True)
            
            group['HEAD_Keyword'] = group.loc[head_idx, keyword_col]
            group['Typ_w_klastrze'] = 'RELATED'
            group.loc[head_idx, 'Typ_w_klastrze'] = 'HEAD'
            group['Liczba_fraz_w_klastrze'] = len(group)
            
        return group
    
    keywords_df = keywords_df.groupby('Klaster_ID', group_keys=False).apply(get_head_keyword)
    
    # Dodanie informacji o jakości klastra
    def calculate_cluster_quality(group):
        if len(group) <= 1:
            return group
        # Średnia probability w klastrze
        group['Cluster_Quality'] = group['Cluster_Probability'].mean()
        return group
    
    keywords_df = keywords_df.groupby('Klaster_ID', group_keys=False).apply(calculate_cluster_quality)
    keywords_df['Cluster_Quality'] = keywords_df.get('Cluster_Quality', 1.0)
    
    return keywords_df

def analyze_cluster_coherence(keywords_df, embeddings):
    """
    Analizuje spójność semantyczną klastrów.
    Zwraca statystyki pomocne w ocenie jakości klasteryzacji.
    """
    if 'Klaster_ID' not in keywords_df.columns:
        return {}
    
    embeddings_array = np.array(embeddings)
    stats = {
        'total_clusters': keywords_df['Klaster_ID'].nunique(),
        'outliers': keywords_df['Jest_Outlier'].sum(),
        'avg_cluster_size': keywords_df.groupby('Klaster_ID').size().mean(),
        'max_cluster_size': keywords_df.groupby('Klaster_ID').size().max(),
        'clusters_details': []
    }
    
    # Analiza każdego klastra
    for cluster_id in keywords_df['Klaster_ID'].unique():
        cluster_data = keywords_df[keywords_df['Klaster_ID'] == cluster_id]
        if len(cluster_data) <= 1:
            continue
            
        # Pobieramy embeddingi dla tego klastra
        cluster_indices = cluster_data.index.tolist()
        cluster_embeddings = embeddings_array[cluster_indices]
        
        # Obliczamy średnie podobieństwo wewnątrz klastra
        similarity_matrix = util.cos_sim(
            torch.tensor(cluster_embeddings),
            torch.tensor(cluster_embeddings)
        ).numpy()
        
        # Usuwamy diagonalę (podobieństwo do samego siebie)
        mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
        avg_similarity = similarity_matrix[mask].mean()
        
        keyword_col = 'Keyword' if 'Keyword' in cluster_data.columns else 'Słowo kluczowe'
        
        stats['clusters_details'].append({
            'cluster_id': cluster_id,
            'size': len(cluster_data),
            'avg_similarity': round(avg_similarity, 3),
            'keywords': cluster_data[keyword_col].tolist()[:5],  # Pierwsze 5 fraz
            'total_volume': cluster_data['Wolumen'].sum()
        })
    
    return stats

def detect_search_intent(keyword):
    """
    Wykrywa intencję wyszukiwania na podstawie wzorców w frazie kluczowej.
    """
    keyword_lower = keyword.lower()
    
    # Wzorce informacyjne
    info_patterns = ['jak', 'co to', 'dlaczego', 'kiedy', 'gdzie', 'czy', 'jakie', 'różnica', 'czym jest']
    # Wzorce transakcyjne
    trans_patterns = ['kup', 'cena', 'sklep', 'oferta', 'promocja', 'tani', 'najleps', 'gdzie kupić']
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
    Wzór: (Wolumen × 0.4) + (Pozycja × 0.3) + (Podobieństwo × 0.2) + (Cluster_Quality × 0.1)
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
    
    # Bonus za wysoką jakość klastra (wysoka coherence)
    cluster_quality_score = row.get('Cluster_Quality', 1.0) * 100
    
    priority = (volume_score * 0.4) + (position_score * 0.3) + (similarity_score * 0.2) + (cluster_quality_score * 0.1)
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

ZASADY OBOWIĄZKOWE:
1. Frazy kluczowe odmień i użyj naturalnie jeśli wymaga tego kontekst:
   ❌ ŹLE: "venum dres - jaki model wybrać"
   ✅ DOBRZE: "Dres Venum - jaki model wybrać"
   ❌ ŹLE: "karate szkoła jaką wybrać"
   ✅ DOBRZE: "Jaką szkołę karate wybrać"

2. Rozpoznawaj nazwy marek i traktuj je jako proper names (pisz wielką literą):
   - Venum, Manto, Adidas, Nike to marki sportowe
   - NIE pisz "Co to jest manto dres" (każdy wie co to dres)
   - PISZ "Manto - dlaczego warto wybrać ubrania od tego producenta"
   - PISZ "Dres Manto - wszystko co musisz wiedzieć przed zakupem"

3. Jeśli fraza sugeruje porównanie/ranking, JEDEN tytuł musi być rankingowy:
   - "rękawice bokserskie" → "TOP 10 rękawic bokserskich - ranking 2025"
   - "ochraniacze na piszczele" → "Najlepsze ochraniacze na piszczele - ranking i porównanie"
   - Używaj formatów: TOP 10, ranking, najlepsze, porównanie

4. Typy tytułów do wykorzystania (zróżnicuj 3 propozycje):
   - Poradnikowy: "Jak wybrać...", "Na co zwrócić uwagę przy..."
   - Rankingowy: "TOP 10...", "Najlepsze...", "Ranking..."
   - Problemowy: "Dlaczego...", "Co musisz wiedzieć o..."
   - Ekspercki: "Przewodnik po...", "Wszystko o..."

5. Stosuj polskie zasady pisowni:
   - Tylko pierwsza litera wielka (poza nazwami własnymi)
   - Zamiast dwukropka używaj myślnika
   - Naturalny, płynny język polski

6. Zwróć odpowiedź WYŁĄCZNIE w formie listy numerowanej (bez dodatkowych komentarzy).

Przykłady DOBRYCH tytułów:
- "Rękawice bokserskie Venum - jak wybrać odpowiedni model dla siebie"
- "TOP 10 najlepszych dresów do MMA - ranking 2025"
- "Jaką wybrać szkołę karate - kompletny przewodnik dla początkujących"
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
st.title("🚀 Planer Treści SEO [Wersja HDBSCAN v9 - ULTIMATE]")
st.markdown("✨ Zaawansowana klasteryzacja z HDBSCAN, analiza intencji, scoring priorytetów i rozszerzony status rankingu.")

col1, col2 = st.columns(2)
with col1:
    st.header("1. Konfiguracja")
    num_to_generate = st.number_input("Liczba nowych artykułów do wygenerowania", min_value=1, value=20)
    similarity_threshold = st.slider("Próg podobieństwa dla optymalizacji", min_value=0.7, max_value=1.0, value=0.8, step=0.01)
    
    with st.expander("⚙️ Zaawansowane ustawienia klasteryzacji"):
        min_cluster_size = st.slider(
            "Minimalna wielkość klastra", 
            min_value=2, 
            max_value=10, 
            value=2,
            help="Im wyższa wartość, tym większe i bardziej ogólne klastry. Wartość 2-3 zalecana dla SEO."
        )
        min_samples = st.slider(
            "Minimalna liczba próbek w sąsiedztwie", 
            min_value=1, 
            max_value=5, 
            value=1,
            help="Im wyższa wartość, tym bardziej konserwatywna klasteryzacja. Wartość 1-2 zalecana."
        )
    
    enable_clustering = st.checkbox("Włącz klasteryzację fraz kluczowych (HDBSCAN)", value=True)
    
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
        
        cluster_stats = None
        
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
                st.info("🔄 Klasteryzuję frazy kluczowe z HDBSCAN...")
                df_semantic = cluster_keywords_hdbscan(
                    df_semantic, 
                    query_embeddings, 
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples
                )
                
                # Analiza jakości klasteryzacji
                cluster_stats = analyze_cluster_coherence(df_semantic, query_embeddings)
            
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
                    result['HEAD_Keyword'] = row_dict.get('HEAD_Keyword', keyword)
                    result['Typ_w_klastrze'] = row_dict.get('Typ_w_klastrze', 'HEAD')
                    result['Liczba_fraz_w_klastrze'] = row_dict.get('Liczba_fraz_w_klastrze', 1)
                    result['Jest_Outlier'] = row_dict.get('Jest_Outlier', False)
                    result['Cluster_Probability'] = round(row_dict.get('Cluster_Probability', 1.0), 3)
                    result['Cluster_Quality'] = round(row_dict.get('Cluster_Quality', 1.0), 3)
                
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
        
        # Statystyki - rozszerzone dla HDBSCAN
        if enable_clustering and cluster_stats:
            st.subheader("📈 Statystyki Klasteryzacji HDBSCAN")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Wykryte klastry", cluster_stats['total_clusters'])
            with col2:
                st.metric("Frazy outlier", cluster_stats['outliers'])
            with col3:
                st.metric("Śr. wielkość klastra", f"{cluster_stats['avg_cluster_size']:.1f}")
            with col4:
                st.metric("Maks. wielkość klastra", cluster_stats['max_cluster_size'])
            with col5:
                st.metric("Nowe tematy", len(df_results[df_results['Status'] == 'Nowy temat']))
            
            # Szczegóły najważniejszych klastrów
            with st.expander("🔍 Szczegóły najważniejszych klastrów (TOP 10)"):
                clusters_df = pd.DataFrame(cluster_stats['clusters_details'])
                if not clusters_df.empty:
                    clusters_df = clusters_df.sort_values('total_volume', ascending=False).head(10)
                    for _, cluster in clusters_df.iterrows():
                        st.markdown(f"**Klaster {cluster['cluster_id']}** (Wielkość: {cluster['size']}, Spójność: {cluster['avg_similarity']:.2f}, Wolumen: {cluster['total_volume']})")
                        st.markdown(f"Przykładowe frazy: {', '.join(cluster['keywords'])}")
                        st.markdown("---")
        else:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Nowe tematy", len(df_results[df_results['Status'] == 'Nowy temat']))
            with col2:
                st.metric("Do optymalizacji", len(df_results[df_results['Status'].str.contains('optymalizacji', na=False)]))
            with col3:
                if enable_clustering:
                    st.metric("Liczba klastrów", df_results.get('Klaster_ID', pd.Series()).nunique())
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
            cols_order.extend([
                'Klaster_ID', 'HEAD_Keyword', 'Typ_w_klastrze', 'Liczba_fraz_w_klastrze',
                'Jest_Outlier', 'Cluster_Probability', 'Cluster_Quality'
            ])
        
        cols_order.extend(['Propozycja_tematu_1', 'Propozycja_tematu_2', 'Propozycja_tematu_3'])
        
        existing_cols = [col for col in cols_order if col in df_results_sorted.columns]
        
        # Dodanie kolorowania dla lepszej wizualizacji
        def highlight_rows(row):
            if row['Status'] == 'Nowy temat':
                return ['background-color: #e8f5e9'] * len(row)
            elif 'TOP1' in str(row['Status']):
                return ['background-color: #fff3e0'] * len(row)
            elif 'Nie rankuje' in str(row['Status']):
                return ['background-color: #ffebee'] * len(row)
            return [''] * len(row)
        
        st.dataframe(
            df_results_sorted[existing_cols].style.apply(highlight_rows, axis=1),
            use_container_width=True,
            height=600
        )
        
        # Legenda kolorów
        st.markdown("""
        **Legenda kolorów:**
        - 🟢 Zielony: Nowy temat
        - 🟠 Pomarańczowy: Blisko TOP1 (optymalizacja)
        - 🔴 Czerwony: Nie rankuje
        """)
        
        # Export do CSV
        csv_buffer = io.StringIO()
        df_results_sorted[existing_cols].to_csv(csv_buffer, index=False, encoding='utf-8')
        csv_bytes = csv_buffer.getvalue().encode('utf-8-sig')

        st.download_button(
            "📥 Pobierz gotowy plan treści jako CSV", 
            csv_bytes, 
            "plan_tresci_hdbscan_ultimate.csv", 
            "text/csv",
            type="primary"
        )
        
        # Dodatkowe eksporty
        col1, col2 = st.columns(2)
        
        with col1:
            if enable_clustering:
                # Export tylko HEAD keywords
                df_head_only = df_results_sorted[df_results_sorted['Typ_w_klastrze'] == 'HEAD']
                csv_head_buffer = io.StringIO()
                df_head_only[existing_cols].to_csv(csv_head_buffer, index=False, encoding='utf-8')
                csv_head_bytes = csv_head_buffer.getvalue().encode('utf-8-sig')
                
                st.download_button(
                    "📥 Pobierz tylko HEAD keywords", 
                    csv_head_bytes, 
                    "plan_tresci_head_only.csv", 
                    "text/csv"
                )
        
        with col2:
            # Export priorytetów
            df_priority = df_results_sorted[df_results_sorted['Status'] == 'Nowy temat'].head(50)
            csv_priority_buffer = io.StringIO()
            df_priority[existing_cols].to_csv(csv_priority_buffer, index=False, encoding='utf-8')
            csv_priority_bytes = csv_priority_buffer.getvalue().encode('utf-8-sig')
            
            st.download_button(
                "📥 Pobierz TOP 50 priorytetów", 
                csv_priority_bytes, 
                "plan_tresci_top50.csv", 
                "text/csv"
            )
        
        # Dodatkowa analiza
        st.header("📊 Dodatkowa Analiza")
        
        tab1, tab2, tab3 = st.tabs(["Rozkład intencji", "Analiza wolumenu", "Jakość klastrów"])
        
        with tab1:
            if 'Intencja' in df_results.columns:
                intent_counts = df_results['Intencja'].value_counts()
                st.bar_chart(intent_counts)
                st.markdown("**Interpretacja:** Dominująca intencja wyszukiwania w analizowanych frazach.")
        
        with tab2:
            volume_by_status = df_results.groupby('Status')['Wolumen'].sum().sort_values(ascending=False)
            st.bar_chart(volume_by_status)
            st.markdown("**Interpretacja:** Łączny potencjał ruchu dla każdej kategorii statusu.")
        
        with tab3:
            if enable_clustering and 'Cluster_Quality' in df_results.columns:
                quality_dist = df_results[df_results['Typ_w_klastrze'] == 'HEAD']['Cluster_Quality']
                st.line_chart(quality_dist.sort_values(ascending=False))
                st.markdown(f"**Średnia jakość klastrów:** {quality_dist.mean():.3f}")
                st.markdown("**Interpretacja:** Im wyższa wartość, tym bardziej spójne semantycznie są frazy w klastrze.")
                
                # Ostrzeżenia o słabych klastrach
                weak_clusters = df_results[
                    (df_results['Typ_w_klastrze'] == 'HEAD') & 
                    (df_results['Cluster_Quality'] < 0.5)
                ]['Słowo kluczowe'].tolist()
                
                if weak_clusters:
                    st.warning(f"⚠️ Znaleziono {len(weak_clusters)} klastrów o niskiej spójności. Rozważ ich weryfikację ręczną.")
                    with st.expander("Zobacz listę"):
                        for kw in weak_clusters[:10]:
                            st.markdown(f"- {kw}")
