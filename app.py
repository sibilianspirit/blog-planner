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
import warnings
import difflib

warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"hdbscan\.robust_single_linkage_")

# -------------------------------------------------------------
# Ustawienia strony Streamlit
# -------------------------------------------------------------
st.set_page_config(page_title="Planer Treści SEO", layout="wide")

# --- Pamięć wyników między rerunami ---
if "plan_df" not in st.session_state:
    st.session_state["plan_df"] = None
if "plan_cols" not in st.session_state:
    st.session_state["plan_cols"] = None

# -------------------------------------------------------------
# Narzędzia pomocnicze
# -------------------------------------------------------------
def ensure_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)

# -------------------------------------------------------------
# Funkcje OpenAI / Embeddings / Klasteryzacja
# -------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_openai_embeddings(texts_tuple, api_key, batch_size=256):
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
            time.sleep(0.8)  # delikatny throttle
        except Exception as e:
            st.error(f"Błąd podczas przetwarzania batcha nr {i+1}: {e}")
            return None
    return all_embeddings

def cluster_keywords_hdbscan(keywords_df, embeddings, min_cluster_size=2, min_samples=1):
    if len(keywords_df) == 0:
        return keywords_df

    if 'Volume' not in keywords_df.columns and 'Wolumen' not in keywords_df.columns:
        keywords_df['Wolumen'] = 0
    elif 'Volume' in keywords_df.columns and 'Wolumen' not in keywords_df.columns:
        keywords_df['Wolumen'] = keywords_df['Volume']
    ensure_numeric(keywords_df, ['Wolumen', 'Volume'])

    embeddings_array = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    embeddings_array = embeddings_array / np.clip(norms, 1e-12, None)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )
    cluster_labels = clusterer.fit_predict(embeddings_array)
    keywords_df['Klaster_ID'] = cluster_labels

    outlier_mask = keywords_df['Klaster_ID'] == -1
    if outlier_mask.any():
        max_cluster_id = keywords_df['Klaster_ID'].max()
        start = 0 if max_cluster_id == -1 else (max_cluster_id + 1)
        unique_outlier_ids = range(start, start + outlier_mask.sum())
        keywords_df.loc[outlier_mask, 'Klaster_ID'] = list(unique_outlier_ids)
        keywords_df.loc[outlier_mask, 'Jest_Outlier'] = True

    keywords_df['Jest_Outlier'] = keywords_df.get('Jest_Outlier', False)
    keywords_df['Jest_Outlier'] = keywords_df['Jest_Outlier'].fillna(False)

    probabilities = clusterer.probabilities_ if hasattr(clusterer, 'probabilities_') else np.ones(len(keywords_df))
    keywords_df['Cluster_Probability'] = probabilities

    def get_head_keyword(group):
        if len(group) == 0:
            return group
        keyword_col = 'Keyword' if 'Keyword' in group.columns else 'Słowo kluczowe'
        if len(group) == 1:
            group['HEAD_Keyword'] = group[keyword_col].iloc[0]
            group['Typ_w_klastrze'] = 'HEAD'
            group['Liczba_fraz_w_klastrze'] = 1
            return group
        if 'Wolumen' in group.columns:
            group['_score'] = group['Wolumen'] * (group['Cluster_Probability'] + 0.1)
            head_idx = group['_score'].idxmax()
            group.drop('_score', axis=1, inplace=True)
            group['HEAD_Keyword'] = group.loc[head_idx, keyword_col]
            group['Typ_w_klastrze'] = 'RELATED'
            group.loc[head_idx, 'Typ_w_klastrze'] = 'HEAD'
            group['Liczba_fraz_w_klastrze'] = len(group)
        return group

    keywords_df = keywords_df.groupby('Klaster_ID', group_keys=False).apply(get_head_keyword)

    def calculate_cluster_quality(group):
        if len(group) <= 1:
            return group
        group['Cluster_Quality'] = group['Cluster_Probability'].mean()
        return group

    keywords_df = keywords_df.groupby('Klaster_ID', group_keys=False).apply(calculate_cluster_quality)
    keywords_df['Cluster_Quality'] = keywords_df.get('Cluster_Quality', 1.0)
    return keywords_df

def analyze_cluster_coherence(keywords_df, embeddings):
    if 'Klaster_ID' not in keywords_df.columns:
        return {}
    embeddings_array = np.array(embeddings, dtype=np.float32)
    stats = {
        'total_clusters': keywords_df['Klaster_ID'].nunique(),
        'outliers': int(keywords_df['Jest_Outlier'].sum()),
        'avg_cluster_size': float(keywords_df.groupby('Klaster_ID').size().mean()),
        'max_cluster_size': int(keywords_df.groupby('Klaster_ID').size().max()),
        'clusters_details': []
    }
    for cluster_id in keywords_df['Klaster_ID'].unique():
        cluster_data = keywords_df[keywords_df['Klaster_ID'] == cluster_id]
        if len(cluster_data) <= 1:
            continue
        cluster_indices = cluster_data.index.tolist()
        cluster_embeddings = embeddings_array[cluster_indices]
        similarity_matrix = util.cos_sim(
            torch.tensor(cluster_embeddings),
            torch.tensor(cluster_embeddings)
        ).numpy()
        mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
        avg_similarity = similarity_matrix[mask].mean()
        keyword_col = 'Keyword' if 'Keyword' in cluster_data.columns else 'Słowo kluczowe'
        stats['clusters_details'].append({
            'cluster_id': int(cluster_id),
            'size': int(len(cluster_data)),
            'avg_similarity': round(float(avg_similarity), 3),
            'keywords': cluster_data[keyword_col].tolist()[:5],
            'total_volume': int(cluster_data['Wolumen'].sum())
        })
    return stats

def detect_search_intent(keyword):
    keyword_lower = keyword.lower()
    info_patterns = ['jak', 'co to', 'dlaczego', 'kiedy', 'gdzie', 'czy', 'jakie', 'różnica', 'czym jest']
    trans_patterns = ['kup', 'cena', 'sklep', 'oferta', 'promocja', 'tani', 'najleps', 'gdzie kupić']
    nav_patterns = ['logowanie', 'kontakt', 'strona', 'oficjalna']
    if any(pattern in keyword_lower for pattern in trans_patterns):
        return 'Transakcyjna'
    elif any(pattern in keyword_lower for pattern in nav_patterns):
        return 'Nawigacyjna'
    elif any(pattern in keyword_lower for pattern in info_patterns):
        return 'Informacyjna'
    else:
        return 'Mieszana'

# --- REKOMENDACJA GRUPOWANIA (odporna na NaN/teksty) ---
def _safe_int(x, default=1):
    try:
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default

def _safe_float(x, default=1.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default

# Normalizacja + podobieństwa leksykalne (precyzja dla wariantów)
_PL_MAP = str.maketrans({
    "ą":"a","ć":"c","ę":"e","ł":"l","ń":"n","ó":"o","ś":"s","ż":"z","ź":"z",
    "Ą":"a","Ć":"c","Ę":"e","Ł":"l","Ń":"n","Ó":"o","Ś":"s","Ż":"z","Ź":"z"
})
_STOPWORDS = set("""
do dla w na z za o od u po przy pod nad bez jak co to jest czym
dla dzieci dziecko dzieciaki dzieciecy dziecieca dziecie
""".split())

def _normalize_kw(s: str) -> list[str]:
    s = (s or "").lower().translate(_PL_MAP)
    tokens = re.split(r"[^a-z0-9]+", s)
    tokens = [t for t in tokens if t and t not in _STOPWORDS and len(t) > 1]
    return tokens

def _lexical_sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(a=(a or "").lower(), b=(b or "").lower()).ratio()

def _jaccard_tokens(a: str, b: str) -> float:
    A, B = set(_normalize_kw(a)), set(_normalize_kw(b))
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def _reco(row: pd.Series) -> str:
    size_ = _safe_int(row.get('Liczba_fraz_w_klastrze', np.nan), default=1)
    qual_ = _safe_float(row.get('Cluster_Quality', np.nan), default=1.0)
    kw = str(row.get('Słowo kluczowe', '') or '')
    head = str(row.get('HEAD_Keyword', kw) or '')

    s_lex = _lexical_sim(kw, head)
    s_jac = _jaccard_tokens(kw, head)

    if max(s_lex, s_jac) >= 0.88:
        return "Jeden artykuł"
    if size_ >= 4 and qual_ >= 0.60:
        return "Jeden artykuł"
    if size_ <= 2 or qual_ < 0.45:
        return "Osobne wpisy"
    return "Do decyzji"

def classify_article_type(keyword: str, cluster_size: int, intent: str) -> str:
    k = (keyword or "").lower()
    if any(x in k for x in ["ranking", "najlepsze", "top ", "top-", "top_", "top10", "top 10", "polecane", "lista", "zestawienie"]):
        return "Ranking/Lista"
    if any(x in k for x in [" vs ", " porównanie", "alternatywy", "alternatywa", "zamiennik", "zamienniki", "konkurencja"]):
        return "Porównanie"
    if any(x in k for x in ["opinie", "recenzja", "recenzje"]):
        return "Recenzja/Opinie"
    if any(x in k for x in ["jak ", "jak zrobić", "jak działa"]):
        return "Poradnik/How-to"
    if any(x in k for x in ["co to", "co to jest", "czym jest", "definicja"]):
        return "Definicja/Co to jest"
    if cluster_size >= 5 and intent in ("Mieszana", "Transakcyjna"):
        return "Ranking/Lista"
    return "Temat ogólny"

def calculate_priority_score(row):
    volume_score = min((row.get('Wolumen', 0) or 0) / 1000, 100)
    position = int(row.get('Aktualna_pozycja', 0) or 0)
    if position == 0:
        position_score = 100
    elif position > 20:
        position_score = 80
    elif position > 10:
        position_score = 60
    elif position > 5:
        position_score = 40
    else:
        position_score = 20
    similarity_score = (1 - float(row.get('Podobieństwo', 0) or 0)) * 100
    cluster_quality_score = float(row.get('Cluster_Quality', 1.0) or 1.0) * 100
    priority = (volume_score * 0.4) + (position_score * 0.3) + (similarity_score * 0.2) + (cluster_quality_score * 0.1)
    return round(priority, 2)

def compute_priority_bucket(series: pd.Series) -> pd.Series:
    n = int(series.shape[0])
    if n == 0:
        return series
    ranks = series.rank(method="first", ascending=False)
    step = max(n / 10.0, 1.0)
    buckets = np.ceil(ranks / step)
    return buckets.clip(1, 10)

def find_first_competitor_url(row):
    for col in row.index:
        if isinstance(col, str) and col.endswith(': URL') and pd.notna(row[col]):
            return row[col]
    return "Brak"

def generate_titles(api_key, keyword, volume, competitor_url, related_keywords="", max_retries=3):
    client = openai.OpenAI(api_key=api_key)
    related_info = f"\n- Powiązane frazy do uwzględnienia: {related_keywords}" if related_keywords else ""
    prompt = f"""
Jesteś ekspertem SEO i copywriterem tworzącym tytuły na polskojęzyczne blogi.
Dane wejściowe:
- Główne słowo kluczowe: "{keyword}"  ← UŻYJ DOKŁADNIE TEJ FRAZY (bez zmian).
- Miesięczny wolumen wyszukiwania: {volume}{related_info}
- Artykuł konkurencji (tylko inspiracja kontekstu): {competitor_url}

Zasady:
1) Każdy tytuł MUSI zawierać dokładnie frazę: "{keyword}" (bez modyfikacji i literówek).
2) Styl informacyjny/poradnikowy: „co to jest…”, „jak…”, „poradnik…”, „ranking…”, „porównanie…” — dobierz naturalnie do intencji.
3) Pisownia zgodna z PL. Unikaj dwukropków; używaj myślnika.
4) Nie wprowadzaj nowych wariantów nazwy własnej/branda; nie poprawiaj frazy wejściowej.
5) Zwróć WYŁĄCZNIE listę numerowaną 1..3 z 3 różnymi propozycjami.
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

            _kw = str(keyword).strip()
            _kw_low = _kw.lower()
            _clean = []
            for t in (titles[:3] if titles else []):
                t = (t or "").strip()
                if _kw_low not in t.lower():
                    if t:
                        t = f"{t} — {_kw}"
                    else:
                        t = f"{_kw} — co to jest i jak działa?"
                _clean.append(t)
            while len(_clean) < 3:
                _clean.append(f"{_kw} — poradnik dla początkujących")
            return _clean[:3]

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            else:
                st.warning(f"Błąd podczas generowania tytułów dla '{keyword}': {e}")
                return ["Błąd API", "Błąd API", "Błąd API"]

def validate_api_key(api_key):
    try:
        client = openai.OpenAI(api_key=api_key)
        client.models.list()
        return True
    except Exception as e:
        st.error(f"Nieprawidłowy klucz API OpenAI: {e}")
        return False

# -------------------------------------------------------------
# UI
# -------------------------------------------------------------
st.title("🚀 Planer Treści SEO [Wersja HDBSCAN v10 - Skondensowana]")
st.markdown("✨ Zaawansowana klasteryzacja z HDBSCAN, agregacja fraz i pytań do tematów głównych.")

col1, col2 = st.columns(2)
with col1:
    st.header("1. Konfiguracja")
    num_to_generate = st.number_input("Liczba nowych artykułów do wygenerowania", min_value=1, value=20)
    similarity_threshold = st.slider("Próg podobieństwa dla optymalizacji", min_value=0.7, max_value=1.0, value=0.8, step=0.01)

    with st.expander("⚙️ Zaawansowane ustawienia klasteryzacji"):
        min_cluster_size = st.slider("Minimalna wielkość klastra", min_value=2, max_value=10, value=2,
                                     help="Im wyższa wartość, tym większe i bardziej ogólne klastry. 2–3 zalecane dla SEO.")
        min_samples = st.slider("Minimalna liczba próbek w sąsiedztwie", min_value=1, max_value=5, value=1,
                                help="Im wyższa wartość, tym bardziej konserwatywna klasteryzacja. 1–2 zalecane.")
    enable_clustering = st.checkbox("Włącz klasteryzację fraz kluczowych (HDBSCAN)", value=True)

with col2:
    st.header("2. Wgraj pliki CSV")
    content_gap_file = st.file_uploader("1. Wgraj plik CSV z analizą Content Gap", type="csv")
    my_articles_file = st.file_uploader("2. Wgraj plik CSV z listą swoich artykułów", type="csv")
    ranking_file = st.file_uploader("3. Wgraj plik CSV z aktualnym rankingiem", type="csv")


# -------------------------------------------------------------
# Logika Aplikacji
# -------------------------------------------------------------
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

            position_col = None
            for col in df_ranking.columns:
                if 'pozycj' in col.lower() or 'position' in col.lower():
                    position_col = col
                    break

            ensure_numeric(df_gap, ['Volume', 'Wolumen'])
            if 'Wolumen' not in df_gap.columns and 'Volume' in df_gap.columns:
                df_gap['Wolumen'] = df_gap['Volume']

            if position_col:
                df_ranking[position_col] = pd.to_numeric(df_ranking[position_col], errors='coerce').fillna(0).astype(int)

            st.info(f"Wczytano {len(df_gap)} słów kluczowych, {len(df_articles)} artykułów i {len(df_ranking)} rankingowych słów kluczowych.")
        except Exception as e:
            st.error(f"Błąd podczas wczytywania plików CSV: {e}")
            st.stop()

        ranking_map = {}
        for _, row in df_ranking.iterrows():
            keyword = str(row['Słowo kluczowe']).lower()
            url = row['Adres URL']
            position = row[position_col] if position_col and pd.notna(row.get(position_col)) else 0
            ranking_map[keyword] = {'url': url, 'position': position}

        results = []
        keywords_for_semantic_check = []

        for _, row in df_gap.iterrows():
            keyword_lower = str(row['Keyword']).lower()
            if keyword_lower in ranking_map:
                rank_data = ranking_map[keyword_lower]
                position = int(rank_data['position'])
                if position == 0:
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
                    'Wolumen': int(row.get('Volume', row.get('Wolumen', 0)) or 0),
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
            if 'Volume' in df_semantic.columns and 'Wolumen' not in df_semantic.columns:
                df_semantic['Wolumen'] = df_semantic['Volume']
            elif 'Wolumen' not in df_semantic.columns:
                df_semantic['Wolumen'] = 0
            ensure_numeric(df_semantic, ['Wolumen', 'Volume'])
            if 'Keyword' in df_semantic.columns and 'Słowo kluczowe' not in df_semantic.columns:
                df_semantic['Słowo kluczowe'] = df_semantic['Keyword']

            st.info("🔄 Generowanie embeddingów dla artykułów...")
            corpus_embeddings = get_openai_embeddings(tuple(df_articles['Title'].tolist()), openai_api_key)

            st.info("🔄 Generowanie embeddingów dla słów kluczowych...")
            query_embeddings = get_openai_embeddings(
                tuple(df_semantic['Keyword' if 'Keyword' in df_semantic.columns else 'Słowo kluczowe'].tolist()),
                openai_api_key
            )

            if not corpus_embeddings or not query_embeddings:
                st.error("Nie udało się wygenerować wektorów. Sprawdź klucz API i spróbuj ponownie.")
                st.stop()

            if enable_clustering:
                st.info("🔄 Klasteryzuję frazy kluczowe z HDBSCAN...")
                df_semantic = cluster_keywords_hdbscan(
                    df_semantic, query_embeddings,
                    min_cluster_size=min_cluster_size, min_samples=min_samples
                )
                cluster_stats = analyze_cluster_coherence(df_semantic, query_embeddings)

            cosine_scores = util.cos_sim(torch.tensor(query_embeddings), torch.tensor(corpus_embeddings))
            semantic_records = df_semantic.to_dict('records')
            for row_dict, scores in zip(semantic_records, cosine_scores):
                top_result = torch.topk(scores, k=1)
                score, corpus_idx = top_result.values[0].item(), top_result.indices[0].item()
                closest_article_url = df_articles.iloc[corpus_idx]['URL']
                closest_article_title = df_articles.iloc[corpus_idx]['Title']
                keyword = row_dict.get('Keyword', row_dict.get('Słowo kluczowe', ''))
                volume = int(row_dict.get('Volume', row_dict.get('Wolumen', 0)) or 0)
                if score > similarity_threshold:
                    status = f'Do optymalizacji (Podobne do: {closest_article_title[:50]}...)'
                    url = closest_article_url
                else:
                    status = 'Nowy temat'
                    url = 'Stwórz nowy artykuł'
                intent = detect_search_intent(keyword)

                result = {
                    'Słowo kluczowe': keyword,
                    'Wolumen': volume,
                    'Status': status,
                    'Akcja / Dopasowany URL': url,
                    'Najbliższy_artykuł': closest_article_url,
                    'Najbliższy_tytuł': closest_article_title,
                    'Podobieństwo': round(float(score), 2),
                    'Intencja': intent,
                    'Aktualna_pozycja': 0
                }

                if enable_clustering:
                    result['Klaster_ID'] = row_dict.get('Klaster_ID', 0)
                    result['HEAD_Keyword'] = row_dict.get('HEAD_Keyword', keyword)
                    result['Typ_w_klastrze'] = row_dict.get('Typ_w_klastrze', 'HEAD')
                    result['Liczba_fraz_w_klastrze'] = row_dict.get('Liczba_fraz_w_klastrze', 1)
                    result['Jest_Outlier'] = bool(row_dict.get('Jest_Outlier', False))
                    result['Cluster_Probability'] = round(float(row_dict.get('Cluster_Probability', 1.0)), 3)
                    result['Cluster_Quality'] = round(float(row_dict.get('Cluster_Quality', 1.0)), 3)

                _cluster_size_for_type = int(result.get('Liczba_fraz_w_klastrze', 1) or 1)
                result['Typ_artykułu'] = classify_article_type(keyword, _cluster_size_for_type, intent)

                results.append(result)

        df_results = pd.DataFrame(results)

        # Agenda: score → bucket 1..10
        df_results['Priorytet_Score'] = df_results.apply(calculate_priority_score, axis=1)
        df_results['Priorytet'] = compute_priority_bucket(df_results['Priorytet_Score']).astype('Int64')

        # HEAD jako nazwa grupy tematycznej
        if 'HEAD_Keyword' in df_results.columns:
            df_results['Grupa_tematyczna'] = df_results['HEAD_Keyword']
        else:
            df_results['Grupa_tematyczna'] = df_results['Słowo kluczowe']

        # Rekomendacja grupowania (przed czyszczeniem typów)
        if 'HEAD_Keyword' in df_results.columns:
            df_results['Rekomendacja_grupowania'] = df_results.apply(_reco, axis=1)

        # =================================================================
        # NOWA SEKCJA: Agregacja fraz RELATED i Pytań do wierszy HEAD
        # =================================================================
        if enable_clustering and 'Klaster_ID' in df_results.columns:
            st.info("🔄 Agreguję frazy RELATED i pytania do głównych tematów (HEAD)...")

            aggregated_data = []
            for cluster_id, group in df_results.groupby('Klaster_ID'):
                head_row = group[group['Typ_w_klastrze'] == 'HEAD']
                related_rows = group[group['Typ_w_klastrze'] == 'RELATED']

                if not head_row.empty:
                    head_keyword = head_row.iloc[0]['Słowo kluczowe']
                    related_keywords_sorted = related_rows.sort_values('Wolumen', ascending=False)['Słowo kluczowe'].tolist()
                    related_keywords_str = "\n".join(related_keywords_sorted)
                    questions = group[group['Intencja'] == 'Informacyjna'].sort_values('Wolumen', ascending=False)['Słowo kluczowe'].tolist()
                    if head_keyword in questions:
                        questions.remove(head_keyword)
                    questions_str = "\n".join(questions)
                    aggregated_data.append({
                        'Słowo kluczowe': head_keyword,
                        'Frazy_RELATED': related_keywords_str,
                        'Powiazane_pytania': questions_str,
                        'Calkowity_wolumen_klastra': group['Wolumen'].sum()
                    })

            if aggregated_data:
                df_aggregated = pd.DataFrame(aggregated_data)
                df_results = pd.merge(df_results, df_aggregated, on='Słowo kluczowe', how='left')
                df_results = df_results[df_results['Typ_w_klastrze'] == 'HEAD'].reset_index(drop=True)
                if 'Calkowity_wolumen_klastra' in df_results.columns:
                     df_results['Wolumen'] = df_results['Calkowity_wolumen_klastra']
                     df_results['Priorytet_Score'] = df_results.apply(calculate_priority_score, axis=1)
                     df_results['Priorytet'] = compute_priority_bucket(df_results['Priorytet_Score']).astype('Int64')
        # =================================================================

        # --- Ujednolicenie typów i uzupełnienie braków (zawsze) ---
        force_int_cols = ['Wolumen', 'Aktualna_pozycja', 'Liczba_fraz_w_klastrze', 'Klaster_ID', 'Priorytet']
        for c in force_int_cols:
            if c in df_results.columns:
                df_results[c] = pd.to_numeric(df_results[c], errors='coerce').fillna(0).astype(int)

        force_float_cols = ['Podobieństwo', 'Cluster_Probability', 'Cluster_Quality', 'Priorytet_Score']
        for c in force_float_cols:
            if c in df_results.columns:
                df_results[c] = pd.to_numeric(df_results[c], errors='coerce')
                df_results[c] = df_results[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        if 'Jest_Outlier' in df_results.columns:
            df_results['Jest_Outlier'] = df_results['Jest_Outlier'].fillna(False).astype(bool)

        obj_cols = df_results.select_dtypes(include=['object']).columns
        df_results[obj_cols] = df_results[obj_cols].fillna('-')


        # ===== Generowanie tytułów dla TOP N nowych tematów (HEAD) =====
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
                # Używamy już zagregowanych fraz
                related_keywords = row.get('Frazy_RELATED', '').replace("\n", ", ")
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

                with st.expander(f"📝 Propozycje tytułów (TOP {len(df_to_process)})", expanded=True):
                    preview_cols = ['Słowo kluczowe', 'Wolumen',
                                    'Propozycja_tematu_1', 'Propozycja_tematu_2', 'Propozycja_tematu_3']
                    _titles_preview = df_results.loc[
                        df_results['Słowo kluczowe'].isin(df_to_process['Słowo kluczowe'])
                    ][preview_cols].sort_values('Wolumen', ascending=False)
                    st.dataframe(_titles_preview, use_container_width=True)

        st.success("✅ Analiza zakończona!")

        # Szybkie metryki
        colm1, colm2, colm3, colm4 = st.columns(4)
        with colm1:
            st.metric("Nowe tematy", len(df_results[df_results['Status'] == 'Nowy temat']))
        with colm2:
            st.metric("Do optymalizacji", len(df_results[df_results['Status'].str.contains('optymalizacji', na=False)]))
        with colm3:
            if enable_clustering and 'Klaster_ID' in df_results.columns:
                st.metric("Liczba klastrów", df_results['Klaster_ID'].nunique())
        with colm4:
            st.metric("Średni Priorytet (1=best)", f"{df_results['Priorytet'].mean():.1f}")

        st.header("📊 Wyniki Analizy i Plan Treści")

        df_results_sorted = df_results.sort_values(
            by=['Priorytet', 'Priorytet_Score', 'Wolumen'],
            ascending=[True, False, False]
        )

        show_advanced = st.checkbox(
            "Pokaż metryki techniczne (HDBSCAN, outliery, podobieństwo)",
            value=False,
            key="show_adv"
        )

        # Definiowanie kolejności kolumn z nowymi kolumnami
        cols_order = [
            'Priorytet',
            'Słowo kluczowe', 'Wolumen', 'Status',
            'Grupa_tematyczna', 'Typ_artykułu', 'Rekomendacja_grupowania',
            'Frazy_RELATED', 'Powiazane_pytania', # NOWE KOLUMNY
            'Akcja / Dopasowany URL', 'Najbliższy_artykuł', 'Podobieństwo',
            'Intencja', 'Aktualna_pozycja'
        ]
        if show_advanced:
            cols_order.extend([
                'Klaster_ID', 'Liczba_fraz_w_klastrze', 'Jest_Outlier',
                'Cluster_Probability', 'Cluster_Quality', 'Priorytet_Score'
            ])
        cols_order.extend(['Propozycja_tematu_1', 'Propozycja_tematu_2', 'Propozycja_tematu_3'])


        existing_cols = [c for c in cols_order if c in df_results_sorted.columns]

        def highlight_rows(row):
            s = str(row['Status'])
            if s == 'Nowy temat':
                return ['background-color: #e8f5e9'] * len(row)
            elif 'TOP1' in s:
                return ['background-color: #fff3e0'] * len(row)
            elif 'Nie rankuje' in s:
                return ['background-color: #ffebee'] * len(row)
            return [''] * len(row)

        display_df = df_results_sorted.copy()

        if 'Jest_Outlier' in display_df.columns:
            display_df['Jest_Outlier'] = display_df['Jest_Outlier'].map({True: 'TAK', False: ''})

        _obj_cols = display_df.select_dtypes(include=['object']).columns
        display_df[_obj_cols] = display_df[_obj_cols].fillna('-')

        st.session_state["plan_df"] = display_df.copy()
        st.session_state["plan_cols"] = existing_cols[:]

        st.dataframe(
            display_df[existing_cols].style.apply(highlight_rows, axis=1),
            width='stretch',
            height=600
        )

        st.markdown("""
**Legenda i wskazówki:**
- **Priorytet**: 1 = najwyższy, 10 = najniższy (agenda działań).
- **Grupa tematyczna**: Główny temat (`HEAD`) dla całego klastra.
- **Frazy_RELATED**: Pozostałe frazy z klastra do uwzględnienia w treści (posortowane wg wolumenu).
- **Powiązane_pytania**: Frazy o intencji informacyjnej, które warto zawrzeć w artykule jako sekcje FAQ lub nagłówki.
- 🟢 Zielony wiersz: Nowy temat
- 🟠 Pomarańczowy: Blisko TOP1 (optymalizacja)
- 🔴 Czerwony: Nie rankuje
""")

        # Uproszczony eksport CSV
        csv_buffer = io.StringIO()
        display_df[existing_cols].to_csv(csv_buffer, index=False, encoding='utf-8')
        csv_bytes = csv_buffer.getvalue().encode('utf-8-sig')
        st.download_button(
            "📥 Pobierz gotowy plan treści jako CSV",
            csv_bytes,
            "plan_tresci_hdbscan_skondensowany.csv",
            "text/csv",
            type="primary"
        )

        # Usunięto dodatkowe przyciski pobierania dla uproszczenia
        # ...

        # Dodatkowa analiza
        st.header("📊 Dodatkowa Analiza")
        tab1, tab2, tab3 = st.tabs(["Rozkład intencji", "Analiza wolumenu", "Jakość klastrów"])
        with tab1:
            if 'Intencja' in df_results.columns:
                intent_counts = df_results['Intencja'].value_counts()
                st.bar_chart(intent_counts)
        with tab2:
            volume_by_status = df_results.groupby('Status')['Wolumen'].sum().sort_values(ascending=False)
            st.bar_chart(volume_by_status)
        with tab3:
            if enable_clustering and 'Cluster_Quality' in df_results.columns:
                quality_dist = pd.to_numeric(df_results['Cluster_Quality'], errors='coerce').dropna()
                if not quality_dist.empty:
                    st.line_chart(quality_dist.sort_values(ascending=False))
                    st.markdown(f"**Średnia jakość klastrów:** {quality_dist.mean():.3f}")
                weak_clusters = df_results[
                    (pd.to_numeric(df_results['Cluster_Quality'], errors='coerce') < 0.5)
                ]['Słowo kluczowe'].dropna().tolist()
                if weak_clusters:
                    st.warning(f"⚠️ Znaleziono {len(weak_clusters)} klastrów o niskiej spójności.")
                    with st.expander("Zobacz listę"):
                        for kw in weak_clusters[:10]: st.markdown(f"- {kw}")
