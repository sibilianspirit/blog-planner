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
import gspread
import warnings

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
def colnum_to_letter(n: int) -> str:
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(r + 65) + s
    return s

def ensure_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)

def _get_gspread_client():
    """Autoryzacja gspread z secrets + fix dla \\n"""
    try:
        creds_dict = st.secrets["gcp_service_account"]
    except Exception:
        st.error("Brak poświadczeń Google w st.secrets['gcp_service_account']. Dodaj JSON konta serwisowego do sekretów.")
        return None
    pk = creds_dict.get("private_key", "")
    if "\\n" in pk and "\n" not in pk:
        creds_dict["private_key"] = pk.replace("\\n", "\n")
    try:
        return gspread.service_account_from_dict(creds_dict)
    except Exception as e:
        st.error(f"Nie można zautoryzować gspread: {e}")
        return None

def _extract_spreadsheet_id(maybe_url_or_id: str) -> str:
    s = maybe_url_or_id.strip()
    if not s:
        return ""
    if "https://docs.google.com" in s:
        try:
            return s.split("/d/")[1].split("/")[0]
        except Exception:
            return s
    return s

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

def _reco(row: pd.Series) -> str:
    size_ = _safe_int(row.get('Liczba_fraz_w_klastrze', np.nan), default=1)
    qual_ = _safe_float(row.get('Cluster_Quality', np.nan), default=1.0)
    if size_ >= 4 and qual_ >= 0.60:
        return "Jeden artykuł"
    if size_ <= 2 or qual_ < 0.45:
        return "Osobne wpisy"
    return "Do decyzji"

def classify_article_type(keyword: str, cluster_size: int, intent: str) -> str:
    """
    Zwraca typ treści: 'Ranking/Lista', 'Porównanie', 'Poradnik/How-to',
    'Definicja/Co to jest', 'Recenzja/Opinie', 'Temat ogólny'.
    """
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
    """
    Na bazie Priorytet_Score tworzy priorytet 1..10 (1 = najwyższy).
    Używa decyli; remisy trzyma spójnie.
    """
    if series.empty:
        return series
    ranks = series.rank(method="min", ascending=False)
    deciles = pd.qcut(ranks, 10, labels=False, duplicates='drop')  # 0..9 (lub mniej przy małych próbach)
    return (deciles + 1)

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

            # Wymuś obecność dokładnej frazy w każdym tytule
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
# Eksport do Google Sheets (update istniejącego lub create do folderu)
# -------------------------------------------------------------
def export_df_to_google_sheets_with_colors(
    df: pd.DataFrame,
    columns_in_order,
    title="Plan treści HDBSCAN",
    existing_spreadsheet_id_or_url: str = "",
    folder_id: str = ""
):
    """
    Jeśli podasz existing_spreadsheet_id_or_url -> zapisze do tego pliku (tworzy/aktualizuje zakładkę 'Plan').
    W przeciwnym razie spróbuje utworzyć nowy plik (jeśli podano folder_id, to w tym folderze).
    """
    gc = _get_gspread_client()
    if gc is None:
        return None

    sh = None
    ss_id = _extract_spreadsheet_id(existing_spreadsheet_id_or_url)

    try:
        if ss_id:
            sh = gc.open_by_key(ss_id)
        else:
            if folder_id.strip():
                sh = gc.create(title, folder_id=folder_id.strip())
            else:
                sh = gc.create(title)
    except Exception as e:
        st.error(f"Nie można otworzyć/utworzyć Spreadsheet: {e}")
        return None

    # Worksheet "Plan"
    try:
        try:
            ws = sh.worksheet("Plan")
            ws.clear()
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title="Plan", rows="1000", cols="26")
    except Exception as e:
        st.error(f"Problem z arkuszem 'Plan': {e}")
        return None

    # Dane
    try:
        data = df[columns_in_order].copy()
        # Zamień Inf/NaN na puste stringi - Google Sheets/JSON nie akceptuje NaN/Inf
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.where(pd.notna(data), "")
        ws.update([data.columns.tolist()] + data.astype(object).values.tolist())
    except Exception as e:
        st.error(f"Nie udało się zaktualizować danych w arkuszu: {e}")
        return None

    # Formatowanie warunkowe wg 'Status'
    try:
        n_rows = data.shape[0] + 1
        n_cols = data.shape[1]
        if 'Status' not in data.columns:
            st.warning("Nie znaleziono kolumny 'Status' – pomijam kolorowanie.")
            return sh.url

        status_col_idx_1 = data.columns.get_loc('Status') + 1
        sheet_id = ws._properties['sheetId']
        table_range = {
            "sheetId": sheet_id,
            "startRowIndex": 1,
            "endRowIndex": n_rows,
            "startColumnIndex": 0,
            "endColumnIndex": n_cols
        }
        col_green = {"red": 0.91, "green": 0.96, "blue": 0.91}
        col_orange = {"red": 1.00, "green": 0.95, "blue": 0.88}
        col_red = {"red": 1.00, "green": 0.92, "blue": 0.93}

        status_col_letter = colnum_to_letter(status_col_idx_1)
        base_cell = f"${status_col_letter}2"

        requests = [
            # Auto-resize kolumn
            {
                "autoResizeDimensions": {
                    "dimensions": {
                        "sheetId": sheet_id,
                        "dimension": "COLUMNS",
                        "startIndex": 0,
                        "endIndex": n_cols
                    }
                }
            },
            # Zielone tło dla dokładnie "Nowy temat"
            {
                "addConditionalFormatRule": {
                    "rule": {
                        "ranges": [table_range],
                        "booleanRule": {
                            "condition": {
                                "type": "CUSTOM_FORMULA",
                                "values": [
                                    {"userEnteredValue": f"={base_cell}=\"Nowy temat\""}
                                ]
                            },
                            "format": {"backgroundColor": col_green}
                        }
                    },
                    "index": 0
                }
            },
            # Pomarańczowe tło gdy Status zawiera "TOP1"
            {
                "addConditionalFormatRule": {
                    "rule": {
                        "ranges": [table_range],
                        "booleanRule": {
                            "condition": {
                                "type": "TEXT_CONTAINS",
                                "values": [
                                    {"userEnteredValue": "TOP1"}
                                ]
                            },
                            "format": {"backgroundColor": col_orange}
                        }
                    },
                    "index": 0
                }
            },
            # Czerwone tło gdy Status zawiera "Nie rankuje"
            {
                "addConditionalFormatRule": {
                    "rule": {
                        "ranges": [table_range],
                        "booleanRule": {
                            "condition": {
                                "type": "TEXT_CONTAINS",
                                "values": [
                                    {"userEnteredValue": "Nie rankuje"}
                                ]
                            },
                            "format": {"backgroundColor": col_red}
                        }
                    },
                    "index": 0
                }
            }
        ]

        sh.batch_update({"requests": requests})
    except Exception as e:
        st.warning(f"Nie udało się ustawić formatowania warunkowego: {e}")

    return sh.url

# -------------------------------------------------------------
# UI
# -------------------------------------------------------------
st.title("🚀 Planer Treści SEO [Wersja HDBSCAN v9 - ULTIMATE]")
st.markdown("✨ Zaawansowana klasteryzacja z HDBSCAN, analiza intencji, agenda priorytetów 1–10 oraz propozycje formatów treści.")

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

# Pola dla eksportu (używane też przez test połączenia)
st.markdown("### ☁️ Ustawienia eksportu do Google Sheets")
existing_sheet_input = st.text_input(
    "Istniejący Spreadsheet (URL lub ID) – zalecane, gdy brakuje miejsca w Drive",
    value="",
    key="existing_spreadsheet_id"
)
target_folder_id = st.text_input(
    "Opcjonalnie: Folder ID (jeśli chcesz tworzyć nowe pliki w konkretnym folderze z wolnym miejscem)",
    value="",
    key="target_folder_id"
)

# --- PRZYCISK TESTOWY: szybkie sprawdzenie połączenia i uprawnień ---
if st.button("🔍 Test połączenia z Google Sheets (A1 -> timestamp)", key="gs_probe"):
    try:
        gc = _get_gspread_client()
        if gc is None:
            st.stop()
        sheet_id_or_url = existing_sheet_input.strip()
        if not sheet_id_or_url:
            st.error("Podaj najpierw Istniejący Spreadsheet (URL lub ID), aby wykonać test.")
        else:
            spreadsheet_id = _extract_spreadsheet_id(sheet_id_or_url)
            sh = gc.open_by_key(spreadsheet_id)
            try:
                ws = sh.worksheet("Plan")
            except gspread.WorksheetNotFound:
                ws = sh.add_worksheet(title="Plan", rows="1000", cols="26")
            import datetime as _dt
            ws.update_acell("A1", f"Połączenie OK @ {_dt.datetime.now().isoformat(timespec='seconds')}")
            st.success(f"OK ✅ Zapisano timestamp do A1. Arkusz: {sh.url}")
    except Exception as e:
        st.exception(e)

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

            st.info(f"Wczytano {len(df_gap)} słów kluczowych, {len[df_articles]} artykułów i {len(df_ranking)} rankingowych słów kluczowych.")
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

        # Rekomendacja grupowania (po czyszczeniu typów)
        df_results['Rekomendacja_grupowania'] = df_results.apply(_reco, axis=1)

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

        # Sort: najpierw agenda (1..10), potem score i wolumen
        df_results_sorted = df_results.sort_values(
            by=['Priorytet', 'Priorytet_Score', 'Wolumen'],
            ascending=[True, False, False]
        )

        # Widok domyślny – biznesowy
        show_advanced = st.checkbox(
            "Pokaż metryki techniczne (HDBSCAN, outliery, prawdopodobieństwo)",
            value=False,
            key="show_adv"
        )

        # =================== (UWAGA: ten blok musi być wewnątrz if-button) ===================
        cols_order = [
            'Priorytet',
            'Słowo kluczowe', 'Wolumen', 'Status',
            'Grupa_tematyczna', 'Typ_artykułu', 'Rekomendacja_grupowania',
            'Akcja / Dopasowany URL', 'Najbliższy_artykuł', 'Podobieństwo',
            'Intencja', 'Aktualna_pozycja'
        ]
        if show_advanced:
            cols_order.extend([
                'Klaster_ID', 'HEAD_Keyword', 'Typ_w_klastrze', 'Liczba_fraz_w_klastrze',
                'Frazy_w_klastrze_TOP5',
                'Jest_Outlier', 'Cluster_Probability', 'Cluster_Quality', 'Priorytet_Score'
            ])
        cols_order.extend(['Propozycja_tematu_1', 'Propozycja_tematu_2', 'Propozycja_tematu_3'])
        # =====================================================================================

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

        # Przygotuj wersję do WYŚWIETLENIA
        display_df = df_results_sorted.copy()

        # Frazy_w_klastrze_TOP5 (podgląd — zbuduj dopiero tutaj, aby mieć pełny set)
        if enable_clustering and 'Klaster_ID' in display_df.columns and 'Frazy_w_klastrze_TOP5' not in display_df.columns:
            top5_map = {}
            for cid, g in display_df.groupby('Klaster_ID'):
                try:
                    gg = g.sort_values('Wolumen', ascending=False)
                except Exception:
                    gg = g
                top5_map[cid] = ", ".join(gg['Słowo kluczowe'].tolist()[:5])
            display_df['Frazy_w_klastrze_TOP5'] = display_df['Klaster_ID'].map(top5_map).fillna('-')
        elif 'Frazy_w_klastrze_TOP5' not in display_df.columns:
            display_df['Frazy_w_klastrze_TOP5'] = '-'

        if 'Jest_Outlier' in display_df.columns:
            display_df['Jest_Outlier'] = display_df['Jest_Outlier'].map({True: 'TAK', False: ''})

        _obj_cols = display_df.select_dtypes(include=['object']).columns
        display_df[_obj_cols] = display_df[_obj_cols].fillna('-')

        # Zapisz do session_state, żeby nie znikło po kliknięciu innych przycisków
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
- **Grupa tematyczna**: nazwa przewodnia (HEAD) dla całego klastra.
- **Rekomendacja grupowania**:
  - *Jeden artykuł* – frazy są spójne; zaplanuj jeden, mocny materiał.
  - *Osobne wpisy* – frazy się rozchodzą; rozbij na kilka treści.
  - *Do decyzji* – spójność umiarkowana; zależy od strategii i zasobów.
- 🟢 Zielony wiersz: Nowy temat
- 🟠 Pomarańczowy: Blisko TOP1 (optymalizacja)
- 🔴 Czerwony: Nie rankuje
""")

        # Eksport CSV (natychmiast)
        csv_buffer = io.StringIO()
        display_df[existing_cols].to_csv(csv_buffer, index=False, encoding='utf-8')
        csv_bytes = csv_buffer.getvalue().encode('utf-8-sig')
        st.download_button(
            "📥 Pobierz gotowy plan treści jako CSV",
            csv_bytes,
            "plan_tresci_hdbscan_ultimate.csv",
            "text/csv",
            type="primary"
        )

        # Dodatkowe eksporty
        colx1, colx2 = st.columns(2)
        with colx1:
            if 'Typ_w_klastrze' in df_results_sorted.columns:
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
        with colx2:
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
            if enable_clustering and 'Cluster_Quality' in df_results.columns and 'Typ_w_klastrze' in df_results.columns:
                quality_dist = pd.to_numeric(
                    df_results.loc[df_results['Typ_w_klastrze'] == 'HEAD', 'Cluster_Quality'],
                    errors='coerce'
                ).dropna()
                if not quality_dist.empty:
                    st.line_chart(quality_dist.sort_values(ascending=False))
                    st.markdown(f"**Średnia jakość klastrów:** {quality_dist.mean():.3f}")
                weak_clusters = df_results[
                    (df_results['Typ_w_klastrze'] == 'HEAD') &
                    (pd.to_numeric(df_results['Cluster_Quality'], errors='coerce') < 0.5)
                ]['Słowo kluczowe'].dropna().tolist()
                if weak_clusters:
                    st.warning(f"⚠️ Znaleziono {len(weak_clusters)} klastrów o niskiej spójności. Rozważ ich weryfikację ręczną.")
                    with st.expander("Zobacz listę"):
                        for kw in weak_clusters[:10]:
                            st.markdown(f"- {kw}")

# =========================================
# STAŁA SEKCJA EKSPORTU / POBIERANIA (poza analizą)
# =========================================
st.subheader("☁️ Eksport / Pobieranie")
if st.session_state.get("plan_df") is None or st.session_state.get("plan_cols") is None:
    st.info("Najpierw uruchom analizę, żeby włączyć eksport i pobieranie.")
else:
    gs_title_default = f"Plan treści HDBSCAN ({time.strftime('%Y-%m-%d %H:%M')})"
    gs_title = st.text_input("Tytuł nowego arkusza (dla NOWEGO pliku)", value=gs_title_default, key="gs_title")

    export_df = st.session_state["plan_df"][st.session_state["plan_cols"]].copy()
    csv_buffer2 = io.StringIO()
    export_df.to_csv(csv_buffer2, index=False, encoding='utf-8')
    csv_bytes2 = csv_buffer2.getvalue().encode('utf-8-sig')
    st.download_button(
        "📥 Pobierz aktualny plan (CSV)",
        data=csv_bytes2,
        file_name="plan_tresci_hdbscan_ultimate.csv",
        mime="text/csv",
        type="primary",
        key="dl_csv_latest"
    )

    if st.button("Wyślij do Google Sheets (z kolorami)", key="export_gs"):
        try:
            with st.spinner("Wysyłam dane do Google Sheets i ustawiam kolorowanie..."):
                url = export_df_to_google_sheets_with_colors(
                    st.session_state["plan_df"],
                    st.session_state["plan_cols"],
                    title=gs_title,
                    existing_spreadsheet_id_or_url=existing_sheet_input,
                    folder_id=target_folder_id
                )
            if url:
                st.success(f"Gotowe! Otwórz arkusz: {url}")
            else:
                st.error("Eksport nie zwrócił adresu URL. Sprawdź logi/sekrety.")
        except Exception as e:
            st.exception(e)
