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
import difflib

# ===== Wycisz wybrane FutureWarningi (3rd party) =====
warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"hdbscan\.robust_single_linkage_")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn\.utils\.deprecation")

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
# OpenAI / Embeddings / Klasteryzacja
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
            time.sleep(0.8)
        except Exception as e:
            st.error(f"Błąd podczas przetwarzania batcha nr {i+1}: {e}")
            return None
    return all_embeddings

def cluster_keywords_hdbscan(keywords_df, embeddings, min_cluster_size=2, min_samples=1):
    if len(keywords_df) == 0:
        return keywords_df

    # Ujednolicenie wolumenu
    if 'Volume' not in keywords_df.columns and 'Wolumen' not in keywords_df.columns:
        keywords_df['Wolumen'] = 0
    elif 'Volume' in keywords_df.columns and 'Wolumen' not in keywords_df.columns:
        keywords_df['Wolumen'] = keywords_df['Volume']
    ensure_numeric(keywords_df, ['Wolumen', 'Volume'])

    # Normalizacja wektorów
    embeddings_array = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    embeddings_array = embeddings_array / np.clip(norms, 1e-12, None)

    # HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )
    cluster_labels = clusterer.fit_predict(embeddings_array)
    keywords_df = keywords_df.copy()
    keywords_df['Klaster_ID'] = cluster_labels

    # -1 → outliery; zamień na nowe, unikalne ID
    outlier_mask = keywords_df['Klaster_ID'] == -1
    if outlier_mask.any():
        max_cluster_id = keywords_df['Klaster_ID'].max()
        start = 0 if max_cluster_id == -1 else (max_cluster_id + 1)
        unique_outlier_ids = range(start, start + outlier_mask.sum())
        keywords_df.loc[outlier_mask, 'Klaster_ID'] = list(unique_outlier_ids)
        if 'Jest_Outlier' not in keywords_df.columns:
            keywords_df['Jest_Outlier'] = False
        keywords_df.loc[outlier_mask, 'Jest_Outlier'] = True

    if 'Jest_Outlier' not in keywords_df.columns:
        keywords_df['Jest_Outlier'] = False
    keywords_df['Jest_Outlier'] = keywords_df['Jest_Outlier'].fillna(False)

    probabilities = clusterer.probabilities_ if hasattr(clusterer, 'probabilities_') else np.ones(len(keywords_df))
    keywords_df['Cluster_Probability'] = probabilities

    # HEAD w klastrze
    def get_head_keyword(group):
        if len(group) == 0:
            return group
        keyword_col = 'Keyword' if 'Keyword' in group.columns else 'Słowo kluczowe'
        group = group.copy()
        if len(group) == 1:
            group['HEAD_Keyword'] = group[keyword_col].iloc[0]
            group['Typ_w_klastrze'] = 'HEAD'
            group['Liczba_fraz_w_klastrze'] = 1
            return group
        group['_score'] = group['Wolumen'] * (group['Cluster_Probability'] + 0.1)
        head_idx = group['_score'].idxmax()
        group.drop('_score', axis=1, inplace=True)
        group['HEAD_Keyword'] = group.loc[head_idx, keyword_col]
        group['Typ_w_klastrze'] = 'RELATED'
        group.loc[head_idx, 'Typ_w_klastrze'] = 'HEAD'
        group['Liczba_fraz_w_klastrze'] = len(group)
        return group

    tmp = keywords_df.groupby('Klaster_ID', group_keys=False).apply(get_head_keyword)

    # Jeżeli Klaster_ID trafił do indeksu — przywróć go jako kolumnę
    if 'Klaster_ID' not in tmp.columns:
        if isinstance(tmp.index, pd.MultiIndex) and 'Klaster_ID' in tmp.index.names:
            tmp = tmp.reset_index(level='Klaster_ID')
        elif tmp.index.name == 'Klaster_ID':
            tmp = tmp.reset_index()
    keywords_df = tmp

    # Jakość klastra = średnia z prawdopodobieństw
    def calculate_cluster_quality(group):
        if len(group) <= 1:
            return group
        group = group.copy()
        group['Cluster_Quality'] = group['Cluster_Probability'].mean()
        return group

    tmp2 = keywords_df.groupby('Klaster_ID', group_keys=False).apply(calculate_cluster_quality)

    # Ponownie dopilnuj, by Klaster_ID był kolumną
    if 'Klaster_ID' not in tmp2.columns:
        if isinstance(tmp2.index, pd.MultiIndex) and 'Klaster_ID' in tmp2.index.names:
            tmp2 = tmp2.reset_index(level='Klaster_ID')
        elif tmp2.index.name == 'Klaster_ID':
            tmp2 = tmp2.reset_index()

    keywords_df = tmp2
    if 'Cluster_Quality' not in keywords_df.columns:
        keywords_df['Cluster_Quality'] = 1.0

    return keywords_df

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

# --- Rekomendacja grupowania + precyzja wariantów
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
    """
    Zwraca priorytet 1..10 (1 = najwyższy). Bez NaN — braki dostają 10.
    """
    s = pd.to_numeric(series, errors='coerce')
    n = int(s.shape[0])
    result = pd.Series(10, index=series.index, dtype='int64')  # domyślnie 10
    if n == 0:
        return result
    mask = s.notna()
    if mask.sum() == 0:
        return result
    ranks = s[mask].rank(method="first", ascending=False)  # 1..k
    step = max(mask.sum() / 10.0, 1.0)
    buckets = np.ceil(ranks / step).astype(int).clip(1, 10)
    result.loc[mask] = buckets.values
    return result

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
1) Każdy tytuł MUSI zawierać dokładnie frazę: "{keyword}".
2) Styl informacyjny/poradnikowy: „co to jest…”, „jak…”, „poradnik…”, „ranking…”, „porównanie…” — dobierz naturalnie do intencji.
3) Pisownia PL. Unikaj dwukropków; używaj myślnika.
4) Nie poprawiaj branda/odmian słowa kluczowego.
5) Zwróć WYŁĄCZNIE listę numerowaną 1..3.
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
                    t = f"{t} — {_kw}" if t else f"{_kw} — co to jest i jak działa?"
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

    try:
        try:
            ws = sh.worksheet("Plan")
            ws.clear()
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title="Plan", rows="1000", cols="26")
    except Exception as e:
        st.error(f"Problem z arkuszem 'Plan': {e}")
        return None

    try:
        data = df[columns_in_order].copy()
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.where(pd.notna(data), "")
        ws.update([data.columns.tolist()] + data.astype(object).values.tolist())
    except Exception as e:
        st.error(f"Nie udało się zaktualizować danych w arkuszu: {e}")
        return None

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
            {
                "addConditionalFormatRule": {
                    "rule": {
                        "ranges": [table_range],
                        "booleanRule": {
                            "condition": {"type": "CUSTOM_FORMULA",
                                "values": [{"userEnteredValue": f"={base_cell}=\"Nowy temat\""}]
                            },
                            "format": {"backgroundColor": col_green}
                        }
                    }, "index": 0
                }
            },
            {
                "addConditionalFormatRule": {
                    "rule": {
                        "ranges": [table_range],
                        "booleanRule": {
                            "condition": {"type": "TEXT_CONTAINS",
                                "values": [{"userEnteredValue": "TOP1"}]
                            },
                            "format": {"backgroundColor": col_orange}
                        }
                    }, "index": 0
                }
            },
            {
                "addConditionalFormatRule": {
                    "rule": {
                        "ranges": [table_range],
                        "booleanRule": {
                            "condition": {"type": "TEXT_CONTAINS",
                                "values": [{"userEnteredValue": "Nie rankuje"}]
                            },
                            "format": {"backgroundColor": col_red}
                        }
                    }, "index": 0
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
st.title("🚀 Planer Treści SEO [Wersja HDBSCAN v10]")
st.markdown("✨ Klasteryzacja HDBSCAN, analiza intencji, priorytety 1–10, **checkboxy do wyboru fraz**, backlog CSV do kolejnych analiz.")

col1, col2 = st.columns(2)
with col1:
    st.header("1. Konfiguracja")
    gen_mode = st.radio("Tryb generowania tytułów", ["Automatycznie (TOP N)", "Ręcznie (checkboxy)"])
    num_to_generate = st.number_input("TOP N (dla trybu automatycznego)", min_value=1, value=20)
    similarity_threshold = st.slider("Próg podobieństwa dla optymalizacji", min_value=0.7, max_value=1.0, value=0.8, step=0.01)

    with st.expander("⚙️ Zaawansowane ustawienia klasteryzacji"):
        min_cluster_size = st.slider("Minimalna wielkość klastra", min_value=2, max_value=10, value=2,
                                     help="Im wyższa wartość, tym większe i bardziej ogólne klastry. 2–3 zalecane dla SEO.")
        min_samples = st.slider("Minimalna liczba próbek w sąsiedztwie", min_value=1, max_value=5, value=1,
                                help="Im wyższa wartość, tym bardziej konserwatywna klasteryzacja. 1–2 zalecane.")
    enable_clustering = st.checkbox("Włącz klasteryzację fraz kluczowych (HDBSCAN)", value=True)

with col2:
    st.header("2. Wgraj pliki CSV")
    content_gap_file = st.file_uploader("1. Content Gap CSV", type="csv")
    my_articles_file = st.file_uploader("2. Twoje artykuły CSV", type="csv")
    ranking_file = st.file_uploader("3. Ranking CSV", type="csv")
    backlog_file = st.file_uploader("4. (Opcjonalnie) Backlog/Historia CSV (kolumny: 'Słowo kluczowe', 'Wyklucz_następnym_razem')", type="csv")

# Eksport do Sheets
st.markdown("### ☁️ Ustawienia eksportu do Google Sheets")
existing_sheet_input = st.text_input("Istniejący Spreadsheet (URL lub ID)", value="", key="existing_spreadsheet_id")
target_folder_id = st.text_input("Opcjonalnie: Folder ID (dla nowych plików)", value="", key="target_folder_id")

# Test połączenia
if st.button("🔍 Test połączenia z Google Sheets (A1 -> timestamp)", key="gs_probe"):
    try:
        gc = _get_gspread_client()
        if gc is None:
            st.stop()
        sheet_id_or_url = existing_sheet_input.strip()
        if not sheet_id_or_url:
            st.error("Podaj URL/ID Spreadsheet, aby wykonać test.")
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

    # Backlog: przygotuj zbiór fraz do pominięcia
    skip_set = set()
    if backlog_file is not None:
        try:
            df_backlog = pd.read_csv(backlog_file)
            if 'Słowo kluczowe' in df_backlog.columns:
                mask = False
                for col in df_backlog.columns:
                    if str(col).lower().strip() in ["wyklucz_następnym_razem", "wyklucz", "skip", "zrealizowano", "wygenerowano_tytuł"]:
                        mask = mask | df_backlog[col].astype(str).str.lower().isin(['true', '1', 'tak', 'yes'])
                if isinstance(mask, (pd.Series, np.ndarray)) is False:
                    mask = df_backlog.get('Wyklucz_następnym_razem', False)
                skip_set = set(df_backlog.loc[mask, 'Słowo kluczowe'].astype(str).str.lower().tolist())
                if len(skip_set) > 0:
                    st.info(f"Z backloga pomijam {len(skip_set)} fraz przy tej analizie.")
        except Exception as e:
            st.warning(f"Nie udało się wczytać backloga: {e}")

    with st.spinner("Przeprowadzam analizę..."):
        try:
            df_gap_raw = pd.read_csv(content_gap_file).dropna(subset=['Keyword']).astype({'Keyword': str})
            # odfiltruj frazy oznaczone do pominięcia
            df_gap = df_gap_raw[~df_gap_raw['Keyword'].str.lower().isin(skip_set)].copy()

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

            st.info(f"Wczytano {len(df_gap)} słów (po odfiltrowaniu backloga), {len(df_articles)} artykułów i {len(df_ranking)} pozycji rankingu.")
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
            kw_raw = str(row['Keyword'])
            keyword_lower = kw_raw.lower()
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
                    'Słowo kluczowe': kw_raw,
                    'Wolumen': int(row.get('Volume', row.get('Wolumen', 0)) or 0),
                    'Status': status,
                    'Akcja / Dopasowany URL': rank_data['url'],
                    'Podobieństwo': 1.00,
                    'Aktualna_pozycja': position
                })
            else:
                if keyword_lower not in skip_set:
                    keywords_for_semantic_check.append(row.to_dict())

        st.info(f"{len(results)} słów zmapowano na podstawie rankingu. Pozostało {len(keywords_for_semantycznej := keywords_for_semantic_check)} do analizy semantycznej.")

        if keywords_for_semantycznej:
            df_semantic = pd.DataFrame(keywords_for_semantycznej)
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

        # Priorytet 1..10 (bez NaN)
        df_results['Priorytet_Score'] = df_results.apply(calculate_priority_score, axis=1)
        df_results['Priorytet'] = compute_priority_bucket(df_results['Priorytet_Score'])

        # Grupa tematyczna
        if 'HEAD_Keyword' in df_results.columns:
            df_results['Grupa_tematyczna'] = df_results['HEAD_Keyword']
        else:
            df_results['Grupa_tematyczna'] = df_results['Słowo kluczowe']

        # Typy i brakujące wartości
        force_int_cols = ['Wolumen', 'Aktualna_pozycja', 'Liczba_fraz_w_klastrze', 'Klaster_ID', 'Priorytet']
        for c in force_int_cols:
            if c in df_results.columns:
                df_results[c] = pd.to_numeric(df_results[c], errors='coerce').fillna(0).astype(int)
        force_float_cols = ['Podobieństwo', 'Cluster_Probability', 'Cluster_Quality', 'Priorytet_Score']
        for c in force_float_cols:
            if c in df_results.columns:
                df_results[c] = pd.to_numeric(df_results[c], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if 'Jest_Outlier' in df_results.columns:
            df_results['Jest_Outlier'] = pd.Series(df_results['Jest_Outlier'], dtype='boolean').fillna(False)

        obj_cols = df_results.select_dtypes(include=['object']).columns
        df_results[obj_cols] = df_results[obj_cols].fillna('-')

        # Rekomendacja grupowania
        df_results['Rekomendacja_grupowania'] = df_results.apply(_reco, axis=1)

        # Kolumny workflow
        df_results['Wybrane_do_tytułów'] = False
        df_results['Wyklucz_następnym_razem'] = False
        df_results['Wygenerowano_tytuł'] = False

        st.success("✅ Analiza zakończona!")

        # Sort domyślny
        df_results_sorted = df_results.sort_values(by=['Priorytet', 'Priorytet_Score', 'Wolumen'], ascending=[True, False, False])

        # === INICJUJ kolumny z propozycjami *przed* edytorem ===
        for col in ['Propozycja_tematu_1','Propozycja_tematu_2','Propozycja_tematu_3']:
            if col not in df_results_sorted.columns:
                df_results_sorted[col] = '-'

        show_advanced = st.checkbox("Pokaż metryki techniczne (HDBSCAN, outliery, prawdopodobieństwo)", value=False, key="show_adv")

        # Kolumny pokazywane w edytorze (ale do edytora przekażemy pełny DF; ukryte kolumny dalej istnieją)
        cols_order = [
            'Priorytet', 'Słowo kluczowe', 'Wolumen', 'Status',
            'Grupa_tematyczna', 'Typ_artykułu', 'Rekomendacja_grupowania',
            'Akcja / Dopasowany URL', 'Najbliższy_artykuł', 'Podobieństwo',
            'Intencja', 'Aktualna_pozycja',
            'Wybrane_do_tytułów', 'Wygenerowano_tytuł', 'Wyklucz_następnym_razem',
            'Propozycja_tematu_1','Propozycja_tematu_2','Propozycja_tematu_3'
        ]
        if show_advanced:
            cols_order.extend([
                'Klaster_ID', 'HEAD_Keyword', 'Typ_w_klastrze', 'Liczba_fraz_w_klastrze',
                'Jest_Outlier', 'Cluster_Probability', 'Cluster_Quality', 'Priorytet_Score'
            ])
        # zapewnij, że Priorytet_Score pozostaje dostępny (do sortowania), nawet jeśli widok tech = OFF
        if 'Priorytet_Score' not in cols_order and 'Priorytet_Score' in df_results_sorted.columns:
            cols_order.append('Priorytet_Score')

        # === Edytowalna tabela z checkboxami ===
        st.subheader("📋 Plan – wybierz frazy checkboxem lub użyj trybu automatycznego")
        edit_cols = [c for c in cols_order if c in df_results_sorted.columns]

        column_config = {
            "Wybrane_do_tytułów": st.column_config.CheckboxColumn("Wybrane_do_tytułów", help="Zaznacz frazy do wygenerowania tytułów"),
            "Wyklucz_następnym_razem": st.column_config.CheckboxColumn("Wyklucz_następnym_razem", help="Omiń przy kolejnej analizie"),
            "Wygenerowano_tytuł": st.column_config.CheckboxColumn("Wygenerowano_tytuł", disabled=True, help="Ustawiane automatycznie po generacji")
        }

        # Przekazujemy pełny DF, a UI kontrolujemy column_order -> ukryte kolumny nadal istnieją (np. Priorytet_Score)
        edited_df = st.data_editor(
            df_results_sorted,
            column_order=edit_cols,
            use_container_width=True,
            num_rows="dynamic",
            column_config=column_config,
            hide_index=True
        )

        st.caption(f"Nowych tematów w widoku: {(edited_df['Status'] == 'Nowy temat').sum()}")

        # ======= GENEROWANIE TYTUŁÓW =======
        st.subheader("📝 Generowanie tytułów")

        if gen_mode == "Automatycznie (TOP N)":
            if st.button("Generuj tytuły – tryb automatyczny (TOP N)"):
                if enable_clustering and 'Typ_w_klastrze' in edited_df.columns:
                    candidates = edited_df[(edited_df['Status'] == 'Nowy temat') & (edited_df['Typ_w_klastrze'] == 'HEAD')].copy()
                else:
                    candidates = edited_df[edited_df['Status'] == 'Nowy temat'].copy()

                # --- solidne sortowanie z fallbackiem ---
                sort_cols, sort_asc = [], []
                if 'Priorytet_Score' in candidates.columns:
                    sort_cols.append('Priorytet_Score'); sort_asc.append(False)  # większy score → wyżej
                if 'Priorytet' in candidates.columns:
                    sort_cols.append('Priorytet');       sort_asc.append(True)   # 1 → wyżej
                if 'Wolumen' in candidates.columns:
                    sort_cols.append('Wolumen');         sort_asc.append(False)  # większy → wyżej
                if sort_cols:
                    candidates = candidates.sort_values(by=sort_cols, ascending=sort_asc)

                to_process = candidates.head(int(num_to_generate))
                if to_process.empty:
                    st.warning("Brak kandydatów z 'Nowy temat'. Zmniejsz próg podobieństwa lub sprawdź wejściowe CSV.")
                else:
                    df_gap_indexed = df_gap_raw.set_index('Keyword')
                    gen_rows = []
                    progress = st.progress(0.0, text="Generowanie…")
                    for i, (_, r) in enumerate(to_process.iterrows()):
                        related_keywords = ""
                        if enable_clustering and 'Klaster_ID' in edited_df.columns and 'Typ_w_klastrze' in edited_df.columns:
                            related = edited_df[
                                (edited_df['Klaster_ID'] == r.get('Klaster_ID')) &
                                (edited_df['Typ_w_klastrze'] == 'RELATED')
                            ]['Słowo kluczowe'].tolist() if 'Typ_w_klastrze' in edited_df.columns else []
                            related_keywords = ", ".join(related[:5])

                        comp_url = "Brak"
                        if r['Słowo kluczowe'] in df_gap_indexed.index:
                            try:
                                comp_url = find_first_competitor_url(df_gap_indexed.loc[r['Słowo kluczowe']])
                            except Exception:
                                comp_url = "Brak"

                        titles = generate_titles(
                            openai_api_key,
                            r['Słowo kluczowe'],
                            r.get('Wolumen', 0),
                            comp_url,
                            related_keywords
                        )
                        gen_rows.append((r['Słowo kluczowe'], titles))
                        progress.progress((i+1)/max(len(to_process),1), text=f"Generowanie ({i+1}/{len(to_process)})")

                    ed = edited_df.set_index('Słowo kluczowe').copy()
                    for kw, titles in gen_rows:
                        for j, col in enumerate(['Propozycja_tematu_1','Propozycja_tematu_2','Propozycja_tematu_3']):
                            ed.loc[kw, col] = titles[j]
                        ed.loc[kw, 'Wygenerowano_tytuł'] = True
                        ed.loc[kw, 'Wyklucz_następnym_razem'] = True

                    edited_df = ed.reset_index()
                    st.success("Tytuły wygenerowane (tryb automatyczny). Przewiń tabelę, by zobaczyć kolumny z propozycjami.")

        else:
            if st.button("Generuj tytuły – dla zaznaczonych checkboxem"):
                selected = edited_df[(edited_df['Wybrane_do_tytułów'] == True) & (edited_df['Status'] == 'Nowy temat')].copy()
                if selected.empty:
                    st.warning("Nie zaznaczono żadnych fraz (muszą mieć Status = 'Nowy temat').")
                else:
                    df_gap_indexed = df_gap_raw.set_index('Keyword')
                    gen_rows = []
                    progress = st.progress(0.0, text="Generowanie…")
                    for i, (_, r) in enumerate(selected.iterrows()):
                        related_keywords = ""
                        if enable_clustering and 'Klaster_ID' in edited_df.columns and 'Typ_w_klastrze' in edited_df.columns:
                            related = edited_df[
                                (edited_df['Klaster_ID'] == r.get('Klaster_ID')) &
                                (edited_df['Typ_w_klastrze'] == 'RELATED')
                            ]['Słowo kluczowe'].tolist() if 'Typ_w_klastrze' in edited_df.columns else []
                            related_keywords = ", ".join(related[:5])
                        comp_url = "Brak"
                        if r['Słowo kluczowe'] in df_gap_indexed.index:
                            try:
                                comp_url = find_first_competitor_url(df_gap_indexed.loc[r['Słowo kluczowe']])
                            except Exception:
                                comp_url = "Brak"
                        titles = generate_titles(openai_api_key, r['Słowo kluczowe'], r.get('Wolumen', 0), comp_url, related_keywords)
                        gen_rows.append((r['Słowo kluczowe'], titles))
                        progress.progress((i+1)/max(len(selected),1), text=f"Generowanie ({i+1}/{len(selected)})")

                    ed = edited_df.set_index('Słowo kluczowe').copy()
                    for kw, titles in gen_rows:
                        for j, col in enumerate(['Propozycja_tematu_1','Propozycja_tematu_2','Propozycja_tematu_3']):
                            ed.loc[kw, col] = titles[j]
                        ed.loc[kw, 'Wygenerowano_tytuł'] = True
                        ed.loc[kw, 'Wyklucz_następnym_razem'] = True
                    edited_df = ed.reset_index()
                    st.success("Tytuły wygenerowane (tryb ręczny).")

        # ====== Wyświetl wynikową tabelę (po ewentualnej generacji) ======
        for col in ['Propozycja_tematu_1','Propozycja_tematu_2','Propozycja_tematu_3']:
            if col not in edited_df.columns:
                edited_df[col] = '-'

        st.subheader("📊 Podsumowanie (po generowaniu)")
        summary_cols = [
            'Priorytet','Słowo kluczowe','Wolumen','Status',
            'Typ_artykułu','Rekomendacja_grupowania',
            'Propozycja_tematu_1','Propozycja_tematu_2','Propozycja_tematu_3',
            'Wyklucz_następnym_razem','Wygenerowano_tytuł'
        ]
        extra_cols = [c for c in ['Grupa_tematyczna','Akcja / Dopasowany URL','Podobieństwo','Intencja','Aktualna_pozycja'] if c in edited_df.columns]
        final_df = edited_df.copy()
        st.dataframe(final_df[summary_cols + extra_cols], use_container_width=True)

        # Zapis do session_state dla eksportów
        st.session_state["plan_df"] = final_df.copy()
        base_export_cols = [
            'Priorytet','Słowo kluczowe','Wolumen','Status',
            'Grupa_tematyczna','Typ_artykułu','Rekomendacja_grupowania',
            'Akcja / Dopasowany URL','Najbliższy_artykuł','Podobieństwo','Intencja','Aktualna_pozycja',
            'Propozycja_tematu_1','Propozycja_tematu_2','Propozycja_tematu_3',
            'Wybrane_do_tytułów','Wygenerowano_tytuł','Wyklucz_następnym_razem'
        ]
        st.session_state["plan_cols"] = [c for c in base_export_cols if c in final_df.columns]

# =========================================
# STAŁA SEKCJA EKSPORTU / POBIERANIA
# =========================================
st.subheader("☁️ Eksport / Pobieranie")
if st.session_state.get("plan_df") is None or st.session_state.get("plan_cols") is None:
    st.info("Najpierw uruchom analizę, żeby włączyć eksport i pobieranie.")
else:
    gs_title_default = f"Plan treści HDBSCAN ({time.strftime('%Y-%m-%d %H:%M')})"
    gs_title = st.text_input("Tytuł nowego arkusza (dla NOWEGO pliku)", value=gs_title_default, key="gs_title")

    export_df = st.session_state["plan_df"][st.session_state["plan_cols"]].copy()

    # 1) Eksport pełnego planu
    csv_buffer2 = io.StringIO()
    export_df.to_csv(csv_buffer2, index=False, encoding='utf-8')
    csv_bytes2 = csv_buffer2.getvalue().encode('utf-8-sig')
    st.download_button("📥 Pobierz aktualny plan (CSV)", data=csv_bytes2, file_name="plan_tresci_hdbscan.csv", mime="text/csv", type="primary", key="dl_csv_latest")

    # 2) Eksport backloga do kolejnych analiz
    backlog_cols = ['Słowo kluczowe','Wyklucz_następnym_razem']
    backlog_df = export_df[backlog_cols].copy()
    if 'Wygenerowano_tytuł' in export_df.columns:
        wmask = export_df['Wygenerowano_tytuł'] == True
        backlog_df.loc[wmask, 'Wyklucz_następnym_razem'] = True
    csv_buffer3 = io.StringIO()
    backlog_df.to_csv(csv_buffer3, index=False, encoding='utf-8')
    st.download_button(
        "🗂️ Pobierz Backlog (CSV do następnej analizy)",
        data=csv_buffer3.getvalue().encode('utf-8-sig'),
        file_name="backlog_planer.csv",
        mime="text/csv"
    )

    # 3) Google Sheets
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
