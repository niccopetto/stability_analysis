"""
main_analysis.py — Pipeline di Analisi Dati LWFA
=================================================
Script di esecuzione che gestisce I/O, DataFrame pandas, batch processing
e visualizzazione statistica per esperimenti Laser-Plasma (LWFA).

Fasi:
  1. Data Ingestion & Feature Engineering (Excel)
  2. File Mapping & Extraction (TIFF)
  3. Batch Processing (diagnostics_lib)
  4. Statistica e Visualizzazione (Seaborn)
"""

import re
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
from matplotlib.widgets import RectangleSelector
import seaborn as sns
from scipy.stats import linregress
from skimage import io as skio

import diagnostics_lib as diag

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURAZIONE — Aggiorna questi percorsi prima dell'esecuzione
# ═══════════════════════════════════════════════════════════════════════════════

RUN_MODE = "STEP_2_ANALYZE"  # Modalità operative: "STEP_1_CLASSIFY" oppure "STEP_2_ANALYZE"
TS_ON = False                # Fallback: se True, forza analisi Top+Side anche senza tag TS_ON/TS_IN nel CSV

# Cartella radice contenente le sotto-cartelle strumento (Andor_Lanex/, ecc.)
ROOT_DIR = Path(r"c:\Users\ILILUser\Desktop\Stability_Analysis")

MASTER_CSV_PATH = ROOT_DIR / "master_classification.csv"

# Cartella contenente i file Excel (uno per giorno) con i parametri degli shot
EXCEL_DIR = ROOT_DIR / "External_User_run_excel"

# Cartella di output per grafici e risultati
OUTPUT_DIR = ROOT_DIR / "output"

# Percorso dedicato per le immagini Pointing Lanex (Feature 2)
POINTING_LANEX_DIR = Path(r"U:\unwrapped_pointing_lanex")

# Percorso per le immagini Top Imaging (stesso layout di ROOT_DIR se non separato)
TOP_IMAGING_DIR = ROOT_DIR

# ── Fattori di Conversione Pixel → Millimetri ──
PX_TO_MM = 0.064           # Pointing Lanex: 1 pixel = 64 µm = 0.064 mm
PX_TO_MM_SIDE = 1.0 / 179  # Side Imaging: 179 px = 1 mm ≈ 0.00559 mm/px
PX_TO_MM_TOP = 1.0 / 100       # Top Imaging: 100 px = 1 mm

# Soglia frazionaria per la lunghezza del plasma (10% del massimo per includere più coda)
PLASMA_THRESHOLD_FRACTION = 0.1
FWHM_FRACTION_SATURATED = 0.8  # Frazione per la larghezza delle colonne saturate (80%)

# ── Parametri Classificazione Fascio (Pointing Lanex) ──
BEAM_PEAK_THRESHOLD_HI = 0.25     # Soglia alta (25%) per separare blob multipli
BEAM_PEAK_THRESHOLD_LO = 0.15     # Soglia bassa (15%) per misurare estensione diffusa
BEAM_MIN_BLOB_AREA_PX = 2500      # Area minima blob (pixel)
BEAM_DIFFUSE_AREA_FRAC = 0.08     # 8% dell'immagine: soglia area per Diffused
BEAM_DIFFUSE_COMPACT_MIN = 0.10   # Compattezza minima per confermare Diffused
BEAM_SMOOTHING_SIGMA = 2.0        # σ gaussiana per smoothing pre-labeling

# Nomi degli strumenti e colonne corrispondenti nel DataFrame
INSTRUMENTS = {
    'Andor_Lanex':    'Path_Andor',
    'Pointing_Lanex': 'Path_Pointing',
    'SideImaging':    'Path_SideImaging',
    'TopImaging':     'Path_TopImaging',
    'TopView':        'Path_TopImaging',   # alias per TopImaging
}

# Pattern regex per il parsing dei nomi file TIFF
FILENAME_RE = re.compile(
    r'^(Andor_Lanex|SideImaging|Pointing_Lanex|TopImaging|TopView)_(\d{8})_(\d+)\.tiff?$',
    re.IGNORECASE
)

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITÀ
# ═══════════════════════════════════════════════════════════════════════════════

def safe_imread(path) -> np.ndarray | None:
    """Carica un'immagine TIFF in modo sicuro, restituendo None se fallisce."""
    try:
        img = skio.imread(str(path))
        # Se multi-canale (RGB), converti in scala di grigi (media)
        if img.ndim == 3:
            img = np.mean(img, axis=2)
        return img
    except Exception as e:
        log.warning("Impossibile caricare %s: %s", path, e)
        return None


def draw_confidence_ellipse(x, y, ax, n_std=2.0, **kwargs):
    """Disegna un'ellisse di confidenza (n_std deviazioni standard).

    Parameters
    ----------
    x, y : array-like — Coordinate dei punti.
    ax : matplotlib Axes
    n_std : float — Numero di deviazioni standard per il raggio.
    **kwargs : passati a matplotlib.patches.Ellipse.
    """
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # eigh restituisce autovalori in ordine CRESCENTE.
    # L'asse principale corrisponde all'autovalore più grande (indice -1).
    # L'angolo è calcolato dall'autovettore della colonna corrispondente.
    angle = np.degrees(np.arctan2(eigenvectors[1, -1], eigenvectors[0, -1]))
    width  = 2 * n_std * np.sqrt(max(eigenvalues[-1], 0))   # asse maggiore
    height = 2 * n_std * np.sqrt(max(eigenvalues[-2], 0))   # asse minore
    ellipse = Ellipse(
        xy=(np.mean(x), np.mean(y)),
        width=width, height=height, angle=angle,
        **kwargs
    )
    ax.add_patch(ellipse)


def interactive_roi_selection(image: np.ndarray, title: str = "Select ROI") -> dict | None:
    """Mostra un'immagine e lascia all'utente la selezione interattiva di un ROI.

    Usa matplotlib.widgets.RectangleSelector. Chiudere la finestra per confermare.

    Returns
    -------
    dict con chiavi 'y_min', 'y_max', 'x_min', 'x_max', 'center_x', 'center_y'
    oppure None se nessuna selezione è stata fatta.
    """
    roi_coords = {}

    def on_select(eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        roi_coords['x_min'] = min(x1, x2)
        roi_coords['x_max'] = max(x1, x2)
        roi_coords['y_min'] = min(y1, y2)
        roi_coords['y_max'] = max(y1, y2)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image, cmap='gray')
    ax.grid(False)  # Disabilita la griglia di seaborn che potrebbe "sporcare" l'immagine
    ax.set_title(f"{title}\nDraw a rectangle, then press 'Enter' to confirm.")

    selector = RectangleSelector(
        ax, on_select, interactive=True,
        useblit=True, button=[1],
        props=dict(facecolor='none', edgecolor='lime', linewidth=2, fill=False)
    )
    
    def on_key(event):
        if event.key == 'enter':
            plt.close(fig)
            
    fig.canvas.mpl_connect('key_press_event', on_key)
    # Mantieni riferimento per evitare garbage collection
    fig._roi_selector = selector
    plt.show(block=True)

    if not roi_coords:
        log.warning("Nessun ROI selezionato, verrà usata l'intera immagine")
        return None

    # Centro geometrico dell'area croppata
    h = roi_coords['y_max'] - roi_coords['y_min']
    w = roi_coords['x_max'] - roi_coords['x_min']
    roi_coords['center_x'] = w / 2.0
    roi_coords['center_y'] = h / 2.0
    log.info("ROI selezionato: x=[%d, %d], y=[%d, %d], centro=(%d, %d)",
             roi_coords['x_min'], roi_coords['x_max'],
             roi_coords['y_min'], roi_coords['y_max'],
             roi_coords['center_x'], roi_coords['center_y'])
    return roi_coords



def find_first_mask_free_pointing(df: pd.DataFrame) -> str | None:
    """Trova il primo shot mask_free con un'immagine Pointing_Lanex valida."""
    mask = (
        (df['Tipo_Maschera'] == 'free') &
        (df['Path_Pointing'].notna())
    )
    candidates = df[mask]
    if candidates.empty:
        return None
    return candidates.iloc[0]['Path_Pointing']


def crop_image(image: np.ndarray, roi: dict) -> np.ndarray:
    """Ritaglia un'immagine usando le coordinate ROI."""
    return image[roi['y_min']:roi['y_max'], roi['x_min']:roi['x_max']]



def save_plasma_debug_image(
        raw_image: np.ndarray,
        cleaned_image: np.ndarray,
        result: dict,
        output_path: Path,
        title_prefix: str = "",
        roi_circle: dict = None):
    """Salva un'immagine di debug per l'analisi del canale di plasma.

    Sovrappone all'immagine:
    - Contorno del nozzle (se trovato)
    - Punta del nozzle (croce rossa)
    - Centroide del plasma (croce gialla)
    - Asse del canale fitto (punti ciano, per Top View)
    - Cerchio ROI (se fornito)
    - Profilo Z robusto (subplot)
    """
    debug = result.get('debug_data', {})
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ── Left: immagine con overlay ──
    ax = axes[0]
    ax.imshow(cleaned_image, cmap='inferno', aspect='auto')
    ax.grid(False)

    nozzle_contour = debug.get('nozzle_contour')
    if nozzle_contour is not None:
        ax.plot(nozzle_contour[:, 1], nozzle_contour[:, 0],
                color='cyan', linewidth=1.5, alpha=0.8, label='Nozzle contour')

    nozzle_tip = debug.get('nozzle_tip')
    if nozzle_tip is not None:
        ax.plot(nozzle_tip[0], nozzle_tip[1], 'rx', markersize=12,
                markeredgewidth=2.5, label='Nozzle tip', zorder=10)

    z_c = result.get('Plasma_Z_Centroid', result.get('Plasma_Z_Position'))
    y_c = result.get('Plasma_Y_Centroid', result.get('Plasma_Y_Position'))
    if z_c is not None and y_c is not None:
        if not np.isnan(z_c) and not np.isnan(y_c):
            ax.plot(z_c, y_c, 'y+', markersize=14, markeredgewidth=2.5,
                    label='Plasma centroid', zorder=10)

    # Disegna l'asse del canale estratto e il fit
    axis_coords = debug.get('axis_coords')
    fit_coeffs = debug.get('fit_coeffs')
    
    if axis_coords is not None:
        axis_z, axis_y = axis_coords
        if len(axis_z) > 0:
            # Mostra comunque i punti calcolati (per Side View saranno i punti medi, per Top View l'asse)
            ax.plot(axis_z, axis_y, 'c.', markersize=2, alpha=0.6, label='Extracted Axis points')
            
            # Disegna i bordi del canale se disponibili
            axis_y_l = debug.get('axis_y_left')
            axis_y_r = debug.get('axis_y_right')
            if axis_y_l is not None and axis_y_r is not None and len(axis_y_l) == len(axis_z):
                ax.plot(axis_z, axis_y_l, 'g.', markersize=1, alpha=0.5, label='Channel edges')
                ax.plot(axis_z, axis_y_r, 'g.', markersize=1, alpha=0.5)

            # Se abbiamo i coefficienti del fit traccia la linea
            if fit_coeffs is not None:
                slope, intercept = fit_coeffs
                z_min, z_max = np.min(axis_z), np.max(axis_z)
                y_min = slope * z_min + intercept
                y_max = slope * z_max + intercept
                
                # Linea spessa blu sui punti validi
                ax.plot([z_min, z_max], [y_min, y_max], color='blue', linewidth=2.5, label='Fitted Axis')
                
                # Linea tratteggiata estesa su tutta l'immagine
                w_img = cleaned_image.shape[1]
                y_0 = intercept
                y_end = slope * w_img + intercept
                ax.plot([0, w_img], [y_0, y_end], color='blue', linewidth=0.5, linestyle='--', alpha=0.8)

    # Disegna il cerchio ROI
    if roi_circle is not None:
        circle_patch = Circle(
            (roi_circle['center_x'], roi_circle['center_y']),
            roi_circle['radius'],
            facecolor='none', edgecolor='lime', linewidth=1.5, linestyle=':',
            label='ROI Circle'
        )
        ax.add_patch(circle_patch)

    # Disegna Interactive ROI e Mask (se presenti)
    interactive_roi = debug.get('interactive_roi')
    if interactive_roi is not None:
        import matplotlib.patches as patches
        roi_x, roi_y, roi_w, roi_h = interactive_roi
        rect = patches.Rectangle((roi_x, roi_y), roi_w, roi_h, linewidth=2, 
                                 edgecolor='yellow', facecolor='none', linestyle='--', label='Interactive ROI')
        ax.add_patch(rect)
        
    interactive_mask = debug.get('interactive_mask')
    if interactive_mask is not None:
        # Crea un overlay rosso semitrasparente per i blob validi
        overlay = np.zeros((*cleaned_image.shape, 4), dtype=np.float32)
        overlay[interactive_mask, 0] = 1.0  # R
        overlay[interactive_mask, 3] = 0.5  # Alpha
        ax.imshow(overlay, origin='upper', aspect='auto')

    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(f"{title_prefix} Plasma Channel Overlay")

    # ── Right: Z profile ──
    ax2 = axes[1]
    z_profile = debug.get('z_profile_robust')
    if z_profile is not None:
        ax2.plot(z_profile, label='Robust Z profile (90th pctl)')
        threshold_val = result.get('Max_Intensity', 0) * PLASMA_THRESHOLD_FRACTION
        ax2.axhline(threshold_val, color='r', ls='--', alpha=0.7,
                     label=f'Threshold ({PLASMA_THRESHOLD_FRACTION*100:.0f}%)')
        if z_c is not None and not np.isnan(z_c):
            ax2.axvline(z_c, color='gold', ls=':', alpha=0.7,
                         label=f'Z centroid = {z_c:.1f}')
        ax2.set_xlabel('Z (pixel)')
        ax2.set_ylabel('Intensity (90th pctl)')
        ax2.legend(fontsize=8)
    ax2.set_title('Robust Z Profile')

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    log.debug("Debug image salvata: %s", output_path)


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 1: DATA INGESTION & FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def phase1_ingest_excel(excel_path: Path) -> pd.DataFrame:
    """Legge il file Excel (.xlsx) e crea la colonna Tipo_Maschera con forward-fill.

    La prima riga è un header di sezione e viene saltata.
    Cerca in General_Comments le stringhe mask_free, mask_circular, mask_round,
    mask_square e propaga con ffill() (mappa 'circular' su 'round').

    Richiede openpyxl come dipendenza (pip install openpyxl).
    """
    log.info("═══ FASE 1: Lettura Excel ═══")

    # Auto-detect: alcuni file hanno una riga di intestazione di sezione
    # prima dei nomi colonna reali. Leggiamo prima senza index_col per
    # capire se 'Shot' è già tra le colonne o se serve saltare una riga.
    df_probe = pd.read_excel(excel_path, nrows=2)
    if 'Shot' in df_probe.columns:
        df = pd.read_excel(excel_path, index_col='Shot')
    else:
        log.info("Header di sezione rilevato, salto prima riga")
        df = pd.read_excel(excel_path, header=1, index_col='Shot')

    if 'General comments' in df.columns:
        df.rename(columns={'General comments': 'General_Comments'}, inplace=True)
    log.info("Excel caricato: %d shot, colonne: %s", len(df), list(df.columns))

    # ── Rinomina colonna carica per coerenza interna ──
    charge_col = [c for c in df.columns if 'charge' in c.lower()]
    if charge_col:
        df.rename(columns={charge_col[0]: 'Turbo_ICT_Charge'}, inplace=True)
        log.info("Colonna carica rinominata: '%s' → 'Turbo_ICT_Charge'", charge_col[0])

    # ── Taglio dell'Excel (da 'mask_free' a 'END') ──
    if 'General_Comments' in df.columns:
        comments_str = df['General_Comments'].astype(str)
        
        # Inizia da mask_free
        mask_free_matches = df.index[comments_str.str.contains('mask_free', case=False, na=False)]
        if len(mask_free_matches) > 0:
            idx_start = mask_free_matches.min()
            df = df.loc[idx_start:]
            comments_str = df['General_Comments'].astype(str)
            log.info("Excel tagliato: inizio da shot %s (trovato mask_free)", idx_start)
            
        # Termina a END (escluso)
        end_matches = df.index[comments_str.str.contains('END', case=False, na=False)]
        if len(end_matches) > 0:
            idx_end = end_matches.min()
            df = df.loc[:idx_end-1]
            comments_str = df['General_Comments'].astype(str)
            log.info("Excel tagliato: fine a shot %s (trovato END, escluso)", idx_end)

        # Riconoscimento fallimenti iniezione e filamentazione
        df['Injection_Success'] = ~comments_str.str.contains(r'\bNo\b', case=True, na=False)
        df['Is_Filamented'] = comments_str.str.contains('filamented', case=False, na=False)
        
        n_fails = (~df['Injection_Success']).sum()
        n_fils = df['Is_Filamented'].sum()
        if n_fails > 0:
            log.info("Trovati %d shot con fallimento iniezione (tag 'No')", n_fails)
        if n_fils > 0:
            log.info("Trovati %d shot filamentati", n_fils)

        # ── Tagging Variabili ──
        mask_pattern = r'(mask_free|mask_circular|mask_round|mask_square)'
        df['Tipo_Maschera'] = (
            df['General_Comments']
            .astype(str)
            .str.extract(mask_pattern, expand=False)
            .str.replace('mask_', '', regex=False)
            .replace('circular', 'round')
            .ffill()
        )
        
        # Estrazione dello stato del Magnete (Magnet_IN / Magnet_OUT)
        magnet_pattern = r'Magnet_(IN|OUT)'
        df['Magnet_State'] = (
            df['General_Comments']
            .astype(str)
            .str.extract(magnet_pattern, expand=False, flags=re.IGNORECASE)
            .str.upper()
            .ffill()
        )
        
        # Estrazione dello stato del Pointing Lanex (Pointing_IN / Pointing_OUT)
        pointing_pattern = r'Pointing_(IN|OUT)'
        df['Pointing_State'] = (
            df['General_Comments']
            .astype(str)
            .str.extract(pointing_pattern, expand=False, flags=re.IGNORECASE)
            .str.upper()
            .ffill()
        )

        # Estrazione della Posizione del target (P1, P2, ecc.)
        posizione_pattern = r'\b(P\d+)\b'
        df['Posizione'] = (
            df['General_Comments']
            .astype(str)
            .str.extract(posizione_pattern, expand=False, flags=re.IGNORECASE)
            .str.upper()
            .ffill()
        )

        # Estrazione dello stato Top+Side Imaging (TS_ON / TS_IN / TS_OFF)
        ts_pattern = r'TS_(ON|IN|OFF)'
        df['TS_State'] = (
            df['General_Comments']
            .astype(str)
            .str.extract(ts_pattern, expand=False, flags=re.IGNORECASE)
            .str.upper()
            .ffill()
        )
    else:
        df['Tipo_Maschera'] = np.nan
        df['Magnet_State'] = np.nan
        df['Pointing_State'] = np.nan
        df['TS_State'] = np.nan
        df['Posizione'] = np.nan
        log.warning("Colonna 'General_Comments' non trovata: le variabili di stato saranno NaN")


    n_masks = df['Tipo_Maschera'].notna().sum()
    log.info("Tipo_Maschera assegnato a %d/%d shot", n_masks, len(df))
    n_mag = df['Magnet_State'].notna().sum()
    log.info("Magnet_State assegnato a %d/%d shot", n_mag, len(df))
    n_pt = df['Pointing_State'].notna().sum()
    log.info("Pointing_State assegnato a %d/%d shot", n_pt, len(df))
    n_ts = df['TS_State'].isin(['ON', 'IN']).sum()
    log.info("TS_State ON/IN per %d/%d shot", n_ts, len(df))

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 2: FILE MAPPING & EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def phase2_map_files(df: pd.DataFrame, root_dir: Path, target_date: str = None) -> pd.DataFrame:
    """Cerca TIFF nella ROOT_DIR, mappa i percorsi nel DataFrame per Shot.

    Struttura attesa (ricorsiva):
        ROOT_DIR/Strumento/Data/NomeFile.tif(f)
    Oppure flat:
        ROOT_DIR/NomeFile.tif(f)

    Estrae la data dal nome file e crea la colonna 'Giorno'.
    Crea le colonne Path_Andor, Path_Pointing, Path_SideImaging.
    """
    log.info("═══ FASE 2: Mapping File TIFF ═══")

    # Inizializza colonne path a NaN come object per permettere l'inserimento di stringhe
    for col in INSTRUMENTS.values():
        df[col] = pd.Series(dtype=object)
    df['Giorno'] = pd.Series(dtype=object)

    # Cerca ricorsivamente file TIFF:
    # - SideImaging e Andor_Lanex in ROOT_DIR
    # - Pointing_Lanex in POINTING_LANEX_DIR (Feature 2)
    # - TopImaging in TOP_IMAGING_DIR (se diverso da ROOT_DIR)
    tiff_files = list(root_dir.rglob('*.tif')) + list(root_dir.rglob('*.tiff'))
    log.info("Trovati %d file TIFF in %s", len(tiff_files), root_dir)
    if POINTING_LANEX_DIR.exists():
        pointing_files = (list(POINTING_LANEX_DIR.rglob('*.tif'))
                          + list(POINTING_LANEX_DIR.rglob('*.tiff')))
        log.info("Trovati %d file TIFF Pointing in %s", len(pointing_files), POINTING_LANEX_DIR)
        tiff_files.extend(pointing_files)
    else:
        log.warning("Percorso Pointing Lanex non trovato: %s", POINTING_LANEX_DIR)

    # Top Imaging: aggiungi file da TOP_IMAGING_DIR se diverso da ROOT_DIR
    if TOP_IMAGING_DIR.resolve() != root_dir.resolve() and TOP_IMAGING_DIR.exists():
        top_files = (list(TOP_IMAGING_DIR.rglob('*.tif'))
                     + list(TOP_IMAGING_DIR.rglob('*.tiff')))
        log.info("Trovati %d file TIFF TopImaging in %s",
                 len(top_files), TOP_IMAGING_DIR)
        tiff_files.extend(top_files)
    elif not TOP_IMAGING_DIR.exists():
        log.warning("Percorso Top Imaging non trovato: %s", TOP_IMAGING_DIR)

    # Dizionario per raccogliere i background per strumento+data.
    # Preferisce shot 0; se non disponibile, usa shot 1 come fallback.
    backgrounds = {}   # (strumento, data) -> path
    fallback_bg = {}   # (strumento, data) -> path  (shot 1, usato solo se 0 manca)

    for fpath in tiff_files:
        match = FILENAME_RE.match(fpath.name)
        if not match:
            continue

        instrument = match.group(1)
        date_str = match.group(2)
        shot_num = int(match.group(3))
        
        if target_date and date_str != target_date:
            continue
            
        col_name = INSTRUMENTS.get(instrument)
        if col_name is None:
            continue

        # Registra background: shot 0 (prioritario) o shot 1 (fallback)
        if shot_num == 0:
            backgrounds[(instrument, date_str)] = fpath
            continue
        if shot_num == 1:
            fallback_bg[(instrument, date_str)] = fpath

        # Associa il file alla riga del DataFrame se lo shot esiste
        if shot_num in df.index:
            df.at[shot_num, col_name] = str(fpath.resolve())
            # Imposta Giorno solo se ancora vuoto (evita sovrascritture
            # in caso di file di strumenti diversi con date diverse)
            if pd.isna(df.at[shot_num, 'Giorno']):
                df.at[shot_num, 'Giorno'] = date_str

    # Integra fallback (shot 1) dove shot 0 non è disponibile
    for key, fpath in fallback_bg.items():
        if key not in backgrounds:
            backgrounds[key] = fpath
            log.info("Background fallback (shot 1) usato per %s/%s", key[0], key[1])

    # Salva i background come attributo del DataFrame (metadata)
    df.attrs['backgrounds'] = backgrounds

    n_mapped = {col: df[col].notna().sum() for col in INSTRUMENTS.values()}
    log.info("File mappati: %s", n_mapped)
    log.info("Background trovati: %d", len(backgrounds))

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 3: BATCH PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════


def phase3_batch_process(df: pd.DataFrame, roi: dict = None,
                         run_mode: str = "STEP_1_CLASSIFY",
                         debug_output_dir: Path = None,
                         ts_on: bool = False) -> pd.DataFrame:
    """Itera sugli shot, applica le funzioni di diagnostics_lib.

    Per ogni shot:
    - Injection_Success == False ("No") → salta TUTTO, Beam_Type = 'Null'
    - Is_Filamented → calcola centroide/nozzle (Side+Top) ma salta
      Plasma_Length; salta Pointing/Andor, Beam_Type = 'Null'
    - SideImaging → analyze_plasma_channel_side
    - TopImaging → analyze_plasma_channel_top
    - Pointing_Lanex → classify_beam → analyze_pointing_profile
    - Andor_Lanex → [TEMPORANEAMENTE SOSPESO]

    Parameters
    ----------
    roi : dict or None
        Coordinate ROI rettangolare per il crop Pointing Lanex.
    debug_output_dir : Path or None
        Se impostato, salva immagini di debug per i primi shot.
    """
    log.info("═══ FASE 3: Batch Processing ═══")

    backgrounds = df.attrs.get('backgrounds', {})

    # Pre-carica i background per evitare letture ripetute
    bg_cache = {}
    for key, path in backgrounds.items():
        bg_cache[key] = safe_imread(path)

    # ── Colonne risultato (esistenti + nuove) ──
    result_cols_str = ['Beam_Type']
    result_cols_num = [
        # Side View
        'Plasma_Z_Position', 'Plasma_Z_Position_err',
        'Plasma_Z_Position_rel',
        'Plasma_Y_Position', 'Plasma_Y_Position_err',
        'Plasma_Length', 'Plasma_Length_err',
        'Plasma_Length_mm', 'Plasma_Length_mm_err',
        'Max_Intensity',
        'Nozzle_Tip_Z', 'Nozzle_Tip_Y',
        'Nozzle_Distance_Y', 'Nozzle_Distance_Y_err',
        'Nozzle_Distance_Y_mm', 'Nozzle_Distance_Y_mm_err',
        'Side_Beam_Angle_mrad', 'Side_Beam_Angle_mrad_err',
        'Side_Channel_Width_mean', 'Side_Saturated_Length',
        # Top View
        'Top_Plasma_Z_Position', 'Top_Plasma_Z_Position_err',
        'Top_Plasma_Y_Position', 'Top_Plasma_Y_Position_err',
        'Top_Plasma_Length', 'Top_Plasma_Length_err',
        'Top_Plasma_Length_mm', 'Top_Plasma_Length_mm_err',
        'Top_Saturated_Length',
        'Top_Max_Intensity',
        'Top_Beam_Angle_mrad', 'Top_Beam_Angle_mrad_err',
        'Top_Channel_Width_mean', 'Top_Channel_Width_mean_mm',
        # Pointing
        'N_Blobs', 'Compactness',
        'X_c', 'Y_c', 'Sigma_X', 'Sigma_Y', 'Total_Intensity',
        # Energy (sospeso)
        'Peak_X', 'Energy_Spread_px',
    ]
    for col in result_cols_str:
        if col not in df.columns:
            df[col] = pd.Series(dtype=object)
    for col in result_cols_num:
        if col not in df.columns:
            df[col] = np.nan

    total_shots = len(df)
    processed = 0
    debug_saved_side = 0
    debug_processed_side = 0
    debug_saved_top = 0
    debug_processed_top = 0
    interactive_side_roi = None  # Persistent ROI for Side View
    interactive_top_roi = None   # Persistent ROI for Top View
    previous_posizione = None    # Track position changes

    if debug_output_dir is not None:
        debug_output_dir.mkdir(parents=True, exist_ok=True)

    for shot in df.index:
        date_val = df.at[shot, 'Giorno']
        date_str = str(date_val) if pd.notna(date_val) else None
        
        ts_state = df.at[shot, 'TS_State'] if 'TS_State' in df.columns else None
        curr_pos = df.at[shot, 'Posizione'] if 'Posizione' in df.columns else None
        
        # Gestione reset ROI iterativa
        reset_roi = False
        if pd.notna(ts_state) and str(ts_state).upper() in ('OUT', 'OFF'):
            reset_roi = True
            
        if pd.notna(curr_pos):
            if previous_posizione is not None and curr_pos != previous_posizione:
                log.info("Cambio Posizione rilevato: %s -> %s. Reset delle ROI interattive.", previous_posizione, curr_pos)
                reset_roi = True
            previous_posizione = curr_pos
            
        if reset_roi:
            interactive_side_roi = None
            interactive_top_roi = None

        # ── Controllo preventivo: Injection_Success ──
        is_injected = df.at[shot, 'Injection_Success'] \
            if 'Injection_Success' in df.columns else True
        is_filamented = df.at[shot, 'Is_Filamented'] \
            if 'Is_Filamented' in df.columns else False

        if not is_injected:
            # "No" → salta TUTTO
            df.at[shot, 'Beam_Type'] = 'Null'
            processed += 1
            if processed % 50 == 0 or processed == total_shots:
                log.info("Processati %d/%d shot", processed, total_shots)
            continue

        # ── compute_length: True solo se NON filamented ──
        compute_length = not is_filamented

        # ── 3A. TS: Plasma Channel (Side + Top) ───────────────────
        # Stato per-shot: colonna TS_State (ON/IN) dal CSV, con fallback
        # alla costante globale TS_ON
        shot_ts_on = (pd.notna(ts_state) and str(ts_state).upper() in ('ON', 'IN')) \
            or (pd.isna(ts_state) and ts_on)
        if run_mode == "STEP_2_ANALYZE" and shot_ts_on:
            
            # --- Side Imaging Execution ---
            try:
                si_path = df.at[shot, 'Path_SideImaging']
                if pd.notna(si_path):
                    img_si_raw = safe_imread(si_path)
                    if img_si_raw is not None:
                        img_si_clean = img_si_raw.copy().astype(np.float64)

                        # Sottrai background se disponibile
                        if date_str:
                            bg = bg_cache.get(('SideImaging', date_str))
                            if bg is not None and bg.shape == img_si_raw.shape:
                                img_si_clean = diag.subtract_background(
                                    img_si_raw, bg)['cleaned_image']
                            else:
                                log.warning(
                                    "Shot %s | SideImaging: background "
                                    "non disponibile o shape incompatibile",
                                    shot)

                        # UI Interattiva per ROI SideImaging
                        if interactive_side_roi is None:
                            log.info("Shot %s | Seleziona ROI su SideImaging (Premi ENTER per confermare)", shot)
                            roi_dict = interactive_roi_selection(img_si_clean, title=f"Select SideImaging ROI for shot {shot}")
                            
                            if roi_dict is not None:
                                roi_x = roi_dict['x_min']
                                roi_y = roi_dict['y_min']
                                roi_w = roi_dict['x_max'] - roi_dict['x_min']
                                roi_h = roi_dict['y_max'] - roi_dict['y_min']
                                
                                if roi_w > 0 and roi_h > 0:
                                    interactive_side_roi = (roi_x, roi_y, roi_w, roi_h)
                                    log.info("Shot %s | ROI selezionata: %s", shot, interactive_side_roi)
                                else:
                                    log.warning("Shot %s | ROI invalida: w=%s, h=%s", shot, roi_w, roi_h)
                            else:
                                log.warning("Shot %s | Nessuna ROI selezionata.", shot)
                        
                        res_si = diag.analyze_plasma_channel_side(
                            img_si_clean,
                            interactive_roi=interactive_side_roi,
                            threshold_fraction=PLASMA_THRESHOLD_FRACTION,
                            px_to_mm=PX_TO_MM_SIDE,
                        )

                        # Salva risultati nel DataFrame
                        df.at[shot, 'Plasma_Z_Position'] = res_si['Plasma_Z_Position']
                        df.at[shot, 'Plasma_Z_Position_err'] = res_si['Plasma_Z_Position_err']
                        df.at[shot, 'Plasma_Z_Position_rel'] = res_si['Plasma_Z_Position_rel']
                        df.at[shot, 'Plasma_Y_Position'] = res_si['Plasma_Y_Position']
                        df.at[shot, 'Plasma_Y_Position_err'] = res_si['Plasma_Y_Position_err']
                        df.at[shot, 'Plasma_Z_Centroid'] = res_si['Plasma_Z_Centroid']
                        df.at[shot, 'Plasma_Z_Centroid_err'] = res_si['Plasma_Z_Centroid_err']
                        df.at[shot, 'Plasma_Y_Centroid'] = res_si['Plasma_Y_Centroid']
                        df.at[shot, 'Plasma_Y_Centroid_err'] = res_si['Plasma_Y_Centroid_err']
                        df.at[shot, 'Plasma_Length'] = res_si['Plasma_Length']
                        df.at[shot, 'Plasma_Length_err'] = res_si['Plasma_Length_err']
                        df.at[shot, 'Plasma_Length_mm'] = res_si['Plasma_Length_mm']
                        df.at[shot, 'Plasma_Length_mm_err'] = res_si['Plasma_Length_mm_err']
                        df.at[shot, 'Max_Intensity'] = res_si['Max_Intensity']
                        df.at[shot, 'Nozzle_Tip_Z'] = res_si['Nozzle_Tip_Z']
                        df.at[shot, 'Nozzle_Tip_Y'] = res_si['Nozzle_Tip_Y']
                        df.at[shot, 'Nozzle_Distance_Y'] = res_si['Nozzle_Distance_Y']
                        df.at[shot, 'Nozzle_Distance_Y_err'] = res_si['Nozzle_Distance_Y_err']
                        df.at[shot, 'Nozzle_Distance_Y_mm'] = res_si['Nozzle_Distance_Y_mm']
                        df.at[shot, 'Nozzle_Distance_Y_mm_err'] = res_si['Nozzle_Distance_Y_mm_err']
                        df.at[shot, 'Side_Beam_Angle_mrad'] = res_si['Side_Beam_Angle_mrad']
                        df.at[shot, 'Side_Beam_Angle_mrad_err'] = res_si['Side_Beam_Angle_mrad_err']
                        df.at[shot, 'Side_Channel_Width_mean'] = res_si['Side_Channel_Width_mean']
                        df.at[shot, 'Side_Saturated_Length'] = res_si['Side_Saturated_Length']

                        # Debug image (solo per shot esplicitamente TS_ON, max 20)
                        is_explicit_ts = (pd.notna(ts_state) and str(ts_state).upper() in ('ON', 'IN'))
                        if debug_output_dir is not None and is_explicit_ts and debug_saved_side < 20:
                            save_plasma_debug_image(
                                img_si_raw, img_si_clean, res_si,
                                debug_output_dir / f'debug_side_shot{shot}.png',
                                title_prefix=f'Shot {shot} Side')
                            debug_saved_side += 1
            except Exception as e:
                log.warning("Shot %s | SideImaging fallito: %s", shot, e, exc_info=True)

            # --- Top Imaging Execution ---
            try:
                ti_path_col = 'Path_TopImaging'
                if ti_path_col in df.columns:
                    ti_path = df.at[shot, ti_path_col]
                    if pd.notna(ti_path):
                        img_ti = safe_imread(ti_path)
                        if img_ti is not None:
                            img_ti_clean = img_ti.copy().astype(np.float64)

                            # Sottrai background
                            if date_str:
                                bg = bg_cache.get(('TopImaging', date_str))
                                if bg is None:
                                    bg = bg_cache.get(('TopView', date_str))
                                if bg is not None and bg.shape == img_ti.shape:
                                    img_ti_clean = diag.subtract_background(
                                        img_ti, bg)['cleaned_image']
                                else:
                                    log.warning(
                                        "Shot %s | TopImaging: background "
                                        "non disponibile o shape "
                                        "incompatibile", shot)

                            # UI Interattiva per ROI TopImaging
                            if interactive_top_roi is None:
                                log.info("Shot %s | Seleziona ROI su TopImaging (Premi ENTER per confermare)", shot)
                                roi_dict_top = interactive_roi_selection(img_ti_clean, title=f"Select TopImaging ROI for shot {shot}")
                                
                                if roi_dict_top is not None:
                                    roi_x = roi_dict_top['x_min']
                                    roi_y = roi_dict_top['y_min']
                                    roi_w = roi_dict_top['x_max'] - roi_dict_top['x_min']
                                    roi_h = roi_dict_top['y_max'] - roi_dict_top['y_min']
                                    
                                    if roi_w > 0 and roi_h > 0:
                                        interactive_top_roi = (roi_x, roi_y, roi_w, roi_h)
                                        log.info("Shot %s | ROI Top selezionata: %s", shot, interactive_top_roi)
                                    else:
                                        log.warning("Shot %s | ROI Top invalida: w=%s, h=%s", shot, roi_w, roi_h)
                                else:
                                    log.warning("Shot %s | Nessuna ROI Top selezionata.", shot)

                            if interactive_top_roi is not None:
                                roi_x, roi_y, roi_w, roi_h = interactive_top_roi
                                img_ti_clean_roi = img_ti_clean[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
                            else:
                                img_ti_clean_roi = img_ti_clean

                            # La ROI circolare iniziale è disattivata

                            res_ti = diag.analyze_plasma_channel_top(
                                img_ti_clean_roi,
                                threshold_fraction=PLASMA_THRESHOLD_FRACTION,
                                fwhm_fraction_saturated=FWHM_FRACTION_SATURATED,
                                px_to_mm=PX_TO_MM_TOP,
                                compute_length=compute_length,
                            )

                            # Aggiusta le coordinate se è stata applicata la ROI rettangolare
                            if interactive_top_roi is not None:
                                roi_x, roi_y, roi_w, roi_h = interactive_top_roi
                                if not pd.isna(res_ti['Plasma_Z_Position']):
                                    res_ti['Plasma_Z_Position'] += roi_x
                                if not pd.isna(res_ti['Plasma_Y_Position']):
                                    res_ti['Plasma_Y_Position'] += roi_y
                                
                                if 'debug_data' in res_ti:
                                    res_ti['debug_data']['interactive_roi'] = interactive_top_roi
                                    
                                    # Trasliamo le coordinate dell'asse geometrico ai riferimenti dell'immagine intera
                                    axis_z, axis_y = res_ti['debug_data'].get('axis_coords', (np.array([]), np.array([])))
                                    if len(axis_z) > 0:
                                        res_ti['debug_data']['axis_coords'] = (axis_z + roi_x, axis_y + roi_y)
                                        
                                        # Trasliamo anche i bordi del canale per la visualizzazione corretta
                                        axis_y_left = res_ti['debug_data'].get('axis_y_left')
                                        if axis_y_left is not None:
                                            res_ti['debug_data']['axis_y_left'] = axis_y_left + roi_y
                                        axis_y_right = res_ti['debug_data'].get('axis_y_right')
                                        if axis_y_right is not None:
                                            res_ti['debug_data']['axis_y_right'] = axis_y_right + roi_y
                                        
                                    # Paddiamo il profilo Z per centrarlo nell'immagine originaria nel debug
                                    z_prof = res_ti['debug_data'].get('z_profile_robust')
                                    if z_prof is not None:
                                        padded_prof = np.zeros(img_ti_clean.shape[1])
                                        padded_prof[roi_x:roi_x+roi_w] = z_prof
                                        res_ti['debug_data']['z_profile_robust'] = padded_prof

                            df.at[shot, 'Top_Plasma_Z_Position'] = res_ti['Plasma_Z_Position']
                            df.at[shot, 'Top_Plasma_Z_Position_err'] = res_ti['Plasma_Z_Position_err']
                            df.at[shot, 'Top_Plasma_Y_Position'] = res_ti['Plasma_Y_Position']
                            df.at[shot, 'Top_Plasma_Y_Position_err'] = res_ti['Plasma_Y_Position_err']
                            df.at[shot, 'Top_Plasma_Length'] = res_ti['Plasma_Length']
                            df.at[shot, 'Top_Plasma_Length_err'] = res_ti['Plasma_Length_err']
                            df.at[shot, 'Top_Plasma_Length_mm'] = res_ti['Plasma_Length_mm']
                            df.at[shot, 'Top_Plasma_Length_mm_err'] = res_ti['Plasma_Length_mm_err']
                            df.at[shot, 'Top_Saturated_Length'] = res_ti['Top_Saturated_Length']
                            df.at[shot, 'Top_Max_Intensity'] = res_ti['Max_Intensity']
                            df.at[shot, 'Top_Beam_Angle_mrad'] = res_ti['Top_Beam_Angle_mrad']
                            df.at[shot, 'Top_Beam_Angle_mrad_err'] = res_ti['Top_Beam_Angle_mrad_err']
                            df.at[shot, 'Top_Channel_Width_mean'] = res_ti['Top_Channel_Width_mean']
                            df.at[shot, 'Top_Channel_Width_mean_mm'] = res_ti['Top_Channel_Width_mean_mm']

                            # Debug image (solo per shot esplicitamente TS_IN, max 20)
                            is_explicit_ts_in = (pd.notna(ts_state) and str(ts_state).upper() == 'IN')
                            if debug_output_dir is not None and is_explicit_ts_in and debug_saved_top < 20:
                                save_plasma_debug_image(
                                    img_ti, img_ti_clean, res_ti,
                                    debug_output_dir / f'debug_top_shot{shot}.png',
                                    title_prefix=f'Shot {shot} Top')
                                debug_saved_top += 1
            except Exception as e:
                log.warning("Shot %s | TopImaging fallito: %s", shot, e, exc_info=True)

        # ── Filamented → salta analisi a valle ──
        if is_filamented:
            df.at[shot, 'Beam_Type'] = 'Null'
            processed += 1
            if processed % 50 == 0 or processed == total_shots:
                log.info("Processati %d/%d shot", processed, total_shots)
            continue

        # ── 3B. Pointing Lanex ─────────────────────────
        try:
            pt_state = df.at[shot, 'Pointing_State'] \
                if 'Pointing_State' in df.columns else None

            if pd.notna(pt_state) and str(pt_state).upper() == 'IN':
                pt_path = df.at[shot, 'Path_Pointing']
                if pd.notna(pt_path):
                    img_pt = safe_imread(pt_path)
                    if img_pt is not None:
                        # Sottrai background se disponibile
                        if date_str:
                            bg = bg_cache.get(('Pointing_Lanex', date_str))
                            if bg is not None and bg.shape == img_pt.shape:
                                img_pt = diag.subtract_background(
                                    img_pt, bg)['cleaned_image']
                            else:
                                log.warning(
                                    "Shot %s | Pointing: background "
                                    "non disponibile o shape incompatibile",
                                    shot)

                        # ─── Crop ROI (Feature 1) ───
                        if roi is not None:
                            img_pt = crop_image(img_pt, roi)

                        # ─── Classificazione del fascio ───
                        classification = diag.classify_beam(
                            img_pt,
                            peak_threshold_hi=BEAM_PEAK_THRESHOLD_HI,
                            peak_threshold_lo=BEAM_PEAK_THRESHOLD_LO,
                            min_blob_area_px=BEAM_MIN_BLOB_AREA_PX,
                            diffuse_area_frac=BEAM_DIFFUSE_AREA_FRAC,
                            diffuse_compactness_min=BEAM_DIFFUSE_COMPACT_MIN,
                            smoothing_sigma=BEAM_SMOOTHING_SIGMA
                        )

                        if run_mode == "STEP_1_CLASSIFY":
                            df.at[shot, 'Beam_Type'] = classification['label']
                            df.at[shot, 'N_Blobs'] = classification['n_blobs']
                            df.at[shot, 'Compactness'] = classification.get(
                                'compactness', 0.0)

                        elif run_mode == "STEP_2_ANALYZE":
                            beam_label = df.at[shot, 'Beam_Type']
                            if pd.isna(beam_label):
                                beam_label = classification['label']

                            df.at[shot, 'N_Blobs'] = classification['n_blobs']
                            df.at[shot, 'Compactness'] = classification.get(
                                'compactness', 0.0)

                            # Analisi completa SOLO per Collimati
                            if beam_label == 'Collimated':
                                profile = diag.analyze_pointing_profile(
                                    img_pt,
                                    blob_mask=classification['primary_mask']
                                )
                                df.at[shot, 'X_c'] = \
                                    profile['X_c'] * PX_TO_MM
                                df.at[shot, 'Y_c'] = \
                                    profile['Y_c'] * PX_TO_MM
                                df.at[shot, 'Sigma_X'] = \
                                    profile['Sigma_X'] * PX_TO_MM
                                df.at[shot, 'Sigma_Y'] = \
                                    profile['Sigma_Y'] * PX_TO_MM
                                df.at[shot, 'Total_Intensity'] = \
                                    profile['Total_Intensity']
        except Exception as e:
            log.warning("Shot %s | Pointing fallito: %s", shot, e, exc_info=True)

        # ── 3C. Andor Lanex (Spettrometro) ─────────────────────────
        # [TEMPORANEAMENTE SOSPESO — da integrare successivamente]

        processed += 1
        if processed % 50 == 0 or processed == total_shots:
            log.info("Processati %d/%d shot", processed, total_shots)

    log.info("Batch processing completato: %d shot elaborati", processed)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 4: STATISTICA E VISUALIZZAZIONE
# ═══════════════════════════════════════════════════════════════════════════════

def phase4_visualize(df: pd.DataFrame, output_dir: Path, roi_center: dict = None):
    """Genera grafici di confronto raggruppati per Tipo_Maschera.

    Grafici prodotti:
    0.  Probabilità Iniezione & Tasso Filamentazione
    0b. Classificazione Fasci (Distribuzione globale e per maschera)
    1.  Stabilità Plasma: Boxplot Plasma_Z_Position
    2a. Jitter Puntamento: Scatter 2D centroidi + ellissi
    2b. Sigma Jitter: Barplot deviazione standard X_c, Y_c per gruppo
    [SOSPESI] 3a/3b. Energia Picco e Spread (filtro Magnet IN)
    4.  Deriva Temporale: Catplot carica per Giorno e Maschera

    Parameters
    ----------
    roi_center : dict or None
        Dict with 'center_x' and 'center_y' (in pixels from ROI selection).
        Will be converted to µm and plotted as reference on jitter scatter.
    """
    log.info("═══ FASE 4: Visualizzazione ═══")
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style='whitegrid', palette='deep', font_scale=1.1)
    palette = sns.color_palette('Set2')
    mask_palette = {
        'free': '#2ecc71',   # green
        'square': '#e74c3c', # red
        'round': '#3498db'   # blue
    }

    posizioni = np.sort(df['Posizione'].dropna().unique()) if 'Posizione' in df.columns else ['All']
    if len(posizioni) == 0:
        posizioni = ['All']

    # ── 4.0a Injection Probability ───────────────────────────
    if 'Injection_Success' in df.columns:
        fig, axes = plt.subplots(1, len(posizioni), figsize=(5 * len(posizioni), 5), squeeze=False)
        for i, pos in enumerate(posizioni):
            ax = axes[0, i]
            subset = df if pos == 'All' else df[df['Posizione'] == pos]
            if subset.empty:
                log.warning(f"00a_injection_probability: Nessun dato per Posizione={pos}")
                ax.set_title(f"Pos: {pos} (No Data)")
                ax.axis('off')
                continue
                
            inj_rate = subset.groupby('Tipo_Maschera')['Injection_Success'].mean() * 100
            if inj_rate.empty:
                ax.set_title(f"Pos: {pos} (No Data)")
                ax.axis('off')
                continue
                
            sns.barplot(x=inj_rate.index, y=inj_rate.values, ax=ax, hue=inj_rate.index, palette=mask_palette, legend=False)
            ax.set_title(f'Injection Probability (Pos: {pos})')
            ax.set_ylabel('Success Rate (%)' if i == 0 else '')
            ax.set_ylim(0, 100)
            
        fig.tight_layout()
        fig.savefig(output_dir / '00a_injection_probability.png', dpi=150)
        plt.close(fig)
        log.info("Grafico 0a salvato: 00a_injection_probability.png")

    # ── 4.0b Filamentation Rate ──────────────────────────────
    if 'Injection_Success' in df.columns and 'Is_Filamented' in df.columns:
        df_inj = df[df['Injection_Success'] == True]
        if not df_inj.empty:
            fig, axes = plt.subplots(1, len(posizioni), figsize=(5 * len(posizioni), 5), squeeze=False)
            for i, pos in enumerate(posizioni):
                ax = axes[0, i]
                subset = df_inj if pos == 'All' else df_inj[df_inj['Posizione'] == pos]
                if subset.empty:
                    log.warning(f"00b_filamentation_rate: Nessun dato per Posizione={pos}")
                    ax.set_title(f"Pos: {pos} (No Data)")
                    ax.axis('off')
                    continue
                    
                fil_rate = subset.groupby('Tipo_Maschera')['Is_Filamented'].mean() * 100
                if fil_rate.empty:
                    ax.set_title(f"Pos: {pos} (No Data)")
                    ax.axis('off')
                    continue
                    
                sns.barplot(x=fil_rate.index, y=fil_rate.values, ax=ax, hue=fil_rate.index, palette=mask_palette, legend=False)
                ax.set_title(f'Filamentation Rate (Pos: {pos})')
                ax.set_ylabel('Filamentation Rate (%)' if i == 0 else '')
                ax.set_ylim(0, 100)
                
            fig.tight_layout()
            fig.savefig(output_dir / '00b_filamentation_rate.png', dpi=150)
            plt.close(fig)
            log.info("Grafico 0b salvato: 00b_filamentation_rate.png")

    # ── 4.0c Beam Classification Statistics ───────────────────────
    if 'Beam_Type' in df.columns:
        df_beams = df.dropna(subset=['Beam_Type'])
        df_beams = df_beams[df_beams['Beam_Type'] != 'Null']
        if not df_beams.empty and 'Tipo_Maschera' in df_beams.columns:
            fig, axes = plt.subplots(1, len(posizioni), figsize=(6 * len(posizioni), 5), squeeze=False)
            beam_colors = {'Collimated': '#2ecc71', 'Diffused': '#e67e22', 'Multiple': '#3498db'}
            
            for i, pos in enumerate(posizioni):
                ax = axes[0, i]
                subset = df_beams if pos == 'All' else df_beams[df_beams['Posizione'] == pos]
                if subset.empty:
                    log.warning(f"00c_beam_classification: Nessun dato per Posizione={pos}")
                    ax.set_title(f"Pos: {pos} (No Data)")
                    ax.axis('off')
                    continue
                
                ct = pd.crosstab(subset['Tipo_Maschera'], subset['Beam_Type'])
                if ct.empty:
                    ax.set_title(f"Pos: {pos} (No Data)")
                    ax.axis('off')
                    continue
                    
                ct.plot(
                    kind='bar', stacked=True, ax=ax,
                    color=[beam_colors.get(c, 'gray') for c in ct.columns]
                )
                ax.set_title(f'Beam Class by Mask (Pos: {pos})')
                ax.set_ylabel('Shot Count' if i == 0 else '')
                if i == len(posizioni) - 1:
                    ax.legend(title='Beam Type', bbox_to_anchor=(1.05, 1), loc='upper left')
                else:
                    if ax.get_legend() is not None:
                        ax.get_legend().remove()
                ax.set_xlabel('Mask Type')

            fig.tight_layout()
            fig.savefig(output_dir / '00c_beam_classification.png', dpi=150)
            plt.close(fig)
            log.info("Grafico 0c salvato: 00c_beam_classification.png")
        else:
            log.warning("Grafico 0c saltato: dati Beam_Type insufficienti")

    # Filtro: rimuoviamo gli shot non validi dalle statistiche spaziali/energetiche
    df_clean = df.copy()
    if 'Injection_Success' in df_clean.columns:
        df_clean = df_clean[df_clean['Injection_Success'] == True]
    if 'Is_Filamented' in df_clean.columns:
        df_clean = df_clean[df_clean['Is_Filamented'] == False]

    # ── 4.1 Plasma Ignition Stability ─────────────────────────────
    df_plasma = df_clean.dropna(subset=['Plasma_Z_Position_rel', 'Tipo_Maschera'])
    if not df_plasma.empty:
        fig, axes = plt.subplots(1, len(posizioni), figsize=(6 * len(posizioni), 5), squeeze=False)
        for i, pos in enumerate(posizioni):
            ax = axes[0, i]
            subset = df_plasma if pos == 'All' else df_plasma[df_plasma['Posizione'] == pos]
            if subset.empty:
                log.warning(f"01_plasma_stability: Nessun dato per Posizione={pos}")
                ax.set_title(f"Pos: {pos} (No Data)")
                ax.axis('off')
                continue
                
            sns.boxplot(data=subset, x='Tipo_Maschera', y='Plasma_Z_Position_rel',
                        hue='Tipo_Maschera', palette=mask_palette, legend=False, 
                        ax=ax, boxprops={'alpha': 0.4})
            
            sns.stripplot(data=subset, x='Tipo_Maschera', y='Plasma_Z_Position_rel',
                          hue='Tipo_Maschera', palette=mask_palette, legend=False,
                          ax=ax, jitter=True, size=5, alpha=0.8, edgecolor='gray', linewidth=0.5)
                          
            ax.set_title(fr'Plasma Ignition Stability (Pos: {pos})')
            ax.set_xlabel('Mask Type')
            ax.set_ylabel(r'Ignition Start $\Delta Z$ from Nozzle (mm)' if i == 0 else '')
            
        fig.tight_layout()
        fig.savefig(output_dir / '01_plasma_stability.png', dpi=150)
        plt.close(fig)
        log.info("Grafico 1 salvato: 01_plasma_stability.png")
    else:
        log.warning("Grafico 1 saltato: dati insufficienti per Plasma_Z_Position_rel")

    # ── 4.2a Pointing Jitter — 2D Scatter with Ellipses ───────────
    df_jitter = df_clean.dropna(subset=['X_c', 'Y_c', 'Tipo_Maschera'])
    if not df_jitter.empty:
        fig_2a, axes_2a = plt.subplots(1, len(posizioni), figsize=(6 * len(posizioni), 6), squeeze=False)
        fig_2b, axes_2b = plt.subplots(1, len(posizioni), figsize=(6 * len(posizioni), 5), squeeze=False)
        
        for i, pos in enumerate(posizioni):
            ax_2a = axes_2a[0, i]
            ax_2b = axes_2b[0, i]
            subset = df_jitter if pos == 'All' else df_jitter[df_jitter['Posizione'] == pos]
            if subset.empty:
                log.warning(f"02_pointing_jitter: Nessun dato per Posizione={pos}")
                ax_2a.set_title(f"Pos: {pos} (No Data)"); ax_2a.axis('off')
                ax_2b.set_title(f"Pos: {pos} (No Data)"); ax_2b.axis('off')
                continue

            groups = subset.groupby('Tipo_Maschera')
            colors = {name: mask_palette.get(name, 'gray') for name in groups.groups.keys()}

            for name, group in groups:
                sigma_x = group['X_c'].std()
                sigma_y = group['Y_c'].std()
                label_with_sigma = fr"{name} ($\sigma_X={sigma_x:.1f}$, $\sigma_Y={sigma_y:.1f}$)"

                ax_2a.scatter(group['X_c'], group['Y_c'], label=label_with_sigma,
                           color=colors[name], alpha=0.7, edgecolors='w', s=40)
                draw_confidence_ellipse(
                    group['X_c'].values, group['Y_c'].values, ax_2a, n_std=2.0,
                    facecolor=colors[name], alpha=0.15, edgecolor=colors[name], lw=2
                )
                mean_x = group['X_c'].mean()
                mean_y = group['Y_c'].mean()
                ax_2a.plot(mean_x, mean_y, marker='o', markersize=10,
                        color=colors[name], markeredgecolor='k',
                        markeredgewidth=1.5, alpha=0.8, zorder=5)

            if roi_center is not None:
                roi_cx, roi_cy = roi_center.get('roi_cx'), roi_center.get('roi_cy')
                if roi_cx is not None and roi_cy is not None:
                    ax_2a.plot(roi_cx * PX_TO_MM, roi_cy * PX_TO_MM, marker='x', markersize=10, markeredgewidth=2,
                            color='black', zorder=10, label='ROI Center')
                
                img_cx, img_cy = roi_center.get('img_cx'), roi_center.get('img_cy')
                if img_cx is not None and img_cy is not None:
                    ax_2a.plot(img_cx * PX_TO_MM, img_cy * PX_TO_MM, marker='+', markersize=18, markeredgewidth=2,
                            color='black', zorder=10, label='Image Center')

            ax_2a.set_title(fr'Pointing Jitter (Pos: {pos})')
            ax_2a.set_xlabel(r'$X_c$ (mm)')
            ax_2a.set_ylabel(r'$Y_c$ (mm)' if i == 0 else '')
            ax_2a.legend(title='Mask', loc='best')
            ax_2a.set_aspect('equal', adjustable='datalim')
            ax_2a.invert_yaxis()

            # ── 4.2b Barplot sigma X_c and Y_c ──
            sigma_stats = groups[['X_c', 'Y_c']].std().reset_index()
            sigma_stats.rename(columns={'X_c': r'$\sigma_X$', 'Y_c': r'$\sigma_Y$'}, inplace=True)
            sigma_melt = sigma_stats.melt(id_vars='Tipo_Maschera',
                                           value_vars=[r'$\sigma_X$', r'$\sigma_Y$'],
                                           var_name='Axis', value_name=r'$\sigma$ (mm)')
            sns.barplot(data=sigma_melt, x='Tipo_Maschera', y=r'$\sigma$ (mm)',
                        hue='Axis', palette='muted', ax=ax_2b)
            ax_2b.set_title(fr'Centroid $\sigma$ (Pos: {pos})')
            ax_2b.set_xlabel('Mask Type')
            ax_2b.set_ylabel(r'$\sigma$ (mm)' if i == 0 else '')

        fig_2a.tight_layout()
        fig_2a.savefig(output_dir / '02a_pointing_jitter_scatter.png', dpi=150)
        plt.close(fig_2a)
        log.info("Grafico 2a salvato: 02a_pointing_jitter_scatter.png")
        
        fig_2b.tight_layout()
        fig_2b.savefig(output_dir / '02b_pointing_jitter_sigma.png', dpi=150)
        plt.close(fig_2b)
        log.info("Grafico 2b salvato: 02b_pointing_jitter_sigma.png")
    else:
        log.warning("Grafici 2a/2b saltati: dati jitter insufficienti")

    # ── 4.3 Prestazioni Energetiche (filtro Magnet IN) ─────────────
    # [TEMPORANEAMENTE SOSPESO — da integrare successivamente]
    # if 'Magnet_State' in df_clean.columns:
    #     df_energy = df_clean[df_clean['Magnet_State'] == 'IN'].copy()
    # else:
    #     df_energy = df_clean.copy()
    #
    # # 4.3a Boxplot Peak_X
    # df_peak = df_energy.dropna(subset=['Peak_X', 'Tipo_Maschera'])
    # if not df_peak.empty:
    #     fig, ax = plt.subplots(figsize=(8, 5))
    #     sns.boxplot(data=df_peak, x='Tipo_Maschera', y='Peak_X',
    #                 hue='Tipo_Maschera', palette=mask_palette, legend=False, ax=ax)
    #     ax.set_title('Energia di Picco — Peak Position (Magnet IN)')
    #     ax.set_xlabel('Tipo Maschera')
    #     ax.set_ylabel('Peak X (pixel)')
    #     fig.tight_layout()
    #     fig.savefig(output_dir / '03a_energy_peak.png', dpi=150)
    #     plt.close(fig)
    #     log.info("Grafico 3a salvato: 03a_energy_peak.png")
    # else:
    #     log.warning("Grafico 3a saltato: dati Peak_X insufficienti")
    #
    # # 4.3b Boxplot Energy Spread
    # df_spread = df_energy.dropna(subset=['Energy_Spread_px', 'Tipo_Maschera'])
    # if not df_spread.empty:
    #     fig, ax = plt.subplots(figsize=(8, 5))
    #     sns.boxplot(data=df_spread, x='Tipo_Maschera', y='Energy_Spread_px',
    #                 hue='Tipo_Maschera', palette=mask_palette, legend=False, ax=ax)
    #     ax.set_title('Spread Energetico — FWHM (Magnet IN)')
    #     ax.set_xlabel('Tipo Maschera')
    #     ax.set_ylabel('Energy Spread (pixel)')
    #     fig.tight_layout()
    #     fig.savefig(output_dir / '03b_energy_spread.png', dpi=150)
    #     plt.close(fig)
    #     log.info("Grafico 3b salvato: 03b_energy_spread.png")
    # else:
    #     log.warning("Grafico 3b saltato: dati Energy_Spread insufficienti")

    # ── 4.4 Analisi Temporale — Catplot per Giorno e Maschera ─────────
    # Basato sulla carica (Peak_X temporaneamente sospeso)
    temporal_col = None
    if 'Turbo_ICT_Charge' in df.columns and df['Turbo_ICT_Charge'].notna().any():
        temporal_col = 'Turbo_ICT_Charge'
    # [TEMPORANEAMENTE SOSPESO — Peak_X non popolato finché lo spettrometro è disabilitato]
    # elif 'Peak_X' in df.columns and df['Peak_X'].notna().any():
    #     temporal_col = 'Peak_X'

    if temporal_col:
        df[temporal_col] = pd.to_numeric(df[temporal_col], errors='coerce')
        df_temp = df.dropna(subset=[temporal_col, 'Tipo_Maschera', 'Giorno']).copy()
        df_temp['Giorno'] = df_temp['Giorno'].astype(str)
        
        if not df_temp.empty:
            fig, axes = plt.subplots(1, len(posizioni), figsize=(8 * len(posizioni), 5), squeeze=False)
            for i, pos in enumerate(posizioni):
                ax = axes[0, i]
                subset = df_temp if pos == 'All' else df_temp[df_temp['Posizione'] == pos]
                if subset.empty:
                    log.warning(f"04_temporal_drift: Nessun dato per Posizione={pos}")
                    ax.set_title(f"Pos: {pos} (No Data)")
                    ax.axis('off')
                    continue
                
                sns.boxplot(
                    data=subset, x='Giorno', y=temporal_col,
                    hue='Tipo_Maschera', palette=mask_palette, ax=ax
                )
                ax.set_title(f'Temporal Drift (Pos: {pos})')
                ax.set_xlabel('Day')
                ax.set_ylabel(temporal_col if i == 0 else '')
            
            fig.tight_layout()
            fig.savefig(output_dir / '04_temporal_drift.png', dpi=150)
            plt.close(fig)
            log.info("Grafico 4 salvato: 04_temporal_drift.png")
        else:
            log.warning("Grafico 4 saltato: dati temporali insufficienti")
    else:
        log.warning("Grafico 4 saltato: nessuna colonna temporale disponibile")

    # ── 4.4b Temporal Trend of Geometric Mean of Standard Deviations ──
    df_jitter = df_clean.dropna(subset=['X_c', 'Y_c', 'Tipo_Maschera', 'Giorno']).copy()
    if not df_jitter.empty:
        df_jitter['Giorno'] = df_jitter['Giorno'].astype(str)
        fig, axes = plt.subplots(1, len(posizioni), figsize=(8 * len(posizioni), 5), squeeze=False)
        
        for i, pos in enumerate(posizioni):
            ax = axes[0, i]
            subset_pos = df_jitter if pos == 'All' else df_jitter[df_jitter['Posizione'] == pos]
            if subset_pos.empty:
                ax.set_title(f"Pos: {pos} (No Data)"); ax.axis('off')
                continue
                
            giorni_list = sorted(subset_pos['Giorno'].unique())
            mask_list = subset_pos['Tipo_Maschera'].unique()
            records = []
            for giorno in giorni_list:
                for mask in mask_list:
                    subset = subset_pos[(subset_pos['Giorno'] == giorno) & (subset_pos['Tipo_Maschera'] == mask)]
                    if len(subset) > 1:
                        sigma_x = subset['X_c'].std()
                        sigma_y = subset['Y_c'].std()
                        geom_sigma = np.sqrt(sigma_x * sigma_y)
                        records.append({'Giorno': giorno, 'Tipo_Maschera': mask, 'Geom_Sigma': geom_sigma})
            
            df_geom_sigma = pd.DataFrame(records)
            if not df_geom_sigma.empty:
                sns.pointplot(data=df_geom_sigma, x='Giorno', y='Geom_Sigma', hue='Tipo_Maschera', 
                              palette=mask_palette, markers='o', linestyles='-', ax=ax)
                ax.set_title(fr'Geom $\sigma$ Trend (Pos: {pos})')
                ax.set_xlabel('Day')
                ax.set_ylabel(r'$\sqrt{\sigma_X \sigma_Y}$ (mm)' if i == 0 else '')
            else:
                ax.set_title(f"Pos: {pos} (No Data)"); ax.axis('off')

        fig.tight_layout()
        fig.savefig(output_dir / '04b_temporal_geom_sigma.png', dpi=150)
        plt.close(fig)
        log.info("Grafico 4b salvato: 04b_temporal_geom_sigma.png")

    # ── 4.4c Temporal Trend of Injection Probability ──
    if 'Injection_Success' in df.columns and 'Giorno' in df.columns:
        df_inj_temp = df.dropna(subset=['Injection_Success', 'Tipo_Maschera', 'Giorno']).copy()
        if not df_inj_temp.empty:
            df_inj_temp['Giorno'] = df_inj_temp['Giorno'].astype(str)
            fig, axes = plt.subplots(1, len(posizioni), figsize=(8 * len(posizioni), 5), squeeze=False)
            
            for i, pos in enumerate(posizioni):
                ax = axes[0, i]
                subset_pos = df_inj_temp if pos == 'All' else df_inj_temp[df_inj_temp['Posizione'] == pos]
                if subset_pos.empty:
                    ax.set_title(f"Pos: {pos} (No Data)"); ax.axis('off')
                    continue
                    
                giorni_list = sorted(subset_pos['Giorno'].unique())
                mask_list = subset_pos['Tipo_Maschera'].unique()
                records_inj = []
                for giorno in giorni_list:
                    for mask in mask_list:
                        subset = subset_pos[(subset_pos['Giorno'] == giorno) & (subset_pos['Tipo_Maschera'] == mask)]
                        if not subset.empty:
                            inj_prob = subset['Injection_Success'].mean() * 100
                            records_inj.append({'Giorno': giorno, 'Tipo_Maschera': mask, 'Injection_Probability': inj_prob})
                            
                df_inj_trend = pd.DataFrame(records_inj)
                if not df_inj_trend.empty:
                    sns.pointplot(data=df_inj_trend, x='Giorno', y='Injection_Probability', hue='Tipo_Maschera', 
                                  palette=mask_palette, markers='s', linestyles='-', ax=ax)
                    ax.set_title(f'Injection Trend (Pos: {pos})')
                    ax.set_xlabel('Day')
                    ax.set_ylabel('Injection Probability (%)' if i == 0 else '')
                else:
                    ax.set_title(f"Pos: {pos} (No Data)"); ax.axis('off')

            fig.tight_layout()
            fig.savefig(output_dir / '04c_temporal_injection.png', dpi=150)
            plt.close(fig)
            log.info("Grafico 4c salvato: 04c_temporal_injection.png")

    # ── 4.4d Temporal Trend of Spark Length (Top View) ──
    if 'Top_Plasma_Length_mm' in df.columns and 'Giorno' in df.columns:
        df_len_top = df_clean.dropna(subset=['Top_Plasma_Length_mm', 'Tipo_Maschera', 'Giorno']).copy()
        if not df_len_top.empty:
            df_len_top['Giorno'] = df_len_top['Giorno'].astype(str)
            fig, axes = plt.subplots(1, len(posizioni), figsize=(8 * len(posizioni), 5), squeeze=False)
            
            for i, pos in enumerate(posizioni):
                ax = axes[0, i]
                subset = df_len_top if pos == 'All' else df_len_top[df_len_top['Posizione'] == pos]
                if subset.empty:
                    ax.set_title(f"Pos: {pos} (No Data)"); ax.axis('off')
                    continue
                    
                dodge_val = 0.2 if subset['Tipo_Maschera'].nunique() > 1 else False
                sns.pointplot(data=subset, x='Giorno', y='Top_Plasma_Length_mm', hue='Tipo_Maschera', 
                              palette=mask_palette, dodge=dodge_val, markers='D', linestyles='-', errorbar='sd', ax=ax)
                ax.set_title(f'Top View Spark Length (Pos: {pos})')
                ax.set_xlabel('Day')
                ax.set_ylabel('Spark Length (mm)' if i == 0 else '')

            fig.tight_layout()
            fig.savefig(output_dir / '04d_temporal_spark_length_top.png', dpi=150)
            plt.close(fig)
            log.info("Grafico 4d salvato: 04d_temporal_spark_length_top.png")

    # ── 4.4e Temporal Trend of Spark Length (Side View) ──
    if 'Plasma_Length_mm' in df.columns and 'Giorno' in df.columns:
        df_len_side = df_clean.dropna(subset=['Plasma_Length_mm', 'Tipo_Maschera', 'Giorno']).copy()
        if not df_len_side.empty:
            df_len_side['Giorno'] = df_len_side['Giorno'].astype(str)
            fig, axes = plt.subplots(1, len(posizioni), figsize=(8 * len(posizioni), 5), squeeze=False)
            
            for i, pos in enumerate(posizioni):
                ax = axes[0, i]
                subset = df_len_side if pos == 'All' else df_len_side[df_len_side['Posizione'] == pos]
                if subset.empty:
                    ax.set_title(f"Pos: {pos} (No Data)"); ax.axis('off')
                    continue
                    
                dodge_val = 0.2 if subset['Tipo_Maschera'].nunique() > 1 else False
                sns.pointplot(data=subset, x='Giorno', y='Plasma_Length_mm', hue='Tipo_Maschera', 
                              palette=mask_palette, dodge=dodge_val, markers='D', linestyles='-', errorbar='sd', ax=ax)
                ax.set_title(f'Side View Spark Length (Pos: {pos})')
                ax.set_xlabel('Day')
                ax.set_ylabel('Spark Length (mm)' if i == 0 else '')

            fig.tight_layout()
            fig.savefig(output_dir / '04e_temporal_spark_length_side.png', dpi=150)
            plt.close(fig)
            log.info("Grafico 4e salvato: 04e_temporal_spark_length_side.png")

    log.info("═══ Visualizzazione completata ═══")

    # ── 4.5a Top View Spark Width by Mask ───────────────────────
    df_width = df_clean.dropna(subset=['Top_Channel_Width_mean_mm', 'Tipo_Maschera'])
    if not df_width.empty:
        fig, axes = plt.subplots(1, len(posizioni), figsize=(8 * len(posizioni), 5), squeeze=False)
        for i, pos in enumerate(posizioni):
            ax = axes[0, i]
            subset = df_width if pos == 'All' else df_width[df_width['Posizione'] == pos]
            if subset.empty:
                ax.set_title(f"Pos: {pos} (No Data)"); ax.axis('off')
                continue
                
            sns.boxplot(data=subset, x='Tipo_Maschera', y='Top_Channel_Width_mean_mm',
                        hue='Tipo_Maschera', palette=mask_palette, legend=False, ax=ax)
            ax.set_title(f'Top View Spark Width (Pos: {pos})')
            ax.set_xlabel('Mask Type')
            ax.set_ylabel('Spark Width (mm)' if i == 0 else '')
            
        fig.tight_layout()
        fig.savefig(output_dir / '05a_top_spark_width.png', dpi=150)
        plt.close(fig)
        log.info("Grafico 5a salvato: 05a_top_spark_width.png")
    else:
        log.warning("Grafico 5a saltato: dati Top_Channel_Width_mean_mm insufficienti")

    # ── 4.5b Top View Spark Length by Mask ───────────────────────
    df_len = df_clean.dropna(subset=['Top_Plasma_Length_mm', 'Tipo_Maschera'])
    if not df_len.empty:
        fig, axes = plt.subplots(1, len(posizioni), figsize=(8 * len(posizioni), 5), squeeze=False)
        for i, pos in enumerate(posizioni):
            ax = axes[0, i]
            subset = df_len if pos == 'All' else df_len[df_len['Posizione'] == pos]
            if subset.empty:
                ax.set_title(f"Pos: {pos} (No Data)"); ax.axis('off')
                continue
                
            sns.boxplot(data=subset, x='Tipo_Maschera', y='Top_Plasma_Length_mm',
                        hue='Tipo_Maschera', palette=mask_palette, legend=False, ax=ax)
            ax.set_title(f'Top View Spark Length (Pos: {pos})')
            ax.set_xlabel('Mask Type')
            ax.set_ylabel('Spark Length (mm)' if i == 0 else '')
            
        fig.tight_layout()
        fig.savefig(output_dir / '05b_top_spark_length.png', dpi=150)
        plt.close(fig)
        log.info("Grafico 5b salvato: 05b_top_spark_length.png")
    else:
        log.warning("Grafico 5b saltato: dati Top_Plasma_Length_mm insufficienti")

    # ── 4.6 Side View — Nozzle Distance (ΔY) by Mask ─────────────
    df_nozzle = df_clean.dropna(subset=['Nozzle_Distance_Y_mm', 'Tipo_Maschera'])
    if not df_nozzle.empty:
        fig, axes = plt.subplots(1, len(posizioni), figsize=(8 * len(posizioni), 5), squeeze=False)
        for i, pos in enumerate(posizioni):
            ax = axes[0, i]
            subset = df_nozzle if pos == 'All' else df_nozzle[df_nozzle['Posizione'] == pos]
            if subset.empty:
                ax.set_title(f"Pos: {pos} (No Data)"); ax.axis('off')
                continue
                
            sns.boxplot(data=subset, x='Tipo_Maschera', y='Nozzle_Distance_Y_mm',
                        hue='Tipo_Maschera', palette=mask_palette, legend=False, ax=ax)
            ax.set_title(fr'Side View $\Delta Y$ Nozzle (Pos: {pos})')
            ax.set_xlabel('Mask Type')
            ax.set_ylabel(r'$\Delta Y$ (mm)' if i == 0 else '')
            
        fig.tight_layout()
        fig.savefig(output_dir / '06a_nozzle_distance_y.png', dpi=150)
        plt.close(fig)
        log.info("Grafico 6a salvato: 06a_nozzle_distance_y.png")
    else:
        log.warning("Grafico 6a saltato: dati Nozzle_Distance_Y_mm insufficienti")

    # ── 4.6b Side View — Plasma Length by Mask ────────────────────
    df_side_len = df_clean.dropna(subset=['Plasma_Length_mm', 'Tipo_Maschera'])
    if not df_side_len.empty:
        fig, axes = plt.subplots(1, len(posizioni), figsize=(8 * len(posizioni), 5), squeeze=False)
        for i, pos in enumerate(posizioni):
            ax = axes[0, i]
            subset = df_side_len if pos == 'All' else df_side_len[df_side_len['Posizione'] == pos]
            if subset.empty:
                ax.set_title(f"Pos: {pos} (No Data)"); ax.axis('off')
                continue
                
            sns.boxplot(data=subset, x='Tipo_Maschera', y='Plasma_Length_mm',
                        hue='Tipo_Maschera', palette=mask_palette, legend=False, ax=ax)
            ax.set_title(f'Side View Plasma Length (Pos: {pos})')
            ax.set_xlabel('Mask Type')
            ax.set_ylabel('Plasma Length (mm)' if i == 0 else '')
            
        fig.tight_layout()
        fig.savefig(output_dir / '06b_side_plasma_length.png', dpi=150)
        plt.close(fig)
        log.info("Grafico 6b salvato: 06b_side_plasma_length.png")
    else:
        log.warning("Grafico 6b saltato: dati Plasma_Length_mm insufficienti")

    # ── 4.7 Correlazione Carica vs Lunghezza Canale ───────────────────
    # Due pannelli (Side mm + Top px), suddivisi per Giorno,
    # scatter colorato per Tipo_Maschera con fit lineare + R² per maschera.
    if 'Turbo_ICT_Charge' in df.columns:
        df['Turbo_ICT_Charge'] = pd.to_numeric(
            df['Turbo_ICT_Charge'], errors='coerce')
        df_clean['Turbo_ICT_Charge'] = pd.to_numeric(
            df_clean['Turbo_ICT_Charge'], errors='coerce')

        length_configs = [
            ('Plasma_Length_mm', 'Side View — Spark Length (mm)', 'mm'),
            ('Top_Plasma_Length_mm', 'Top View — Spark Length (mm)', 'mm'),
        ]

        for length_col, panel_title, unit in length_configs:
            if length_col not in df.columns:
                log.warning(
                    "Grafico 7 saltato per %s: colonna assente", length_col)
                continue

            df_corr = df_clean.dropna(
                subset=['Turbo_ICT_Charge', length_col,
                        'Tipo_Maschera', 'Giorno']).copy()
            df_corr['Giorno'] = df_corr['Giorno'].astype(str)

            if df_corr.empty:
                log.warning(
                    "Grafico 7 saltato per %s: dati insufficienti",
                    length_col)
                continue

            giorni = sorted(df_corr['Giorno'].unique())
            n_giorni = len(giorni)
            n_pos = len(posizioni)

            fig, axes = plt.subplots(
                n_giorni, n_pos, figsize=(6 * n_pos, 5 * n_giorni),
                sharey=True, squeeze=False)

            for i_g, giorno in enumerate(giorni):
                df_day = df_corr[df_corr['Giorno'] == giorno]
                for i_p, pos in enumerate(posizioni):
                    ax = axes[i_g, i_p]
                    df_m_subset = df_day if pos == 'All' else df_day[df_day['Posizione'] == pos]
                    
                    if df_m_subset.empty:
                        ax.set_title(f"Day {giorno} | Pos: {pos} (No Data)")
                        ax.axis('off')
                        continue

                    for mask_type, color in mask_palette.items():
                        df_m = df_m_subset[df_m_subset['Tipo_Maschera'] == mask_type]
                        if df_m.empty:
                            continue

                        x = df_m['Turbo_ICT_Charge'].values.astype(float)
                        y = df_m[length_col].values.astype(float)

                        ax.scatter(x, y, color=color, alpha=0.7,
                                   edgecolors='white', linewidths=0.5,
                                   s=40, label=mask_type, zorder=3)

                        # Fit lineare (servono almeno 2 punti)
                        if len(x) >= 2:
                            slope, intercept, r_value, p_value, std_err = linregress(x, y)
                            r2 = r_value ** 2
                            x_fit = np.linspace(x.min(), x.max(), 100)
                            y_fit = slope * x_fit + intercept
                            ax.plot(x_fit, y_fit, color=color, linewidth=1.5,
                                    linestyle='--', alpha=0.8)
                            
                            p_str = "p<0.001" if p_value < 0.001 else f"p={p_value:.3f}"
                                
                            ax.annotate(
                                f'{mask_type}: R²={r2:.3f}, {p_str}',
                                xy=(0.05, 0.95 - 0.07 * list(
                                    mask_palette.keys()).index(mask_type)),
                                xycoords='axes fraction',
                                fontsize=9, color=color,
                                fontweight='bold',
                                bbox=dict(boxstyle='round,pad=0.2',
                                          fc='white', alpha=0.7, ec=color))

                    ax.set_title(f'Day {giorno} | Pos: {pos}', fontsize=11)
                    if i_g == n_giorni - 1:
                        ax.set_xlabel('Charge (pC)')
                    if i_p == 0:
                        ax.set_ylabel(f'Spark Length ({unit})')
                    ax.legend(loc='lower right', fontsize=8)

            fig.suptitle(
                f'Charge vs Spark Length — {panel_title}',
                fontsize=13, fontweight='bold', y=1.02)
            fig.tight_layout()

            suffix = 'side' if 'Side' in panel_title else 'top'
            fname = f'07_charge_vs_length_{suffix}.png'
            fig.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
            plt.close(fig)
            log.info("Grafico 7 salvato: %s", fname)
    else:
        log.warning(
            "Grafico 7 saltato: colonna Turbo_ICT_Charge non disponibile")


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITÀ DI SALVATAGGIO
# ═══════════════════════════════════════════════════════════════════════════════

def _save_csv(df: pd.DataFrame, csv_path: Path):
    """Salva il DataFrame in CSV, escludendo colonne con array numpy."""
    array_cols = [c for c in df.columns
                  if df[c].dropna().apply(lambda v: isinstance(v, np.ndarray)).any()]
    if array_cols:
        log.info("Colonne escluse dal CSV (array numpy): %s", array_cols)
    df_export = df.drop(columns=array_cols, errors='ignore')
    # Converte N_Blobs in intero (supportando NaN) per evitare 1.0 → 1,0
    if 'N_Blobs' in df_export.columns:
        df_export['N_Blobs'] = df_export['N_Blobs'].astype('Int64')
    df_export.to_csv(csv_path, sep=';', decimal=',')
    log.info("DataFrame salvato: %s", csv_path)


def _save_beam_debug(df: pd.DataFrame, output_dir: Path):
    """Salva il report e CSV di debug della classificazione fasci."""
    if 'Beam_Type' not in df.columns:
        return
    beam_counts = df['Beam_Type'].value_counts(dropna=False)
    log.info("═══ REPORT CLASSIFICAZIONE FASCI ═══")
    for btype, count in beam_counts.items():
        label = btype if pd.notna(btype) else 'Non classificato'
        log.info("  %-15s : %d shot (%.1f%%)", label, count, 100 * count / len(df))
    debug_cols = ['Beam_Type', 'N_Blobs', 'Tipo_Maschera',
                  'Injection_Success', 'X_c', 'Y_c', 'Sigma_X', 'Sigma_Y']
    debug_cols = [c for c in debug_cols if c in df.columns]
    df_debug = df[debug_cols].copy()
    debug_csv = output_dir / 'beam_classification_debug.csv'
    df_debug.to_csv(debug_csv, sep=';', decimal=',')
    log.info("CSV debug classificazione salvato: %s", debug_csv)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Esegue la pipeline completa: per-giorno + aggregata.

    1. Scansiona EXCEL_DIR per tutti i file .xlsx
    2. Per ogni file: Fasi 1-3 + grafici giornalieri in output/<data>/
    3. Concatena tutti i giorni e genera grafici aggregati in output/aggregated/
    """
    log.info("╔══════════════════════════════════════════════╗")
    log.info("║  LWFA Stability Analysis Pipeline            ║")
    log.info("╚══════════════════════════════════════════════╝")

    master_df = None
    if RUN_MODE == "STEP_2_ANALYZE":
        if MASTER_CSV_PATH.exists():
            master_df = pd.read_csv(MASTER_CSV_PATH, sep=None, engine='python', encoding='utf-8-sig')
            log.info("Caricato master CSV per STEP 2 da: %s", MASTER_CSV_PATH)
        else:
            log.error("File master.csv non trovato: %s", MASTER_CSV_PATH)
            return None

    # ── Scoperta automatica dei file Excel ────────────────────────────
    # Escludi i file temporanei di Excel (~$*.xlsx) che sono bloccati
    excel_files = sorted(f for f in EXCEL_DIR.glob('*.xlsx')
                         if not f.name.startswith('~$'))
    if not excel_files:
        log.error("Nessun file Excel trovato in %s", EXCEL_DIR)
        return None
    log.info("Trovati %d file Excel: %s",
             len(excel_files), [f.name for f in excel_files])

    all_dfs = []

    # ── Loop per-giorno ───────────────────────────────────────────────
    for excel_path in excel_files:
        date_match = re.search(r'\d{8}', excel_path.name)
        if not date_match:
            log.warning("Impossibile estrarre data da '%s', file saltato",
                        excel_path.name)
            continue
        target_date = date_match.group(0)

        log.info("══════════════════════════════════════════════")
        log.info("  GIORNO %s  —  %s", target_date, excel_path.name)
        log.info("══════════════════════════════════════════════")

        # Fase 1
        df = phase1_ingest_excel(excel_path)

        # Fase 2 (filtra TIFF solo per questa data)
        df = phase2_map_files(df, ROOT_DIR, target_date=target_date)

        # Assicura che Giorno sia popolato anche per shot senza TIFF
        if 'Giorno' in df.columns:
            df['Giorno'] = df['Giorno'].fillna(target_date).astype(str)
        else:
            df['Giorno'] = target_date

        if RUN_MODE == "STEP_2_ANALYZE" and master_df is not None:
            day_master = master_df[master_df['Giorno'].astype(str) == target_date]
            if not day_master.empty and 'Shot' in day_master.columns:
                day_master = day_master.set_index('Shot')
                if 'Beam_Type' in day_master.columns:
                    if 'Beam_Type' not in df.columns:
                        df['Beam_Type'] = pd.Series(dtype=object)
                    else:
                        df['Beam_Type'] = df['Beam_Type'].astype(object)
                    df.update(day_master[['Beam_Type']])
                    log.info("Aggiornate etichette Beam_Type da master CSV per il giorno %s", target_date)

        # ── Selezione interattiva ROI per Pointing Lanex (Feature 1) ──
        roi = None
        roi_center = None
        first_pt_path = find_first_mask_free_pointing(df)
        if first_pt_path is not None:
            img_roi = safe_imread(first_pt_path)
            if img_roi is not None:
                log.info("Selezione ROI: immagine %s", first_pt_path)
                roi = interactive_roi_selection(
                    img_roi,
                    title=f"ROI Selection — Day {target_date}"
                )
                if roi is not None:
                    full_h, full_w = img_roi.shape[:2]
                    roi_center = {
                        'roi_cx': roi['center_x'],
                        'roi_cy': roi['center_y'],
                        'img_cx': (full_w / 2.0) - roi['x_min'],
                        'img_cy': (full_h / 2.0) - roi['y_min']
                    }
                    log.info("Centro ROI: (%.1f, %.1f) px | "
                             "Centro Immagine: (%.1f, %.1f) px",
                             roi_center['roi_cx'], roi_center['roi_cy'],
                             roi_center['img_cx'], roi_center['img_cy'])
        else:
            log.warning("Nessuna immagine Pointing mask_free "
                        "trovata per ROI selection")

        # Cartella debug per immagini diagnostiche
        day_output = OUTPUT_DIR / target_date
        debug_dir = day_output / 'debug' \
            if RUN_MODE == "STEP_2_ANALYZE" else None

        # Fase 3
        df = phase3_batch_process(
            df, roi=roi,
            run_mode=RUN_MODE,
            debug_output_dir=debug_dir,
            ts_on=TS_ON,
        )

        # ── Output giornaliero ────────────────────────────────────────
        if RUN_MODE == "STEP_2_ANALYZE":
            day_output.mkdir(parents=True, exist_ok=True)

            _save_csv(df, day_output / 'results_full.csv')
            _save_beam_debug(df, day_output)
            phase4_visualize(df, day_output, roi_center=roi_center)

            log.info("Output giorno %s completato → %s",
                     target_date, day_output)
        
        all_dfs.append(df)

    # ── Aggregazione multi-giorno ─────────────────────────────────────
    if not all_dfs:
        log.error("Nessun giorno elaborato con successo")
        return None

    if RUN_MODE == "STEP_1_CLASSIFY":
        df_all = pd.concat(all_dfs, ignore_index=False)
        df_all_reset = df_all.reset_index(names='Shot') if df_all.index.name == 'Shot' else df_all.reset_index()
        if 'index' in df_all_reset.columns and 'Shot' not in df_all_reset.columns:
            df_all_reset.rename(columns={'index': 'Shot'}, inplace=True)
        
        if 'Pointing_State' in df_all_reset.columns:
            master_export = df_all_reset[df_all_reset['Pointing_State'].astype(str).str.upper() == 'IN']
        else:
            master_export = df_all_reset
            
        cols_to_keep = ['Giorno', 'Shot', 'Beam_Type', 'N_Blobs', 'Compactness']
        cols_export = [c for c in cols_to_keep if c in master_export.columns]
        master_export = master_export[cols_export]
        
        master_export.to_csv(MASTER_CSV_PATH, sep=';', decimal=',', index=False)
        log.info("STEP 1 completato. Master CSV per classificazione salvato in: %s", MASTER_CSV_PATH)
        return df_all

    if len(all_dfs) == 1:
        log.info("Un solo giorno elaborato, skip aggregazione.")
        return all_dfs[0]

    log.info("══════════════════════════════════════════════")
    log.info("  AGGREGAZIONE: %d giorni", len(all_dfs))
    log.info("══════════════════════════════════════════════")

    # ignore_index=True evita indici Shot duplicati tra giorni diversi;
    # la colonna Giorno disambigua l'origine dei dati.
    df_all = pd.concat(all_dfs, ignore_index=True)

    agg_output = OUTPUT_DIR / 'aggregated'
    agg_output.mkdir(parents=True, exist_ok=True)

    _save_csv(df_all, agg_output / 'results_full.csv')
    _save_beam_debug(df_all, agg_output)
    phase4_visualize(df_all, agg_output)

    log.info("Output aggregato completato → %s", agg_output)
    log.info("Pipeline completata con successo per %d giorni.", len(all_dfs))
    return df_all


if __name__ == '__main__':
    df_result = main()
