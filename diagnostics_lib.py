"""
diagnostics_lib.py — Libreria Core per Diagnostica LWFA
=========================================================

Libreria pura di funzioni matematiche/fisiche per l'analisi di immagini
provenienti da esperimenti di accelerazione Laser-Plasma (LWFA).

Questa libreria è INDIPENDENTE da pandas, CSV, percorsi di file e I/O.
Tutte le funzioni accettano array NumPy e restituiscono dizionari Python.

Dipendenze
----------
- numpy
- scipy (ndimage)
"""

import numpy as np
from scipy import ndimage


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PRE-PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def subtract_background(image_array: np.ndarray, bg_array: np.ndarray) -> dict:
    """Sottrae il background da un'immagine e clip i valori negativi a zero.

    Esegue la sottrazione pixel-per-pixel dell'immagine di background
    dall'immagine di segnale. I valori risultanti negativi (dovuti a
    fluttuazioni di rumore) vengono troncati a zero.

    Parameters
    ----------
    image_array : np.ndarray (2D)
        Immagine grezza (segnale + background).
    bg_array : np.ndarray (2D)
        Immagine di background (acquisita senza segnale, tipicamente shot _0).
        Deve avere le stesse dimensioni di image_array.

    Returns
    -------
    dict
        - 'cleaned_image' : np.ndarray (float64, 2D)
            Immagine pulita con background sottratto, clippata >= 0.

    Notes
    -----
    Entrambi gli array vengono convertiti a float64 prima della sottrazione
    per evitare overflow con tipi unsigned integer (es. uint16).
    """
    cleaned = image_array.astype(np.float64) - bg_array.astype(np.float64)
    cleaned = np.clip(cleaned, 0, None)
    return {'cleaned_image': cleaned}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. LANEX STATUS CHECK
# ═══════════════════════════════════════════════════════════════════════════════

def check_lanex_status(image_array: np.ndarray, noise_threshold: float) -> dict:
    """Determina se il Lanex screen è inserito (IN) o estratto (OUT).

    Calcola la somma totale dei pixel dell'immagine. Se questa supera
    la soglia specificata, il Lanex è considerato inserito (stato IN),
    indicando la presenza di un segnale di scintillazione.

    Parameters
    ----------
    image_array : np.ndarray (2D)
        Immagine dal Lanex screen (già pulita dal background).
    noise_threshold : float
        Soglia di conteggi totali per discriminare segnale da rumore.
        Valori tipici dipendono dalla camera e dall'esposizione.

    Returns
    -------
    dict
        - 'is_in'        : bool — True se il Lanex è IN (segnale presente).
        - 'total_counts'  : float — Somma totale dei pixel.
    """
    total = float(np.sum(image_array.astype(np.float64)))
    is_in = total > noise_threshold
    return {'is_in': is_in, 'total_counts': total}


# ═══════════════════════════════════════════════════════════════════════════════
# 2B. BEAM CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def classify_beam(image_array: np.ndarray,
                  peak_threshold_hi: float = 0.35,
                  peak_threshold_lo: float = 0.15,
                  min_blob_area_px: int = 2500,
                  diffuse_area_frac: float = 0.08,
                  diffuse_compactness_min: float = 0.10,
                  smoothing_sigma: float = 2.0) -> dict:
    """Classifica il tipo di fascio di elettroni presente nell'immagine.

    Utilizza un algoritmo a due stadi per la classificazione:

    **Stadio 1 — Soglia alta (peak_threshold_hi):**
    Binarizza a soglia alta per isolare i core dei fasci e conta i blob.
    Se ≥ 2 blob validi → 'Multiple'.

    **Stadio 2 — Soglia bassa (peak_threshold_lo) + Compattezza:**
    Se allo stadio 1 si trova 0 o 1 blob, si ricalcola l'area a soglia
    bassa per catturare l'alone diffuso. Se l'area a soglia bassa supera
    diffuse_area_frac dell'immagine E la compattezza (intensità nel blob
    @soglia_alta / intensità totale) supera diffuse_compactness_min,
    il fascio è 'Diffused'. Altrimenti 'Collimated' (o 'Null').

    Categorie:
    - 'Null'       : nessun segnale significativo
    - 'Diffused'   : fascio con alone esteso (divergenza alta)
    - 'Multiple'   : più blob distinti (beam splitting)
    - 'Collimated' : singolo blob compatto (fascio ideale)

    Parameters
    ----------
    image_array : np.ndarray (2D)
        Immagine dal Pointing Lanex (preferibilmente con background sottratto).
    peak_threshold_hi : float, default 0.35
        Soglia alta (frazione del picco) per la separazione dei fasci multipli.
    peak_threshold_lo : float, default 0.15
        Soglia bassa (frazione del picco) per la misura dell'estensione diffusa.
    min_blob_area_px : int, default 2500
        Area minima in pixel per considerare un blob come fascio reale.
    diffuse_area_frac : float, default 0.08
        Frazione dell'area totale dell'immagine: se l'area dei blob a soglia
        bassa la supera (e la compattezza è sopra soglia), → 'Diffused'.
    diffuse_compactness_min : float, default 0.10
        Soglia minima di compattezza (intensità nel core / intensità totale).
        Se inferiore, il segnale nel core è trascurabile rispetto al fondo
        e il fascio viene considerato 'Collimated' (il fondo non è alone).
    smoothing_sigma : float, default 2.0
        σ del filtro gaussiano applicato solo per la ricerca dei blob.
        L'immagine originale non viene modificata.

    Returns
    -------
    dict
        - 'label'        : str — 'Null', 'Diffused', 'Multiple', 'Collimated'
        - 'n_blobs'      : int — Numero di blob validi trovati (soglia alta).
        - 'blob_areas'   : list[int] — Aree in pixel di ogni blob valido.
        - 'primary_mask' : np.ndarray (bool, 2D) o None
            Maschera del blob per i Collimati (True = pixel del fascio);
            None per tutte le altre categorie.
    """
    img = image_array.astype(np.float64)
    total_pixels = img.size  # H × W
    total_intensity = float(np.sum(img))

    # 1. Smoothing per la sola ricerca dei blob
    img_smooth = ndimage.gaussian_filter(img, sigma=smoothing_sigma)

    # 2. Soglia assoluta antirumore
    peak_val = np.percentile(img_smooth, 99.9)
    if peak_val < 35.0:
        return {'label': 'Null', 'n_blobs': 0, 'blob_areas': [],
                'primary_mask': None}

    # ══════════════════════════════════════════════════════════════════════
    # STADIO 1 — Soglia alta: conta i blob per rilevare i Multiple
    # ══════════════════════════════════════════════════════════════════════
    binary_hi = img_smooth > (peak_threshold_hi * peak_val)
    structure = ndimage.generate_binary_structure(2, 2)  # 8-conn
    labeled_hi, n_raw = ndimage.label(binary_hi, structure=structure)

    # Filtra per area minima
    raw_areas = []
    raw_labels = []
    for lbl in range(1, n_raw + 1):
        area = int(np.sum(labeled_hi == lbl))
        if area >= min_blob_area_px:
            raw_areas.append(area)
            raw_labels.append(lbl)

    # Filtra per dimensione relativa (blob secondari ≥ 5% del primario)
    blob_areas = []
    valid_labels = []
    if len(raw_areas) > 0:
        max_a = max(raw_areas)
        for a, lbl in zip(raw_areas, raw_labels):
            if a >= 0.05 * max_a:
                blob_areas.append(a)
                valid_labels.append(lbl)

    n_blobs = len(valid_labels)

    # Se ≥ 2 blob a soglia alta → potenziale Multiple, ma verifica che
    # non sia un diffuso frammentato (alone spezzato dal rumore)
    if n_blobs >= 2:
        # Calcola area a soglia bassa e compattezza totale di tutti i blob
        binary_lo_m = img_smooth > (peak_threshold_lo * peak_val)
        labeled_lo_m, n_raw_lo_m = ndimage.label(binary_lo_m, structure=structure)
        areas_lo_m = [int(np.sum(labeled_lo_m == lbl))
                      for lbl in range(1, n_raw_lo_m + 1)
                      if int(np.sum(labeled_lo_m == lbl)) >= 1250]
        total_area_lo_m = sum(areas_lo_m)

        # Compattezza totale: intensità in TUTTI i blob@alta / intensità totale
        mask_all = np.zeros(img.shape, dtype=bool)
        for lbl in valid_labels:
            mask_all |= (labeled_hi == lbl)
        compactness_all = float(np.sum(img * mask_all)) / total_intensity if total_intensity > 0 else 0.0

        diffuse_area_limit = diffuse_area_frac * total_pixels
        if total_area_lo_m > diffuse_area_limit and compactness_all < diffuse_compactness_min:
            # Alone enorme + energia nei blob irrisoria → diffuso frammentato
            return {
                'label': 'Diffused',
                'n_blobs': n_blobs,
                'blob_areas': blob_areas,
                'compactness': compactness_all,
                'primary_mask': None
            }

        return {
            'label': 'Multiple',
            'n_blobs': n_blobs,
            'blob_areas': blob_areas,
            'compactness': compactness_all,
            'primary_mask': None
        }

    # ══════════════════════════════════════════════════════════════════════
    # STADIO 2 — Soglia bassa: distingui Diffused da Collimated
    # ══════════════════════════════════════════════════════════════════════
    binary_lo = img_smooth > (peak_threshold_lo * peak_val)
    labeled_lo, n_raw_lo = ndimage.label(binary_lo, structure=structure)

    # Area totale dei blob a soglia bassa (filtro area minima a 1250 px
    # per coerenza con il vecchio algoritmo sulla soglia bassa)
    areas_lo = []
    for lbl in range(1, n_raw_lo + 1):
        area = int(np.sum(labeled_lo == lbl))
        if area >= 1250:
            areas_lo.append(area)
    total_area_lo = sum(areas_lo)

    # Compattezza: quanta intensità è concentrata nel core (blob@soglia_alta)
    if n_blobs == 1 and total_intensity > 0:
        mask_hi = (labeled_hi == valid_labels[0])
        intensity_in_core = float(np.sum(img * mask_hi))
        compactness = intensity_in_core / total_intensity
    else:
        compactness = 0.0

    # Regole di decisione
    diffuse_area_limit = diffuse_area_frac * total_pixels

    if n_blobs == 0:
        # Nessun blob a soglia alta: potrebbe essere un diffuso molto esteso
        # oppure un Null borderline (peak appena sopra 35)
        if total_area_lo > diffuse_area_limit:
            label = 'Diffused'
        else:
            label = 'Null'
        primary_mask = None

    else:  # n_blobs == 1
        # Singolo blob a soglia alta: Collimated o Diffused?
        if total_area_lo > diffuse_area_limit and compactness >= diffuse_compactness_min:
            # L'alone a soglia bassa è vasto E una quota rilevante di intensità
            # è nel core → il fascio è reale ma esteso → Diffused
            label = 'Diffused'
            primary_mask = None
        else:
            # L'alone è piccolo, o la compattezza è troppo bassa (= il fondo
            # largo è solo rumore, non alone reale) → Collimated
            label = 'Collimated'
            primary_mask = (labeled_hi == valid_labels[0])

    return {
        'label': label,
        'n_blobs': n_blobs,
        'blob_areas': blob_areas,
        'compactness': compactness,
        'primary_mask': primary_mask
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. POINTING PROFILE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_pointing_profile(image_array: np.ndarray,
                             blob_mask: np.ndarray = None) -> dict:
    """Analizza il profilo spaziale del fascio dal Pointing Lanex.

    Calcola:
    - Intensità totale integrata
    - Centroide (X_c, Y_c) tramite scipy.ndimage.center_of_mass
    - Dimensioni RMS (Sigma_X, Sigma_Y) come deviazione standard
      spaziale pesata sull'intensità (momento secondo centrato)

    Parameters
    ----------
    image_array : np.ndarray (2D)
        Immagine del Pointing Lanex (preferibilmente con background sottratto).
        Convenzione: asse 0 = righe (Y), asse 1 = colonne (X).
    blob_mask : np.ndarray (bool, 2D) or None, optional
        Maschera del blob identificato da classify_beam().
        True = pixel del fascio (mantieni), False = background (azzera a 0).
        Se None, l'analisi viene eseguita sull'intera immagine.

    Returns
    -------
    dict
        - 'Total_Intensity' : float — Intensità integrata totale.
        - 'X_c'             : float — Posizione X del centroide (colonna, pixel).
        - 'Y_c'             : float — Posizione Y del centroide (riga, pixel).
        - 'Sigma_X'         : float — Larghezza RMS lungo X (pixel).
        - 'Sigma_Y'         : float — Larghezza RMS lungo Y (pixel).

    Notes
    -----
    Le dimensioni RMS sono calcolate come:
        Sigma_X = sqrt( Σ I(x,y) · (x - X_c)² / Σ I(x,y) )
    Se l'intensità totale è <= 0, tutti i parametri geometrici sono NaN.
    """
    img = image_array.astype(np.float64)

    # Se c'è una maschera, MANTIENI solo il fascio (azzera tutto il resto)
    # blob_mask[i,j] == True  → pixel del fascio  → mantieni valore
    # blob_mask[i,j] == False → background/rumore  → azzera a 0
    if blob_mask is not None:
        img = img * blob_mask.astype(np.float64)

    total_intensity = float(np.sum(img))

    if total_intensity <= 0:
        return {
            'Total_Intensity': 0.0,
            'X_c': np.nan, 'Y_c': np.nan,
            'Sigma_X': np.nan, 'Sigma_Y': np.nan
        }

    # scipy.ndimage.center_of_mass → (row_center, col_center) = (Y, X)
    com = ndimage.center_of_mass(img)
    Y_c, X_c = float(com[0]), float(com[1])

    # Griglie di coordinate per il calcolo dei momenti secondi
    rows, cols = np.indices(img.shape, dtype=np.float64)

    # Deviazione standard pesata sull'intensità
    Sigma_Y = float(np.sqrt(np.sum(img * (rows - Y_c) ** 2) / total_intensity))
    Sigma_X = float(np.sqrt(np.sum(img * (cols - X_c) ** 2) / total_intensity))

    return {
        'Total_Intensity': total_intensity,
        'X_c': X_c,
        'Y_c': Y_c,
        'Sigma_X': Sigma_X,
        'Sigma_Y': Sigma_Y
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ENERGY SPECTRUM ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def _find_fwhm(profile: np.ndarray, peak_idx: int) -> tuple:
    """Calcola la FWHM di un profilo 1D cercando dal picco verso l'esterno.

    Utilizza interpolazione lineare per determinare i punti esatti
    in cui il profilo attraversa il livello di half-maximum.

    Parameters
    ----------
    profile : np.ndarray (1D)
        Profilo di intensità.
    peak_idx : int
        Indice del picco nel profilo.

    Returns
    -------
    tuple (fwhm, left_edge, right_edge)
        FWHM in pixel e posizioni interpolate dei bordi.
        Se non determinabili, (NaN, NaN, NaN).
    """
    half_max = profile[peak_idx] / 2.0

    # ── Ricerca bordo sinistro (dal picco verso sinistra) ──
    left_edge = 0.0
    for i in range(peak_idx, 0, -1):
        if profile[i - 1] < half_max:
            y_lo, y_hi = profile[i - 1], profile[i]
            denom = y_hi - y_lo
            if denom > 0:
                left_edge = (i - 1) + (half_max - y_lo) / denom
            else:
                left_edge = float(i)
            break

    # ── Ricerca bordo destro (dal picco verso destra) ──
    # Il crossing è tra pixel i e i+1: interpolazione lineare corretta.
    right_edge = float(len(profile) - 1)
    for i in range(peak_idx, len(profile) - 1):
        if profile[i + 1] < half_max:
            denom = profile[i] - profile[i + 1]
            if denom > 0:
                right_edge = i + (profile[i] - half_max) / denom
            else:
                right_edge = float(i)
            break

    fwhm = right_edge - left_edge
    if fwhm <= 0:
        return (np.nan, np.nan, np.nan)
    return (float(fwhm), float(left_edge), float(right_edge))


def analyze_energy_spectrum(image_array: np.ndarray,
                            calibration_func=None) -> dict:
    """Analizza lo spettro energetico dal Lanex Andor (spettrometro dispersivo).

    Integra verticalmente l'immagine per ottenere un profilo spettrale 1D
    lungo l'asse orizzontale (X). Trova il picco e calcola la FWHM
    (Full Width at Half Maximum) come misura dello spread energetico.

    Parameters
    ----------
    image_array : np.ndarray (2D)
        Immagine dall'Andor Lanex (spettrometro magnetico).
        Asse 0 = spaziale (Y), Asse 1 = dispersivo (X/energia).
    calibration_func : callable or None, optional
        Funzione f(pixel) -> MeV per la conversione da pixel a energia.
        Se None, i risultati sono solo in unità pixel.

    Returns
    -------
    dict
        - 'spectrum_1d'      : np.ndarray — Profilo spettrale 1D integrato.
        - 'Peak_X'           : int — Posizione pixel del picco.
        - 'Energy_Spread_px' : float — FWHM in pixel.
        - 'Peak_MeV'         : float — (solo se calibration_func) Energia picco.
        - 'Spread_MeV'       : float — (solo se calibration_func) Spread in MeV.

    Notes
    -----
    La FWHM viene calcolata cercando dal picco verso l'esterno i punti
    in cui il profilo scende al 50% del massimo, con interpolazione lineare.
    """
    img = image_array.astype(np.float64)

    # Integrazione verticale: somma lungo le righe → profilo 1D(X)
    spectrum_1d = np.sum(img, axis=0)
    peak_val = np.max(spectrum_1d)

    # Caso degenere: nessun segnale
    if peak_val <= 0:
        result = {
            'spectrum_1d': spectrum_1d,
            'Peak_X': np.nan,
            'Energy_Spread_px': np.nan
        }
        if calibration_func is not None:
            result['Peak_MeV'] = np.nan
            result['Spread_MeV'] = np.nan
        return result

    Peak_X = int(np.argmax(spectrum_1d))

    # ── Calcolo FWHM con interpolazione ────────────────────────────────
    fwhm, left_edge, right_edge = _find_fwhm(spectrum_1d, Peak_X)

    result = {
        'spectrum_1d': spectrum_1d,
        'Peak_X': Peak_X,
        'Energy_Spread_px': fwhm
    }

    # ── Conversione opzionale pixel → MeV ──────────────────────────────
    if calibration_func is not None:
        result['Peak_MeV'] = float(calibration_func(Peak_X))
        if not np.isnan(fwhm):
            e_left = calibration_func(left_edge)
            e_right = calibration_func(right_edge)
            result['Spread_MeV'] = float(abs(e_right - e_left))
        else:
            result['Spread_MeV'] = np.nan

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 5. PLASMA CHANNEL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_plasma_channel(image_array: np.ndarray,
                           threshold_fraction: float = 0.2) -> dict:
    """Analizza il canale di plasma dall'immagine laterale (Side Imaging).

    Identifica la "scintilla" di plasma nell'immagine laterale.
    Calcola la posizione del centroide lungo l'asse di propagazione Z
    e la lunghezza del plasma come numero di pixel che superano una
    frazione dell'intensità massima.

    Parameters
    ----------
    image_array : np.ndarray (2D)
        Immagine dal Side Imaging (vista laterale del canale plasma).
        Asse 0 = trasversale (Y), Asse 1 = propagazione (Z).
    threshold_fraction : float, default 0.2
        Frazione dell'intensità massima usata come soglia per definire
        l'estensione del plasma (default: 20%).

    Returns
    -------
    dict
        - 'Plasma_Z_Position' : float — Centroide lungo Z (pixel).
        - 'Plasma_Length'      : int — Lunghezza del plasma in pixel.
        - 'Max_Intensity'      : float — Intensità massima del profilo Z.

    Notes
    -----
    Il profilo lungo Z è ottenuto integrando l'immagine lungo l'asse
    trasversale (somma sulle righe). La Plasma_Length conta il numero
    di pixel il cui valore nel profilo integrato supera
    threshold_fraction × max_intensity.
    """
    img = image_array.astype(np.float64)

    # Integrazione lungo l'asse trasversale (righe) → profilo 1D lungo Z
    profile_z = np.sum(img, axis=0)

    max_intensity = float(np.max(profile_z))

    if max_intensity <= 0:
        # Plasma_Length è NaN (float) nel caso degenere; pandas lo gestisce
        # correttamente come valore mancante nella colonna int/float del DF.
        return {
            'Plasma_Z_Position': np.nan,
            'Plasma_Length': np.nan,
            'Max_Intensity': 0.0
        }

    # Centroide pesato sull'intensità lungo Z
    z_indices = np.arange(len(profile_z), dtype=np.float64)
    total = np.sum(profile_z)
    Plasma_Z_Position = float(np.sum(z_indices * profile_z) / total)

    # Lunghezza: conteggio pixel sopra la soglia frazionaria
    threshold = threshold_fraction * max_intensity
    above_threshold = profile_z >= threshold
    Plasma_Length = int(np.sum(above_threshold))

    return {
        'Plasma_Z_Position': Plasma_Z_Position,
        'Plasma_Length': Plasma_Length,
        'Max_Intensity': max_intensity
    }
