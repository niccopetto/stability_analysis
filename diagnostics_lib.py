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
- scipy (ndimage, optimize, special)
- skimage (measure)
"""

import logging

import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.special import voigt_profile
import cv2

log = logging.getLogger(__name__)


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
# 5. PLASMA CHANNEL ANALYSIS — HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def _robust_z_profile(image: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    """Compute a hot-spot-immune 1D profile along Z.

    Instead of summing or taking the max along Y (axis 0), this uses
    the given percentile (default 99.5%), which is robust to isolated bright 
    pixels while correctly capturing the true plasma signal even for 
    narrow channels.

    Parameters
    ----------
    image : np.ndarray (2D)
        Cleaned plasma image (float64).
    percentile : float, default 99.5
        Percentile to use.

    Returns
    -------
    np.ndarray (1D)
        One value per column (Z pixel).
    """
    return np.percentile(image.astype(np.float64), percentile, axis=0)


def _voigt_model(x, center, sigma, gamma, peak_height, background):
    """Voigt profile model for ``curve_fit``.

    Parameters
    ----------
    x : array-like   — pixel coordinates (Y positions).
    center : float   — profile centre.
    sigma : float    — Gaussian width (>0).
    gamma : float    — Lorentzian width (>0).
    peak_height : float — peak amplitude.
    background : float — constant offset.
    """
    from scipy.special import voigt_profile
    v = voigt_profile(x - center, sigma, gamma)
    v_max = voigt_profile(0, sigma, gamma)
    return peak_height * (v / v_max) + background


def _find_fwhm_edges(profile: np.ndarray, peak_idx: int,
                     half_max: float) -> tuple:
    """Find left and right edges where *profile* drops below *half_max*.

    Walks outward from *peak_idx* and uses linear interpolation between
    the last pixel above *half_max* and the first below it to obtain
    sub-pixel accuracy.

    Parameters
    ----------
    profile : np.ndarray (1D)
        Intensity profile (smoothed or raw).
    peak_idx : int
        Index of the peak / centre of the plateau.
    half_max : float
        Threshold intensity to define the edges.

    Returns
    -------
    (y_left, y_right) : tuple of float
        Sub-pixel positions of the two edges, or ``np.nan`` when an
        edge cannot be found.
    """
    n = len(profile)

    # ── Left edge ──
    y_left = np.nan
    for i in range(peak_idx, 0, -1):
        if profile[i] < half_max <= profile[i - 1]:
            # should not happen (rising towards peak), skip
            continue
        if profile[i - 1] < half_max <= profile[i]:
            # linear interpolation
            frac = (half_max - profile[i - 1]) / (profile[i] - profile[i - 1])
            y_left = (i - 1) + frac
            break
    # If we hit the edge of the array without crossing, check boundary
    if np.isnan(y_left) and profile[0] < half_max:
        y_left = 0.0

    # ── Right edge ──
    y_right = np.nan
    for i in range(peak_idx, n - 1):
        if profile[i] >= half_max > profile[i + 1]:
            # linear interpolation
            frac = (half_max - profile[i + 1]) / (profile[i] - profile[i + 1])
            y_right = (i + 1) - frac
            break
    # If we hit the edge of the array without crossing, check boundary
    if np.isnan(y_right) and profile[-1] < half_max:
        y_right = float(n - 1)

    return y_left, y_right

def _compute_centroid_with_errors(image: np.ndarray) -> dict:
    """Compute 2D intensity-weighted centroid and its uncertainty.

    The standard error on the centroid is estimated as:
        ``σ_c = σ_spatial / sqrt(N_eff)``
    where ``N_eff ≈ ΣI / I_max`` is the effective number of
    independent intensity samples.

    Parameters
    ----------
    image : np.ndarray (2D, float64, non-negative)

    Returns
    -------
    dict
        ``Y_c``, ``Z_c``           — centroid (px)
        ``Y_c_err``, ``Z_c_err``   — standard errors (px)
        ``Sigma_Y``, ``Sigma_Z``   — RMS widths (px)
        ``total_intensity``        — ΣI
    """
    total = float(np.sum(image))
    if total <= 0:
        return {
            'Y_c': np.nan, 'Z_c': np.nan,
            'Y_c_err': np.nan, 'Z_c_err': np.nan,
            'Sigma_Y': np.nan, 'Sigma_Z': np.nan,
            'total_intensity': 0.0,
        }

    com = ndimage.center_of_mass(image)
    Y_c, Z_c = float(com[0]), float(com[1])

    rows, cols = np.indices(image.shape, dtype=np.float64)
    Sigma_Y = float(np.sqrt(np.sum(image * (rows - Y_c) ** 2) / total))
    Sigma_Z = float(np.sqrt(np.sum(image * (cols - Z_c) ** 2) / total))

    max_int = float(np.max(image))
    N_eff = max(total / max_int, 1.0) if max_int > 0 else 1.0

    Y_c_err = Sigma_Y / np.sqrt(N_eff)
    Z_c_err = Sigma_Z / np.sqrt(N_eff)

    log.debug("Centroid: Z_c=%.2f±%.2f, Y_c=%.2f±%.2f, N_eff=%.0f",
              Z_c, Z_c_err, Y_c, Y_c_err, N_eff)

    return {
        'Y_c': Y_c, 'Z_c': Z_c,
        'Y_c_err': Y_c_err, 'Z_c_err': Z_c_err,
        'Sigma_Y': Sigma_Y, 'Sigma_Z': Sigma_Z,
        'total_intensity': total,
    }


def _compute_plasma_length(profile_z: np.ndarray,
                           threshold_fraction: float,
                           virtual_peak: float) -> dict:
    """Compute plasma length as the longest contiguous segment above threshold.

    Instead of counting *all* pixels above threshold (which includes
    isolated noise clusters), this function identifies the longest
    uninterrupted run of pixels above the threshold level.  This
    provides a physically meaningful measurement of the continuous
    plasma channel.

    The uncertainty is estimated by counting the pixels that lie
    within ±5 % of the virtual peak from the threshold level
    ("boundary pixels") at the edges of the longest segment.

    Returns
    -------
    dict
        ``length``      — length of the longest contiguous segment (px)
        ``length_err``  — uncertainty estimate (px)
        ``threshold``   — absolute threshold value used
        ``plasma_start``— start index of the longest segment (px)
        ``plasma_end``  — end index (inclusive) of the longest segment (px)
    """
    threshold = threshold_fraction * virtual_peak
    above = profile_z >= threshold

    # ── Find the longest contiguous run of True values ──
    # Label connected regions in the 1D boolean array
    labeled, n_segments = ndimage.label(above)

    if n_segments == 0:
        log.debug("Plasma length: 0 px (no segment above threshold)")
        return {'length': 0, 'length_err': 1,
                'threshold': threshold,
                'plasma_start': np.nan, 'plasma_end': np.nan}

    # Find the largest segment by pixel count
    best_label = 0
    best_length = 0
    for seg_id in range(1, n_segments + 1):
        seg_mask = (labeled == seg_id)
        seg_len = int(np.sum(seg_mask))
        if seg_len > best_length:
            best_length = seg_len
            best_label = seg_id

    # Extract start and end indices of the best segment
    seg_indices = np.where(labeled == best_label)[0]
    plasma_start = int(seg_indices[0])
    plasma_end = int(seg_indices[-1])
    length = plasma_end - plasma_start + 1

    # ── Uncertainty: boundary pixels near the threshold at segment edges ──
    boundary_band = 0.05 * virtual_peak
    # Only count boundary pixels within a small neighbourhood of the edges
    edge_margin = max(length // 10, 5)  # look ±10% of length or ±5 px
    left_zone = slice(max(plasma_start - edge_margin, 0),
                      min(plasma_start + edge_margin, len(profile_z)))
    right_zone = slice(max(plasma_end - edge_margin, 0),
                       min(plasma_end + edge_margin + 1, len(profile_z)))
    boundary_left = np.sum(
        np.abs(profile_z[left_zone] - threshold) < boundary_band)
    boundary_right = np.sum(
        np.abs(profile_z[right_zone] - threshold) < boundary_band)
    length_err = max(int(boundary_left + boundary_right), 1)

    log.debug("Plasma length: %d px [%d–%d] (err: ±%d px, threshold: %.2f, "
              "%d segments found)",
              length, plasma_start, plasma_end, length_err, threshold,
              n_segments)
    return {'length': length, 'length_err': length_err,
            'threshold': threshold,
            'plasma_start': plasma_start, 'plasma_end': plasma_end}


# ═══════════════════════════════════════════════════════════════════════════════
# 5A-HELPER. SHARED TRANSVERSE AXIS EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_transverse_axis(
        image: np.ndarray,
        active_cols: np.ndarray,
        saturation_value: float = 255.0,
        fwhm_fraction_saturated: float = 0.8,
        fwhm_fraction_unsaturated: float = 0.5,
        smoothing_sigma: float = 1.0) -> dict:
    """Extract the transverse centre and width for each active column.

    This helper is shared between Side View and Top View analyses.
    For each column Z in ``active_cols`` it determines whether the
    transverse profile is saturated or not and applies the appropriate
    centre-finding and width-measuring strategy:

    * **Saturated columns:** centre = midpoint of the saturated
      plateau; width measured at ``fwhm_fraction_saturated ×
      saturation_value``.
    * **Unsaturated columns:** centre = smoothed maximum (parabolic
      interpolation for sub-pixel accuracy); width measured at
      ``fwhm_fraction_unsaturated × peak_value``.

    Parameters
    ----------
    image : np.ndarray (2D, float64)
        The cleaned image.
    active_cols : np.ndarray (1D, int)
        Column indices (Z positions) to analyse.
    saturation_value : float
        ADC saturation level (default 255 for 8-bit).
    fwhm_fraction_saturated : float
        Fraction of ``saturation_value`` for FWHM edges in saturated
        columns.
    fwhm_fraction_unsaturated : float
        Fraction of column peak for FWHM edges in unsaturated columns.
    smoothing_sigma : float
        Gaussian σ for smoothing unsaturated profiles.

    Returns
    -------
    dict
        ``axis_z``       — np.ndarray, Z positions analysed
        ``axis_y``       — np.ndarray, transverse centre per column
        ``axis_y_err``   — np.ndarray, uncertainty on centre
        ``axis_width``   — np.ndarray, FWHM per column
        ``axis_y_left``  — np.ndarray, left FWHM edge
        ``axis_y_right`` — np.ndarray, right FWHM edge
        ``saturated_count`` — int, number of saturated columns
    """
    n_rows = image.shape[0]
    sat_thresh = saturation_value - 1.0

    axis_z_list = []
    axis_y_list = []
    axis_y_err_list = []
    axis_width_list = []
    axis_y_left_list = []
    axis_y_right_list = []
    saturated_count = 0

    for z in active_cols:
        col_profile = image[:, z]

        # — Check for saturated plateau —
        saturated = col_profile >= sat_thresh
        sat_indices = np.where(saturated)[0]

        if len(sat_indices) >= 3:
            # ── SATURATED COLUMN ──
            y_start = int(sat_indices[0])
            y_end = int(sat_indices[-1])
            center = (y_start + y_end) / 2.0

            half_max = fwhm_fraction_saturated * saturation_value
            mid_idx = int(round(center))
            y_left, y_right = _find_fwhm_edges(
                col_profile, mid_idx, half_max)

            if not np.isnan(y_left) and not np.isnan(y_right):
                width = y_right - y_left
                center_err = width / 2.0
            else:
                width = float(y_end - y_start)
                center_err = width / 2.0

            saturated_count += 1

        else:
            # ── UNSATURATED COLUMN ──
            col_smooth = ndimage.gaussian_filter1d(
                col_profile, sigma=smoothing_sigma)
            max_idx = int(np.argmax(col_smooth))

            # Parabolic interpolation for sub-pixel accuracy
            if 0 < max_idx < n_rows - 1:
                y1 = col_smooth[max_idx - 1]
                y2 = col_smooth[max_idx]
                y3 = col_smooth[max_idx + 1]
                denom = (y1 - 2 * y2 + y3)
                if denom != 0:
                    center = max_idx - 0.5 * (y3 - y1) / denom
                else:
                    center = float(max_idx)
            else:
                center = float(max_idx)

            peak_val = col_smooth[max_idx]
            if peak_val > 0:
                half_max = fwhm_fraction_unsaturated * peak_val
                y_left, y_right = _find_fwhm_edges(
                    col_smooth, max_idx, half_max)
            else:
                y_left, y_right = np.nan, np.nan

            if not np.isnan(y_left) and not np.isnan(y_right):
                width = y_right - y_left
                center_err = width / 2.0
            else:
                width = np.nan
                center_err = np.nan

        axis_z_list.append(z)
        axis_y_list.append(center)
        axis_y_err_list.append(center_err)
        axis_width_list.append(width)
        axis_y_left_list.append(y_left if not np.isnan(y_left)
                                else np.nan)
        axis_y_right_list.append(y_right if not np.isnan(y_right)
                                 else np.nan)

    return {
        'axis_z': np.array(axis_z_list, dtype=np.float64),
        'axis_y': np.array(axis_y_list, dtype=np.float64),
        'axis_y_err': np.array(axis_y_err_list, dtype=np.float64),
        'axis_width': np.array(axis_width_list, dtype=np.float64),
        'axis_y_left': np.array(axis_y_left_list, dtype=np.float64),
        'axis_y_right': np.array(axis_y_right_list, dtype=np.float64),
        'saturated_count': saturated_count,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5A. PLASMA CHANNEL — SIDE VIEW
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_plasma_channel_side(
        cleaned_image: np.ndarray,
        interactive_roi: tuple = None,
        threshold_fraction: float = 0.05,
        px_to_mm: float = 1.0 / 179.0) -> dict:
    """Analyse the plasma channel from Side Imaging using interactive ROI.

    Uses OpenCV connected components to identify plasma blobs within the
    interactive ROI, extracts the geometric axis by computing the mean Y
    for each Z column, and fits a linear model to determine the beam angle.

    Parameters
    ----------
    cleaned_image : np.ndarray (2D)
        Background-subtracted Side Imaging image.
    interactive_roi : tuple of (x, y, w, h) or None
        ROI rectangle in pixel coordinates.  If None, analysis is skipped
        and all-NaN results are returned.
    threshold_fraction : float, default 0.05
        Fraction of the ROI maximum intensity used for blob thresholding.
    px_to_mm : float, default 1.0/179.0
        Pixel-to-mm conversion factor for Side Imaging.

    Returns
    -------
    dict
        Numeric results including plasma position, length, beam angle,
        nozzle distance, and a ``'debug_data'`` sub-dict with axis
        coordinates, ROI info, blob mask, and fit coefficients.
    """
    log.info("── Side View analysis (geometric blob extraction) ──")
    img = cleaned_image.astype(np.float64)

    nan_result = {
        'Plasma_Z_Position': np.nan, 'Plasma_Z_Position_err': np.nan,
        'Plasma_Z_Position_rel': np.nan,
        'Plasma_Y_Position': np.nan, 'Plasma_Y_Position_err': np.nan,
        'Plasma_Z_Centroid': np.nan, 'Plasma_Z_Centroid_err': np.nan,
        'Plasma_Y_Centroid': np.nan, 'Plasma_Y_Centroid_err': np.nan,
        'Plasma_Length': np.nan, 'Plasma_Length_err': np.nan,
        'Plasma_Length_mm': np.nan, 'Plasma_Length_mm_err': np.nan,
        'Max_Intensity': 0.0,
        'Nozzle_Tip_Z': np.nan, 'Nozzle_Tip_Y': np.nan,
        'Nozzle_Distance_Y': np.nan, 'Nozzle_Distance_Y_err': np.nan,
        'Nozzle_Distance_Y_mm': np.nan,
        'Nozzle_Distance_Y_mm_err': np.nan,
        'Side_Beam_Angle_mrad': np.nan,
        'Side_Beam_Angle_mrad_err': np.nan,
        'Side_Channel_Width_mean': np.nan,
        'Side_Saturated_Length': np.nan,
        'debug_data': {
            'axis_coords': (np.array([]), np.array([])),
            'interactive_roi': interactive_roi,
            'interactive_mask': None,
            'fit_coeffs': None
        }
    }

    if interactive_roi is None:
        log.warning("Side View: No interactive ROI provided. Skipping analysis.")
        return nan_result

    roi_x, roi_y, roi_w, roi_h = interactive_roi
    img_roi = img[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    
    if img_roi.size == 0:
        return nan_result
        
    max_val = np.max(img_roi)
    if max_val <= 0:
        return nan_result
        
    thresh_val = threshold_fraction * max_val
    mask = (img_roi >= thresh_val).astype(np.uint8)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    valid_labels = []
    for i in range(1, num_labels):
        if not np.any(labels[roi_h - 1, :] == i):
            valid_labels.append(i)
            
    if not valid_labels:
        log.warning("Side View: No valid plasma blobs found in ROI.")
        return nan_result
        
    valid_mask = np.isin(labels, valid_labels)
    interactive_mask_full = np.zeros(img.shape, dtype=bool)
    interactive_mask_full[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] = valid_mask
    
    # Estrazione asse geometrico
    y_indices, x_indices = np.where(interactive_mask_full)
    
    # Raggruppa Y per ogni Z e calcola il punto medio
    unique_z = np.unique(x_indices)
    axis_z_list = []
    axis_y_list = []
    
    for z in unique_z:
        ys_for_z = y_indices[x_indices == z]
        axis_z_list.append(float(z))
        axis_y_list.append(float(np.mean(ys_for_z)))
        
    axis_z = np.array(axis_z_list)
    axis_y = np.array(axis_y_list)
    
    # Centroide Geometrico (Media dei punti asse)
    Z_centroid = float(np.mean(axis_z))
    Y_centroid = float(np.mean(axis_y))
    
    # Lunghezza Plasma dal blob
    plasma_length = float(np.max(axis_z) - np.min(axis_z)) if len(axis_z) > 0 else 0.0
    plasma_length_mm = plasma_length * px_to_mm
    
    # Altezza
    nozzle_dist_y = float((roi_y + roi_h) - Y_centroid)
    nozzle_dist_y_mm = nozzle_dist_y * px_to_mm
    
    # Fit Lineare sull'asse
    beam_angle_mrad = np.nan
    beam_angle_mrad_err = np.nan
    fit_coeffs = None
    
    if len(axis_z) > 2:
        coeffs = np.polyfit(axis_z, axis_y, deg=1, cov=True)
        fit_coeffs = coeffs[0]  # slope, intercept
        slope = coeffs[0][0]
        slope_err = np.sqrt(coeffs[1][0, 0])
        beam_angle_mrad = float(np.arctan(slope) * 1000.0)
        beam_angle_mrad_err = float(slope_err / (1 + slope**2) * 1000.0)
    elif len(axis_z) == 2:
        coeffs = np.polyfit(axis_z, axis_y, deg=1, cov=False)
        fit_coeffs = coeffs
        slope = coeffs[0]
        beam_angle_mrad = float(np.arctan(slope) * 1000.0)
        beam_angle_mrad_err = np.nan
        
    log.info("Geometric Side Analysis: Z_c=%.1f, Y_c=%.1f, Height=%.1f px, Angle=%.1f mrad",
             Z_centroid, Y_centroid, nozzle_dist_y, beam_angle_mrad)

    return {
        'Plasma_Z_Position': Z_centroid,
        'Plasma_Z_Position_err': np.nan,
        'Plasma_Z_Position_rel': np.nan,
        'Plasma_Y_Position': Y_centroid,
        'Plasma_Y_Position_err': np.nan,
        'Plasma_Z_Centroid': Z_centroid,
        'Plasma_Z_Centroid_err': np.nan,
        'Plasma_Y_Centroid': Y_centroid,
        'Plasma_Y_Centroid_err': np.nan,
        'Plasma_Length': plasma_length,
        'Plasma_Length_err': np.nan,
        'Plasma_Length_mm': plasma_length_mm,
        'Plasma_Length_mm_err': np.nan,
        'Max_Intensity': float(max_val),
        'Nozzle_Tip_Z': np.nan,
        'Nozzle_Tip_Y': np.nan,
        'Nozzle_Distance_Y': nozzle_dist_y,
        'Nozzle_Distance_Y_err': np.nan,
        'Nozzle_Distance_Y_mm': nozzle_dist_y_mm,
        'Nozzle_Distance_Y_mm_err': np.nan,
        'Side_Beam_Angle_mrad': beam_angle_mrad,
        'Side_Beam_Angle_mrad_err': beam_angle_mrad_err,
        'Side_Channel_Width_mean': np.nan,
        'Side_Saturated_Length': np.nan,
        'debug_data': {
            'axis_coords': (axis_z, axis_y),
            'interactive_roi': interactive_roi,
            'interactive_mask': interactive_mask_full,
            'fit_coeffs': fit_coeffs
        }
    }

# ═══════════════════════════════════════════════════════════════════════════════
# 5B. PLASMA CHANNEL — TOP VIEW
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_plasma_channel_top(
        cleaned_image: np.ndarray,
        threshold_fraction: float = 0.2,
        saturation_value: float = 255.0,
        fwhm_fraction_saturated: float = 0.6,
        fwhm_fraction_unsaturated: float = 0.5,
        smoothing_sigma: float = 1.0,
        px_to_mm: float = 1.0,
        compute_length: bool = True) -> dict:
    """Analyse the plasma channel from Top Imaging.

    Uses a **geometric FWHM** approach instead of fitting:

    * **Saturated columns:** centre = midpoint of the saturated plateau;
      width measured at ``fwhm_fraction_saturated × saturation_value``.
    * **Unsaturated columns:** centre = smoothed maximum (parabolic
      interpolation for sub-pixel accuracy); width measured at
      ``fwhm_fraction_unsaturated × peak_value``.

    After extracting the axis, a linear fit is performed to obtain the
    beam entrance angle.

    Parameters
    ----------
    cleaned_image : np.ndarray (2D)
        Background-subtracted Top Imaging image.
    threshold_fraction : float
        Fraction of the virtual peak for length thresholding.
    saturation_value : float
        ADC saturation value (255 for 8-bit cameras).
    fwhm_fraction_saturated : float
        Fraction of ``saturation_value`` used to define the FWHM
        edges in saturated columns (default 0.6 = 60%).
    fwhm_fraction_unsaturated : float
        Fraction of the column peak used to define the FWHM edges
        in unsaturated columns (default 0.5 = classic half-max).
    smoothing_sigma : float
        Gaussian smoothing sigma (pixels) applied to unsaturated
        column profiles before peak detection.
    px_to_mm : float
        Pixel-to-mm conversion for Top Imaging.  Default 1.0
        (uncalibrated; update when calibration is known).
    compute_length : bool
        If False (filamented shot), length is NaN.

    Returns
    -------
    dict
        Numeric results + ``'debug_data'`` with axis and width details.
    """
    log.info("── Top View analysis (geometric FWHM) ──")
    img = cleaned_image.astype(np.float64)

    # ── 1. Robust Z profile ───────────────────────────────────────
    profile_z_robust = _robust_z_profile(img)
    virtual_peak = float(np.max(profile_z_robust))
    log.debug("Virtual peak (99.5th pctl): %.2f", virtual_peak)

    nan_result = {
        'Plasma_Z_Position': np.nan, 'Plasma_Z_Position_err': np.nan,
        'Plasma_Y_Position': np.nan, 'Plasma_Y_Position_err': np.nan,
        'Plasma_Length': np.nan, 'Plasma_Length_err': np.nan,
        'Plasma_Length_mm': np.nan, 'Plasma_Length_mm_err': np.nan,
        'Max_Intensity': 0.0,
        'Top_Beam_Angle_mrad': np.nan,
        'Top_Beam_Angle_mrad_err': np.nan,
        'Top_Channel_Width_mean': np.nan,
        'Top_Channel_Width_mean_mm': np.nan,
        'Top_Saturated_Length': np.nan,
        'debug_data': {
            'z_profile_robust': profile_z_robust,
            'axis_coords': (np.array([]), np.array([])),
            'axis_coords_err': np.array([]),
            'axis_width': np.array([]),
            'axis_y_left': np.array([]),
            'axis_y_right': np.array([]),
        }
    }

    if virtual_peak <= 0:
        log.warning("Top View: empty image (virtual_peak=0)")
        return nan_result

    # ── 2. Centroid (for backward compat Y if needed) ─────────────
    centroid = _compute_centroid_with_errors(img)
    Y_c = centroid['Y_c']
    Y_c_err = centroid['Y_c_err']

    # ── 3. Plasma length and Ignition Point ───────────────────────
    plasma_length = np.nan
    plasma_length_err = np.nan
    Z_c = np.nan
    Z_c_err = np.nan
    
    if compute_length:
        lres = _compute_plasma_length(
            profile_z_robust, threshold_fraction, virtual_peak
        )
        plasma_length = lres['length']
        plasma_length_err = lres['length_err']
        Z_c = float(lres['plasma_start'])
        Z_c_err = float(lres['length_err'])
    else:
        log.info("Plasma length skipped (filamented shot)")
        
    log.info("Plasma ignition Z=%.1f±%.1f, centroid Y=%.1f±%.1f px",
             Z_c, Z_c_err, Y_c, Y_c_err)

    # ── 4. Transverse axis extraction (shared helper) ─────────────
    threshold = threshold_fraction * virtual_peak
    active_cols = np.where(profile_z_robust >= threshold)[0]

    axis_data = _extract_transverse_axis(
        img, active_cols,
        saturation_value=saturation_value,
        fwhm_fraction_saturated=fwhm_fraction_saturated,
        fwhm_fraction_unsaturated=fwhm_fraction_unsaturated,
        smoothing_sigma=smoothing_sigma,
    )

    axis_z = axis_data['axis_z']
    axis_y = axis_data['axis_y']
    axis_y_err = axis_data['axis_y_err']
    axis_width = axis_data['axis_width']
    axis_y_left = axis_data['axis_y_left']
    axis_y_right = axis_data['axis_y_right']
    saturated_columns_count = axis_data['saturated_count']

    log.info("Top View axis extracted: %d columns analysed "
             "(%d saturated)", len(axis_z), saturated_columns_count)

    # ── 5. Beam entrance angle (linear fit + sigma-clipping) ──────
    beam_angle_mrad = np.nan
    beam_angle_mrad_err = np.nan

    valid = np.isfinite(axis_y)
    if np.sum(valid) >= 2:
        z_fit = axis_z[valid].copy()
        y_fit = axis_y[valid].copy()

        # Iterative sigma-clipping (2 iterations, 3σ threshold)
        for clip_iter in range(2):
            if len(z_fit) < 2:
                break
            coeffs_tmp = np.polyfit(z_fit, y_fit, deg=1)
            slope_tmp = coeffs_tmp[0]
            intercept_tmp = coeffs_tmp[1]
            residuals = y_fit - (slope_tmp * z_fit + intercept_tmp)
            res_std = float(np.std(residuals))
            if res_std <= 0:
                break
            inlier_mask = np.abs(residuals) <= 3.0 * res_std
            n_outliers = int(np.sum(~inlier_mask))
            if n_outliers > 0:
                log.debug("Sigma-clip iter %d: removed %d outliers "
                          "(σ=%.2f px)", clip_iter + 1, n_outliers,
                          res_std)
            z_fit = z_fit[inlier_mask]
            y_fit = y_fit[inlier_mask]

        # Final fit on cleaned data (with covariance)
        if len(z_fit) > 2:
            coeffs = np.polyfit(z_fit, y_fit, deg=1, cov=True)
            slope = coeffs[0][0]
            slope_err = np.sqrt(coeffs[1][0, 0])
            beam_angle_mrad = float(np.arctan(slope) * 1000.0)
            beam_angle_mrad_err = float(
                slope_err / (1 + slope**2) * 1000.0)
            log.info("Beam angle (sigma-clipped): %.3f ± %.3f mrad "
                     "(%d/%d points used)",
                     beam_angle_mrad, beam_angle_mrad_err,
                     len(z_fit), int(np.sum(valid)))
        elif len(z_fit) == 2:
            coeffs = np.polyfit(z_fit, y_fit, deg=1, cov=False)
            slope = coeffs[0]
            beam_angle_mrad = float(np.arctan(slope) * 1000.0)
            beam_angle_mrad_err = np.nan
            log.info("Beam angle: %.3f mrad (2 points used, no error est)",
                     beam_angle_mrad)

    # ── 6. Mean channel width ─────────────────────────────────────
    valid_width = axis_width[np.isfinite(axis_width)]
    mean_fwhm = float(np.mean(valid_width)) if len(valid_width) > 0 \
        else np.nan

    # ── 7. Convert to mm ──────────────────────────────────────────
    plasma_length_mm = np.nan
    plasma_length_mm_err = np.nan
    if not np.isnan(plasma_length):
        plasma_length_mm = plasma_length * px_to_mm
        plasma_length_mm_err = plasma_length_err * px_to_mm

    mean_fwhm_mm = np.nan
    if not np.isnan(mean_fwhm):
        mean_fwhm_mm = mean_fwhm * px_to_mm

    return {
        'Plasma_Z_Position': Z_c,
        'Plasma_Z_Position_err': Z_c_err,
        'Plasma_Y_Position': Y_c,
        'Plasma_Y_Position_err': Y_c_err,
        'Plasma_Length': plasma_length,
        'Plasma_Length_err': plasma_length_err,
        'Plasma_Length_mm': plasma_length_mm,
        'Plasma_Length_mm_err': plasma_length_mm_err,
        'Max_Intensity': virtual_peak,
        'Top_Beam_Angle_mrad': beam_angle_mrad,
        'Top_Beam_Angle_mrad_err': beam_angle_mrad_err,
        'Top_Channel_Width_mean': mean_fwhm,
        'Top_Channel_Width_mean_mm': mean_fwhm_mm,
        'Top_Saturated_Length': float(saturated_columns_count),
        'debug_data': {
            'z_profile_robust': profile_z_robust,
            'axis_coords': (axis_z, axis_y),
            'axis_coords_err': axis_y_err,
            'axis_width': axis_width,
            'axis_y_left': axis_y_left,
            'axis_y_right': axis_y_right,
            'fit_coeffs': (slope, intercept) if 'slope' in locals() and 'intercept' in locals() else None,
        }
    }



# ═══════════════════════════════════════════════════════════════════════════════
# 5C. PLASMA CHANNEL — BACKWARD-COMPATIBLE DISPATCHER
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_plasma_channel(image_array: np.ndarray,
                           threshold_fraction: float = 0.2,
                           is_top_view: bool = False,
                           **kwargs) -> dict:
    """Dispatch to the appropriate plasma-channel analysis.

    This wrapper preserves backward compatibility with existing code.

    * ``is_top_view=False`` → :func:`analyze_plasma_channel_side`
    * ``is_top_view=True``  → :func:`analyze_plasma_channel_top`

    Parameters
    ----------
    image_array : np.ndarray (2D)
        Cleaned (background-subtracted) plasma image.
    threshold_fraction : float
        Fraction of the virtual peak used for thresholding.
    is_top_view : bool
        Select Top View analysis when True.
    **kwargs
        Forwarded to the selected implementation.

    Returns
    -------
    dict — see ``analyze_plasma_channel_side`` or
    ``analyze_plasma_channel_top``.
    """
    if is_top_view:
        return analyze_plasma_channel_top(
            image_array, threshold_fraction=threshold_fraction, **kwargs
        )
    else:
        return analyze_plasma_channel_side(
            image_array,
            threshold_fraction=threshold_fraction, **kwargs
        )

