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
from skimage.measure import find_contours

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
# 5. PLASMA CHANNEL ANALYSIS — HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def _detect_nozzle_tip(raw_image: np.ndarray,
                       contour_level_fraction: float = 0.15,
                       intensity_upper_limit: float = None) -> dict | None:
    """Detect the nozzle tip from a raw Side View image.

    The nozzle appears as a dark shape with localised reflections,
    touching the **bottom edge** of the image.  We use
    ``skimage.measure.find_contours`` on the (optionally intensity-
    clipped) raw image, keep only contours that reach the bottom
    edge, and identify the tip as the topmost point (minimum Y).

    Parameters
    ----------
    raw_image : np.ndarray (2D)
        Raw (non-background-subtracted) Side View image.
    contour_level_fraction : float, default 0.15
        Fraction of image maximum used as the contour level.
    intensity_upper_limit : float or None
        If set, pixel values above this are clipped **before**
        contouring.  This prevents bright reflections on the nozzle
        from producing spurious separate contours.

    Returns
    -------
    dict or None
        ``{'nozzle_contour', 'nozzle_tip_y', 'nozzle_tip_z'}``
        or ``None`` if detection fails.
    """
    img = raw_image.astype(np.float64)
    H = img.shape[0]

    # Clip bright reflections to merge them with the nozzle body
    if intensity_upper_limit is not None:
        img = np.clip(img, 0, intensity_upper_limit)
        log.debug("Nozzle detection: clipped intensities above %.1f",
                  intensity_upper_limit)

    max_val = float(np.max(img))
    if max_val <= 0:
        log.warning("Nozzle detection: image is empty (max=0)")
        return None

    contour_level = contour_level_fraction * max_val
    log.debug("Nozzle detection: contour_level=%.2f (%.1f%% of max=%.1f)",
              contour_level, contour_level_fraction * 100, max_val)

    contours = find_contours(img, level=contour_level)
    log.debug("Nozzle detection: found %d raw contours", len(contours))

    if not contours:
        log.warning("Nozzle detection: no contours at level=%.2f",
                    contour_level)
        return None

    # Keep contours that touch the bottom edge (row >= H - 2)
    bottom_contours = []
    for i, c in enumerate(contours):
        if np.any(c[:, 0] >= H - 2):
            bottom_contours.append(c)
            log.debug("  Contour %d: %d pts, touches bottom edge", i, len(c))

    if not bottom_contours:
        log.warning("Nozzle detection: no contours touch the bottom edge")
        return None
    log.debug("Nozzle detection: %d contours touch bottom",
              len(bottom_contours))

    # Select the largest contour (most vertices) as the nozzle
    nozzle_contour = max(bottom_contours, key=len)
    log.debug("Nozzle detection: selected contour with %d points",
              len(nozzle_contour))

    # Tip = point with minimum Y (topmost point, pointing up)
    tip_idx = int(np.argmin(nozzle_contour[:, 0]))
    nozzle_tip_y = float(nozzle_contour[tip_idx, 0])
    nozzle_tip_z = float(nozzle_contour[tip_idx, 1])
    log.info("Nozzle tip detected at (Z=%.1f, Y=%.1f) px",
             nozzle_tip_z, nozzle_tip_y)

    return {
        'nozzle_contour': nozzle_contour,
        'nozzle_tip_y': nozzle_tip_y,
        'nozzle_tip_z': nozzle_tip_z,
    }


def _robust_z_profile(image: np.ndarray, percentile: float = 90) -> np.ndarray:
    """Compute a hot-spot-immune 1D profile along Z.

    Instead of summing or taking the max along Y (axis 0), this uses
    the given percentile, which is robust to isolated bright pixels
    while still capturing the true plasma signal.

    Parameters
    ----------
    image : np.ndarray (2D)
        Cleaned plasma image (float64).
    percentile : float, default 90
        Percentile to use.

    Returns
    -------
    np.ndarray (1D)
        One value per column (Z pixel).
    """
    return np.percentile(image.astype(np.float64), percentile, axis=0)


def _voigt_model(x, center, sigma, gamma, amplitude, background):
    """Voigt profile model for ``curve_fit``.

    Parameters
    ----------
    x : array-like   — pixel coordinates (Y positions).
    center : float   — profile centre.
    sigma : float    — Gaussian width (>0).
    gamma : float    — Lorentzian width (>0).
    amplitude : float — peak amplitude.
    background : float — constant offset.
    """
    return amplitude * voigt_profile(x - center, sigma, gamma) + background


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
    """Compute plasma length from a 1D Z profile with uncertainty.

    The uncertainty is estimated by counting the pixels that lie
    within ±5 % of the virtual peak from the threshold level
    ("boundary pixels").

    Returns
    -------
    dict
        ``length``, ``length_err``, ``threshold``
    """
    threshold = threshold_fraction * virtual_peak
    above = profile_z >= threshold
    length = int(np.sum(above))

    boundary_band = 0.05 * virtual_peak
    boundary = np.abs(profile_z - threshold) < boundary_band
    length_err = max(int(np.sum(boundary)), 1)

    log.debug("Plasma length: %d px (err: ±%d px, threshold: %.2f)",
              length, length_err, threshold)
    return {'length': length, 'length_err': length_err,
            'threshold': threshold}


# ═══════════════════════════════════════════════════════════════════════════════
# 5A. PLASMA CHANNEL — SIDE VIEW
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_plasma_channel_side(
        cleaned_image: np.ndarray,
        raw_image: np.ndarray = None,
        threshold_fraction: float = 0.2,
        contour_level_fraction: float = 0.15,
        intensity_upper_limit: float = None,
        px_to_mm: float = 1.0 / 179.0,
        compute_length: bool = True) -> dict:
    """Analyse the plasma channel from Side Imaging.

    Workflow
    --------
    1. Compute a robust (90th-percentile) Z profile immune to hot
       spots.
    2. Always compute the 2D intensity-weighted centroid (Z_c, Y_c)
       with uncertainty.
    3. If a raw image is provided, detect the nozzle tip via
       contouring and compute the **vertical distance** ΔY from
       the nozzle tip to the plasma centroid.
    4. If ``compute_length`` is True (i.e. the shot is NOT
       filamented), compute the plasma length from the robust
       profile.

    Parameters
    ----------
    cleaned_image : np.ndarray (2D)
        Background-subtracted Side Imaging image.
        Axis 0 = transverse (Y), Axis 1 = propagation (Z).
    raw_image : np.ndarray (2D) or None
        Raw (un-subtracted) image for nozzle detection.
    threshold_fraction : float
        Fraction of the virtual peak for plasma-length thresholding.
    contour_level_fraction : float
        Fraction of the raw-image maximum for nozzle contouring.
    intensity_upper_limit : float or None
        Upper clip for nozzle contouring (prevents bright-reflection
        artefacts).
    px_to_mm : float
        Pixel-to-mm conversion for Side Imaging.  Default is
        1/179 ≈ 0.00559 mm/px (179 px = 1 mm).
    compute_length : bool
        If False (filamented shot), the plasma length is set to NaN.

    Returns
    -------
    dict
        Numeric results + ``'debug_data'`` sub-dict for visual
        inspection.
    """
    log.info("── Side View analysis ──")
    img = cleaned_image.astype(np.float64)

    # ── 1. Robust Z profile ───────────────────────────────────────
    profile_z_robust = _robust_z_profile(img)
    virtual_peak = float(np.max(profile_z_robust))
    log.debug("Virtual peak (90th pctl): %.2f", virtual_peak)

    # Degenerate case
    if virtual_peak <= 0:
        log.warning("Side View: empty image (virtual_peak=0)")
        return {
            'Plasma_Z_Position': np.nan, 'Plasma_Z_Position_err': np.nan,
            'Plasma_Z_Position_rel': np.nan,
            'Plasma_Y_Position': np.nan, 'Plasma_Y_Position_err': np.nan,
            'Plasma_Length': np.nan, 'Plasma_Length_err': np.nan,
            'Max_Intensity': 0.0,
            'Nozzle_Tip_Z': np.nan, 'Nozzle_Tip_Y': np.nan,
            'Nozzle_Distance_Y': np.nan, 'Nozzle_Distance_Y_err': np.nan,
            'Nozzle_Distance_Y_mm': np.nan,
            'Nozzle_Distance_Y_mm_err': np.nan,
            'debug_data': {
                'nozzle_contour': None, 'nozzle_tip': None,
                'z_profile_robust': profile_z_robust,
            }
        }

    # ── 2. Centroid (always, even for filamented) ─────────────────
    centroid = _compute_centroid_with_errors(img)
    Z_c = centroid['Z_c']
    Y_c = centroid['Y_c']
    Z_c_err = centroid['Z_c_err']
    Y_c_err = centroid['Y_c_err']
    log.info("Plasma centroid: Z=%.1f±%.1f, Y=%.1f±%.1f px",
             Z_c, Z_c_err, Y_c, Y_c_err)

    # ── 3. Nozzle detection (from raw image) ──────────────────────
    nozzle_tip_y = np.nan
    nozzle_tip_z = np.nan
    nozzle_contour = None
    nozzle_tip = None

    if raw_image is not None:
        nozzle_res = _detect_nozzle_tip(
            raw_image, contour_level_fraction, intensity_upper_limit
        )
        if nozzle_res is not None:
            nozzle_tip_y = nozzle_res['nozzle_tip_y']
            nozzle_tip_z = nozzle_res['nozzle_tip_z']
            nozzle_contour = nozzle_res['nozzle_contour']
            nozzle_tip = (nozzle_tip_z, nozzle_tip_y)
        else:
            log.warning("Side View: nozzle detection failed")
    else:
        log.warning("Side View: raw_image not provided, "
                    "nozzle detection skipped")

    # ── 4. Z position relative to nozzle ──────────────────────────
    Z_pos_rel = np.nan
    if not np.isnan(nozzle_tip_z):
        Z_pos_rel = (Z_c - nozzle_tip_z) * px_to_mm
        log.info("Z relative to nozzle: %.3f mm", Z_pos_rel)

    # ── 5. Vertical distance ΔY (pure height, no diagonal) ───────
    nozzle_dist_y = np.nan
    nozzle_dist_y_err = np.nan
    nozzle_dist_y_mm = np.nan
    nozzle_dist_y_mm_err = np.nan

    if not np.isnan(nozzle_tip_y):
        nozzle_dist_y = abs(Y_c - nozzle_tip_y)
        nozzle_tip_err_px = 1.0          # ≈1 px uncertainty on tip
        nozzle_dist_y_err = float(
            np.sqrt(Y_c_err ** 2 + nozzle_tip_err_px ** 2)
        )
        nozzle_dist_y_mm = nozzle_dist_y * px_to_mm
        nozzle_dist_y_mm_err = nozzle_dist_y_err * px_to_mm
        log.info("Nozzle distance (ΔY): %.1f±%.1f px = "
                 "%.3f±%.3f mm",
                 nozzle_dist_y, nozzle_dist_y_err,
                 nozzle_dist_y_mm, nozzle_dist_y_mm_err)

    # ── 6. Plasma length (skipped for filamented) ─────────────────
    plasma_length = np.nan
    plasma_length_err = np.nan

    if compute_length:
        lres = _compute_plasma_length(
            profile_z_robust, threshold_fraction, virtual_peak
        )
        plasma_length = lres['length']
        plasma_length_err = lres['length_err']
    else:
        log.info("Plasma length skipped (filamented shot)")

    return {
        'Plasma_Z_Position': Z_c,
        'Plasma_Z_Position_err': Z_c_err,
        'Plasma_Z_Position_rel': Z_pos_rel,
        'Plasma_Y_Position': Y_c,
        'Plasma_Y_Position_err': Y_c_err,
        'Plasma_Length': plasma_length,
        'Plasma_Length_err': plasma_length_err,
        'Max_Intensity': virtual_peak,
        'Nozzle_Tip_Z': nozzle_tip_z,
        'Nozzle_Tip_Y': nozzle_tip_y,
        'Nozzle_Distance_Y': nozzle_dist_y,
        'Nozzle_Distance_Y_err': nozzle_dist_y_err,
        'Nozzle_Distance_Y_mm': nozzle_dist_y_mm,
        'Nozzle_Distance_Y_mm_err': nozzle_dist_y_mm_err,
        'debug_data': {
            'nozzle_contour': nozzle_contour,
            'nozzle_tip': nozzle_tip,
            'z_profile_robust': profile_z_robust,
        }
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5B. PLASMA CHANNEL — TOP VIEW
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_plasma_channel_top(
        cleaned_image: np.ndarray,
        threshold_fraction: float = 0.2,
        saturation_value: float = 255.0,
        voigt_margin_px: int = 18,
        voigt_center_tol_px: float = 3.0,
        px_to_mm: float = 1.0,
        compute_length: bool = True) -> dict:
    """Analyse the plasma channel from Top Imaging.

    The image typically exhibits strong central saturation (pixels
    clipped at ``saturation_value``) surrounded by diffraction /
    scattering rings.  A coarse-to-fine strategy is used to recover
    the transverse axis position even inside the saturated plateau:

    * **Coarse:** geometric midpoint of the saturated zone.
    * **Fine:**  Voigt fit on the wings (excluding the plateau),
      constrained around the coarse estimate.

    Parameters
    ----------
    cleaned_image : np.ndarray (2D)
        Background-subtracted Top Imaging image.
    threshold_fraction : float
        Fraction of the virtual peak for length thresholding.
    saturation_value : float
        ADC saturation value (255 for 8-bit cameras).
    voigt_margin_px : int
        Number of pixels beyond the plateau edges to include in
        the fitting window.
    voigt_center_tol_px : float
        Allowed deviation of the Voigt centre from the coarse
        midpoint (bounds for ``curve_fit``).
    px_to_mm : float
        Pixel-to-mm conversion for Top Imaging.  Default 1.0
        (uncalibrated; update when calibration is known).
    compute_length : bool
        If False (filamented shot), length is NaN.

    Returns
    -------
    dict
        Numeric results + ``'debug_data'`` with fit details.
    """
    log.info("── Top View analysis ──")
    img = cleaned_image.astype(np.float64)
    n_rows, n_cols = img.shape

    # ── 1. Robust Z profile ───────────────────────────────────────
    profile_z_robust = _robust_z_profile(img)
    virtual_peak = float(np.max(profile_z_robust))
    log.debug("Virtual peak (90th pctl): %.2f", virtual_peak)

    if virtual_peak <= 0:
        log.warning("Top View: empty image (virtual_peak=0)")
        return {
            'Plasma_Z_Position': np.nan, 'Plasma_Z_Position_err': np.nan,
            'Plasma_Y_Position': np.nan, 'Plasma_Y_Position_err': np.nan,
            'Plasma_Length': np.nan, 'Plasma_Length_err': np.nan,
            'Max_Intensity': 0.0,
            'debug_data': {
                'z_profile_robust': profile_z_robust,
                'axis_coords': (np.array([]), np.array([])),
                'axis_coords_err': np.array([]),
                'voigt_params_log': [],
            }
        }

    # ── 2. Centroid (always) ──────────────────────────────────────
    centroid = _compute_centroid_with_errors(img)
    Z_c = centroid['Z_c']
    Y_c = centroid['Y_c']
    Z_c_err = centroid['Z_c_err']
    Y_c_err = centroid['Y_c_err']
    log.info("Plasma centroid: Z=%.1f±%.1f, Y=%.1f±%.1f px",
             Z_c, Z_c_err, Y_c, Y_c_err)

    # ── 3. Plasma length (skipped for filamented) ─────────────────
    plasma_length = np.nan
    plasma_length_err = np.nan
    if compute_length:
        lres = _compute_plasma_length(
            profile_z_robust, threshold_fraction, virtual_peak
        )
        plasma_length = lres['length']
        plasma_length_err = lres['length_err']
    else:
        log.info("Plasma length skipped (filamented shot)")

    # ── 4. Coarse-to-Fine transverse axis extraction ──────────────
    threshold = threshold_fraction * virtual_peak
    active_cols = np.where(profile_z_robust >= threshold)[0]

    axis_z_list = []
    axis_y_list = []
    axis_y_err_list = []
    voigt_params_log = []

    sat_thresh = saturation_value - 1.0   # ≥ this → saturated

    for z in active_cols:
        col_profile = img[:, z]

        # — Coarse: find saturated plateau —
        saturated = col_profile >= sat_thresh
        sat_indices = np.where(saturated)[0]

        if len(sat_indices) >= 3:
            y_start = int(sat_indices[0])
            y_end = int(sat_indices[-1])
            y_mid = (y_start + y_end) / 2.0

            # — Fine: Voigt fit on wings only —
            win_lo = max(y_start - voigt_margin_px, 0)
            win_hi = min(y_end + voigt_margin_px + 1, n_rows)
            y_coords = np.arange(win_lo, win_hi, dtype=np.float64)
            data = col_profile[win_lo:win_hi]

            # Mask: True = use this pixel, False = saturated → skip
            mask = data < sat_thresh
            y_fit = y_coords[mask]
            d_fit = data[mask]

            fit_center = y_mid
            fit_center_err = np.nan
            fit_params_dict = None

            if len(y_fit) >= 5:
                try:
                    wing_max = float(np.max(d_fit)) if len(d_fit) > 0 else 1.0
                    p0 = [y_mid, 2.0, 1.0, wing_max, 0.0]
                    bounds_lo = [y_mid - voigt_center_tol_px, 0.1, 0.01,
                                 0.0, -np.inf]
                    bounds_hi = [y_mid + voigt_center_tol_px, 30.0, 30.0,
                                 np.inf, np.inf]

                    popt, pcov = curve_fit(
                        _voigt_model, y_fit, d_fit,
                        p0=p0, bounds=(bounds_lo, bounds_hi),
                        maxfev=2000
                    )
                    perr = np.sqrt(np.diag(pcov))

                    fit_center = popt[0]
                    fit_center_err = perr[0]

                    fit_params_dict = {
                        'center': popt[0], 'center_err': perr[0],
                        'sigma': popt[1], 'sigma_err': perr[1],
                        'gamma': popt[2], 'gamma_err': perr[2],
                        'amplitude': popt[3], 'amplitude_err': perr[3],
                        'background': popt[4], 'background_err': perr[4],
                    }
                    log.debug(
                        "Z=%d Voigt fit: center=%.2f±%.2f, "
                        "σ=%.3f±%.3f, γ=%.3f±%.3f, "
                        "A=%.1f±%.1f, bg=%.1f±%.1f",
                        z, popt[0], perr[0],
                        popt[1], perr[1], popt[2], perr[2],
                        popt[3], perr[3], popt[4], perr[4],
                    )
                except (RuntimeError, ValueError) as e:
                    log.debug("Z=%d Voigt fit failed (%s), "
                              "using coarse midpoint %.1f", z, e, y_mid)
                    fit_center = y_mid
                    # Coarse error ≈ half plateau width
                    fit_center_err = (y_end - y_start) / 2.0
            else:
                log.debug("Z=%d too few wing pixels (%d), "
                          "using coarse midpoint %.1f",
                          z, len(y_fit), y_mid)
                fit_center_err = (y_end - y_start) / 2.0

        else:
            # No saturation in this column → simple weighted centroid
            total_col = float(np.sum(col_profile))
            if total_col > 0:
                y_indices = np.arange(n_rows, dtype=np.float64)
                fit_center = float(
                    np.sum(y_indices * col_profile) / total_col
                )
                # Error ≈ σ / sqrt(N_eff)
                sigma_col = float(np.sqrt(
                    np.sum(col_profile * (y_indices - fit_center) ** 2)
                    / total_col
                ))
                max_col = float(np.max(col_profile))
                n_eff_col = max(total_col / max_col, 1.0) \
                    if max_col > 0 else 1.0
                fit_center_err = sigma_col / np.sqrt(n_eff_col)
            else:
                fit_center = np.nan
                fit_center_err = np.nan
            fit_params_dict = None

        axis_z_list.append(z)
        axis_y_list.append(fit_center)
        axis_y_err_list.append(fit_center_err)
        voigt_params_log.append({
            'z': z,
            'y_center': fit_center,
            'y_center_err': fit_center_err,
            'params': fit_params_dict,
        })

    axis_z = np.array(axis_z_list, dtype=np.float64)
    axis_y = np.array(axis_y_list, dtype=np.float64)
    axis_y_err = np.array(axis_y_err_list, dtype=np.float64)

    log.info("Top View axis extracted: %d columns analysed",
             len(axis_z))

    return {
        'Plasma_Z_Position': Z_c,
        'Plasma_Z_Position_err': Z_c_err,
        'Plasma_Y_Position': Y_c,
        'Plasma_Y_Position_err': Y_c_err,
        'Plasma_Length': plasma_length,
        'Plasma_Length_err': plasma_length_err,
        'Max_Intensity': virtual_peak,
        'debug_data': {
            'z_profile_robust': profile_z_robust,
            'axis_coords': (axis_z, axis_y),
            'axis_coords_err': axis_y_err,
            'voigt_params_log': voigt_params_log,
        }
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5C. PLASMA CHANNEL — BACKWARD-COMPATIBLE DISPATCHER
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_plasma_channel(image_array: np.ndarray,
                           threshold_fraction: float = 0.2,
                           is_top_view: bool = False,
                           raw_image: np.ndarray = None,
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
    raw_image : np.ndarray (2D) or None
        Raw image for nozzle detection (Side View only).
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
        if raw_image is None:
            log.warning("Side View: raw_image not provided, "
                        "nozzle detection will be disabled")
        return analyze_plasma_channel_side(
            image_array, raw_image=raw_image,
            threshold_fraction=threshold_fraction, **kwargs
        )
