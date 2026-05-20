# LWFA Stability Analysis Pipeline

Pipeline in Python per l'analisi dei dati di stabilità degli esperimenti di accelerazione Laser-Plasma (LWFA). Il progetto gestisce l'intero flusso di lavoro: dall'ingestione dei log sperimentali (da file Excel), alla classificazione delle immagini diagnostiche, fino all'elaborazione delle metriche fisiche e alla loro visualizzazione statistica.

## Caratteristiche Principali

La pipeline è divisa in quattro macro-fasi, orchestrate dallo script principale `main_analysis.py`:

1. **Data Ingestion & Feature Engineering**:
   - Analisi e parsing automatico dei log di esperimento in formato Excel (situati, ad esempio, in `External_User_run_excel`).
   - Riconoscimento di tag e configurazioni sperimentali come la tipologia di maschera (es. `mask_free`, `mask_round`), stato dei magneti, puntamento Lanex e stima automatica dei fallimenti di iniezione o filamentazione.

2. **File Mapping & Extraction**:
   - Ricerca e associazione automatica delle immagini (formato TIFF) per ogni colpo di laser con le corrispondenti righe di log, distinguendo tra varie diagnostiche (Andor Lanex, Pointing Lanex, Side Imaging).
   - Supporto nativo per la sottrazione del background e fallback intelligenti sulle acquisizioni di riferimento (es. colpi in assenza di plasma).

3. **Batch Processing (`diagnostics_lib.py`)**:
   - **Side Imaging**: Stima dell'estensione del plasma (lunghezza e posizione longitudinale Z).
   - **Pointing Lanex**: Classificazione automatica della qualità del fascio (Collimato, Diffuso, Multiplo) ed estrazione della deviazione e del jitter del puntamento spaziale ($X_c$, $Y_c$, $\sigma$).
   - **Andor Lanex (Spettrometro)**: Supporto in via di integrazione per l'analisi dello spettro di energia degli elettroni (posizione e dispersione del picco di energia).

4. **Statistica e Visualizzazione (EDA)**:
   - Aggregazione dell'intero dataset sperimentale per estrarre visualizzazioni chiare della stabilità del plasma, rate di successo dell'iniezione, tasso di filamentazione e andamento spaziale del fascio.
   - Generazione automatica di boxplot, scatter plot con ellissi di confidenza (jitter del centroide) e andamenti nel tempo, esportati nella cartella configurabile di output.

## Struttura della Repository
- `main_analysis.py`: Punto di ingresso della pipeline. Contiene logica di orchestrazione, mapping di file e generazione di figure con `matplotlib`/`seaborn`.
- `diagnostics_lib.py`: Modulo con gli algoritmi centrali per l'elaborazione delle immagini spaziali e del canale di plasma (sottrazione di fondo, fit ellittici, analisi ROI).
- `External_User_run_excel/`: Contiene esempi o log attuali dei run in formato foglio di calcolo, utilizzati per importare la meta-struttura dei dati.

## Utilizzo

Per avviare l'analisi, è necessario configurare le path nella sezione `CONFIGURAZIONE` di `main_analysis.py`:
- Modificare `ROOT_DIR` alla cartella base che contiene l'organizzazione delle directory.
- Impostare `RUN_MODE` in base all'obiettivo dell'esecuzione (`STEP_1_CLASSIFY` per esplorare o `STEP_2_ANALYZE` per l'elaborazione del dataset completo).

Una volta configurato, lo script può essere eseguito normalmente. Verrà prodotta una sotto-cartella con i risultati grafici dell'analisi statistica.
