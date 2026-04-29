# GTZAN Zenei Műfajfelismerés – Projekt Összefoglaló

> **Feladat:** Gépi tanulás alapú zenei műfajfelismerő rendszer fejlesztése a GTZAN adathalmazon (1000 klip, 10 műfaj).  
> **Eszközök:** Python, librosa, scikit-learn, PyTorch, torchvision  
> **Hardver:** NVIDIA GeForce GTX 1660 SUPER (6 GB VRAM) / CUDA

---

## 1. fázis – EDA (Feltáró adatelemzés)
**Notebook:** `01_EDA_with_theory.ipynb`

### Adathalmaz
| Tulajdonság | Érték |
|---|---|
| Klipek száma | 1000 |
| Műfajok száma | 10 |
| Klip hossza | 30 s |
| Mintavételi frekvencia | 22 050 Hz |
| Feature CSV | `features30sec.csv` (1000 sor × 60 oszlop) |

**10 műfaj:** blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock

### Elemzett jellemzők
- **Hullámforma (waveform)** – amplitúdó az időben
- **MFCC** (Mel-Frequency Cepstral Coefficients) – 13 együttható × mean/var
- **Chroma** – 12 hangmagasság-osztály energiája
- **Spektrális centroid** – a spektrum "súlypontja"
- **Spektrális sávszélesség** – a spektrum szóródása
- **RMS energia** – hangosság
- **Tempo** – BPM becslés

### Főbb megfigyelések
- A **classical** és **metal** műfajok könnyen elválaszthatók (eltérő frekvencia-profil)
- A **country–rock** és **disco–pop** párok nehezebben különíthetők el
- A **jazz** a legkomplexebb (magas MFCC variancia)
- Osztályegyensúly: pontosan 100 klip/műfaj – **stratified split** szükséges

---

## 2. fázis – Baseline ML (Hagyományos gépi tanulás)
**Notebook:** `02_Baseline_ML-2.ipynb`

### Módszertan
| Lépés | Eszköz |
|---|---|
| Feature forrás | `features30sec.csv` (előre kiszámított) |
| Előfeldolgozás | `StandardScaler` (csak train-en fit) |
| Split | 80/20, `stratify=y`, `random_state=42` |
| Mentés | `joblib` (.pkl) |

### Modellek és eredmények (teszt halmaz)

| Modell | Test Accuracy | Megjegyzés |
|---|---|---|
| Logistic Regression | **74,00%** | `max_iter=1000` |
| Random Forest | **77,50%** ✅ | `n_estimators=100`, legjobb |
| SVC (RBF kernel) | **76,50%** | `C=10`, `gamma='scale'` |

### Random Forest – részletes eredmények (classification report)

| Műfaj | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| blues | 0.789 | 0.750 | 0.769 | 20 |
| classical | 0.864 | 0.950 | 0.905 | 20 |
| country | 0.714 | 0.750 | 0.732 | 20 |
| disco | 0.706 | 0.600 | 0.649 | 20 |
| hiphop | 0.652 | 0.750 | 0.698 | 20 |
| jazz | 0.773 | 0.850 | 0.810 | 20 |
| metal | 0.895 | 0.850 | 0.872 | 20 |
| pop | 0.857 | 0.900 | 0.878 | 20 |
| reggae | 0.762 | 0.800 | 0.780 | 20 |
| rock | 0.733 | 0.550 | 0.629 | 20 |
| **macro avg** | **0.775** | **0.775** | **0.772** | 200 |

**Tipikus tévesztések:** country↔rock, disco↔pop (hasonló hangszerelés/ritmus)

### Mentett artifaktumok
- `models/scaler.pkl` – StandardScaler
- `models/random_forest.pkl` (feltételezett)

---

## 3. fázis – CNN mélytanulás (Mel-spektrogram alapú)
**Notebook:** `03_CNN_Model_v2-3.ipynb`

### Adatfeldolgozás – szegmentált Mel-spektrogramok

| Paraméter | Érték | Indoklás |
|---|---|---|
| Szegmens hossza | 3,0 s | Tízszerese a mintaszámnak (10 szeg/klip) |
| Szegmens lépésköz | 1,5 s | Átfedéses augmentáció |
| n_mels | 128 | Frekvencia felbontás |
| hop_length | 512 | ~23 ms idő felbontás |
| Képméret | 224 × 224 px | CNN input |
| Generált szegmensek | ~9 990 / 10 000 | 10 hibs (jazz.00054.wav) |

**Pipeline:** `.wav` → librosa → Mel-spektrogram (dB) → normalizálás [0, 255] → PNG (224×224)

### Dataset és DataLoader
- **Külön** Dataset példányok train/val/test halmazokra (javított logika)
- Train: augmentáció aktív (crop, kis forgatás)
- Val/Test: csak resize + normalize (tiszta mérés)
- `pin_memory=True` csak CUDA esetén

### CNN Architektúra (VGG-stílusú)

```
Bemenet: (batch, 3, 224, 224)
  │
  ├── ConvBlock 1: Conv(3→32) → BN → ReLU → Conv(32→32) → BN → ReLU → MaxPool(2×2) → Dropout2D(0.25)
  ├── ConvBlock 2: Conv(32→64) → BN → ReLU → Conv(64→64) → BN → ReLU → MaxPool(2×2) → Dropout2D(0.25)
  ├── ConvBlock 3: Conv(64→128) → BN → ReLU → Conv(128→128) → BN → ReLU → MaxPool(2×2) → Dropout2D(0.25)
  └── ConvBlock 4: Conv(128→256) → BN → ReLU → Conv(256→256) → BN → ReLU → MaxPool(2×2) → Dropout2D(0.25)
  │
  ├── Global Average Pooling → (batch, 256, 1, 1)
  └── Classifier: Flatten → Linear(256→128) → ReLU → Dropout(0.5) → Linear(128→10)

Kimenet: (batch, 10) logit – softmax a CrossEntropyLoss-ban
Összes paraméter: 1 208 362
```

### Training konfiguráció

| Paraméter | Érték |
|---|---|
| Loss | `CrossEntropyLoss` (belső softmax) |
| Optimizer | Adam (`lr=1e-3`, `weight_decay=1e-4`) |
| Scheduler | `ReduceLROnPlateau` (`patience=8`, `factor=0.5`) |
| Early stopping | `patience=15` epoch |
| Max epoch | 100 |
| Batch size | 16 (CUDA OOM miatt csökkentve 64-ről) |
| AMP | `torch.cuda.amp.GradScaler` (CUDA esetén) |
| Gradient accumulation | `GRAD_ACCUM_STEPS=1` (előkészítve) |

### Memória-optimalizálások (CUDA OOM kezelés)
- Batch size: 64 → **16**
- AMP (Automatic Mixed Precision) bekapcsolva
- `set_to_none=True` gradiens-nullázás
- `non_blocking=True` device-transzfer
- `pin_memory` csak CUDA esetén

### CNN Eredmények (teszt halmaz – korai checkpoint)

> ⚠️ **Megjegyzés:** A teljes training CUDA OOM hibával megszakadt (GTX 1660 SUPER, 6 GB). Az alábbi eredmények egy korai, nem optimális checkpointból származnak. A notebook memória-safe verzióra lett átállítva; a végleges futtatás szükséges.

| Műfaj | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| blues | 1.000 | 0.125 | 0.222 | 16 |
| classical | 0.833 | 0.909 | 0.870 | 11 |
| country | 0.733 | 0.579 | 0.647 | 19 |
| disco | 0.500 | 0.286 | 0.364 | 21 |
| hiphop | 0.875 | 0.538 | 0.667 | 13 |
| jazz | 0.421 | 0.667 | 0.516 | 12 |
| metal | 0.750 | 1.000 | 0.857 | 12 |
| pop | 0.444 | 0.941 | 0.604 | 17 |
| reggae | 0.524 | 0.733 | 0.611 | 15 |
| rock | 0.500 | 0.333 | 0.400 | 15 |
| **accuracy** | | | **0.583 (58.28%)** | 151 |
| macro avg | 0.658 | 0.611 | 0.576 | 151 |

> A korai eredmény alacsony (58%) az OOM megszakadás miatt. A memória-safe konfiguráció után a várt pontosság ~80–85%+.

### Mentett artifaktumok
- `models/cnn_best.pt` – legjobb checkpoint
- `Data/spectrograms_segmented/` – ~9 990 PNG szegmens
- `models/cnn_training_curves.png` – loss/accuracy görbék (ha lefutott)

### ONNX export (tervezett)
```python
torch.onnx.export(model, dummy_input, "models/gtzan_cnn.onnx", ...)
```

---

## Összehasonlítás

| Megközelítés | Feature | Test Accuracy | Státusz |
|---|---|---|---|
| Logistic Regression | CSV features | 74,00% | ✅ Kész |
| SVC (RBF) | CSV features | 76,50% | ✅ Kész |
| **Random Forest** | **CSV features** | **77,50%** | ✅ **Legjobb baseline** |
| CNN (VGG-stílusú) | Mel-spektrogram | ~58% (OOM) / ~80-85% (várt) | ⚠️ Futtatás szükséges |

---

## 4. fázis – Streamlit App (tervezett)
**Notebook:** `04_Streamlit_App.ipynb` *(még elkészítendő)*

### Tervezett funkciók
- `.wav` fájl feltöltése
- Mel-spektrogram generálás (librosa)
- ONNX inference (gyors, deployment-barát)
- Műfaj predikció + konfidencia diagram
- Local és Cloud Deploy opciók

---

## Fájlstruktúra

```
gtzan-music-genre-recognition/
├── data/
│   ├── genres_original/         # 1000 × .wav fájl
│   └── features30sec.csv        # Előre kiszámított feature-ök
├── Data/
│   └── spectrograms_segmented/  # ~9990 × 224×224 PNG
├── models/
│   ├── scaler.pkl               # StandardScaler
│   ├── cnn_best.pt              # Legjobb CNN checkpoint
│   └── cnn_training_curves.png  # Training görbék
├── notebooks/
│   ├── 01_EDA_with_theory.ipynb
│   ├── 02_Baseline_ML-2.ipynb
│   ├── 03_CNN_Model_v2-3.ipynb
│   └── 04_Streamlit_App.ipynb   ← következő
└── README.md
```

---

*Generálva: 2026-04-29*
