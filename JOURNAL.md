# Projekt Napló (JOURNAL)

Létrehozva: 2026-05-15

Ez a fájl retrospektív napló a projekt eddigi haladásáról. Minden bejegyzés megőrzendő: további bejegyzések hozzáadása csak az állomány végéhez történjen (append-only). Semmit ne töröljünk vagy helyettesítsünk a korábbi bejegyzésekből.

## Összefoglaló (státusz 2026-05-15)

- **Adatok:** A projekthez szükséges fájlok és mappák megtalálhatók a repositoryban: nyers hangok a `music/`-ban, eredeti címkék a `data/genres_original/`-ban.
- **Feldolgozás:** Készültek származtatott adatok: `features_3_sec.csv`, `features_30_sec.csv` a gyökérben; spektrális képek a `Data/spectrograms/` és szegmentált változatuk a `Data/spectrograms_segmented/` mappákban.
- **Fedezetes elemzés (EDA):** Az EDA és elméleti jegyzetek a `notebooks/01_EDA_with_theory.ipynb` fájlban találhatók.
- **Baseline ML:** Baseline modellek és kísérletek a `notebooks/02_Baseline_ML.ipynb`-ben.
- **Konvolúciós háló (CNN):** Tréning, validáció és finomhangolás a `notebooks/03_CNN_Model_v2.ipynb` és `notebooks/03_CNN_Model.ipynb` (archív) fájlokban. Legjobb modell: `models/cnn_best.pt`. ONNX export: `models/cnn_gtzan.onnx`.
- **Deploy:** Egyszerű Streamlit alkalmazás megtalálható: `deploy/04_Streamlit_App.py`.
- **Segédscriptek és környezet:** A környezet és függőségek a `requirements.txt` és `gtzan_env.yaml` fájlokban; letöltő script: `download_by_genre.sh`.

## Korábbi mérföldkövek (rövid, fordított időrendben)

- 2026-05-15: Alapvető projektstruktúra, EDA és modell célok dokumentálva; `JOURNAL.md` létrehozva.
- (korábbi munkák összefoglalása a repositoryból):
  - Feature-extrahálás és CSV-ek generálása (`features_3_sec.csv`, `features_30_sec.csv`).
  - Spektrál-képek és szegmentálás előállítása `Data/spectrograms/` és `Data/spectrograms_segmented/` mappákba.
  - Több kísérleti notebook futtatása és eredmények mentése a `notebooks/output/`-ba (`compare_models_summary.csv`).
  - Modell mentése PyTorch formátumban és ONNX-be exportálás.

## Append-only szabály és használat

1. Minden új bejegyzést a fájl végéhez kell hozzáfűzni.
2. Soha ne szerkesszük vagy töröljük a meglévő bejegyzéseket — kisebb javítások esetén adjunk új bejegyzést, amely hivatkozik az előzőre.
3. Ajánlott bejegyzés-formátum (másolj és töltsd ki):

```
2026-05-15 14:30 | Szerző: <név>
Rövid cím: Pl. "Spektrók generálása"
Összefoglaló: Egy rövid mondat a végrehajtott munkáról.
Részletek: Pontokba szedve a pontos fájlok, parancsok, paraméterek, eredmények.
Kapcsolódó fájlok: relatív elérési utak (pl. notebooks/03_CNN_Model_v2.ipynb)

```

## Gyors példa: hogyan adjunk hozzá egy bejegyzést (terminál)

Hasznos parancsok (példa):

```bash
echo "$(date '+%Y-%m-%d %H:%M') | Szerző: Anna" >> JOURNAL.md
echo "Rövid cím: Új kísérlet - adat augmentáció" >> JOURNAL.md
echo "Összefoglaló: ..." >> JOURNAL.md
git add JOURNAL.md && git commit -m "JOURNAL: hozzáadva bejegyzés" && git push
```

---

Készítette: projekt csapata

## Fázisok és Roadmap

- **1. Adatgyűjtés (Data Acquisition):** Gyűjtsd össze a nyers audio fájlokat és metaadatokat. Artefaktumok: `music/`, `data/genres_original/`, `music_links.txt`.
  - Ellenőrző pontok: fájlok megléte, hash/ellenőrzés, hiányzó adatok listája.

- **2. Előfeldolgozás (Preprocessing):** Hangfeldolgozás, normalizálás, szegmentálás, spectrogram generálás.
  - Artefaktumok: `Data/spectrograms/`, `Data/spectrograms_segmented/`, `features_*.csv`.
  - Ellenőrző pontok: zajszintek, hosszak, duplikátumok, szegmens-eloszlás.

- **3. Feltáró adatelemzés (EDA):** Megismerni az adat eloszlását, jellemzők szerepét, osztály egyensúlyt.
  - Artefaktumok: `notebooks/01_EDA_with_theory.ipynb`, vizualizációk, `notebooks/output/`.

- **4. Feature engineering & baseline modellek:** Jellemzők kiválasztása/átalakítása és egyszerű baseline modellek (ML).
  - Artefaktumok: `notebooks/02_Baseline_ML.ipynb`, feature CSV-ek.

- **5. Modellépítés (Modeling):** CNN/egyéb architektúrák tervezése, tréning, hyperparaméter keresés.
  - Artefaktumok: `notebooks/03_CNN_Model_v2.ipynb`, `models/cnn_best.pt`, `models/*.onnx`.

- **6. Értékelés és összehasonlítás (Evaluation & Comparison):** Modellek összehasonlítása, metrikák, hibaforrások feltárása.
  - Artefaktumok: `notebooks/04_Compare_Models.ipynb`, `notebooks/output/compare_models_summary.csv`.

- **7. Telepítés (Deployment):** Model exportálása (ONNX), egyszerű app (Streamlit), inference pipeline.
  - Artefaktumok: `models/cnn_gtzan.onnx`, `deploy/04_Streamlit_App.py`.

- **8. Monitorozás és karbantartás:** Teljesítmény figyelése, új adatok hozzáadása, re-tréning pipeline (ha szükséges).

- **9. Dokumentáció & Reprodukció:** README, környezeti fájlok, `JOURNAL.md` append-only napló.

### Ajánlott checkpointok minden fázisban
 - Jegyezd fel dátumot, szerzőt, pontos parancsokat és a kapott eredményeket a `JOURNAL.md` végéhez.

---

2026-05-15 16:20 | Szerző: Automatikus bejegyzés
Rövid cím: Átfogó haladás-összegzés
Összefoglaló: Az eddigi munkafázisok befejezett alapjai: adatgyűjtés, előfeldolgozás, feature-extrahálás, EDA, baseline modellezés és CNN tréning; rendelkezésre állnak a fő artefaktumok (CSV-ek, spektrum-képek, notebookok, modellek, deploy script).
Részletek:

- Adat és strukturális artefaktumok:
  - `music/` — nyers audio fájlok rendezve genre szerint.
  - `data/genres_original/` — eredeti címkék és szerkezet.

- Előfeldolgozás és jellemzők:
  - Generált fájlok: `features_3_sec.csv`, `features_30_sec.csv` a projekt gyökérben.
  - Spektrális képek: `Data/spectrograms/`, `Data/spectrograms_segmented/`.

- Elemzés és modellezés:
  - EDA notebook: `notebooks/01_EDA_with_theory.ipynb`.
  - Baseline notebookok: `notebooks/02_Baseline_ML.ipynb`.
  - CNN notebookok és modell mentések: `notebooks/03_CNN_Model_v2.ipynb`, `models/cnn_best.pt`, `models/cnn_gtzan.onnx`.

- Deploy és reprodukció:
  - Streamlit demo: `deploy/04_Streamlit_App.py`.
  - Környezet fájlok: `gtzan_env.yaml`, `requirements.txt`.

Megjegyzés: A projekt naplóként `JOURNAL.md` append-only módon tárolja a további bejegyzéseket; minden további mérföldkőhöz kérjük, adj hozzá egy új bejegyzést a fenti sablon szerint.

---

2026-05-15 16:40 | Szerző: Automatikus bejegyzés
Rövid cím: Kezdő EDA összegzés
Összefoglaló: Rövid, gyakorlatias összegzés az EDA fő megállapításairól és a következő lépésekről.
Részletek:

- Adatkészlet mérete és struktúra:
  - 10 műfaj, rendezett könyvtárstruktúra a `music/` alatt.
  - Előfeldolgozott jellemzők: `features_30_sec.csv` (~30s alapú jellemzők) és `features_3_sec.csv` (szegmentált jellemzők).

- Osztály-eloszlás:
  - Az osztályok viszonylag kiegyensúlyozottak, de érdemes ellenőrizni ritka hiányosságokat és esetleges duplikátumokat (pl. azonos fájl több mappában).

- Jellemzők és korrelációk:
  - Az alapvető akusztikus jellemzők (MFCC-átlagok, kromagram, spectral centroid stb.) különböző mértékben járulnak hozzá a megkülönböztetéshez; részletes korrelációs mátrix a `notebooks/01_EDA_with_theory.ipynb`-ben található.

- Hibaforrások és kockázatok:
  - Zaj és minőségi különbségek a források között; lehetséges metaadat-problémák a fájlok címkézésénél.
  - A 30s klipek belső variabilitása miatt szegmentálás / augmentáció hasznos lehet.

- Javasolt következő lépések:
  1. Ellenőrizni és rögzíteni az esetleges duplikátumokat (`md5sum` vagy hasonló).
  2. Készíteni részletes osztály-eloszlás kimutatást és felvenni a naplóba.
  3. Kísérletezni egyszerű augmentációval (pitch shift, time stretch) a CNN tréning stabilitásának növelésére.
  4. Rögzíteni minden parancsot és paramétert a `JOURNAL.md`-ben a reprodukálhatóság érdekében.

Kapcsolódó fájlok: notebooks/01_EDA_with_theory.ipynb, data/features_30_sec.csv, Data/spectrograms/



