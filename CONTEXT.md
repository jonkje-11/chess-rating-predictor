# Kontekst og overlevering (draft v1)

Dette repoet er **første utkast (draft v1)**: en komplett, fungerende versjon generert med Claude Code 6.–7. oktober 2026. Planen videre er å **bygge prosjektet på nytt, rolig og skikkelig**, med egen research og egen rapport, og hente kode og ideer herfra når det er nyttig. Denne filen samler det som ikke står i koden: beslutninger, resultater, kilder, fallgruver og hva som gjenstår.

Spesifikasjonen for prosjektet ligger i [AGENTS.md](AGENTS.md).

---

## 1. Kursets krav (fra Canvas, "ML assignment 2: Project work")

- Frist **30. oktober 2026 23:59**, 20 poeng. Innlevering = lenke til offentlig GitHub-repo.
- Nettside: brukerinput → ML-resultat. **Live** (foretrukket) **eller video**.
- Modell bygget fra bunnen, etter ML-livssyklusen fra forelesningene (Géron kap. 2 + **Appendix A: ML Project Checklist**).
- Rapport etter malen "Template DAT158 assignment 2.docx": **maks 3500 ord**, 5 seksjoner: 1 Describe the problem (Scope, Metrics), 2 Data, 3 Modeling, 4 Deployment, 5 References.
- Alt reproduserbart fra README. AI-verktøy er lov, men kilder og AI-bruk må siteres.
- Grupper på 2–3. **Jeg jobber alene, må bekreftes med faglærer.**
- Referansestil: ingen krav i oppgaven → **APA 7** (HVL-biblioteket / Kildekompasset). AI siteres som referanse + én setning i teksten om hva det ble brukt til.
- Pensum vi knytter til: beslutningstrær → random forest → gradient boosting, ensembler, regularisering, bias/variance, læringskurver, hyperparametere (random vs. grid search), loss-funksjon/gradient descent, end-to-end-system, monitorering.

## 2. Viktige beslutninger (og hvorfor)

| Beslutning | Begrunnelse |
|---|---|
| Lichess open database, sept. 2026, CC0 | Alle partier, klokketider, ~10 % med engine-evals, fri lisens. Alternativer vurdert: Chess.com API (bare per spiller, restriksjoner), Kaggle Lichess-datasett (20k partier, uten klokker), OTB/FIDE (smalt ratingspenn, ingen klokker) |
| Strømme filen, aldri laste ned (29 GB) | Stopper etter N partier; ~180 MB overføres |
| De første 250 000 blitz-partiene | Nok iflg. læringskurven. Ulempe: alle er fra 1. sept. kl. 00:00–07:15 UTC |
| Regresjon, ikke klassifisering | Rating er ordnet/kontinuerlig; et tall med ± er mer informativt |
| 2 rader per parti (egen + motstanders features + farge) | Dobler data, én modell for begge farger; motstanderens spill er informativt fordi Lichess parer like ratinger (korrelasjon 0.95) |
| Split gruppert på parti-ID | Begge rader fra samme parti havner samme sted |
| Lekkasjebeskyttelse i lag | `features.py` kan bare lese 4 headere; appen fjerner ratingheadere i backend; tester beviser identiske features |
| MAE som hovedmål | Direkte tolkbar ("bommer med 225 poeng") og vises som ± i appen |
| Modellvalg på valideringssett, test kun til rapportering | Unngår å velge modell etter testsettet |
| `predict.py` adskilt fra UI | Gjør det billig å bytte frontend |
| **Hosting: Streamlit Community Cloud** (ikke gjort ennå) | Hugging Face krever nå betalt PRO for Gradio Spaces (oppdaget 7. okt.). Streamlit er gratis, står på kursets liste og deployer fra GitHub |

## 3. Resultater fra draft v1 (250k partier, 100k testrader)

| Modell | Test MAE | R² |
|---|---:|---:|
| Gjennomsnitt (baseline) | 292 | 0.00 |
| Snitt per tidskontroll (heuristikk) | 281 | 0.07 |
| Ridge | 235 | 0.34 |
| Random forest | 239 | 0.32 (78 MB, for stor) |
| HistGradientBoosting, tunet (brukt i appen) | 225 | 0.39 |
| Samme modell på spillere aldri sett i trening | 225 | 0.39 |
| HGB + engine-features (set B), bare partier med evals | 243 mot 262 uten | 0.51 mot 0.42 |

Funn verdt å skrive om:
- **Tuning ga ingenting** (224.8 mot 224.7); feilen skyldes bias/støy, ikke varians.
- **Læringskurve:** 235 → 225 fra 10k til 200k partier, avtagende gevinst. Ridge flat på ~235 (høy bias).
- **Regresjon mot snittet:** <1000 overestimeres +437, 2000+ underestimeres −341.
- **Viktigste features:** åpning (ECO), deretter klokke (variasjon i tid per trekk, snittid, tid på første 10 trekk). Sterke spillere bruker 17 s på åpningen mot 42 s for <1000.
- **Engine-features** gir størst forbedring (−20 MAE): bedre informasjon per parti slår mer data.
- Latens i appen: ~25 ms. Modellfil: 0.6 MB.
- Sammenlign: Omori & Tadepalli (2024), dyp CNN-LSTM på 1M+ partier: MAE 182 (ikke direkte sammenlignbart).

## 4. Kilder funnet (sjekk selv før bruk)

- Lichess open database: https://database.lichess.org/ (CC0)
- Lichess FAQ, Glicko-2, provisorisk rating ved RD > 110: https://lichess.org/faq
- lichess-org/chess-openings (CC0): https://github.com/lichess-org/chess-openings
- Omori & Tadepalli (2024), *Chess Rating Estimation from Moves and Clock Times Using a CNN-LSTM*, CG 2024: https://arxiv.org/abs/2409.11506 (**nærmeste relaterte arbeid, les det**)
- Tijhuis, Mavromoustakos Blom & Spronck (2023), *Predicting Chess Player Rating Based on a Single Game*, IEEE CoG: https://research.tilburguniversity.edu/en/publications/predicting-chess-player-rating-based-on-a-single-game/
- McIlroy-Young et al. (2020), Maia, KDD '20: https://arxiv.org/abs/2006.01855
- Kaggle *Finding Elo*: https://www.kaggle.com/c/finding-elo (bare tittelen kunne bekreftes)
- Kaggle *Chess Game Dataset (Lichess)*, datasnaek: https://www.kaggle.com/datasnaek/chess
- Chess.com Game Review: https://support.chess.com/article/364-how-does-the-game-report-analysis-work
- Chess.com Published-Data API: https://support.chess.com/en/articles/9650547-published-data-api
- Guess The Elo (itch.io): https://hieuimba.itch.io/guess-the-elo
- Géron (2022), *Hands-On ML* 3rd ed., kap. 2 + Appendix A. Pedregosa et al. (2011), scikit-learn, JMLR.

## 5. Fallgruver vi har truffet (spar tid neste gang)

- **Swiss-partier** har `Event "Blitz swiss"` uten ordet "Rated", men er ratede. Filtrer bort bare "Casual".
- **HistGradientBoosting** tåler maks 255 kategorier per feature; ECO har ~500. Bruk `OrdinalEncoder(max_categories=250)`.
- **Early stopping i HGB** splitter valideringen tilfeldig per rad, så de to radene fra samme parti kan havne på hver side. Slått av under tuning.
- **Gradio 6:** `theme` og `css` sendes til `launch()`, ikke `gr.Blocks()`.
- En `app.py` i roten samtidig med en `app/`-mappe gir importkonflikt.
- **Hugging Face:** Gradio/Docker-Spaces på gratis CPU krever nå PRO. Bare statiske Spaces er gratis.
- `python-chess` parser fritekst som et parti med 0 trekk. Sjekk at det finnes trekk.
- Månedsfilen er kronologisk, så de første 250k er fra noen få timer. Vurder å sample over hele måneden (f.eks. behold hvert 10. parti) hvis tid.
- Ikke lim tokens/passord inn i chatten (HF-tokenet fra 7. okt. bør være slettet).

## 6. Hva som gjenstår

- [ ] Spørre faglærer: alene OK? Gjelder "AI coding tools" også rapporttekst? Teller referanselisten i 3500 ord?
- [ ] Bygge på nytt med egen research og egen rapport (dette repoet er utkast).
- [ ] Bytte UI til Streamlit og deploye på Streamlit Community Cloud.
- [ ] Demovideo som backup.
- [ ] Rydde: tabeller bare som `.md`, fjerne ubrukte figurer og 20k-testfiler.
- [ ] Valgfritt: eksperiment uten klokke-features (hvor mye dårligere uten klokker?).
- [ ] Én setning om gradient boosting = gradient descent på en loss-funksjon (knytter til modul 2).
- [ ] Suksesskriterium (≥15 % bedre enn baseline) er satt etter at resultatene var kjent. Enten sett et eget kriterium før ny kjøring, eller skriv det ærlig.
- [ ] Endelig AI-erklæring skrives av meg selv, i APA 7.

## 7. Komme i gang på Mac

```bash
# Python 3.13 (hvis ikke installert): brew install python@3.13
git clone https://github.com/jonkje-11/chess-rating-predictor.git
cd chess-rating-predictor
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make test            # på Mac fungerer make
make data features train evaluate   # full kjøring ~15 min (8 kjerner: noe lengre)
make app             # http://127.0.0.1:7860
```

Prøvekjøring: `python -m src.download --n 20000 && python -m src.dataset --n 20000 && python -m src.train --n 20000 --no-tune && python -m src.evaluate --n 20000`.

Ny Claude-økt: be den lese `AGENTS.md` og `CONTEXT.md` først.
