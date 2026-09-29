# itg-charter

Génère des charts ITGmania / In The Groove (`.sm`, `dance-single`) à partir d'un fichier audio,
pour les difficultés choisies. Reproductible : même audio + même seed + mêmes options ⇒ `.sm`
identique à l'octet.

```
itg-charter gen song.mp3 -d easy,medium,hard -s 42 -o ~/.itgmania/Songs/Generated
itg-charter analyze song.mp3            # BPM / offset détectés
itg-charter train <Songs/>              # réentraîne model/model.json
```

## Commandes

- `cargo build --release` — le binaire est `target/release/itg-charter`.
- `cargo test` — rapide (le profil test est en `opt-level = 3`, l'analyse audio est lente sinon).
- `cargo test -- --ignored` — lance aussi le vrai Demucs (venv requis, voir plus bas).
- `cargo fmt --all --check` et `cargo clippy --all-targets -- -D warnings` — exigés par la CI.
  Un cargo de distribution (ex. `/usr/bin/cargo` d'Ubuntu) peut ne fournir ni rustfmt ni clippy :
  utiliser alors une toolchain rustup stable (même version que la CI), par exemple
  `~/.cargo/bin/cargo fmt --all` et `~/.cargo/bin/cargo clippy ...`.
- Évaluations sur une bibliothèque de chansons ITGmania (`SONGS=$ITGMANIA_DIR/Songs`, où
  `$ITGMANIA_DIR` est le dossier d'installation du jeu) :
  - `cargo run --release --example eval_sync -- $SONGS --max 400 --split test [--stems] [--tag T]` —
    BPM/offset vs simfiles humains ; split train/test déterministe par titre ; CSV par morceau dans
    `target/eval/`, résumé versionné dans `eval/` ;
  - `fit_sync` / `fit_octave` (sur les CSV d'`eval_sync --split train`) — recalibrent les poids du
    demi-temps et le prior de tempo ; toujours rapporter le score sur le split **test** ;
  - `cargo run --release --example eval_charts -- $SONGS 40 [parity_weight temperature repeat_bonus]` —
    stats des charts générés vs distribution humaine (objectif : `outside human range: 0`) ;
  - `cargo run --release --example eval_placement -- $SONGS --max 300 --split test [--tag T]` —
    nos lignes tombent-elles là où l'humain les met (grille humaine imposée) : précision / rappel /
    F, corrélation des densités par mesure, reprise du rythme des mesures similaires ;
  - `cargo run --release --example fit_placement -- $SONGS` — réajuste `model/placement.json`
    (régression logistique par difficulté, train ; log-loss rapportée sur test) ;
  - `cargo run --release --example music_signal -- $SONGS` — la musique prédit-elle les flèches ?
  - `ITGMANIA_DIR=… cargo run --release --example parity_check` — notre parité vs `#TECHCOUNTS`
    du cache du jeu (`~/.itgmania/Cache/Songs`).
  Relancer l'éval concernée après toute modification de l'analyse, du générateur ou de la parité.

## Architecture (`src/`)

| module | rôle |
|---|---|
| `audio.rs` | décodage symphonia 0.6 → mono f32 + tags titre/artiste |
| `stems.rs` | Demucs (`htdemucs`) en sous-processus, cache par sha256 du fichier audio |
| `analysis.rs` | aubio (onsets spectral-flux, beat tracker) + fit BPM constant multi-résolution, phase, temps fort (kick) |
| `model.rs` | stats apprises sur les charts humains : n-gramme de flèches, densités, snaps, sauts, holds, tech, meter |
| `parity.rs` | **port Rust de la parité d'ITGmania** (`StepParity*.cpp`, `TechCounts.cpp`) |
| `placement.rs` | placement appris : P(un humain met une ligne ici \| 17 indices audio/métriques), `model/placement.json` |
| `chart.rs` | placement des notes (probabilités apprises, quotas par mesure, reprise du rythme) + choix des flèches (beam search) + meter |
| `music.rs` | indices musicaux par note (hauteur yinfast, type de frappe, accent) ; split train/test partagé (`split_of`) |
| `simfile.rs` | lecture `.sm`/`.ssc`, timing, écriture `.sm` |
| `difficulty.rs` | les 5 difficultés, parsing, sel de seed par difficulté |
| `synth.rs` | signaux de test synthétiques (tests + fixture) |

Principe : **aucune règle de pattern inventée à la main**. Le « naturel » vient du modèle appris
(`model/model.json`, entraîné sur 19 321 charts dance-single — Songs du jeu + 125 packs de
`training/packs.tsv`, voir `training/PLAN.md` et l'exemple `count_charts` —, embarqué
dans le binaire via `include_str!`), le « jouable » vient des coûts de parité d'ITGmania.

## Règles à respecter

- **Déterminisme** : toute aléa passe par `ChaCha8Rng::seed_from_u64(seed ^ difficulty.seed_salt())`.
  Jamais d'itération sur un `HashMap`/`HashSet` qui influence la sortie (lookup seulement ; ordonner
  avec `Vec`/`BTreeMap`). Tris avec départage explicite (`then(pos)`). Pas de temps/horloge/threads
  dans la génération. Un chart ne dépend pas des autres difficultés demandées (test dédié).
- **Demucs** : toujours `--shifts 0` (sinon décalage aléatoire). Les stems sont mis en cache
  (`~/.cache/itg-charter/stems/<sha256>-htdemucs/`) : c'est ce qui garantit la reproductibilité,
  l'inférence GPU n'étant pas bit-exacte.
- **Parité** : garder `parity.rs` au plus près de l'upstream (mêmes noms, même ordre, `f32`) pour
  pouvoir reporter les évolutions d'ITGmania. Écarts connus documentés en tête de fichier.
  Validation actuelle : 97,3 % des 6832 charts du cache ont des tech counts identiques au jeu.
- **Modèle** : `train` est déterministe (fichiers triés, compteurs entiers). Changer le format ⇒
  incrémenter `MODEL_VERSION` et réentraîner.
- **Calibrations** mesurées, pas devinées : `ENVELOPE_LATENCY` (17 ms, via `eval_sync`) ;
  `HALF_BEAT_WEIGHTS` (via `fit_sync`) ; `TEMPO_PRIOR_*` (via `fit_octave`) ;
  `GenOptions::default()` (via `eval_charts`). Ajuster sur train, mesurer sur test, documenter.
- **Un commit par expérience**, y compris celles qui régressent, avec les chiffres mesurés dans le
  message et le résumé `eval/*.txt` ajouté (voir `eval/README.md`).
- **MP3** : symphonia est gapless, le jeu non ; `audio.rs` ajoute en tête le silence que le jeu joue
  (trame Info + délai LAME) et `SongAnalysis::compute` fait de même pour les stems.
- Les grilles utilisées pour charter sont arrondies comme dans le `.sm` (BPM et offset à 0,001).
- Code et commentaires en anglais ; licence GPL-3.0 (aubio et le code de parité d'ITGmania sont GPL).

## Environnement

- `.cargo/config.toml` définit `CFLAGS=-D_DEFAULT_SOURCE` : sans ça, le C d'aubio (compilé par
  `aubio-rs` feature `builtin`) ne compile pas avec GCC ≥ 14.
- Venv Demucs (Python 3.11, torch CUDA ; ≈ 20 s par morceau sur un GPU de 4 Go) :
  ```
  uv venv --python 3.11 ~/.local/share/itg-charter/demucs-venv
  VIRTUAL_ENV=~/.local/share/itg-charter/demucs-venv uv pip install demucs torch torchaudio soundfile
  ```
  Autre interpréteur : `ITG_CHARTER_PYTHON=/chemin/python`. Sans Demucs : `--no-stems`.
- Le jeu : installation ITGmania dans `$ITGMANIA_DIR` (contient `Songs/`) ; données utilisateur et
  cache dans `~/.itgmania/` (Linux). Pour tester en jeu : `-o ~/.itgmania/Songs/itg-charter`.

## Résultats de référence (à maintenir / améliorer)

- Sync (split test, 400 morceaux, sans stems) : BPM exact 89,2 %, octave 10,2 % ; sur BPM exact,
  97,5 % calés (< 30 ms), 1,7 % d'erreurs d'un demi-temps, phase médiane 5,0 ms, 94 % sous 20 ms.
  Limites connues : choix d'octave (le prior seul n'y fait rien, cf. `fit_octave`) ; boucles EDM
  synthétiques à contretemps (tests `#[ignore]` dans `analysis.rs`).
- Placement (split test, 300 morceaux, grille humaine, corpus `~/itg-train` de 19 321 charts) :
  F-score 69,4 % avec notre nombre de lignes, 72,7 % à nombre égal (hasard 49,9 %) ; corrélation
  des densités par mesure 0,28 ; reprise du rythme des mesures similaires 42,7 % (humain 47,3 %).
  Plus de données n'y change rien (×4,9 charts, `fit_placement --max-train 1500`) : la régression
  logistique est limitée par sa capacité (log-loss train ≈ test), pas par les données.
- Styles (`training/STYLES.md`, exemple `pack_styles`) : 5 styles de packs ; les Songs du jeu sont
  toutes « classique ITG », les packs ajoutés surtout « stream » (plus denses, peu de sauts et de
  crossovers). Sur les Songs du jeu, le modèle entraîné sur elles seules bat celui du corpus élargi
  (perplexité 1,729 contre 1,766 bit/ligne, `eval/perplexity-test-game-songs.txt`) ; plafond humain
  du placement : F médian 71 % entre deux charters (`eval/human-agreement.txt`).
- Flèches ↔ musique : la hauteur du mix ne prédit pas la direction (corrélation ≈ 0) ; seul
  l'accent compte (sauts ×2), cf. `eval/music-signal-train.txt`.
- Données : `Simfile::is_generated()` exclut nos propres simfiles (liens vers des chansons
  générées dans Songs/) de tout entraînement et de toute évaluation.
- Charts (40 morceaux × 5 difficultés) : toutes les médianes (NPS, sauts, holds, crossovers,
  footswitches, jacks, double-steps, meter) dans l'intervalle p10–p90 humain.
