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
  La toolchain Ubuntu (`/usr/bin/cargo`) n'a ni rustfmt ni clippy : utiliser la toolchain rustup
  installée sans toucher au PATH, `~/.cargo/bin/cargo fmt --all` et `~/.cargo/bin/cargo clippy ...`
  (même version que la CI : stable).
- Évaluations sur la bibliothèque du jeu (`SONGS=~/Downloads/ITGmania-1.1.0-Linux-no-songs/itgmania/Songs`) :
  - `cargo run --release --example eval_sync -- $SONGS --max 400 --split test [--stems] [--tag T]` —
    BPM/offset vs simfiles humains ; split train/test déterministe par titre ; CSV par morceau dans
    `target/eval/`, résumé versionné dans `eval/` ;
  - `fit_sync` / `fit_octave` (sur les CSV d'`eval_sync --split train`) — recalibrent les poids du
    demi-temps et le prior de tempo ; toujours rapporter le score sur le split **test** ;
  - `cargo run --release --example eval_charts -- $SONGS 40 [parity_weight temperature repeat_bonus]` —
    stats des charts générés vs distribution humaine (objectif : `outside human range: 0`) ;
  - `cargo run --release --example parity_check` — notre parité vs `#TECHCOUNTS` du cache du jeu
    (`~/.itgmania/Cache/Songs`).
  Relancer l'éval concernée après toute modification de l'analyse, du générateur ou de la parité.

## Architecture (`src/`)

| module | rôle |
|---|---|
| `audio.rs` | décodage symphonia 0.6 → mono f32 + tags titre/artiste |
| `stems.rs` | Demucs (`htdemucs`) en sous-processus, cache par sha256 du fichier audio |
| `analysis.rs` | aubio (onsets spectral-flux, beat tracker) + fit BPM constant multi-résolution, phase, temps fort (kick) |
| `model.rs` | stats apprises sur les charts humains : n-gramme de flèches, densités, snaps, sauts, holds, tech, meter |
| `parity.rs` | **port Rust de la parité d'ITGmania** (`StepParity*.cpp`, `TechCounts.cpp`) |
| `chart.rs` | placement des notes (onsets → grille) + choix des flèches (beam search) + meter |
| `simfile.rs` | lecture `.sm`/`.ssc`, timing, écriture `.sm` |
| `difficulty.rs` | les 5 difficultés, parsing, sel de seed par difficulté |
| `synth.rs` | signaux de test synthétiques (tests + fixture) |

Principe : **aucune règle de pattern inventée à la main**. Le « naturel » vient du modèle appris
(`model/model.json`, entraîné sur les ~6800 charts dance-single du dossier Songs du jeu, embarqué
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
- Venv Demucs (Python 3.11, torch CUDA ; GTX 1650 ≈ 20 s par morceau) :
  ```
  uv venv --python 3.11 ~/.local/share/itg-charter/demucs-venv
  VIRTUAL_ENV=~/.local/share/itg-charter/demucs-venv uv pip install demucs torch torchaudio soundfile
  ```
  Autre interpréteur : `ITG_CHARTER_PYTHON=/chemin/python`. Sans Demucs : `--no-stems`.
- Le jeu : `~/Downloads/ITGmania-1.1.0-Linux-no-songs/itgmania` ; données utilisateur et cache dans
  `~/.itgmania/`. Pour tester en jeu : `-o ~/.itgmania/Songs/itg-charter`.

## Résultats de référence (à maintenir / améliorer)

- Sync (split test, 400 morceaux, sans stems) : BPM exact 89,2 %, octave 10,2 % ; sur BPM exact,
  97,5 % calés (< 30 ms), 1,7 % d'erreurs d'un demi-temps, phase médiane 5,0 ms, 94 % sous 20 ms.
  Limites connues : choix d'octave (le prior seul n'y fait rien, cf. `fit_octave`) ; boucles EDM
  synthétiques à contretemps (tests `#[ignore]` dans `analysis.rs`).
- Charts (40 morceaux × 5 difficultés) : toutes les médianes (NPS, sauts, holds, crossovers,
  footswitches, jacks, double-steps, meter) dans l'intervalle p10–p90 humain.
