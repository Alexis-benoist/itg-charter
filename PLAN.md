# itg-charter — plan

## Context
Programme Rust : `itg-charter song.mp3 -d easy,medium,hard --seed 42` → dossier de chanson
ITGmania (`<out>/<Titre>/<Titre>.sm` + audio copié) avec un chart `dance-single` par difficulté.
Même entrée + même seed ⇒ `.sm` identique à l'octet. Choix utilisateur : analyse via **aubio (FFI)**,
séparation de pistes **Demucs dès la v1**. Une installation ITGmania (`$ITGMANIA_DIR`) et ses
`Songs/` (packs communautaires : mp3/ogg + `.sm`) servent de vérité terrain. Point de départ : un
projet vide (`cargo init` + deps symphonia 0.6 / rustfft / clap / rand 0.8 / rand_chacha 0.3 / anyhow).

Recherches : TempoSync+FootGraph (netraular/stepmania-generator, docs/METHOD.md) — Viterbi pieds +
densité par couches métriques, bat AutoStepper et DDC ; DDC (arXiv 1703.06891) — placement puis
sélection, densité Beginner < 1 NPS → Expert > 7 NPS. aubio-rs 0.2 : feature `builtin` compile
aubio embarqué, **licence GPL-3.0**.

## Architecture (lib + bin)
- `audio.rs` — décodage symphonia 0.6 (API : `get_probe().probe(..)`, `default_track(TrackType::Audio)`,
  `make_audio_decoder`, `copy_to_slice_interleaved`) → mono f32 ; tags titre/artiste. mp3 + ogg.
- `analysis/` — trait `Analyzer` → `Analysis { bpm, offset, onsets: Vec<(t, strength, low)>, rms }`.
  - backend `aubio` : `Tempo` (bpm) + `Onset` (mode `specflux`/`hfc`) ; filtre passe-bas maison pour la
    bande kick. Refinement maison : BPM fin + arrondi entier si score quasi égal, phase par peigne,
    downbeat par énergie basse, `#OFFSET = -beat0`.
  - `stems.rs` — **Demucs (v1)** : sous-processus `python -m demucs -n htdemucs --shifts 0
    -o <cache> song.mp3` (shifts=0 : sinon décalage aléatoire, non déterministe) dans un venv dédié
    `uv venv --python 3.11 ~/.local/share/itg-charter/demucs-venv` + `uv pip install demucs torch`
    (CUDA sur un GPU de 4 Go, `--segment 7` ; fallback `--device cpu`, ~2 min/morceau).
    Stems (drums, bass, vocals, other) **mis en cache** dans `~/.cache/itg-charter/<sha256 audio>/`
    → le déterminisme seed ⇒ octets tient même si le GPU n'est pas bit-exact (commande
    `--device cpu` pour un calcul strictement reproductible). `--no-stems` pour s'en passer ;
    si Demucs absent : avertissement + analyse sur le mix complet.
  - Usage des stems : tempo/phase sur mix + drums ; onsets **drums** = rythme de base (kick → sauts
    / temps forts, snare, hats pour les 16ths en Hard+) ; onsets **bass/vocals/other** = couches
    mélodiques ajoutées aux difficultés hautes ; notes tenues (énergie soutenue sur vocals/other)
    → holds. Chaque difficulté choisit ses couches (Beginner : kick ; Challenge : tout).
- `grid.rs` — grille à 1/48 de mesure (4ths, 8ths, 12ths, 16ths) ; chaque onset quantifié à la
  subdivision la plus grossière dans la tolérance.
- `difficulty.rs` — Beginner/Easy/Medium/Hard/Challenge : NPS cible (≈1.1 / 2 / 3.3 / 5 / 7, modulé
  par la densité naturelle du morceau), subdivisions autorisées, sauts, holds, crossovers, poids de coûts.
- `placement.rs` — densité par couches (temps → croches → doubles) suivant l'énergie locale ;
  holds sur notes fortes suivies d'un silence ; sauts sur kicks forts.
Principe : **aucune règle biomécanique inventée par le LLM**. Deux sources éprouvées :
- `parity/` — **port Rust de `StepParityGenerator.cpp` d'ITGmania** (branche beta, dérivé du travail
  de tillvit / mjvotaw, 14 coûts : distance, facing, doublestep, twisted foot, slow bracket,
  footswitch, jack, sideswitch, holdswitch, spin…, poids réglés par la communauté). Vérifier la
  licence du fichier avant le port. Sert (a) à annoter les charts humains (quel pied, crossover,
  footswitch), (b) de coût dans la génération, (c) de métrique de jouabilité.
- `model/` — **modèle de patterns appris** : `itg-charter train <Songs/>` parse les ~1791 simfiles
  du jeu (`.sm`/`.ssc`, dance-single), annote la parité, et compte des n-grammes de transitions
  (flèches + pied + saut/hold) conditionnés par difficulté et écart temporel (bucket 4th/8th/16th).
  Sortie : `model.json` versionné (déterministe : entrées triées). Idem pour les densités par
  difficulté et le lien meter ↔ NPS (remplace les valeurs codées en dur).
- `generate.rs` — beam search / Viterbi : score = log-prob du n-gramme − coût de parité, bruit
  ChaCha8 seedé pour la variété. RNG par difficulté = seed ⊕ constante(difficulté) → Hard identique
  qu'on demande Easy ou non. Crossovers/footswitchs limités par difficulté selon les stats apprises.
- `repeat.rs` — sections répétées (auto-similarité chroma/énergie par mesure) → réutilise le pattern.
- `meter.rs` — meter 1–13 depuis NPS des mesures chargées (p75) + sauts/crossovers.
- `sm.rs` — écriture `.sm` (format vérifié sur les fichiers JBEAN) + parseur minimal (tests/éval).
- `main.rs` — clap : `<AUDIO> -d … --seed N (défaut 0) -o DIR --bpm --offset --title --artist`.

## Étapes
1. Vérifier `aubio-rs` feature `builtin` compile (sinon `apt install libaubio-dev` + pkg-config).
2. audio + sm (écriture/parse) + tests. Installer le venv Demucs, valider sur un mp3 du jeu
   (temps, VRAM, `--shifts 0` bit-exact sur CPU).
3. analyse aubio + refinement ; exemple `examples/eval.rs` : BPM/offset détectés vs `.sm` réels
   (BPM unique) du dossier ITGmania → taux de BPM exact, erreur d'offset en ms ; itérer.
4. port de la parité ITGmania (tests : reproduire les compteurs tech du jeu sur quelques charts) ;
   `train` sur Songs/ → model.json ; placement + génération + meter appris.
   Métriques : coût de parité moyen, crossovers, footswitchs, équilibre, NPS — comparés aux charts
   humains de même difficulté (objectif : dans leur distribution).
5. repeat.rs, CLI, copie audio, CLAUDE.md (commandes, archi, règles de déterminisme).
6. **GitHub Actions** (`.github/workflows/ci.yml`, push + PR) : `cargo fmt --check`,
   `cargo clippy -- -D warnings`, `cargo test` (ubuntu-latest, cache cargo, ffmpeg installé pour
   les fixtures). Sans Demucs ni chansons du jeu : ces tests sont `#[ignore]` et tournent en local.
   Job de déterminisme : génère la fixture 2× et compare les `.sm` (sha256).

## Tests
- `.sm` round-trip et compression des mesures ; parse d'un vrai `.sm` JBEAN.
- BPM/offset sur click-tracks synthétiques (plusieurs tempos) + fixture mp3 générée par ffmpeg.
- Déterminisme : même seed ⇒ octets identiques ; seed différente ⇒ différent ; indépendance des difficultés.
- Jouabilité : NPS croissant Beginner→Challenge, Beginner uniquement noires sans sauts, jamais de note
  sur colonne tenue, 0 crossover ≤ Medium.
- Stems : tests sans Demucs via un dossier de stems factices (cache pré-rempli) ; test `#[ignore]`
  qui lance vraiment Demucs ; éval comparative avec/sans stems (erreur d'onset, BPM).
- Parité : cas connus (stream LDUR sans crossover, LDR-LUR = crossover, jack vs footswitch).
- Modèle : `train` deux fois ⇒ model.json identique ; petit corpus de fixtures .sm.
- Intégration : binaire complet sur la fixture, sortie relue par le parseur.
- Éval `#[ignore]`/exemple sur les chansons du jeu (release).

## Vérification
`cargo test`, `cargo run --release --example eval_sync -- $ITGMANIA_DIR/Songs`,
génération d'un morceau dans `~/.itgmania/Songs/itg-charter/` puis test en jeu par l'utilisateur.

## État (2026-09-27)

Fait : étapes 1 à 6 (sauf `repeat.rs` intégré dans `chart.rs::find_repeats`). Mesures :
- parité : 97,3 % de tech counts identiques au jeu sur 6832 charts (`examples/parity_check.rs`) ;
- sync : BPM exact 83 %, à l'octave près 98,6 %, phase 85 % < 20 ms (`examples/eval_sync.rs`) ;
- charts : toutes les stats dans p10–p90 humain (`examples/eval_charts.rs`).
Écart au plan : Beginner n'est pas limité aux noires sans sauts — les données humaines ont 9 %
de sauts et quelques croches en Beginner, on suit les données.
Reste : erreurs d'un demi-temps (~11 %), BPM variables, notes pendant les holds, test en jeu.

## Synchro (2026-09-27, suite)

Résultats (split test, 400 morceaux ; détails dans `eval/`) :
- décision demi-temps apprise (`fit_sync`) : 79,8 % → 98,3 % de bonnes décisions ;
- MP3 à tag LAME alignés sur le jeu : MP3 calés 47 % → 95 % ;
- au total, morceaux calés sur BPM exact 76,8 % → 97,5 %, phase médiane 5,0 ms ;
- octave : aucun prior (appris ou ajusté) ne bat 140 / 0,6 octave sur test (89,2 % BPM exact).
Demucs n'intervient plus dans la phase (features calculées sur le mix) ; l'utiliser pour le
demi-temps demanderait de réentraîner avec stems (~4 h GPU) — non fait.
