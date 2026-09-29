# Doubler le corpus d'entraînement (packs ITG ≤ 10 Go) puis réentraîner

## Contexte
`model/model.json` est entraîné sur `/opt/itgmania/Songs` : 13 packs, 847 simfiles, ≈ 6 800 charts
dance-single. On télécharge des packs listés dans la feuille « ITG Packs Release Spreadsheet »
(https://docs.google.com/spreadsheets/d/1F1IURV1UAYiICTLhAOKIJfwUN1iG12ZOufHZuDKiP48/htmlview),
**dans un dossier séparé** (`~/itg-packs`, pas le Songs du jeu), jusqu'à **au moins doubler le
nombre de charts** (≥ 13 600 au total), sans dépasser 10 Go de téléchargement. Puis on réentraîne.

## Source des packs (vérifiée)
- Les liens `search.stepmaniaonline.net` de la feuille sont morts (redirection vers l'accueil).
- `https://nnty.fun/downloads/game/stepmania/stepmania_packs/` : miroir SMO ouvert (index h5ai,
  6 391 zips avec leur taille). 1 011 packs de la feuille y sont (appariement par nom normalisé).
- Filtres : format Singles / Singles + Doubles (pad 4 panneaux) ; pas de Content Type Mods/gimmick ;
  ≥ 3 charts singles par chanson en moyenne (colonne Difficulties, par ex. `1-5S/0-4D`) pour couvrir
  toutes les difficultés ; pack entre 5 Mo et 900 Mo ; packs déjà présents exclus.
- Tri glouton par charts estimés par Mo → premier lot (`training/packs.tsv`) : **66 packs, 4,6 Go,
  ≈ 1 900 chansons, ≈ 9 000 charts estimés**.

## Étapes
1. ✅ Worktree `worktree-more-training-packs` ; nouveaux fichiers seulement :
   - `training/packs.tsv` : pack, nom du zip sur le miroir, taille, chansons, charts singles/chanson ;
   - `training/fetch_packs.py` : télécharge et dézippe dans `~/itg-packs/<pack>/` (reprise possible,
     4 en parallèle, zip supprimé après extraction, refuse si la liste dépasse `--max-gb 10`).
2. ⏳ Téléchargement : `python3 training/fetch_packs.py --dest ~/itg-packs`.
3. ⏳ Comptage réel des charts dance-single (mêmes critères que `Model::train` : difficulté
   parsable, ≥ 16 lignes, `is_generated` exclu). Si < ×2, ajouter les candidats suivants à
   `packs.tsv`, toujours sous 10 Go, et relancer le script.
4. ⏳ Dossier combiné `~/itg-train/` : liens symboliques vers `/opt/itgmania/Songs/*` et
   `~/itg-packs/*` (`find_simfiles` suit les liens). Aucun changement de code ; sert aussi de
   `$SONGS` pour les évals.
5. ⏳ Réentraînement :
   - `target/release/itg-charter train ~/itg-train` → `model/model.json` (format inchangé,
     `MODEL_VERSION` inchangé) ;
   - `cargo run --release --example fit_placement -- ~/itg-train` → `model/placement.json`.
6. ⏳ Évals avant/après sur le nouveau corpus (le split test change avec le corpus : on mesure aussi
   l'ancien modèle sur le nouveau corpus pour comparer à armes égales) :
   - `cargo run --release --example eval_charts -- ~/itg-train 40` (objectif `outside human range: 0`) ;
   - `cargo run --release --example eval_placement -- ~/itg-train --max 300 --split test`.
7. ⏳ Commits (un par expérience, chiffres dans le message, résumés `eval/*.txt`) : liste + script ;
   nouveau `model.json` ; nouveau `placement.json`. Mise à jour des chiffres de référence dans
   CLAUDE.md. Push de la branche, sans merge dans main.

## Passation (pour l'agent qui reprend)
**État au 2026-09-29** : les étapes 2 à 7 n'ont pas été faites. Aucun téléchargement, rien n'est
commité. Les commandes Bash de la session précédente étaient bloquées (le classifieur du mode auto
ne répondait plus).

- **Où** : worktree `/home/alexis/itg-charter/.claude/worktrees/more-training-packs`, branche
  `worktree-more-training-packs` (partie de `main` @ 60e7a49). Pour y entrer :
  `EnterWorktree(path=…)`. Sinon, recréer une branche depuis main et y copier `training/`.
- **Fichiers** (non commités) :
  - `training/PLAN.md` : ce fichier ;
  - `training/packs.tsv` : premier lot figé (66 packs, 4,6 Go) ;
  - `training/fetch_packs.py` : téléchargement + extraction (n'a jamais tourné, à tester sur un petit
    pack d'abord ; l'URL = `MIRROR + quote(zip + ".zip")`) ;
  - `training/select_packs.py` : régénère la liste classée de tous les candidats, pour compléter
    si le lot 1 ne double pas le corpus (réécrit de mémoire à partir de l'analyse d'origine :
    vérifier que ses premières lignes correspondent bien à `packs.tsv`).
- **Premier commit à faire** : `training/` (liste + scripts + plan), message avec les chiffres
  ci-dessus. Ne pas pousser sur main, ne pas merger.
- **Compter les charts** : un petit script Python qui parcourt les `.ssc`/`.sm` (préférer `.ssc`
  dans un même dossier, comme `find_simfiles` dans `src/model.rs`), et compte les blocs `#NOTES`
  (ou `#STEPSTYPE` en `.ssc`) en `dance-single`. Référence : même méthode sur `/opt/itgmania/Songs`
  (≈ 6 800 attendus). Autre option : ajouter un `eprintln!` temporaire dans `Model::train`, non
  commité.
- **Pièges** :
  - certains zips peuvent contenir un dossier de pack imbriqué, ou des `.rar`/fichiers audio
    manquants : ce n'est pas grave, `find_simfiles` est récursif et l'entraînement ne lit que les
    simfiles. Pour `eval_placement`, il faut en revanche l'audio ;
  - chansons en double entre packs : acceptable, mais à signaler dans le commit ;
  - le split train/test (`music::split_of`, par titre) change avec le corpus : toujours comparer
    ancien et nouveau modèle sur **le même** `~/itg-train` ;
  - `eval_*` écrivent leurs CSV dans `target/eval/` et leurs résumés dans `eval/` (voir
    `eval/README.md`) ;
  - `cargo fmt`/`clippy` : utiliser `~/.cargo/bin/cargo` si le cargo système n'a pas ces outils.
- **Règles du projet** (CLAUDE.md + mémoire) : un commit par expérience, y compris celles qui
  régressent, avec les chiffres dans le message ; déterminisme ; ne pas refactorer les fichiers
  partagés si un autre agent travaille (ajouter des fichiers seulement) ; lancer les téléchargements
  en parallèle.
- **Rapport final attendu par l'utilisateur** : nombre de charts avant/après, taille téléchargée,
  diff des évals ancien/nouveau modèle, branche poussée.

## Vérification
- `cargo test`, `cargo clippy --all-targets -- -D warnings`, `cargo fmt --all --check`.
- `itg-charter gen` lancé deux fois sur un même morceau : `.sm` identique à l'octet.
- `du -sh ~/itg-packs` ≤ 10 Go ; nombre de charts ≥ 2 × l'ancien.
