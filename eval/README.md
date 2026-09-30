# Évaluations

Résumés produits par les exemples d'évaluation, committés avec le changement qu'ils mesurent.
Les CSV par morceau sont dans `target/eval/` (non versionnés).

| fichier | changement mesuré |
|---|---|
| `sync-train.txt` | référence, split train (poids demi-temps = kick seul) |
| `sync-test-baseline.txt` | référence, split test |
| `sync-test-halfbeat.txt` | décision demi-temps apprise (régression logistique) |
| `sync-test-mp3pad.txt` | + alignement des MP3 à tag LAME sur le jeu |
| `sync-test-prior.txt` | + prior de BPM appris appliqué à toute la recherche (régression : ×4/3) |
| `sync-test-octave.txt` | prior appris limité au choix d'octave (encore sous la référence) |
| `sync-test-prior-sigma{4,8,12}.txt` | prior appris sur toute la recherche, lissage σ = 4/8/12 bins |
| `sync-{train,test}-candidates.txt` | candidats de tempo enregistrés (prior log-normal 140 / 0,6 inchangé) — base de `fit_octave` |

`fit_octave` (candidats ci-dessus), BPM exact train / test :
actuel 140 / 0,6 oct : 82,5 % / 89,2 % — sans prior : 68,5 % / 68,5 % —
histogramme appris : 84,5 % / 87,0 % — meilleur sur train (132 / 0,8) : 86,5 % / 86,8 %.
| `music-signal-train.txt` | la musique prédit-elle les flèches humaines ? (contexte : Δhauteur du mix, type de frappe, accent) |
| `placement-test-baseline.txt` | nos notes tombent-elles là où l'humain les met ? (grille humaine imposée) — référence avant placement appris |
| `placement-fit.txt` | ajustement du placement appris (log-loss train/test par difficulté, poids) |
| `placement-test-learned.txt` | placement appris (régression logistique par difficulté) vs humains |
| `placement-test-learned-clean.txt` | idem après exclusion des chansons générées (lien Songs/YouTube) et réentraînement ; + corrélation des densités par mesure |
| `placement-test-1a-expected-count.txt` | 1a : nombre naturel = somme des probabilités du placement appris |
| `placement-test-1b-measure-quota.txt` | 1b : quotas par mesure au prorata de la densité attendue |
| `placement-test-1b-share{0.5,0.75}.txt` | quotas partiels : retombent sur la répartition de 1a (pas d'entre-deux) |
| `placement-test-rhythm-baseline.txt`, `placement-test-2-reuse{0.0,0.1,0.3,1.0}.txt` | reprise du rythme des mesures similaires : mesure + bonus β |
| `placement-test-2-reuse-challenge-only.txt` | bonus de reprise du rythme en Challenge seulement (retenu) |
| `charts-corpus{3,4}-{old,new}-model.txt`, `placement-test-corpus{3,4}-{old,new}-model.txt` | ancien vs nouveau modèle sur le corpus élargi `~/itg-train` (13 635 puis 19 321 charts, voir `training/PLAN.md`) |
| `human-agreement.txt` | plafond : deux charters humains sur la même chanson et la même difficulté (F après alignement, ±25 ms) |
| `perplexity-test.txt` | bits par ligne des charts humains du split test d'`~/itg-train` selon le n-gramme de flèches, modèles entraînés sur le split train seul (Songs du jeu ; courbe d'apprentissage 1/8 → 1/1 d'`~/itg-train`) |
| `perplexity-test-game-songs.txt` | idem sur le split test des Songs du jeu : le modèle des Songs du jeu y bat celui du corpus élargi |
| `charts-style-{stream,tech-pw*}.txt` | modèles `--style` sur les packs de leur style ; balayage du poids de parité du style tech (0,02 / 0,01 / 0,005 retenu / 0) |
| `pack-styles.txt` | styles de charting par pack (z-scores à meter égal, k-means à 5 groupes), décrits dans `training/STYLES.md` |
| `perplexity-test{,-game-songs}-classic.txt` | perplexité des modèles Songs du jeu / style classique / corpus complet, sur le split test des Songs du jeu et du style classique |
| `charts-classic-*.txt`, `placement-test-classic-*.txt` | ancien modèle, corpus complet, style classique (flèches + placement), et combinaison retenue (`combo` : flèches classique, placement corpus complet) sur `~/itg-train-classic` |
| `context-{classic,stream,tech}.txt` | information des lignes précédentes : bits par ligne des modèles de flèches d'ordre 1 à 5 (0 à 4 lignes précédentes) par style, courbe d'apprentissage |
| `patterns-{classic,stream,tech}.txt` | patterns au-delà de 2 lignes (escaliers, drills, candles, runs de 16es, diversité, répétition) : nos charts vs les humains du style |
| `patterns-*-order4.txt`, `charts-style-*-order4.txt`, `perplexity-test-classic-order4.txt` | modèle de flèches d'ordre 4 (3 lignes précédentes, format 5) : perplexité, stats, patterns |
| `patterns-sweep-*.txt`, `patterns-diag-*.txt` | températures 0,4 → 1,5 et bonus de répétition ; diagnostic beam / température / parité : le modèle seul échantillonné (beam 1, T 1, parité 0) est humain sur tous les patterns, la parité crée les drills |
| `patterns-grid-*.txt`, `charts-grid-*.txt` | grille température × poids de parité (patterns + stats) ; retenu : T 0,7, parité 0,01 (tech 0,0025) |
| `perplexity-test-{classic,stream,tech}-split.txt` | modèle du style vs modèle unique (tout le corpus), ordre 4, sur le split test de chaque style |
| `charts-fsconst-classic-fsp*.txt`, `charts-fscum-classic-fsp*.txt` | pénalité de footswitch (nats) dans le beam, constante, style classique : sur le pas qui vient d'être posé (`fsconst`, sans effet : la parité révise après coup jack ↔ footswitch) puis sur le nombre de footswitches du meilleur chemin de parité (`fscum`, retenu) |
| `charts-fsp90-{classic,stream,tech}-fsp{1,2,4}.txt`, `patterns-fsp90-*.txt` | pénalité de footswitch divisée par 1 + p90 humain des footswitches /100 de la difficulté, base 1 / 2 / 4 : 0 écart partout en stats ; patterns : base 4 met les drills Challenge classique hors plage (18,4 > 18,1), base 2 retenue |
| `charts-style-{classic,stream,tech}-fs.txt` | options retenues (`eval_charts --style`, pénalité de footswitch 2) : 0 écart dans les 3 styles ; `patterns-{classic,stream,tech}.txt` refaits avec ces options (0 écart) |
