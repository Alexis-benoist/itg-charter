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
| `placement-test-A1-baseline.txt` | référence `HEAD` relancée sur la bibliothèque actuelle (le nombre de morceaux a changé depuis les résumés précédents) |
| `placement-test-A1-section.txt` | A1 : + 6 indices de section (volume mesure / phrase, bande kick, onsets par mesure, changement de section, position) — log-loss test meilleure partout, mais corrélation par mesure 0,316 → 0,293 (régression) |
