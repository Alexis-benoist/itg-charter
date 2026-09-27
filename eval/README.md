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
