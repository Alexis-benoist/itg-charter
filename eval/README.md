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
