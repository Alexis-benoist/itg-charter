# Styles de charting du corpus d'entraînement

Mesurés par `cargo run --release --example pack_styles -- ~/itg-train 5` (sortie brute :
`eval/pack-styles.txt`, déterministe).

**Méthode.** Pour chaque chart dance-single : 13 indices de style (densité en notes/s, part de
streams en 16es, triolets, sauts, holds, rolls, mines, crossovers, footswitches, jacks, brackets,
changements de BPM / stops / delays, BPM principal). Chaque indice est converti en z-score par
rapport aux charts **du même meter** : un pack de charts durs n'apparaît pas « dense » à cause de sa
difficulté. Le profil d'un pack est la moyenne de ces z-scores (131 packs d'au moins 20 charts),
regroupés en 5 styles par k-means (initialisation déterministe). Le nombre de groupes (5) est un
choix, pas une mesure.

Valeurs typiques : médianes des charts de meter 8 à 10. Référence, tous packs confondus :
4,2 notes/s, 5,4 % de sauts, 4,8 % de holds, 1,0 crossover / 100 notes, 0 mine, BPM 142.

| style | packs | charts | notes/s | sauts | holds | crossovers /100 | mines /100 | BPM |
|---|---|---|---|---|---|---|---|---|
| Classique ITG | 48 | 9 246 | 4,1 | 5,7 % | 3,6 % | 1,1 | 0 | 140 |
| Stream / pad moderne | 60 | 7 868 | 4,4 | 4,7 % | 7,5 % | 0,35 | 0,3 | 155 |
| Tech / crossovers | 12 | 1 554 | 4,4 | 7,4 % | 2,7 % | 3,8 | 0 | 145 |
| Homebrew « gimmick » | 9 | 590 | 4,3 | 8,3 % | 13 % | 3,0 | 9,0 | 147 |
| Kyzentun (expérimental) | 2 | 76 | 3,7 | 1,1 % | 9,2 % | 4,1 | 4,9 | 145 |

## Classique ITG

Tous les packs des **Songs du jeu** (JBEAN Originals 2018–2023, Jbean3535 Exclusives, Sexuality
Violation 1–3, DDR vs PIU), In The Groove 3, In The Groove Rebirth 1 et 2, Banzaitv's ITG Home,
Hopscotch Mix, Albumix2, Dancefreak's Originals, Rynker's Hard Techno, PandemiXium…

Proche de la moyenne sur tout : un peu plus de sauts, peu de holds, pas de mines, crossovers
modérés. Les charts JBEAN sont **moins denses que leur meter** (notation généreuse), sur des BPM
plutôt lents, avec très peu de crossovers : lisibles et propres.

## Stream / pad moderne

TalonMix IV–VIII, Valex's Magical 4-Arrow Adventure 1–7, FlightMix 1–4, Sudziosis II–5, Tachyon,
Touhou (Kousaikai, Gensouyou), Condor's Overly Cool & Kawaii Simfiles, The Nu-Virus Files,
StreamVoltex, Anime Extravaganza, Vocaloid Project Pad Pack…

**Plus dense que son meter** et plus rapide, **peu de sauts**, **presque pas de crossovers** (3 fois
moins que le classique), davantage de holds. Priorité au flux : streams en alternance des pieds,
peu de technique. C'est l'essentiel des packs ajoutés (lots 2 et 3), d'où l'écart avec les Songs du
jeu mesuré par `eval_perplexity`.

## Tech / crossovers

Crossover Spectrum I–III, KoL DDR Before-Max et Post-5th Edit Packs, Stomp Exceed, NEMORIGINAL
Collection 2–4 et Soulja Boy, R21, The NES Simfile Collab.

**Crossovers 4 fois plus fréquents** que la moyenne, plus de footswitches, de jacks et de brackets,
peu de holds. Jeu de jambes technique, souvent d'inspiration DDR.

## Homebrew « gimmick »

CMGs Firsts, Second Coming, The 3rd Chapter, r21twins, r2112, Gazebo de Brute, Monster Energy
Pack 3, Zimlord's Pre-staff Originals, JHF-Mix 1.

Un peu de tout : **beaucoup de holds et de rolls, 9 mines / 100 notes**, en médiane **5
changements de BPM ou stops** par chart, brackets, sauts, crossovers. Style chargé et à surprises
des homebrews de l'époque ITG2.

## Kyzentun (expérimental)

Kyzentun's Krap, Kyzentuns Handjobs : triolets, rolls, mines, crossovers, très peu de sauts. Un
style d'auteur isolé.

## Utilisation : `--style`

`itg-charter gen song.mp3 --style classic|stream|tech` (défaut : `classic`). Un modèle de flèches
par style, embarqué ; les deux petits groupes (gimmick, 590 charts ; Kyzentun, 76) n'en ont pas.
`training/styles.tsv` associe chaque pack à son style : déplacer un pack = modifier sa ligne, puis
`cargo run --release --example train_styles -- ~/itg-train` (dossier de liens vers tous les packs)
réécrit `model/model.json` (classique) et `model/styles/*.json`. Le placement reste commun.

Réglages de génération par style : `Style::gen_options` (température 0,7 ; parité 0,01, tech
0,0025), voir `CLAUDE.md`.

Contrôle initial (`eval_charts`, 40 morceaux du style) : classique et stream 0 statistique hors de
l'intervalle p10–p90 humain. Tech : au poids de parité par défaut (0,02) le générateur ne fait
presque pas de crossovers (0,3 / 100 lignes en Medium, humain 4,7) ; poids balayé
(`eval/charts-style-tech-pw*.txt`) : 0,01 → 1,9 ; **0,005 → 4,2 (retenu, 0 hors intervalle)** ;
0 → 12,6 (au-dessus du p90).

## Conséquences pour le modèle

- `eval_perplexity` : sur le split test des Songs du jeu, le modèle entraîné sur ces seules Songs
  (1,729 bit/ligne) bat celui entraîné sur tout `~/itg-train` (1,766) malgré 5 fois moins de charts.
  Le corpus élargi est dominé par le style stream et tire le modèle hors du style classique.
- Pistes : entraîner sur le seul style classique (9 246 charts, ×2,3 par rapport aux Songs du jeu),
  ou conditionner le modèle par le style pour le choisir à la génération.
