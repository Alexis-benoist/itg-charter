# itg-charter

Génère des charts **ITGmania / In The Groove** (`.sm`, dance-single) à partir d'un fichier audio
(mp3, ogg, flac, wav), pour les difficultés choisies, de façon reproductible avec une seed.

```sh
cargo build --release
./target/release/itg-charter gen chanson.mp3 -d easy,medium,hard -s 42 -o ~/.itgmania/Songs/itg-charter
```

Le dossier `~/.itgmania/Songs/itg-charter/<Titre>/` contient le `.sm` et une copie de l'audio ;
relancer ITGmania pour le voir apparaître.

## Comment ça marche

1. **Séparation des sources** avec [Demucs](https://github.com/facebookresearch/demucs)
   (batterie, basse, voix, autres ; optionnel, `--no-stems`).
2. **Analyse** avec [aubio](https://aubio.org) : onsets et tempo, puis ajustement d'un BPM constant
   et de l'offset sur tout le morceau.
3. **Placement** des notes sur la grille (noires → doubles-croches) avec les densités, sauts et
   holds appris sur de vrais charts.
4. **Choix des flèches** : recherche en faisceau qui combine un modèle de patterns appris sur les
   charts humains du jeu et le **calcul de parité d'ITGmania** (confort des pieds : crossovers,
   footswitches, double-steps…), avec un bruit contrôlé par la seed.

Options utiles : `--bpm`, `--offset` (forcer la synchro), `--title`, `--artist`, `--device cpu`,
`--model` (autre modèle, voir `itg-charter train`).

## Précision mesurée

Sur la bibliothèque ITGmania (charts synchronisés par des humains) : BPM exact dans 83 % des cas
(98,6 % à l'octave près), 85 % des morceaux calés à moins de 20 ms. Les charts générés ont des
statistiques (densité, sauts, holds, crossovers, footswitches, meter) dans la distribution des
charts humains de chaque difficulté.

## Licence

GPL-3.0 (aubio et le code de parité porté depuis ITGmania sont sous GPL-3.0).
