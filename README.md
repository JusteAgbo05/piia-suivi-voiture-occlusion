# Suivi d'une voiture à travers une occlusion 

##  Niche

**Détecter et suivre une voiture avant et après une occlusion**, principalement causée par un poids lourd. Une voiture cachée temporairement derrière un poids lourd ne doit pas changer d'identifiant (ID) à sa réapparition.

##  Groupe 4

| Membre | GitHub |
|---|---|
| Juste AGBO (responsable) | `@JusteAgbo05` |
| EGUE Richnel | `@EgueRichnel2025` |
| Zozerigué TRAORE | `@TRAORE352` |

## Pipeline - briques du projet

| Brique | Objectif | Responsable |
|---|---|---|
| 0. Mise en place | Repo propre, dépendances, test de fumée | JusteAgbo05 |
| 1. Mesurer le problème | Compter les ID Switches avec ByteTrack seul (baseline) | JusteAgbo05, EgueRichnel2025, TRAORE352 |
| 2. Détection (si besoin) | Vérifier/fine-tuner la détection sur véhicules partiellement cachés | À déterminer après la brique 1 |
| 3. Module ReID | Dataset + entraînement d'un module de ré-identification | EgueRichnel2025 |
| 4. Logique anti-occlusion | Garder l'empreinte d'une voiture disparue, la comparer à sa réapparition | EgueRichnel2025 |
| 5. Évaluation | Comparer ByteTrack seul / +ReID classique / notre version | JusteAgbo05 |
| 6. Démo + rapport | Vidéo de démo + rapport final | TRAORE352 |

##  Outils utilisés

- [Ultralytics](https://docs.ultralytics.com) — détection (YOLOv8) et suivi natif (ByteTrack / BoT-SORT)
- [Supervision](https://github.com/roboflow/supervision) — annotation et gestion vidéo


```bash
pip install -r requirements.txt
```

##  Structure du repo

```
piia-suivi-voiture-occlusion/
├── data/videos/        (non versionné - vidéos en local)
├── notebooks/
├── src/
│   └── test.py   ← test de démarrage (brique 0)
├── rapport/figures/
└── runs/
    ├── detect/
        └──results
           └── videos_annotees/
               └── test
```

## Test de démarrage (brique 0)

```bash
python src/test.py
```
Si une vidéo annotée apparaît dans `results/detect/results/videos_annotees/test/`, l'environnement est prêt.

## Indicateur d'évaluation

Nombre d'**ID Switches** observés sur une voiture donnée à travers une occlusion, comparé entre les 3 versions du système (ByteTrack seul / +ReID classique / notre version).

## Autheurs

- @JusteAgbo05
- @EgueRichnel2025
- @TRAORE352