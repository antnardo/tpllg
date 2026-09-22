# tpllg

Acquisition à la centrale **Sysam SP5** et exploitation des mesures en TP de
physique, en classes préparatoires. Le paquet réunit ce qui revient d'un TP à
l'autre : la centrale en une ligne, l'ajustement d'un modèle avec ses
incertitudes, le repérage d'un front dans un signal acquis, un diagramme de
Bode, un spectre, la lecture d'un fichier de mesures.

Il est écrit pour être copié tel quel à côté des scripts d'un TP, sur des
postes sans droits d'installation, et pour tourner sans centrale — un
simulateur prend le relais, en le disant.

## Sommaire

- [Installation](#installation)
- [Prise en main en cinq minutes](#prise-en-main-en-cinq-minutes)
- [Les modules](#les-modules)
- [Cas d'usage, de bout en bout](#cas-dusage-de-bout-en-bout)
- [Un module par TP, hors du paquet](#un-module-par-tp-hors-du-paquet)
- [Déployer au lycée](#déployer-au-lycée)
- [Développer, tester](#développer-tester)

La documentation détaillée est dans `doc/` :

| Fiche | Contenu |
| --- | --- |
| [doc/centrale.md](doc/centrale.md) | la Sysam SP5 : voies, calibres, échantillonnage, déclenchement, sorties, acquisition sans centrale |
| [doc/ajustements.md](doc/ajustements.md) | `curvefit` avec incertitudes, `curve_fit_complex`, incertitudes-types, Student, Monte-Carlo |
| [doc/signaux.md](doc/signaux.md) | fronts et régime libre, diagramme de Bode, spectres, gain d'un filtre, Bode automatique |
| [doc/fichiers.md](doc/fichiers.md) | lire un export Latis Pro, Regressi ou un CSV quelconque |
| [doc/tp.md](doc/tp.md) | écrire le module d'un TP, distribuer et déployer, avec un TP complet en exemple |

## Installation

Trois façons, selon le poste.

**Copier le dossier**, au lycée ou chez soi, sans pip ni réseau : le dossier
`tpllg/` de ce dépôt (celui qui contient `sysam.py`) se pose à côté des
scripts du TP, et `from tpllg.sysam import Sysam` fonctionne. C'est ce que
font les dossiers distribués aux élèves.

**Avec pip**, quand le réseau et pip sont disponibles :

```bash
pip install git+https://github.com/antnardo/tpllg
```

**Pour développer**, une installation éditable depuis un clone :

```bash
git clone https://github.com/antnardo/tpllg
pip install -e tpllg
```

Python 3.7 ou plus, avec numpy, scipy et matplotlib : la distribution
Anaconda suffit. Le code évite délibérément ce que Python 3.7 ne connaît pas.

La centrale demande **pycanum**, le module Python de Frédéric Legrand pour la
Sysam SP5, disponible sous Windows :
[interpy](https://www.f-legrand.fr/scidoc/docmml/sciphys/caneurosmart/interpy/interpy.html).
Sans pycanum, `tpllg.sysam` se rabat sur `tpllg.sysam_factice`, qui a la même
interface, et l'annonce à l'ouverture :

```text
[SYSAM] ATTENTION : pycanum absent, centrale simulée (tpllg.sysam_factice)
```

Les scripts tournent alors jusqu'au bout ; les acquisitions ne rendent que du
bruit de quantification.

## Prise en main en cinq minutes

Acquérir deux voies, les enregistrer, les tracer :

```python
import matplotlib.pyplot as plt
import numpy as np
from tpllg.acquisition import acquerir, sauvegarder

fe = 200_000.0                       # Hz
temps, tensions = acquerir([0, 1], calibre=5, te=1/fe, nbpoints=6000)
sauvegarder("essai", [0, 1], temps, tensions)   # essai_EA0.txt, essai_EA1.txt

plt.plot(temps[0], tensions[0], label="EA0")
plt.plot(temps[1], tensions[1], label="EA1")
plt.legend()
plt.show()
```

Ajuster un modèle sur des mesures, avec les incertitudes-types :

```python
from tpllg.ajustement import curvefit, resume_parametres

def modele(x, a, b):
    return a*x + b

pfit, err, chi2 = curvefit(modele, x, y, p0=[1, 0], datayerrors=sigma_y)
print(resume_parametres(("a", "b"), pfit, sigmas=err))
```

```text
a = 1.879 ± 0.078
b = -0.762 ± 0.095
```

Ajuster une fonction de transfert sur un diagramme de Bode mesuré, gain et
phase ensemble, et la tracer :

```python
from tpllg.ajustement import curve_fit_complex, resume_parametres
from tpllg.bode import tracer_bode

def gain(f, H0, f0, Q):
    return H0/(1 + 1j*Q*(f/f0 - f0/f))

pfit, pcov = curve_fit_complex(gain, f, norm=H, phase=phi, p0=[-5, 2000, 6])
print(resume_parametres(("H0", "f0", "Q"), pfit, pcov, unites=("", "Hz", "")))
tracer_bode(f, H, phi, gain, pfit, pcov, noms=("$H_0$", "$f_0$", "$Q$"),
            unites=("", "Hz", ""), fichier="bode.pdf")
```

## Les modules

| Module | Contenu |
| --- | --- |
| `tpllg.sysam` | `Sysam`, la classe de pycanum avec un `with`, des calibres simples et des acquisitions qui rendent directement temps et tensions ; se rabat sur le simulateur sans pycanum |
| `tpllg.sysam_factice` | le simulateur : même interface, sans matériel |
| `tpllg.acquisition` | `acquerir(voies, calibre, te, nbpoints, trigger=…)` en une ligne, déclenchement compris ; `sauvegarder` |
| `tpllg.ajustement` | `curvefit` (incertitudes sur y, ou sur x et y par la variance effective, χ² réduit), `curve_fit_complex` (module et phase ajustés ensemble), `ecarts_types`, `formater`, `resume_parametres`, `residus_complexes` |
| `tpllg.incertitudes` | `incertitudes` (moyenne, incertitude-type de la moyenne, écart-type estimé), `student_coef`, `loi_normale`, `loi_normale_cumulee` |
| `tpllg.montecarlo` | `Point` et `SerieLineaire`, la propagation des incertitudes par tirages |
| `tpllg.signaux` | `fronts_montants`, `fronts_descendants`, `front_utile`, `fenetre`, `frequence_pic`, `extremums`, `decrement_logarithmique` |
| `tpllg.bode` | `tracer_bode`, `phase_0_360` |
| `tpllg.fft` | `calcule_DFT`, `spectre` |
| `tpllg.traitement` | `gain`, `gain_std`, `choix_echantillonnage`, `interpolation_fft`, `indices_plages`, `detecte_maxima_secondaires`, `valeurs_correspondantes` |
| `tpllg.fichiers` | `import_latispro`, `import_regressi`, `readcsv` |

Les scripts d'exemple sont dans `exemples/`. Les notices de la centrale sont
celles d'Eurosmart, à demander au fabricant ou à retrouver sur les postes qui
l'ont installée ; la documentation de pycanum est sur le site de son auteur.

## Cas d'usage, de bout en bout

Chacun est détaillé dans une fiche de `doc/`, avec le code complet.

| Cas | Fiche |
| --- | --- |
| Acquérir un signal, l'enregistrer, le tracer | [centrale](doc/centrale.md#une-acquisition-simple) |
| Acquérir en déclenchant sur un front d'une voie | [centrale](doc/centrale.md#déclencher-sur-une-voie) |
| Générer un signal sur la sortie et acquérir en même temps | [centrale](doc/centrale.md#générer-et-acquérir-en-même-temps) |
| Faire tourner un script sans centrale | [centrale](doc/centrale.md#sans-centrale) |
| Ajuster une droite, avec ou sans incertitudes, lire le χ² réduit | [ajustements](doc/ajustements.md#curvefit-avec-les-incertitudes) |
| Ajuster une fonction de transfert sur gain et phase mesurés | [ajustements](doc/ajustements.md#curve_fit_complex-module-et-phase-ensemble) |
| Estimer une valeur et son incertitude sur une série de mesures | [ajustements](doc/ajustements.md#une-série-de-mesures) |
| Propager des incertitudes par Monte-Carlo | [ajustements](doc/ajustements.md#monte-carlo) |
| Repérer un front et ajuster le régime libre qui le suit | [signaux](doc/signaux.md#le-régime-libre-après-un-front) |
| Tracer un diagramme de Bode avec son ajustement | [signaux](doc/signaux.md#tracer-un-diagramme-de-bode) |
| Calculer le spectre d'un signal acquis | [signaux](doc/signaux.md#le-spectre-dun-signal) |
| Relever un diagramme de Bode automatiquement avec la centrale | [signaux](doc/signaux.md#le-bode-automatique) |
| Lire un fichier Latis Pro, Regressi ou un CSV | [fichiers](doc/fichiers.md) |
| Écrire le module d'un TP et le distribuer | [tp](doc/tp.md) |

## Un module par TP, hors du paquet

Le paquet ne porte rien qui soit propre à un TP. Ce dont un TP a besoin en
plus — une simulation de son montage, une fenêtre d'analyse particulière, des
indications imprimées pour les élèves — va dans un module `tpNN.py` déposé
dans la copie de `tpllg/` distribuée avec les scripts de ce TP, et importé
par `from tpllg.tpNN import …`. Il n'est pas publié ici, et le déploiement
le laisse en place. La fiche [doc/tp.md](doc/tp.md) en donne un exemple
complet.

## Déployer au lycée

Sur les postes du lycée, on ne suppose ni pip ni réseau : chaque dossier de
TP distribué aux élèves embarque **sa copie** du dossier `tpllg/`. Mettre à
jour, c'est recopier les modules du dépôt dans chaque copie, en gardant les
`tpNN.py` qui s'y trouvent ; un script fait cela en une fois, voir
[doc/tp.md](doc/tp.md#déployer).

## Développer, tester

```bash
pip install -e .
python -m pytest tests
```

Les tests n'ont pas besoin de la centrale : ils passent par le simulateur.
Une fonction générique va dans le module qui lui correspond ; ce qui est
propre à un TP reste dehors. Le code doit tourner sous Python 3.7, celui des
postes du lycée : pas d'opérateur `:=`, pas de `f"{x=}"`, pas de
`list[float]` dans les signatures, `np.random.RandomState` plutôt que
`default_rng`.

## Licence

Le code est sous [licence MIT](LICENSE). Auteur : Antonin Marchand, lycée
Louis-le-Grand.

Deux choses n'en relèvent pas. `pycanum` est une dépendance, pas une partie
du dépôt : il appartient à Frédéric Legrand, est publié sous licence CeCILL
sur PyPI, et n'est jamais redistribué ici. Et `exemples/Acquisition.py` et
`exemples/Bode.py` dérivent de ses exemples, publiés sur son site sous
CC BY-NC-SA 2.0 FR : ces deux fichiers-là restent sous cette licence, ce que
leur en-tête rappelle.
