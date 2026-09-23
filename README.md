# tpllg

Acquisition à la centrale **Sysam SP5** et exploitation des mesures en TP de
physique, en classes préparatoires. Le paquet réunit ce qui revient d'un TP à
l'autre : la centrale en une ligne, l'ajustement d'un modèle avec ses
incertitudes, le repérage d'un front dans un signal acquis, un diagramme de
Bode, un spectre, la lecture d'un fichier de mesures, la propagation des
incertitudes.

Il est écrit pour être copié tel quel à côté des scripts d'un TP, sur des
postes sans droits d'installation, et pour tourner sans centrale : un
simulateur prend le relais, en le disant.

## Sommaire

- [Installation](#installation)
- [Prise en main](#prise-en-main)
- [La documentation](#la-documentation)
- [Les modules](#les-modules)
- [Les exemples](#les-exemples)
- [Ce qu'il faut savoir avant d'écrire un script](#ce-quil-faut-savoir-avant-décrire-un-script)
- [Développer, tester](#développer-tester)
- [Licence](#licence)

## Installation

Trois façons, selon le poste.

**Copier le dossier**, au lycée ou chez soi, sans pip ni réseau : le dossier
`tpllg/` de ce dépôt (celui qui contient `sysam.py`) se pose à côté des
scripts, et `from tpllg.sysam import Sysam` fonctionne, à condition
d'exécuter le script **depuis son dossier** (sous Spyder, « Exécuter dans le
répertoire du fichier »). C'est ce que font les dossiers distribués aux
élèves : les scripts et `tpllg/` ensemble, rien à installer.

**Avec pip**, quand le réseau et pip sont disponibles :

```bash
pip install git+https://github.com/antnardo/tpllg
```

**Pour développer**, une installation éditable depuis un clone :

```bash
git clone https://github.com/antnardo/tpllg
pip install -e tpllg
```

Python 3.7 ou plus, avec numpy, scipy et matplotlib : une distribution
Anaconda suffit. Le code évite délibérément ce que Python 3.7 ne connaît pas.

La centrale demande **pycanum**, le module Python de Frédéric Legrand pour la
Sysam SP5, disponible sous Windows :
[interpy](https://www.f-legrand.fr/scidoc/docmml/sciphys/caneurosmart/interpy/interpy.html).
Sans pycanum, `tpllg.sysam` se rabat sur `tpllg.sysam_factice`, qui a la même
interface et rend des données de la même forme, et l'annonce à l'ouverture :

```text
[SYSAM] ATTENTION : pycanum absent, centrale simulée (tpllg.sysam_factice)
```

Les scripts tournent alors jusqu'au bout ; les acquisitions ne rendent que du
bruit de quantification. Sur un poste qui a la centrale, ce message signifie
que pycanum n'est pas installé pour ce Python-là, pas que le script est cassé.

## Prise en main

Acquérir deux voies, les enregistrer, les tracer :

```python
import matplotlib.pyplot as plt
from tpllg.acquisition import acquerir, sauvegarder

fe = 200000.0                                                      # Hz
temps, tensions = acquerir([0, 1], calibre=5, te=1/fe, nbpoints=6000)
sauvegarder("essai", [0, 1], temps, tensions)   # essai_EA0.txt, essai_EA1.txt

plt.plot(temps[0], tensions[0], label="EA0")
plt.plot(temps[1], tensions[1], label="EA1")
plt.legend()
plt.show()
```

`temps` et `tensions` sont deux tableaux de forme `(2, 6000)` : une ligne par
voie, le temps en secondes, la tension en volts.

Ajuster un modèle sur des mesures, avec des incertitudes-types :

```python
import numpy as np
from tpllg.ajustement import curvefit, resume_parametres

def modele(x, a, b):
    return a*x + b

x = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9])
y = np.array([-0.85, -0.42, 0.11, 0.35, 0.84, 1.15, 1.70, 1.95, 2.36, 2.85])
pfit, err, chi2 = curvefit(modele, x, y, p0=[1, 0], datayerrors=0.15*np.ones(x.size))
print(resume_parametres(("a", "b"), pfit, sigmas=err))
```

```text
Least square method
a = 2.010 ± 0.083
b = -1.006 ± 0.095
```

Ajuster une fonction de transfert sur un diagramme de Bode mesuré, gain et
phase ensemble, et tracer :

```python
import numpy as np
from tpllg.ajustement import curve_fit_complex, resume_parametres
from tpllg.bode import tracer_bode

def passe_bande(f, H0, f0, Q):
    return H0/(1 + 1j*Q*(f/f0 - f0/f))

f = np.array([500, 1000, 1500, 1800, 2000, 2200, 2700, 5000.])       # Hz
H = np.array([0.21, 0.53, 1.33, 3.06, 5.13, 3.16, 1.24, 0.39])      # |Vs/Ve|
phi = np.radians([-95, -97, -106, -122, 176, 126, 104, 96])         # radians

pfit, pcov = curve_fit_complex(passe_bande, f, norm=H, phase=phi, p0=[-5, 2000, 6])
print(resume_parametres(("H0", "f0", "Q"), pfit, pcov, unites=("", "Hz", "")))
tracer_bode(f, H, phi, passe_bande, pfit, pcov, noms=("$H_0$", "$f_0$", "$Q$"),
            unites=("", "Hz", ""), fichier="bode.pdf")
```

```text
H0 = -5.14 ± 0.10
f0 = 1988.4 ± 3.0 Hz
Q = 6.65 ± 0.28
```

## La documentation

Une fiche par famille de fonctions, dans `doc/`. Chaque fonction y est
donnée avec sa signature, le sens de chaque argument, ce qu'elle rend, ce
qui la fait échouer, et un exemple exécuté avec ce qu'il imprime. Les fiches
se terminent par des cas complets, de bout en bout.

| Fiche | Contenu |
| --- | --- |
| [doc/centrale.md](doc/centrale.md) | la Sysam SP5 : voies, calibres, modes de conversion, échantillonnage, acquisition, déclenchement sur une voie, sorties analogiques, enregistrement, le simulateur, les pannes courantes |
| [doc/ajustement.md](doc/ajustement.md) | ce que fait `curve_fit` ; `curvefit` avec les incertitudes sur y, sur x et y, le χ² réduit ; `curve_fit_complex` pour un module et une phase ; présenter un résultat ; choisir les valeurs de départ ; lire les résidus |
| [doc/incertitudes.md](doc/incertitudes.md) | une série de mesures, l'écart-type, l'incertitude de la moyenne, le coefficient de Student ; la propagation par Monte-Carlo : `Point`, la droite ajustée sur tous les tirages sans boucle, un modèle quelconque ajusté par tirages |
| [doc/signaux.md](doc/signaux.md) | repérer les fronts d'un créneau, découper une fenêtre, estimer une fréquence, les extremums et le décrément d'une oscillation amortie ; le régime libre d'un filtre après un front, ajusté |
| [doc/bode.md](doc/bode.md) | tracer un diagramme de Bode, mesures et modèle ; la chaîne complète mesures, ajustement, figure ; le gain d'un filtre mesuré sans ajustement ; le Bode automatique par la centrale |
| [doc/spectres.md](doc/spectres.md) | le spectre d'un signal, brut ou fenêtré ; relever les harmoniques ; mesurer une fonction de transfert sur les harmoniques d'un créneau |
| [doc/fichiers.md](doc/fichiers.md) | lire un CSV de tableur, un export Latis Pro ou Regressi ; relire ce que `sauvegarder` écrit |

## Les modules

| Module | Contenu |
| --- | --- |
| `tpllg.sysam` | `Sysam`, la classe de pycanum avec un `with`, des calibres simples et des acquisitions qui rendent directement temps et tensions ; se rabat sur le simulateur sans pycanum |
| `tpllg.sysam_factice` | le simulateur : même interface, mêmes formes de données, sans matériel |
| `tpllg.acquisition` | `acquerir(voies, calibre, te, nbpoints, trigger=…)` en une ligne, déclenchement compris ; `sauvegarder` |
| `tpllg.ajustement` | `curvefit` (incertitudes sur y, ou sur x et y par la variance effective, χ² réduit), `curve_fit_complex` (module et phase ajustés ensemble), `ecarts_types`, `formater`, `resume_parametres`, `residus_complexes` |
| `tpllg.incertitudes` | `incertitudes` (moyenne, incertitude-type de la moyenne, écart-type estimé), `student_coef`, `loi_normale`, `loi_normale_cumulee` |
| `tpllg.montecarlo` | `Point`, `SerieLineaire`, `ajuster_modele` : la propagation des incertitudes par tirages, la droite sans boucle |
| `tpllg.signaux` | `fronts_montants`, `fronts_descendants`, `front_utile`, `fenetre`, `frequence_pic`, `extremums`, `decrement_logarithmique` |
| `tpllg.bode` | `tracer_bode`, `phase_0_360` |
| `tpllg.fft` | `calcule_DFT`, `spectre` |
| `tpllg.traitement` | `gain_std`, `gain`, `choix_echantillonnage`, `interpolation_fft`, `indices_plages`, `detecte_maxima_secondaires`, `valeurs_correspondantes` |
| `tpllg.fichiers` | `readcsv`, `import_latispro`, `import_regressi` |

## Les exemples

Les scripts d'`exemples/` sont complets et tournent sans centrale : quand une
acquisition est nécessaire, un interrupteur `SIMULATION` en tête fabrique des
données comme la centrale les rendrait, et `False` acquiert pour de bon. Ils
s'exécutent depuis n'importe quel dossier, y écrivent leurs fichiers et leurs
figures, et sont les mêmes que ceux des fiches.

| Script | Ce qu'il fait |
| --- | --- |
| `acquisition_simple.py` | acquiert deux voies, enregistre un fichier par voie, trace |
| `regime_libre.py` | un créneau attaque un filtre ; repère un front, découpe le régime libre qui le suit, l'ajuste, trace acquisition, ajustement et résidus |
| `bode_ajustement.py` | des mesures point par point d'un diagramme de Bode, l'ajustement simultané du gain et de la phase, les résidus, la figure |
| `spectre_harmoniques.py` | le spectre d'un créneau, ses harmoniques, le gain d'un filtre mesuré sur chaque harmonique et ajusté |
| `lecture_fichiers.py` | écrit puis relit un CSV de tableur, un export Latis Pro, un export Regressi, un fichier de `sauvegarder` |
| `ajustement_incertitudes.py` | une droite ajustée sans incertitudes, avec celles de y, avec celles de x et y, à bruit constant puis variable |
| `montecarlo.py` | g par un pendule, un quotient à loi dissymétrique, une droite ajustée sur cent mille tirages face à `curvefit`, une exponentielle ajustée par tirages |
| `Acquisition.py`, `Bode.py` | deux scripts dérivés des exemples de Frédéric Legrand, sous leur licence (voir plus bas) : l'acquisition avec spectre, le Bode automatique |

## Ce qu'il faut savoir avant d'écrire un script

**Les données sont des tableaux 2D.** `acquerir`, comme `can.acquerir()`,
rend `temps` et `tensions` de forme `(nombre de voies, nombre de points)`,
en `float64` : `temps[0]` et `tensions[0]` sont la première voie
configurée, quel que soit son numéro. Le temps est en secondes et commence à
zéro.

**Les temps sont en secondes** partout dans `tpllg` : la période
d'échantillonnage donnée à `config_echantillon` ou à `acquerir`, les fronts
rendus par `fronts_montants`, les fenêtres. Les méthodes héritées de pycanum
que `tpllg` ne réécrit pas (`config_sortie`, `config_echantillon_permanent`)
prennent, elles, des **microsecondes**.

**Les phases sont en radians** pour l'ajustement (`curve_fit_complex`,
`residus_complexes`) et en degrés dans les résultats lisibles
(`residus_complexes` les rend en degrés, `tracer_bode` les affiche en
degrés). Une phase mesurée en degrés se convertit par `np.radians` avant
d'ajuster ; oubliée, l'ajustement ne converge pas ou rend n'importe quoi.

**Les valeurs de départ se lisent sur un tracé** avant tout ajustement. Une
descente locale part de là et n'en va pas loin : la fiche des ajustements
montre, sur un cas réel, ce qu'un `p0` faux d'une décade produit.

**Un script pour la centrale tourne sans elle.** Le simulateur répond avec
des tableaux de la bonne forme, remplis de bruit ; un script qui veut
s'essayer avec des données plausibles les fabrique lui-même derrière un
interrupteur `SIMULATION`, comme le font les exemples.

**Python 3.7.** Les postes de TP ont souvent une Anaconda ancienne. Pas
d'opérateur `:=`, pas de `f"{x=}"`, pas d'annotation `list[float]` ;
`np.random.RandomState(graine)` plutôt que `default_rng`.

## Développer, tester

```bash
pip install -e .
python -m pytest tests
```

Les tests n'ont pas besoin de la centrale : ils passent par le simulateur.
Une fonction générique va dans le module qui lui correspond ; ce qui est
propre à un TP donné, une simulation de son montage, une aide à l'énoncé,
reste dans les scripts de ce TP ou dans un module posé à côté, pas dans le
paquet.

## Licence

Le code est sous [licence MIT](LICENSE). Auteur : Antonin Marchand, lycée
Louis-le-Grand.

Deux choses n'en relèvent pas. `pycanum` est une dépendance, pas une partie
du dépôt : il appartient à Frédéric Legrand, est publié sous licence CeCILL
sur PyPI, et n'est jamais redistribué ici. Et `exemples/Acquisition.py` et
`exemples/Bode.py` dérivent de ses exemples, publiés sur son site sous
CC BY-NC-SA 2.0 FR : ces deux fichiers-là restent sous cette licence, ce que
leur en-tête rappelle. La méthode de mesure du gain de `tpllg.traitement`
(`gain_std`) lui est due aussi, réécrite ici.
