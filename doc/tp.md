# Écrire le module d'un TP, distribuer, déployer

## La règle

Le paquet `tpllg` ne porte que du générique. Ce dont un TP a besoin en plus
— une simulation de son montage pour essayer les scripts sans paillasse, une
classe qui découpe le signal comme ce TP le demande, les indications qu'on
veut imprimer aux élèves — va dans un module `tpNN.py`, où `NN` est le
numéro du TP, déposé **dans la copie de `tpllg/` distribuée avec les scripts
de ce TP** :

```text
TP3_eleves/
├── README.txt
├── TP3_PasseBande.py          le script élève, à compléter
├── TP3_RegimeLibre.py         idem
└── tpllg/
    ├── __init__.py
    ├── sysam.py, acquisition.py, ajustement.py, …   le paquet, tel quel
    └── tp3.py                 ce qui est propre au TP3
```

Les scripts l'importent comme le reste, `from tpllg.tp3 import …`. Le
module n'est pas publié dans ce dépôt, et le déploiement (voir plus bas) le
laisse en place quand il met le paquet à jour.

Trois raisons à cette règle. Le paquet reste lisible, et un collègue qui
l'installe n'hérite pas de nos TP. Un TP peut évoluer sans toucher au paquet,
et un paquet mis à jour ne casse pas un TP figé. Et ce qui est générique se
voit : si une fonction de `tpNN.py` servirait à un autre TP, c'est qu'elle
a sa place dans le paquet.

## Ce qu'un tpNN.py contient, sur l'exemple du TP3

Le TP3 étudie un filtre passe-bande actif : relevé du diagramme de Bode et
ajustement simultané du gain et de la phase, puis acquisition à la centrale
de la réponse à un échelon et ajustement du régime libre. Son `tp3.py` porte :

| Fonction | Rôle |
| --- | --- |
| `simuler_bode(f, …)` | des mesures bidon du diagramme de Bode aux fréquences `f`, bruitées, pour essayer le script sans montage |
| `simuler_echelon(te, N, calibre, …)` | une acquisition bidon des deux voies, construite comme la centrale la rendrait : créneau de phase quelconque, sonnerie après chaque front, bruit, offset, quantification 12 bits |
| `FenetreRegimeLibre(t, ve, vs, fe)` | la fenêtre du régime libre après le premier front utile, avec `indications()` — l'instant du front, la pseudo-fréquence, l'offset, le premier extremum — que le script imprime pour aider à choisir les valeurs de départ |
| `rapport_maximums(fen, v_off)` | la mesure « à la main » de l'énoncé, deux maximums successifs dans le rapport $e^{-\pi/Q}$ |
| `comparer_au_bode(f0, Q, f0_bode, Q_bode)` | une ligne d'écart relatif entre les deux méthodes |
| `tracer_regime_libre(t, ve, vs, fen, modele, pfit, pcov)` | trois graphes : acquisition brute, régime libre ajusté avec son enveloppe, résidus |

Tout ce que ces fonctions emploient de générique vient du paquet :
`fronts_montants`, `front_utile`, `fenetre`, `frequence_pic`, `extremums`,
`decrement_logarithmique` de `tpllg.signaux`, `formater` et `resume_parametres`
de `tpllg.ajustement`. Le module ne fait que les composer pour ce TP.

Le script élève qui en résulte tient en une page, et ne garde que ce qu'on
veut faire écrire aux élèves — ici les réglages d'acquisition, le modèle et
les valeurs de départ :

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from tpllg.acquisition import acquerir, sauvegarder
from tpllg.ajustement import resume_parametres
from tpllg.tp3 import FenetreRegimeLibre, simuler_echelon, tracer_regime_libre

SIMULATION = False           # True pour essayer sans centrale
CALIBRE = ...                # À COMPLÉTER
fe = ...                     # À COMPLÉTER
T = ...                      # À COMPLÉTER
te, N = 1/fe, int(fe*T)

if SIMULATION:
    temps, tensions = simuler_echelon(te, N, calibre=CALIBRE)
else:
    temps, tensions = acquerir([0, 1], CALIBRE, te, N, trigger=(0, 0.0, 50))
t, ve, vs = (np.asarray(x, dtype=float) for x in (temps[0], tensions[0], tensions[1]))

fen = FenetreRegimeLibre(t, ve, vs, fe)
print(fen.indications())

def regime_libre(t, A, f0, Q, t0, v_off):
    tau = t - t0
    fp = ...                 # À COMPLÉTER
    return ...               # À COMPLÉTER

PARAM_INIT = [..., ..., ..., ..., ...]   # À COMPLÉTER

pfit, pcov = curve_fit(regime_libre, fen.t, fen.v, p0=PARAM_INIT)
print(resume_parametres(("A", "f0", "Q", "t0", "v_off"), pfit, pcov,
                        unites=("V", "Hz", "", "s", "V")))
tracer_regime_libre(t, ve, vs, fen, regime_libre, pfit, pcov, fichier="TP3_regimelibre.pdf")
plt.show()
```

Le professeur garde à côté une version complète du même script, sur le même
`tpllg`, et un script qui produit les figures du corrigé — tous les deux
hors du dossier distribué.

## Écrire pour les postes du lycée

Les scripts élèves tournent sous Spyder, avec le Python 3.7 d'une Anaconda
qui n'est pas mise à jour, et parfois sans réseau. D'où quelques règles pour
un `tpNN.py` comme pour le paquet :

- pas d'opérateur `:=`, pas de `f"{x=}"`, pas d'annotations `list[float]` ;
  les `dataclass` existent mais sans `slots=True` ;
- `np.random.RandomState(graine)` plutôt que `default_rng`, absent des vieux
  numpy ;
- `scipy.signal.windows.blackman` et non `scipy.signal.blackman`, qui a
  disparu des scipy récents — on écrit pour les deux ;
- une simulation dans le module, pour que le script s'essaie chez soi et se
  teste avant la séance, avec un interrupteur `SIMULATION` en tête du script ;
- des messages d'erreur qui disent quoi faire — « le GBF est-il branché ? » —
  plutôt qu'une trace Python.

## Déployer

Chaque dossier distribué embarque sa copie de `tpllg/`. Mettre le paquet à
jour, c'est recopier les modules du dépôt dans chaque copie, en gardant les
`tpNN.py` qui s'y trouvent et en retirant les modules que le dépôt n'a plus.
Sur la machine du professeur, un script fait cela pour tous les dossiers de
TP d'un coup — au lycée Louis-le-Grand, `TPLLG_DEPLOYER.command` dans le
dossier des TP :

```text
✓ TP2_eleves/tpllg : 12 modules
✓ TP3_eleves/tpllg : 13 modules (propres au TP : tp3.py)
✓ TP10/tpllg : 12 modules
```

La logique tient en quelques lignes, à adapter à son arborescence :

```bash
MAITRE="chemin/du/clone/tpllg"          # le dossier tpllg/ du dépôt
for d in TP*/tpllg; do
    for f in "$d"/*.py; do                       # les modules disparus du dépôt
        b=$(basename "$f")
        [[ "$b" == tp[0-9]*.py ]] && continue    # sauf ceux du TP
        [[ -e "$MAITRE/$b" ]] || rm "$f"
    done
    cp "$MAITRE"/*.py "$d"/                      # le paquet, tel quel
    rm -rf "$d/__pycache__"
done
```

Sur les postes du lycée, il n'y a rien à installer : les élèves reçoivent le
dossier du TP, scripts et `tpllg/` ensemble, et l'exécutent depuis ce dossier
(sous Spyder, « Exécuter dans le répertoire du fichier »). Quand pycanum est
absent d'un poste, la centrale simulée prend le relais et le dit ; c'est le
signe qu'il faut installer pycanum, pas que le script est cassé.
