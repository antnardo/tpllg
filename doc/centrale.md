# La centrale Sysam SP5

`tpllg.sysam.Sysam` est la classe `Sysam` de pycanum, avec trois commodités :
un `with` qui ouvre et ferme la centrale, des calibres qu'on donne une fois
pour toutes les voies, et des acquisitions qui rendent directement temps et
tensions. Tout ce que pycanum sait faire reste accessible, la classe en
hérite.

## Ouvrir la centrale

```python
from tpllg.sysam import Sysam

with Sysam([0, 1], 5) as can:        # EA0 et EA1, calibre 5 V sur les deux
    ...                              # can est ouvert ici
                                     # et fermé ici, même en cas d'erreur
```

`Sysam(voies=None, calibres=None, diff=None)` :

| Argument | Sens |
| --- | --- |
| `voies` | les entrées analogiques à acquérir, numérotées de 0 à 7 |
| `calibres` | la tension maximale en valeur absolue, en volts, une pour toutes les voies ou une liste ; 10 V par défaut. Seuls 0,2, 1, 5 et 10 V existent : une autre valeur prend le calibre immédiatement supérieur |
| `diff` | la liste des voies en mode différentiel (EA0 avec EA4, EA1 avec EA5, …) |

Sans argument, `Sysam()` ouvre la centrale sans rien configurer ; on appelle
ensuite `config_entrees(voies, calibres, diff)`.

Deux modes de conversion, choisis par la centrale d'après les voies : en mode
**direct**, jusqu'à 10 MHz, il faut qu'un seul module sur deux soit actif —
EA0 ou EA4, EA1 ou EA5, etc. ; dès que les deux entrées d'un même module sont
actives en mode simple, on passe en mode **multiplexé**, à 500 kHz au plus.
Les constantes de classe donnent les limites :

| Constante | Valeur | Sens |
| --- | --- | --- |
| `Sysam.TE_MIN_DIRECT` | 1e-7 s | période d'échantillonnage minimale, mode direct |
| `Sysam.TE_MIN_MULTIPLEX` | 2e-6 s | idem, mode multiplexé |
| `Sysam.TE_MIN_SORTIE` | 2e-7 s | idem quand une sortie est utilisée |
| `Sysam.N_MAX` | 262 144 | points au total, toutes voies confondues |
| `Sysam.CALIBRES` | 0,2, 1, 5, 10 | les calibres accessibles par pycanum |

Le convertisseur est sur 12 bits : la résolution vaut le calibre divisé par
2048, soit 2,4 mV au calibre 5 V. Un signal de 100 mV se mesure au calibre
0,2 V ou 1 V, pas 10 V.

## Échantillonner

```python
can.config_echantillon(te, N)        # te en secondes, N points par voie
```

La période d'échantillonnage `te` est en **secondes** — pycanum la prend en
microsecondes, le wrapper convertit. Le nombre de points `N` vaut pour chaque
voie, et le total ne doit pas dépasser `N_MAX` : 130 000 points pour deux
voies.

Le choix se fait toujours de la même façon : assez de points par période
pour dessiner le signal — une centaine si l'on ajuste point par point, dix
suffisent pour un spectre —, et une durée qui contient ce qu'on veut voir.
`tpllg.traitement.choix_echantillonnage` fait ce calcul pour une fréquence
donnée, voir [signaux.md](signaux.md#le-bode-automatique).

## Une acquisition simple

```python
temps, tensions = can.acquerir()
```

`acquerir(reduction=1)` lance l'acquisition, attend qu'elle soit finie, et
rend deux listes : `temps[i]` et `tensions[i]` sont les tableaux de la
i-ième voie configurée, en secondes et en volts. `reduction` sous-échantillonne
au retour (1 point sur `reduction`).

Le script complet d'une acquisition, `exemples/Acquisition.py` :

```python
import matplotlib.pyplot as plt
import numpy as np
from tpllg.sysam import Sysam

FILE_PREFIX = "signaltest"
ENTREES = [0]
CALIBRE = 1
fe = 20000.0                 # Hz
te = 1/fe
T = 1.0                      # s
N = int(fe*T)

with Sysam(ENTREES, CALIBRE) as can:
    can.config_echantillon(te, N)
    t, u = can.acquerir()

for i, ea in enumerate(ENTREES):
    np.savetxt(f"{FILE_PREFIX}_{ea:02d}.txt", [t[i], u[i]])
    plt.figure()
    plt.plot(t[i], u[i])
    plt.xlabel("t (s)")
    plt.ylabel("u (V)")
    plt.grid()
    plt.savefig(f"{FILE_PREFIX}_{ea:02d}.pdf")
plt.show()
```

La même chose en une ligne, par `tpllg.acquisition` :

```python
from tpllg.acquisition import acquerir, sauvegarder

temps, tensions = acquerir(ENTREES, CALIBRE, te, N)
sauvegarder(FILE_PREFIX, ENTREES, temps, tensions)   # un .txt par voie
```

`sauvegarder` écrit `<prefixe>_EA<n>.txt` par voie, deux lignes, temps et
tensions : `np.loadtxt` les relit en `t, u = np.loadtxt("essai_EA0.txt")`.

## Déclencher sur une voie

L'acquisition démarre normalement à l'appel, donc n'importe où dans le
signal. La centrale sait attendre le passage d'une voie par un seuil :

```python
can.config_trigger(voie, seuil, montant=1, pretrigger=1,
                   pretriggerSouple=0, hysteresis=0)
```

| Argument | Sens |
| --- | --- |
| `voie` | l'entrée surveillée ; elle doit être parmi les voies configurées |
| `seuil` | la tension de seuil, en volts, dans le calibre de la voie |
| `montant` | 1 pour un front montant, 0 pour un front descendant |
| `pretrigger` | le nombre de points gardés avant le front (255 au plus) |
| `pretriggerSouple` | 1 pour déclencher même si le prétrig n'est pas rempli |
| `hysteresis` | 1 pour ajouter une hystérésis au seuil |

`config_trigger_externe(pretrigger, pretriggerSouple)` déclenche sur l'entrée
de synchronisation externe. `config_trigger(-1, 0)` désactive le déclenchement.

Par `tpllg.acquisition`, le déclenchement est un tuple :

```python
temps, tensions = acquerir([0, 1], 5, te, N, trigger=(0, 0.0, 50))
#                                                  voie, seuil, prétrig, [montant]
```

Le front se retrouve alors vers le cinquantième point de chaque voie. Un
script qui dépend de la position du front a intérêt à la vérifier dans les
données plutôt que de la supposer : `tpllg.signaux.fronts_montants` le fait,
voir [signaux.md](signaux.md#le-régime-libre-après-un-front).

## Générer et acquérir en même temps

La centrale a deux sorties analogiques, ±10 V, 5 MHz, 12 bits. On peut y
envoyer un signal échantillonné et acquérir les entrées de façon synchrone,
à la même période d'échantillonnage :

```python
import numpy as np
from tpllg.sysam import Sysam

N, Np = 20000, 100                        # points, points par période
e1 = 1.7*np.cos(2*np.pi*np.arange(N)/Np)  # une sinusoïde d'amplitude 1,7 V
with Sysam([0, 1], [2, 2]) as can:
    can.config_echantillon(2e-7*50, N)    # multiple de TE_MIN_SORTIE
    temps, tensions = can.acquerir_avec_sorties(e1, 0)   # sortie 1 : e1, sortie 2 : 0
```

`acquerir_avec_sorties(sortie1=0, sortie2=0)` prend un tableau par sortie, ou
un entier pour une valeur constante. Un câble relie la sortie S1 à l'entrée
EA0 pour lire ce qu'on envoie, et la sortie du montage étudié va sur EA1 :
c'est la base du Bode automatique, [signaux.md](signaux.md#le-bode-automatique).

Le nombre total de points, entrées et sorties, est limité par la mémoire de
la centrale, 262 142.

## Le reste de pycanum

La classe hérite de tout pycanum. Ce qui sert le plus, au-delà de ce qui
précède :

| Méthode | Rôle |
| --- | --- |
| `config_quantification(nbits)` | nombre de bits, 12 au plus |
| `temps(reduction)`, `entrees(reduction)` | relire la dernière acquisition |
| `config_echantillon_permanent(te_us, N)`, `acquerir_permanent()`, `paquet(premier)` | l'acquisition continue, par paquets, en liste circulaire |
| `lancer()`, `stopper_acquisition()` | lancer sans attendre, arrêter |
| `config_sortie(nsortie, techant_us, valeurs, repetition)`, `declencher_sorties`, `stopper_sorties` | les sorties seules, avec ou sans acquisition |
| `ecrire`, `lire`, `activer_lecture`, `portB_*`, `portC_*` | sorties et entrées instantanées, ports logiques |
| `config_compteur`, `compteur`, `config_chrono`, `chrono` | compteur et chronomètre |

Ces méthodes-là sont celles de pycanum, et prennent leurs temps en
**microsecondes**. Les exemples du site de Frédéric Legrand les emploient directement, sans
passer par `tpllg` ; ils ne sont pas reproduits ici, leur licence (CC BY-NC-SA)
n'étant pas celle du dépôt. La documentation
complète est celle de pycanum, [interpy](https://www.f-legrand.fr/scidoc/docmml/sciphys/caneurosmart/interpy/interpy.html),
et les registres de la centrale sont décrits dans la notice de programmation d'Eurosmart, le constructeur.

## Sans centrale

Quand pycanum n'est pas installé — sur un Mac, sur un poste sans carte, dans
les tests —, `tpllg.sysam` importe `tpllg.sysam_factice` à sa place, et
l'ouverture l'annonce :

```text
[SYSAM] ATTENTION : pycanum absent, centrale simulée (tpllg.sysam_factice)
[SYSAM] Configuration entrées [0, 1] : calibres=[5.0, 5.0], diff=[]
[SYSAM] Configuration échantillonnage techant=5.0e+00µs, nbpoints=6000
[SYSAM] Acquisition...
[SYSAM] Terminée.
```

Le simulateur a la même interface : les méthodes de configuration impriment
ce qu'elles reçoivent, `acquerir` attend la durée de l'acquisition, et
`temps` et `entrees` rendent des tableaux de la bonne forme, remplis d'un
bruit de quantification. Un script écrit pour la centrale tourne donc
partout ; c'est à lui de fournir des données de remplacement quand il en a
besoin — un module de TP le fait, voir [tp.md](tp.md).
