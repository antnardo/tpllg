# Lire et écrire des fichiers de mesures

`tpllg.fichiers` lit trois formats : un CSV quelconque dont il devine le
délimiteur et la virgule décimale (`lire_csv`), l'export CSV de Latis Pro
(`lire_latispro`), et l'export de Regressi (`lire_regressi`) ; il écrit le
premier (`ecrire_csv`). Les fichiers qu'écrit `tpllg.acquisition.sauvegarder`
se relisent par `charger`, du même module, ou par numpy directement. Les
trois lecteurs prennent un nom de fichier ou un `pathlib.Path`, lisent
toutes les colonnes sauf à en demander un nombre, et **rendent la même
chose** : une liste de tableaux numpy, un par colonne, dans l'ordre du
fichier, ce qui se dépaquette d'un coup, `t, u = lire_csv("mesures.csv")`.
Ils se taisent, sauf `verbose=True`. `exemples/lecture_fichiers.py` écrit un
petit fichier de chaque sorte, tous préfixés `demo_` pour n'écraser aucune
mesure, et le relit.

## Sommaire

- [Un CSV de tableur : lire_csv](#un-csv-de-tableur--lire_csv)
- [Écrire un CSV : ecrire_csv](#écrire-un-csv--ecrire_csv)
- [Latis Pro](#latis-pro)
- [Regressi](#regressi)
- [Les fichiers de sauvegarder](#les-fichiers-de-sauvegarder)
- [Le dossier d'exécution](#le-dossier-dexécution)
- [Anciens noms](#anciens-noms)

## Un CSV de tableur : lire_csv

```python
from tpllg.fichiers import lire_csv

colonnes = lire_csv(filename, entete=1, dtypes=None, encoding=None, verbose=False)
```

| Argument | Sens |
| --- | --- |
| `filename` | le chemin du fichier, un nom ou un `pathlib.Path` |
| `entete` | le nombre de lignes à écarter en tête, celles des titres ; 1 par défaut |
| `dtypes` | la liste des types des colonnes, `float`, `int` ou `str`, une entrée par colonne ; `float` partout par défaut |
| `encoding` | l'encodage, si on le connaît (`"cp1252"` pour un fichier écrit sous Windows par un vieux tableur, `"mac_roman"` pour un vieux fichier Mac) ; sans, l'UTF-8 puis l'encodage de Windows sont essayés, ce qui suffit presque toujours |
| `verbose` | `True` pour imprimer ce que la fonction a compris du fichier : nombre de colonnes, types, lignes d'en-tête écartées, chacune tronquée à 80 caractères |

Le **délimiteur** est reconnu sur la première ligne : le premier de
point-virgule, tabulation, virgule qui s'y trouve — la virgule en dernier,
puisqu'elle peut être décimale ; aucun, et le fichier n'a qu'une colonne. La
**virgule décimale** est convertie. Un fichier sans en-tête, à une seule
colonne et à virgule décimale, est ambigu (`1,5` : deux colonnes ou un
nombre ?) : on lui donne une ligne de titre. Les lignes vides sont sautées.
Le retour est une liste de tableaux, un par colonne, dans l'ordre du
fichier, ce qui permet de les nommer d'un coup.

C'est la façon la plus courte de rentrer des mesures faites en séance : un
tableur avec une colonne par grandeur, exporté en CSV, et une ligne pour
tout charger. Le fichier `mesures.csv`, tel qu'un tableur français l'écrit :

```text
f (Hz);Ve (V);Vs (V);phi (deg)
500;1,00;0,60;-98
2000;1,00;4,80;178
8000;1,00;0,70;97
```

```python
f, Ve, Vs, phi = lire_csv("mesures.csv")
print("f =", f, " Vs =", Vs, " phi =", phi)
```

```text
f = [ 500. 2000. 8000.]  Vs = [0.6 4.8 0.7]  phi = [-98. 178.  97.]
```

Avec `verbose=True`, pour vérifier d'un coup d'œil qu'elle a lu ce qu'on
croit :

```text
Lecture de mesures.csv
En-tête écarté : f (Hz);Ve (V);Vs (V);phi (deg)
4 colonne(s), types float, float, float, float
```

Une colonne de texte se déclare, sans quoi sa conversion en nombre échoue :

```python
n, nom, x = lire_csv("mixte.csv", dtypes=[int, str, float])
```

Une cellule qui ne se convertit pas, ou une ligne qui n'a pas le bon
nombre de colonnes, arrête la lecture, en disant où :

```text
ValueError: file bad.csv, line 3, column 2 : could not convert string to float: 'abc'
ValueError: file bad.csv, line 4 : 1 colonnes au lieu de 2
```

C'est voulu : un tableau à trous se lit avec `lire_latispro`, qui met
`nan` ; un CSV de mesures rentrées à la main ne doit pas en avoir, et une
cellule illisible est une faute de frappe à corriger dans le fichier.

## Écrire un CSV : ecrire_csv

```python
from tpllg.fichiers import ecrire_csv

ecrire_csv(filename, colonnes, noms=None, delimiter=";", decimale=".", encoding="utf8")
```

| Argument | Sens |
| --- | --- |
| `colonnes` | la liste des colonnes à écrire, des tableaux ou des listes de même longueur, nombres ou textes |
| `noms` | facultatif : les titres des colonnes, un par colonne, qui font la première ligne |
| `delimiter` | `";"` par défaut, celui des tableurs français ; `","` ou `"\t"` |
| `decimale` | `"."` par défaut ; `","` pour qu'un tableur français lise les nombres directement |
| `encoding` | `"utf8"` par défaut |

L'inverse de `lire_csv` : une ligne de titres s'il y en a, puis une ligne
par point, les nombres avec toute leur précision. `lire_csv` relit le
fichier tel quel (`entete=0` s'il n'a pas de titres), un tableur l'ouvre,
Regressi et Latis Pro l'importent.

```python
ecrire_csv("resultats.csv", [f, Vs / Ve], noms=["f (Hz)", "G"], decimale=",")
f2, G2 = lire_csv("resultats.csv")
```

```text
f (Hz);G
500,0;0,6
2000,0;4,8
8000,0;0,7
```

## Latis Pro

Dans Latis Pro, menu Fichier, Exporter, CSV ; glisser les courbes voulues de
« Courbes disponibles » vers « Courbes à exporter ». Le fichier a une ligne
de titres, un point-virgule entre les colonnes, une virgule décimale, et
**une colonne de temps par courbe** : deux courbes EA0 et EA1 font quatre
colonnes.

```python
from tpllg.fichiers import lire_latispro

colonnes = lire_latispro(filename, colonnes=None, delimiter=";", verbose=False)
```

| Argument | Sens |
| --- | --- |
| `filename` | le chemin du fichier |
| `colonnes` | le nombre de colonnes à lire, en comptant celles de temps ; toutes par défaut |
| `delimiter` | `";"` par défaut |
| `verbose` | `True` pour l'écho du fichier |

Rend une liste de tableaux, un par colonne. Le fichier est lu en UTF-8 ou,
à défaut, dans l'encodage de Windows, celui de Latis Pro. Une cellule vide,
absente ou illisible devient `nan` et un avertissement (`UserWarning`)
nomme la colonne : les courbes exportées ensemble n'ont pas toujours le même
nombre de points, et les colonnes courtes finissent par des cellules vides.
Une ligne vide est sautée.

```text
Temps;EA0;Temps;EA1
0;1,2;0;0,5
0,001;1,3;0,001;0,4
0,002;1,1;0,002;0,3
```

```python
t0, ea0, t1, ea1 = lire_latispro("demo_latispro.csv")
print("t =", t0, " EA0 =", ea0, " EA1 =", ea1)
```

```text
t = [0.    0.001 0.002]  EA0 = [1.2 1.3 1.1]  EA1 = [0.5 0.4 0.3]
```

Un nombre de colonnes plus petit que le fichier n'en a laisse les dernières
de côté ; plus grand, les colonnes en trop sont remplies de `nan`, avec
l'avertissement.

## Regressi

Dans Regressi, Fichier, Enregistrer sous, format CSV, case « vrai CSV »
cochée. Le fichier a **trois lignes d'en-tête** (le mot Regressi, les noms,
les unités), une tabulation entre les colonnes, et le temps en première
colonne ; le point comme la virgule décimale sont lus, et le fichier, en
UTF-8 ou dans l'encodage de Windows.

```python
from tpllg.fichiers import lire_regressi

colonnes = lire_regressi(filename, colonnes=None, delimiter="\t", verbose=False)
```

| Argument | Sens |
| --- | --- |
| `colonnes` | le nombre de colonnes à lire, celle du temps comprise ; toutes par défaut |

Rend une liste de tableaux, le temps en tête, ce qui s'écrit d'un coup :

```text
Regressi
t	Vs	Ve
s	V	V
0	0.5	1
0.001	0.6	1
```

```python
t, Vs, Ve = lire_regressi("demo_regressi.txt")
print("t =", t, " Vs =", Vs, " Ve =", Ve)
```

```text
t = [0.    0.001]  Vs = [0.5 0.6]  Ve = [1. 1.]
```

Les cellules illisibles ou absentes deviennent `nan`, avec un
avertissement.

## Les fichiers de sauvegarder

`tpllg.acquisition.sauvegarder(prefixe, voies, temps, tensions)` écrit un
fichier texte par voie, `<prefixe>_EA<n>.txt`, sur deux lignes : les
instants puis les tensions, séparés par des espaces, avec toute la précision
de numpy. `tpllg.acquisition.charger(prefixe, voies)` les relit et rend ce
que `acquerir` rendait, `(temps, tensions)`, deux tableaux `(voies, N)` ;
`np.loadtxt` relit un fichier, en un tableau `(2, N)` qu'on dépaquette :

```python
import numpy as np
from tpllg.acquisition import charger

temps, tensions = charger("demo", [0])  # ce que acquerir avait rendu
t, u = np.loadtxt("demo_EA0.txt")  # une voie, directement
```

```text
t = [0.    0.001 0.002 0.003]  u = [0.1  0.2  0.15 0.05]
```

Ce format lit et écrit vite, et un tableur l'ouvre après transposition. Pour
un fichier en colonnes avec un en-tête, qu'un tableur ouvre tel quel,
`ecrire_csv("mesures.csv", [t, u], noms=["t (s)", "u (V)"])`.

## Le dossier d'exécution

Toutes ces fonctions cherchent le fichier **dans le dossier d'exécution**,
qui n'est pas forcément celui du script : sous Spyder, c'est le dossier
courant de la console, sauf à cocher « Exécuter dans le répertoire du
fichier ». Un fichier introuvable donne une erreur qui rappelle où l'on est :

```text
ValueError: Le fichier absent.csv n'existe pas : vérifiez le dossier d'exécution (actuellement /Users/…/TP3)
```

Le plus sûr est de construire le chemin depuis le script ; les trois
fonctions acceptent un `Path` :

```python
from pathlib import Path

DOSSIER = Path(__file__).resolve().parent
f, Ve, Vs, phi = lire_csv(DOSSIER / "mesures.csv")
```

## Anciens noms

Les noms de 2026.9 fonctionnent encore, et rendent ce qu'ils rendaient,
avec un `DeprecationWarning` qui nomme le remplaçant. Ils ne sont pas dans
`__all__` : `from tpllg.fichiers import *` ne les donne pas.

| Ancien nom | Rendait | Remplaçant |
| --- | --- | --- |
| `readcsv(filename, encoding="utf8", entete=1, dtypes=None)` | la liste des colonnes, en imprimant ce qu'il lisait | `lire_csv(filename, entete, dtypes, encoding)`, muet par défaut |
| `import_latispro(filename, colonnes=2, delimiter=";")` | les deux premières colonnes seulement, par défaut | `lire_latispro(filename)`, toutes les colonnes |
| `import_regressi(filename, colonnes=2, delimiter="\t")` | `(t, [colonne 1, colonne 2])`, `colonnes` comptant les colonnes **après** le temps | `lire_regressi(filename)`, une seule liste, `[t, colonne 1, colonne 2…]` |

`fpointformat`, qui convertit une virgule décimale en point dans une
chaîne, reste importable mais n'est plus dans `__all__` : c'est un détail de
`lire_csv`.
