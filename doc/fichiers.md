# Lire des fichiers de mesures

`tpllg.fichiers` lit trois formats : un CSV quelconque dont il devine le
délimiteur et la virgule décimale, l'export CSV de Latis Pro, et l'export de
Regressi. Les fichiers qu'écrit `tpllg.acquisition.sauvegarder` se relisent
par numpy directement. Toutes ces fonctions rendent des tableaux numpy, un
par colonne. `exemples/lecture_fichiers.py` écrit un petit fichier de chaque
sorte et le relit.

## Sommaire

- [Un CSV de tableur : readcsv](#un-csv-de-tableur--readcsv)
- [Latis Pro](#latis-pro)
- [Regressi](#regressi)
- [Les fichiers de sauvegarder](#les-fichiers-de-sauvegarder)
- [Le dossier d'exécution](#le-dossier-dexécution)

## Un CSV de tableur : readcsv

```python
from tpllg.fichiers import readcsv

colonnes = readcsv(filename, encoding="utf8", entete=1, dtypes=None)
```

| Argument | Sens |
| --- | --- |
| `filename` | le chemin du fichier |
| `encoding` | `"utf8"` par défaut ; `"latin1"` pour un fichier écrit sous Windows par un vieux tableur, `"mac_roman"` pour un vieux fichier Mac. Un mauvais encodage se voit à des accents illisibles dans la ligne d'en-tête imprimée |
| `entete` | le nombre de lignes à écarter en tête, celles des titres ; 1 par défaut |
| `dtypes` | la liste des types des colonnes, `float`, `int` ou `str`, une entrée par colonne ; `float` partout par défaut |

Le **délimiteur** est reconnu automatiquement sur les premiers 1024
caractères, point-virgule, virgule ou tabulation, et la **virgule décimale**
est convertie. Le retour est une liste de tableaux, un par colonne, dans
l'ordre du fichier, ce qui permet de les nommer d'un coup. La fonction
imprime ce qu'elle a compris du fichier, nombre de colonnes, types, ligne
d'en-tête écartée, ce qui permet de vérifier d'un coup d'œil qu'elle a lu
ce qu'on croit.

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
f, Ve, Vs, phi = readcsv("mesures.csv")
print("f =", f, " Vs =", Vs, " phi =", phi)
```

```text
Lecture de mesures.csv
4 colonnes
Formats de conversion :  [<class 'float'>, <class 'float'>, <class 'float'>, <class 'float'>]
Entete exclus :  ['f (Hz)', 'Ve (V)', 'Vs (V)', 'phi (deg)']
f = [ 500. 2000. 8000.]  Vs = [0.6 4.8 0.7]  phi = [-98. 178.  97.]
```

Une colonne de texte se déclare, sans quoi sa conversion en nombre échoue :

```python
n, nom, x = readcsv("mixte.csv", dtypes=[int, str, float])
```

Une cellule qui ne se convertit pas arrête la lecture, en disant où :

```text
ValueError: file bad.csv, line 3, column 2 : could not convert string to float: 'abc'
```

C'est voulu : un tableau à trous se lit avec `import_latispro`, qui met
`nan` ; un CSV de mesures rentrées à la main ne doit pas en avoir, et une
cellule illisible est une faute de frappe à corriger dans le fichier.

## Latis Pro

Dans Latis Pro, menu Fichier, Exporter, CSV ; glisser les courbes voulues de
« Courbes disponibles » vers « Courbes à exporter ». Le fichier a une ligne
de titres, un point-virgule entre les colonnes, une virgule décimale, et
**une colonne de temps par courbe** : deux courbes EA0 et EA1 font quatre
colonnes.

```python
from tpllg.fichiers import import_latispro

colonnes = import_latispro(filename, colonnes=2, delimiter=";")
```

| Argument | Sens |
| --- | --- |
| `filename` | le chemin du fichier |
| `colonnes` | le nombre de colonnes à lire, en comptant celles de temps |
| `delimiter` | `";"` par défaut |

Rend une liste de tableaux, un par colonne. Le fichier est lu en ISO 8859,
l'encodage de Latis Pro. Une cellule vide ou illisible devient `nan` et un
avertissement nomme la colonne : les courbes exportées ensemble n'ont pas
toujours le même nombre de points, et les colonnes courtes finissent par
des cellules vides.

```text
Temps;EA0;Temps;EA1
0;1,2;0;0,5
0,001;1,3;0,001;0,4
0,002;1,1;0,002;0,3
```

```python
t0, ea0, t1, ea1 = import_latispro("latispro.csv", colonnes=4)
print("t =", t0, " EA0 =", ea0, " EA1 =", ea1)
```

```text
t = [0.    0.001 0.002]  EA0 = [1.2 1.3 1.1]  EA1 = [0.5 0.4 0.3]
```

Une colonne de moins que le fichier n'en a laisse la dernière de côté ; une
de plus lève `IndexError` à la première ligne.

## Regressi

Dans Regressi, Fichier, Enregistrer sous, format CSV, case « vrai CSV »
cochée. Le fichier a **trois lignes d'en-tête** (le mot Regressi, les noms,
les unités), une tabulation entre les colonnes, le point décimal, et le
temps en première colonne.

```python
from tpllg.fichiers import import_regressi

t, colonnes = import_regressi(filename, colonnes=2, delimiter="\t")
```

| Argument | Sens |
| --- | --- |
| `colonnes` | le nombre de colonnes **après** celle du temps |

Rend le temps, puis la liste des autres colonnes, ce qui s'écrit d'un coup :

```text
Regressi
t	Vs	Ve
s	V	V
0	0.5	1
0.001	0.6	1
```

```python
t, (Vs, Ve) = import_regressi("regressi.txt", colonnes=2)
print("t =", t, " Vs =", Vs, " Ve =", Ve)
```

```text
t = [0.    0.001]  Vs = [0.5 0.6]  Ve = [1. 1.]
```

Les cellules illisibles deviennent `nan`, avec un avertissement.

## Les fichiers de sauvegarder

`tpllg.acquisition.sauvegarder(prefixe, voies, temps, tensions)` écrit un
fichier texte par voie, `<prefixe>_EA<n>.txt`, sur deux lignes : les
instants puis les tensions, séparés par des espaces, avec toute la précision
de numpy. Ils se relisent par `np.loadtxt`, qui rend un tableau `(2, N)`
qu'on dépaquette :

```python
import numpy as np

t, u = np.loadtxt("essai_EA0.txt")
```

```text
t = [0.    0.001 0.002 0.003]  u = [0.1  0.2  0.15 0.05]
```

Ce format lit et écrit vite, et un tableur l'ouvre après transposition. Pour
un fichier lisible en colonnes, avec un en-tête, `np.savetxt("mesures.txt",
np.array([t, u]).T, header="t (s)  u (V)")` ; `np.loadtxt` le relit avec
`unpack=True`.

## Le dossier d'exécution

Toutes ces fonctions cherchent le fichier **dans le dossier d'exécution**,
qui n'est pas forcément celui du script : sous Spyder, c'est le dossier
courant de la console, sauf à cocher « Exécuter dans le répertoire du
fichier ». Un fichier introuvable donne une erreur qui rappelle où l'on est :

```text
ValueError: Le fichier absent.csv n'existe pas : vérifiez le dossier d'exécution (actuellement /Users/…/TP3)
```

Le plus sûr est de construire le chemin depuis le script :

```python
from pathlib import Path

DOSSIER = Path(__file__).resolve().parent
f, Ve, Vs, phi = readcsv(DOSSIER/"mesures.csv")
```
