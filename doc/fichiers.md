# Lire des fichiers de mesures

`tpllg.fichiers` lit trois formats : l'export CSV de Latis Pro, l'export de
Regressi, et un CSV quelconque dont il devine le délimiteur.

## Latis Pro

Dans Latis Pro, menu Fichier, Exporter, CSV ; glisser les courbes voulues de
« Courbes disponibles » vers « Courbes à exporter ».

```python
from tpllg.fichiers import import_latispro

t, u = import_latispro("acquisition.csv", colonnes=2, delimiter=";")
```

Rend un tableau numpy par colonne. Le fichier est lu en ISO 8859, la virgule
décimale est convertie, et une cellule illisible devient `nan`. Un chemin
qui n'existe pas donne une erreur qui rappelle le dossier d'exécution, la
cause habituelle sous Spyder.

## Regressi

```python
from tpllg.fichiers import import_regressi

t, u = import_regressi("mesures.txt", colonnes=2, delimiter="\t")
```

Même comportement, avec les trois lignes d'en-tête de Regressi.

## Un CSV quelconque

```python
from tpllg.fichiers import readcsv

f, Ve, Vs, phi = readcsv("mesures.csv", encoding="utf8", entete=1, dtypes=None)
```

Le délimiteur est reconnu automatiquement, la virgule décimale aussi.
`entete` est le nombre de lignes à écarter en tête, `dtypes` la liste des
types par colonne (`float` partout par défaut ; `int` et `str` sont
acceptés). Le retour est une liste de tableaux, une par colonne, dans l'ordre
du fichier, ce qui permet de les nommer d'un coup comme ci-dessus. Un fichier
Windows se lit avec `encoding="latin1"`, un vieux fichier Mac avec
`"mac_roman"`.

C'est la façon la plus courte de rentrer des mesures faites en séance :
un tableur avec une colonne par grandeur, exporté en CSV, et une ligne pour
tout charger.
