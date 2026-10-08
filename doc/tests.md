# Tests

`pytest` depuis la racine du dépôt : 273 tests, douze fichiers, quelques
secondes, sans centrale (`tests/conftest.py` rend pycanum introuvable,
supprime l'attente des acquisitions simulées et rend matplotlib muet). Ils
passent sous Python 3.8 (numpy 1.24, scipy 1.10, matplotlib 3.7), 3.11 et
3.14 (numpy 2.5, scipy 1.18, matplotlib 3.11). Chaque fonction publique est
confrontée à un résultat théorique, pas à une simple absence d'erreur, et
chaque défaut corrigé a le test qui l'aurait montré :

| Fichier | Ce qui est vérifié |
| --- | --- |
| `test_sysam_factice.py` | que le simulateur refuse ce que la centrale refuse (calibre au-delà de 10 V, nombre de calibres, période sous le minimum direct ou multiplexé, mémoire partagée entre voies et sorties, points des sorties comptés jusqu'à la fermeture, sortie trop longue ou trop rapide, tableau de sortie mal formé, voie de déclenchement absente) et en reproduise les arrondis (période en flottant 32 bits tronquée, points plafonnés, voies rangées, `floor(N/r)`, seuil converti à la position de la voie) |
| `test_sysam.py`, `test_acquisition.py` | la forme de pycanum (tableaux 2D de double, temps en secondes), la période arrondie au dixième de µs, `n_max` qui tient dans la mémoire quand la formule naïve dépasse, les calibres plafonnés à 10 V, les voies dans le désordre qui gardent leur calibre et leur ligne, le seuil de déclenchement corrigé, une liste ou un nombre en sortie, la période multiple de 0,2 µs exigée avec des sorties, la sauvegarde relue à l'octet |
| `test_fft.py` | l'amplitude en volts exacte quelle que soit la durée, un nombre impair de points, les phases, la composante continue égale à la moyenne, la fenêtre de Blackman qui rattrape les fuites, le pas `fe/((p+1)N)` et le miroir |
| `test_traitement.py` | la fonction de transfert par détection synchrone à 10⁻⁴ degré près quel que soit le nombre de points par période, avec des offsets, un nombre non entier de périodes, sans la fréquence, dans le bruit ; l'interpolation par FFT exacte ; les contraintes d'échantillonnage et le pas juste malgré les arrondis ; l'harmonique `n` d'un créneau échantillonné, `4/(Np sin(πn/Np))` ; les plages centrées sur chaque harmonique d'un fondamental hors grille |
| `test_signaux.py` | les instants des fronts aux zéros du sinus, des impulsions de 2 % de rapport cyclique, une acquisition pleine en moins d'une demi-seconde, la période médiane, la fenêtre semi-ouverte, la fréquence exacte, le taux d'amortissement avec et sans offset, le seuil qui écarte les bosses du bruit, `(t, v)` partout |
| `test_ajustement.py` | les incertitudes des moindres carrés, `σ/√Sxx` et `σ√(1/P + x̄²/Sxx)`, la variance effective (mêmes paramètres, incertitudes dans le rapport des sigmas effectifs, pente prise sur le modèle sans dérivée, au moins une itération, le meilleur gardé), `curve_fit_complex` sans biais et aux incertitudes justes sur deux cents jeux simulés, une phase à 2π près, un signe de H0 faux corrigé, la régression de York sur les données de Pearson et York et ses incertitudes sur quatre cents jeux, l'arrondi à deux chiffres, les longueurs refusées |
| `test_incertitudes.py` | la densité normalisée, son intégrale face à `loi_normale_cumulee`, le coefficient de Student comme quantile bilatéral, `sigma` pris en compte sans Student |
| `test_montecarlo.py` | les incertitudes composées, les corrélations gardées, un tirage jamais refait, deux N différents refusés, un Point non tiré qui prend le N de l'autre, les lois uniforme, triangulaire et arcsinus, l'intervalle le plus court, la droite sans atténuation et égale à York, `ajuster_modele` égal à `curvefit` en valeur et en incertitude, un tirage qui ne converge pas remplacé, les indices de Sobol de deux cas analytiques, `show` dans un repère donné |
| `test_bode.py` | la phase continue d'un passe-bande, d'un inverseur (de 270° à 90°, même mesuré à ±3° près), d'un passe-bas, des mesures voisines de 0°, la phase de chaque mesure au tour du modèle, la figure écrite avec le modèle et la légende |
| `test_harmoniques.py` | la synthèse du créneau, du triangle et de la dent de scie, la valeur efficace (Parseval, composante continue entière), le signe de H(0), un H qui ne se calcule pas en 0, un spectre en entiers, des listes de coefficients non modifiées, la valeur de chaque filtre à sa fréquence caractéristique |
| `test_fichiers.py` | le délimiteur et la virgule devinés, un Path, un fichier sans en-tête ou à une colonne, les lignes vides ou courtes, un fichier Regressi de Windows à virgule décimale, une erreur située à la ligne et à la colonne, le nombre de colonnes vérifié sur un fichier qui existe |

Pour lancer les tests sous un Python donné, avec uv :

```bash
uv run --no-project --python 3.8 --with pytest --with numpy --with scipy --with matplotlib python -m pytest
```
