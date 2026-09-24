# Tests

`pytest` depuis la racine du dépôt : 54 tests, dix fichiers, moins de deux
secondes, sans centrale (`tests/conftest.py` rend pycanum introuvable et
matplotlib muet). Chaque fonction publique est confrontée à un résultat
théorique, pas à une simple absence d'erreur :

| Fichier                                | Ce qui est vérifié                                                                                                                                                                                                                                          |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_sysam.py`, `test_acquisition.py` | la forme de pycanum (tableaux 2D de double, temps en secondes), la conversion en µs, le calibre immédiatement supérieur, la normalisation des calibres, le déclenchement, la sauvegarde relue à l'octet                                                     |
| `test_fft.py`                          | l'amplitude en volts quelle que soit la durée, un nombre impair de points, la fenêtre de Blackman qui rattrape les fuites, le pas `fe/((p+1)N)` et le miroir du spectre                                                                                     |
| `test_traitement.py`                   | le gain et la phase retrouvés sur des sinusoïdes exactes, l'interpolation par FFT exacte sur un signal à bande limitée, les contraintes d'échantillonnage, l'harmonique `n` d'un créneau échantillonné, `4/(Np sin(πn/Np))`                                 |
| `test_signaux.py`                      | les instants des fronts montants et descendants aux zéros du sinus, la période médiane, la fenêtre semi-ouverte, la fréquence exacte sur un nombre entier de périodes, le décrément avec et sans offset                                                     |
| `test_ajustement.py`                   | les incertitudes des moindres carrés, `σ/√Sxx` et `σ√(1/P + x̄²/Sxx)`, la variance effective qui rend les mêmes paramètres et des incertitudes dans le rapport des sigmas effectifs, le χ² réduit, les résidus nuls sur le modèle, l'arrondi à deux chiffres |
| `test_incertitudes.py`                 | la densité normalisée quel que soit l'écart-type, l'intégrale face à `loi_normale_cumulee`, le coefficient de Student comme quantile bilatéral                                                                                                              |
| `test_montecarlo.py`                   | les incertitudes composées, les corrélations gardées, les quantiles qui suivent `x → x²`, `E[x²] = 1 + u²`, la régression par tirages face aux formules des moindres carrés                                                                                 |
| `test_bode.py`                         | la phase repliée dans [0, 360[, la figure écrite avec le modèle et la légende                                                                                                                                                                               |
| `test_fichiers.py`                     | le délimiteur et la virgule devinés, les types, une erreur située à la ligne et à la colonne                                                                                                                                                                |
