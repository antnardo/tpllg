# Journal des modifications

Toutes les modifications notables de `tpllg` sont consignées ici. Le format
suit [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/) ; les versions
sont calendaires, `année.mois.numéro`.

## [2026.10.1] - 2026-10-10

Les actions de l'audit du 10 octobre 2026 (section « tpllg »), et la
rétrocompatibilité avec 2026.9 que la 2026.10.0 avait rompue : tout ce qui
s'écrivait avant s'écrit encore, avec un avertissement.

### Ajouté

- **Les anciens noms de 2026.9 fonctionnent de nouveau**, rendent ce qu'ils
  rendaient, et lèvent un `DeprecationWarning` qui désigne la ligne du
  script et nomme le remplaçant. Ils ne sont pas dans `__all__`
  (`from … import *` ne les donne pas) et sont décrits dans la section
  « Anciens noms » de chaque fiche de `doc/` :

  | Ancien nom | Remplaçant |
  | --- | --- |
  | `traitement.gain_std(t, e, s, Np=0, ninter=0)` → `(G, phi)` | `H = fonction_transfert(t, e, s)` ; `abs(H)`, `np.angle(H)` |
  | `traitement.gain(t, e, s, freq, Np, method, **kwargs)` → `(G, phi)` | `H = fonction_transfert(t, e, s, freq)` |
  | `bode.phase_0_360(phase)` → degrés dans `[0, 360[` | `phase_continue(f, phase)` |
  | `signaux.decrement_logarithmique(t_pics, v_pics, offset)` | `taux_amortissement`, même résultat |
  | `signaux.frequence_pic(v, te)` (second argument un nombre) | `frequence_pic(t, v)` |
  | `signaux.extremums(v, fe, f, offset)` (second argument un nombre) | `extremums(t, v, f, offset, seuil)` |
  | `Sysam.N_MAX`, la mémoire entière (vaut `Sysam.MEMOIRE`, 262 143), sur la classe ou une instance | `Sysam.n_max(nb_voies, nb_sorties)` |
  | `can.acquerir_avec_sorties(signal, 0)` : l'entier `0` ne génère rien | `None` ; `0.0` pour une tension nulle |
  | `fichiers.readcsv(filename, encoding, entete, dtypes)` | `lire_csv(filename, entete, dtypes, encoding)` |
  | `fichiers.import_latispro(filename, colonnes=2)` | `lire_latispro(filename)`, toutes les colonnes |
  | `fichiers.import_regressi(filename, colonnes=2)` → `(t, [colonnes])` | `lire_regressi(filename)` → `[t, colonnes…]` |

  `Sysam.MODULES_ANALOG` redevient le dictionnaire de 2026.9. `curvefit`
  se dépaquette toujours en `pfit, err, chi2`. `gain(method="fit")`, qui
  n'a jamais été implémentée, lève `NotImplementedError`.
- `fichiers.lire_csv`, `lire_latispro`, `lire_regressi` : les trois lecteurs
  rendent la même chose, **une liste de tableaux, un par colonne** — la
  forme que deux des trois avaient déjà, celle de `np.loadtxt(…,
  unpack=True)`, qui se dépaquette d'un coup (`t, u = lire_csv(…)`) ; le
  temps de Regressi est le premier élément de la liste. `colonnes=None`
  lit toutes les colonnes, `verbose=False` par défaut (`import_latispro`
  rendait 2 colonnes sur 4 sans rien dire, `readcsv` imprimait un en-tête de
  50 000 caractères) ; avec `verbose=True`, chaque ligne d'en-tête est
  tronquée à 80 caractères ; l'encodage est deviné (UTF-8 puis cp1252) ou
  donné. Une cellule vide ou illisible lue `NaN` est signalée par un
  `UserWarning`.
- `fichiers.ecrire_csv(filename, colonnes, noms, delimiter, decimale)` :
  l'inverse de `lire_csv`, qu'un tableur, Regressi et Latis Pro ouvrent.
- `acquisition.charger(prefixe, voies)` : l'inverse de `sauvegarder`, qui
  rend `(temps, tensions)` comme `acquerir`.
- `bode.phase_repliee(degres)` : l'inverse de `phase_continue`, une phase
  en degrés ramenée en radians dans `]-π, π]`.
- `montecarlo.Point.normale(val, u)` : la loi normale écrite comme les
  trois autres lois (`uniforme`, `triangulaire`, `arcsinus`).
- `curvefit` et `curve_fit_complex` acceptent `u_y` et `u_x`, les noms
  courts de `datayerrors` et `dataxerrors` (ceux de `regression_york` et de
  `montecarlo`) ; les deux noms ensemble sont refusés.
- `ajustement.formater` écrit en **notation scientifique** hors de
  [10⁻³, 10⁵[, avec une puissance de dix commune à la valeur et à
  l'incertitude : `(6.6260 ± 0.0010) × 10⁻³⁴`, là où elle écrivait quarante
  zéros ; `formater(1234567.0, 5432.0)` donne `(1.2346 ± 0.0054) × 10⁶`.
- Huit exemples venus de `pyphysique`, qui n'utilisent que `tpllg` :
  `regression_lineaire.py`, `ajustement_incertitudes_x_y.py`,
  `importer_donnees.py` (et `exemples/donnees/mesures.csv`),
  `boussole_champ_terrestre.py`, `frequence_defilement_franges.py`,
  `spectre_signal_periodique.py`, `ecarts_moyennes_correcteurs.py`,
  `loi_student_incertitudes.py`.
- L'intégration continue : `.github/workflows/ci.yml` (push et pull
  request ; tests sous 3.8, 3.11 et 3.14 avec `-W error` ; `ruff check`,
  `ruff format --check`) et `publish.yml` (à chaque release publiée,
  `uv build` et les artefacts joints à la release ; pas de PyPI).
- `MANIFEST.in` : le sdist embarque `doc/` (fiches et figures),
  `exemples/` et ses données, la licence et ce journal, ce à quoi le README
  renvoie ; pas les tests. `install_requires` épingle `numpy>=1.17`.
- Dans le README : une section « Point ou grandeur ? » qui trace la
  frontière entre `tpllg.montecarlo.Point` (Monte-Carlo, sans unités,
  autonome, pour les postes du lycée et les corrigés) et `grandeurs`
  (propagation linéaire avec unités et sortie siunitx pour pythontex) ; les
  deux ne s'importent pas l'un l'autre. L'installation par la roue d'une
  release.

### Modifié

- **`curvefit` se tait par défaut** (`verbose=False`) ; avec
  `verbose=True`, les messages sont en français (« Moindres carrés »,
  « Variance effective ») et non plus « Least square method ».
- `fichiers.fpointformat` n'est plus dans `__all__` (toujours importable).
- Le simulateur et les corrections de `Sysam` reproduisent **pycanum 4.x**
  (le pilote par DLL) ; pycanum 5.0, qui passe par un serveur HTTP, reste à
  vérifier sur un poste : le README et `doc/centrale.md` le disent.
- Les trois `_tableaux` et `_tableau` de `fft`, `signaux`, `ajustement` et
  `montecarlo` sont réunis dans un module interne, `tpllg/_interne.py`, qui
  porte aussi l'avertissement des anciens noms ; le message d'un signal mal
  formé est le même partout (« t et v : deux tableaux 1D de même longueur,
  une seule voie »).

### Corrigé

- **Une phase en degrés passait en silence** dans `curve_fit_complex`,
  `residus_complexes` et `tracer_bode`, avec un résultat faux (sur les
  mesures du README : H0 = 7,3, f0 = 2094, Q = 9,7 au lieu de −5,26, 1993,
  6,67) : une phase qui dépasse 2π en valeur absolue est refusée,
  `ValueError: les phases doivent être en radians`.
- `student_coef(7, 0.95)`, les arguments inversés, rendait `nan` : `n` doit
  être un entier au moins égal à 2 et `sigma` strictement positif, sans quoi
  `ValueError` rappelle l'ordre des arguments.
- `.gitignore` : les CSV écrits par les exemples, les roues et les archives
  construites.

## [2026.10.0] - 2026-10-08

Une relecture complète du paquet, de ses tests et de ses exemples, faite le
8 octobre 2026, et l'intégration de `traitementsignal` et de ce que
l'analyse de données de l'oral CCS avait de générique.

### Ajouté

- `tpllg.harmoniques` : un signal périodique par ses harmoniques (`Signal`,
  `spectre_carre`, `spectre_triangle`, `spectre_dent_de_scie`), les filtres
  usuels (`passe_bas_1`, `passe_haut_1`, `passe_bas_2`, `passe_haut_2`,
  `passe_bande`, `coupe_bande`), le filtrage par le calcul et la valeur
  efficace ; il remplace le module `traitementsignal`.
- `ajustement.regression_york` : la droite quand x et y sont incertains,
  solution exacte de York (2004), qui remplace `scipy.odr`, retiré de
  SciPy 1.19.
- `ajustement.Ajustement` : le résultat de `curvefit`, `curve_fit_complex` et
  `regression_york`, qui se dépaquette toujours en `pfit, err, chi2` et
  donne en plus `pcov`.
- `montecarlo` : `Point.uniforme`, `Point.triangulaire`, `Point.arcsinus`
  (les lois du GUM, pour une tolérance ou une résolution),
  `Point.intervalle_le_plus_court`, `indices_sobol` (la part de chaque
  grandeur dans la variance), `fixer_graine`.
- `traitement.fonction_transfert` : la fonction de transfert par détection
  synchrone.
- `bode.phase_continue` : une phase sans saut d'une fréquence à l'autre.
- `signaux.taux_amortissement`, et un `seuil` pour `extremums`.
- `fft.calcule_DFT(..., phases=True)` rend aussi les phases.
- `Sysam.n_max`, `Sysam.te_effectif`, `Sysam.MEMOIRE`,
  `Sysam.POINTS_SORTIE_MAX`, `Sysam.PAS_TE`.
- `curvefit` accepte des incertitudes sur x sans la dérivée du modèle (la
  pente est prise sur le modèle), ou sans incertitudes sur y.
- Exemples `harmoniques.py`, `regression_york.py`, `ajustement_equadiff.py`,
  `ajustement_bruit_variable.py` ; fiche `doc/harmoniques.md` ; ce journal.

### Modifié

- **`signaux.frequence_pic(t, v)` et `signaux.extremums(t, v, f, offset,
  seuil)`** prennent le signal comme l'acquisition le rend : l'un prenait
  `te`, l'autre `fe`, et les confondre donnait 509 extremums au lieu de 41.
  Remplacer `frequence_pic(v, te)` par `frequence_pic(t, v)` et
  `extremums(v, fe, f, offset)` par `extremums(t, v, f, offset)`.
- **`signaux.decrement_logarithmique` devient `taux_amortissement`** : elle
  rendait alpha (s⁻¹), pas le décrément delta = alpha T.
- **`traitement.gain` et `traitement.gain_std` sont remplacés par
  `fonction_transfert(t, e, s, freq)`**, qui rend le complexe H :
  `G, phi = abs(H), np.angle(H)`.
- **`bode.phase_0_360` est remplacée par `phase_continue(f, phase)`** ;
  `tracer_bode` trace une phase continue, la plus basse fréquence entre
  -45° et 315° : un passe-bande inverseur va toujours de 270° à 90°, mais
  un passe-bande non inverseur ne saute plus de 360° à la résonance.
- **`Sysam.N_MAX` disparaît** : c'était la mémoire entière, traitée comme
  une limite par voie. `Sysam.n_max(nb_voies, nb_sorties)` la partage.
- `Sysam.acquerir_avec_sorties(sortie1=None, sortie2=None)` : `None` pour
  ne rien générer, un nombre pour une tension constante, une liste
  convertie.
- `Sysam.MODULES_ANALOG` devient un tuple de couples, `CALIBRES` un tuple.
- `curve_fit_complex` ajuste ln|H| et la phase au lieu des parties réelle
  et imaginaire ; les résultats sans incertitudes déclarées changent un peu.
- `SerieLineaire.ajuster` et `ajuster_modele` rendent la valeur de
  l'ajustement des mesures, les tirages n'en donnant que la dispersion ;
  `SerieLineaire` ajuste la droite de York sur chaque tirage.
- Un `Point` mesuré est tiré à sa première utilisation, avec le nombre de
  tirages des `Point` qu'il rencontre ; deux nombres différents déjà fixés
  lèvent `ValueError` au lieu de retirer l'un des deux.
- `montecarlo` a son propre générateur : `np.random.seed` n'y a plus
  d'effet, `fixer_graine` le remplace.
- `incertitudes(liste, sigma=2)` multiplie l'incertitude par 2 sans
  `advanced`.
- `fft.spectre` et `fft.calcule_DFT` n'emploient plus que numpy.fft ;
  `spectre` normalise par la somme exacte de la fenêtre.
- Python 3.8 au minimum (`python_requires`), celui que les exemples
  exigent déjà par `f"{x=}"` ; testé sous 3.8, 3.11 et 3.14.
- Le code est formaté par ruff ; `assert` remplacés par des `ValueError` ;
  `__all__` dans chaque module.

### Corrigé

- La mémoire de la Sysam (0x3FFFF mots) est partagée entre les voies et les
  sorties : un Bode à 2 voies et 1 sortie dépassait dès 870 Hz.
- Un calibre au-delà de 10 V, refusé par la centrale, est ramené à 10 V en
  le disant.
- La période passée à pycanum, en flottant 32 bits puis tronquée au dixième
  de µs, donnait 0,6 µs pour 0,7 et 1,3 µs pour 1,4 (et désynchronisait les
  sorties) : elle est arrondie avec un demi-pas de marge. Avec des sorties,
  une période qui n'est pas un multiple de 0,2 µs est refusée.
- Des voies données dans le désordre recevaient les calibres dans le mauvais
  ordre ; le seuil de déclenchement était converti avec le calibre d'une
  autre position. Une sortie liste ou nombre était ignorée sans message.
- Le simulateur refuse ce que la centrale refuse et en reproduit les
  arrondis ; `reduction` rend floor(N/r) points ; ses 24 méthodes vides
  lèvent `NotImplementedError`.
- La phase de `gain_std` : toujours -45° ou 135° avec `Np=0`, biaisée de
  0,15 à 3° par le quart de période tronqué, de 0,6° par des signaux non
  centrés.
- `indices_plages` manquait 16 harmoniques impaires sur 28 dès n = 25
  (arrondi accumulé) ; `OverflowError` pour un fondamental hors du spectre.
- `spectre` doublait la composante continue.
- `choix_echantillonnage` prenait 49 pas pour 50.
- La variance effective : `UnboundLocalError` quand le chi2 de départ était
  indéfini ou `chi_limit <= 0` ; la dernière itération gardée même moins
  bonne.
- `curve_fit_complex` : incertitudes fausses de 15 % et biais de 0,7 sigma
  sur H0 et Q (corrélation de Re et Im ignorée).
- `formater(1, 0.0996)` écrivait trois chiffres ; un sigma infini
  disparaissait ; `resume_parametres` tronquait en silence.
- `montecarlo` : deux `Point` de N différents (corrélations cassées,
  `TypeError`), pente atténuée de 5 % avec une incertitude sur x,
  `ajuster_modele` non pondéré qui plantait au premier tirage divergent,
  `show(ax)` qui rendait la figure courante, `coefs` instable.
- `incertitudes` ignorait `sigma` sans `advanced`.
- `fronts_montants` ne détectait rien sous 5 % de rapport cyclique ; il est
  vectorisé.
- `import_latispro`, `import_regressi` : `IndexError` sur une ligne vide ou
  courte ; Regressi lu en UTF-8 seulement et sans virgule décimale.
  `readcsv(Path)` : `TypeError` ; un fichier à une colonne : `csv.Error` ;
  le point-virgule pris pour la virgule sans en-tête.
- `traitementsignal`, en passant dans `harmoniques` : les fabriques de
  filtres plantaient, `Veff` comptait la moyenne pour moitié, le filtrage
  perdait le signe de H(0), un spectre en entiers plantait,
  `spectre_to_func` modifiait la liste de l'appelant, `fourier` n'était
  juste que pour une seconde de signal.
- Exemples : PDF vides de `Bode.py`, `METHODE='fit'`, `CALIBRE` en liste dans
  `Acquisition.py`, spectre tracé jusqu'à fe, une seule voie dans
  `acquisition_simple.py`, `essai_EA0.txt` écrasé par `lecture_fichiers.py`,
  décrément et extremums dans le bruit de `regime_libre.py`, tolérance prise
  pour un écart-type, « 95 % » pour 95,45 %, et les incohérences relevées
  par l'audit.
- Documentation : le nombre de tests, l'affirmation que le code évitait les
  f-strings, `python_requires >=3.7`, une image manquante, des coquilles.

## [2026.9.3] - 2026-10-03

L'état de `master` avant ce journal : le paquet installable, le simulateur
de Sysam, la documentation complète et ses figures, la licence
CC BY-NC-SA 4.0.
