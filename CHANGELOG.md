# Journal des modifications

Toutes les modifications notables de `tpllg` sont consignées ici. Le format
suit [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/) ; les versions
sont calendaires, `année.mois.numéro`.

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
  -180° et 180° : un passe-bande inverseur va de -90° à -270°.
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
