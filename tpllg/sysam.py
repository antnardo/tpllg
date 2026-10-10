"""
La Sysam SP5 par pycanum, avec un `with`, des calibres simples, des temps en
secondes et des acquisitions qui rendent directement temps et tensions.

Sans pycanum (un Mac, un poste sans carte), la classe hérite du simulateur
tpllg.sysam_factice, qui a la même interface et refuse les mêmes choses.

La classe corrige au passage quatre pièges du pilote de pycanum, lus dans son
source C (SysamSP5Link.c, pysysam.c) — non vérifiés sur la centrale :

- la période est passée en flottant 32 bits puis tronquée au dixième de
  microseconde : 0,7 µs demandées donnaient 0,6 µs, 1,4 µs donnaient 1,3 µs ;
  on passe la valeur arrondie au dixième, avec un demi-pas de marge ;
- les voies sont rangées dans l'ordre croissant mais les calibres appliqués
  dans l'ordre donné : on range les couples (voie, calibre) ensemble, et on
  rend les lignes dans l'ordre demandé ;
- le seuil de déclenchement est converti avec le calibre rangé à la position
  du numéro de la voie : on le corrige d'autant ;
- un tableau de sortie qui n'est pas un ndarray de flottants était ignoré sans
  message : on convertit, et une sortie sur une période qui n'est pas un
  multiple de 0,2 µs (les sorties tournent alors à une autre cadence que les
  entrées) est refusée.

Ces règles et le simulateur reproduisent pycanum 4.x, qui pilote la carte par
sa DLL. pycanum 5.0 (2024) passe par un serveur HTTP (sysamhttp) et n'a pas
été vérifié : ses refus et ses arrondis peuvent différer.

Pour plus de détails sur pycanum :
https://www.f-legrand.fr/scidoc/docmml/sciphys/caneurosmart/interpy/interpy.html

Ce qui a changé en 2026.10 fonctionne toujours sous son ancien nom, sans
avertissement : `Sysam.N_MAX` (la mémoire entière, que `n_max` partage
maintenant entre voies et sorties), et `acquerir_avec_sorties(signal, 0)`,
où 0 voulait dire « pas de sortie » (c'est None à présent).

@author: a. marchand
"""

from numbers import Integral

import numpy as np

try:
    import pycanum.main as pycan
except ImportError:  # pas de pycanum sur cette machine : centrale simulée
    from tpllg import sysam_factice as pycan

__all__ = ["CAL_DEFAUT", "SYSAM_TYPE", "Sysam"]

SYSAM_TYPE = "SP5"  # et non "PCI", qui n'existe pas au lycée
CAL_DEFAUT = 10  # calibre par défaut, en volts


class Sysam(pycan.Sysam):
    """La classe de pycanum, en plus simple :

    - `Sysam(voies=[0, 1], calibres=5)` configure les entrées dès l'ouverture,
      un calibre seul vaut pour toutes les voies, et 'SP5' est implicite ;
    - `with Sysam([0, 1]) as can:` ferme la centrale en sortie de bloc ;
    - les périodes sont en secondes ;
    - `acquerir` et `acquerir_avec_sorties` rendent (temps, tensions), deux
      tableaux 2D, une ligne par voie dans l'ordre des voies demandées.

    Exemple minimal :

        with Sysam([0, 1]) as can:
            can.config_echantillon(1e-5, 1000)
            temps, tensions = can.acquerir()

    La centrale : quatre modules de conversion 12 bits (EA0 et EA4, EA1 et EA5,
    EA2 et EA6, EA3 et EA7), 10 MHz quand chaque module n'a qu'une entrée
    active (ou est en différentiel), 500 kHz sinon ; deux sorties 12 bits à
    5 MHz, ±10 V ; une mémoire de 0x3FFFF = 262 143 mots de 12 bits partagée
    entre les entrées (voies × points) et les sorties ; calibres 0,2, 1, 5 et
    10 V accessibles par pycanum ; impédance d'entrée 1 MΩ.
    """

    TE_MIN_DIRECT = 1e-7  # s, un module, une entrée
    TE_MIN_MULTIPLEX = 2e-6  # s, deux entrées d'un même module en mode simple
    TE_MIN_SORTIE = 2e-7  # s, et les sorties ne tournent qu'à un multiple de 0,2 µs
    PAS_TE = 1e-7  # s, la centrale compte la période en dixièmes de microseconde
    MEMOIRE = 0x3FFFF  # mots de 12 bits, entrées et sorties ensemble
    POINTS_SORTIE_MAX = 0x1FFFF
    CALIBRES = (0.2, 1, 5, 10)
    # les deux entrées de chaque module, un dict comme en 2026.9 (on ne le modifie pas)
    MODULES_ANALOG = {0: (0, 4), 1: (1, 5), 2: (2, 6), 3: (3, 7)}  # noqa: RUF012
    N_MAX = MEMOIRE  # l'ancien nom (2026.9) de la mémoire entière, que n_max partage

    @classmethod
    def get_calibre(cls, valeur):
        """Le calibre que la centrale prend pour une tension maximale `valeur` :
        le plus petit qui la contient, et 10 V au-delà de 10 V."""
        for cal in cls.CALIBRES:
            if valeur <= cal:
                return cal
        return max(cls.CALIBRES)

    @classmethod
    def te_min(cls, voies, diff=()):
        """La période minimale pour ces voies : 0,1 µs, ou 2 µs (mode
        multiplexé) quand les deux entrées d'un même module sont actives en
        mode simple."""
        diff = {d % 4 for d in diff}
        for module, (a, b) in cls.MODULES_ANALOG.items():
            if a in voies and b in voies and module not in diff:
                return cls.TE_MIN_MULTIPLEX
        return cls.TE_MIN_DIRECT

    @classmethod
    def n_max(cls, nb_voies, nb_sorties=0):
        """Le nombre de points le plus grand qu'une acquisition sur `nb_voies`
        voies peut demander, quand `nb_sorties` sorties tournent avec autant de
        points que les entrées (le Bode automatique : 2 voies, 1 sortie).

        La mémoire est partagée : voies × N + points des sorties ≤ 0x3FFFF. La
        centrale arrondit N au paquet de sa FIFO (255 mots au plus) : on garde
        255 mots de marge."""
        return (cls.MEMOIRE - 255) // (nb_voies + nb_sorties)

    def __init__(self, voies=None, calibres=None, diff=None):
        self.voies = []
        self.calibres = []
        self.diff = []
        self.te = None
        self.nbpoints = None
        self._ordre = []
        self._calibres_pilote = [10.0] * 8  # le tableau calibreEA du pilote, par position
        super().__init__(SYSAM_TYPE)
        if voies is not None:
            self.config_entrees(voies, calibres, diff)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.fermer()
        return False

    def config_entrees(self, voies, calibres=None, diff=None):
        """Les entrées analogiques et leurs calibres.

        voies : les numéros des entrées, de 0 à 7 (par ex. [0, 1]) ;
        calibres : la tension maximale attendue sur chaque voie, en volts, une
            par voie ou une seule pour toutes (10 V par défaut). La centrale
            prend le plus petit calibre parmi 0,2, 1, 5 et 10 V qui la contient ;
            au-delà de 10 V, 10 V, en le disant ; une valeur illisible ou nulle
            donne 10 V ;
        diff : les modules en mode différentiel : [0] met EA0 en différentiel,
            EA4 étant l'entrée moins ; ne mettez alors pas EA4 dans `voies`.

        Pour la fréquence maximale (10 MHz), une seule entrée par module : EA0
        à EA3, sans EA4 à EA7.
        """
        voies = [int(v) for v in voies]
        diff = [int(d) for d in (diff or [])]
        if len(set(voies)) != len(voies):
            raise ValueError(f"voies {voies} : une voie apparaît deux fois")
        for v in voies:
            if v >= 4 and (v - 4) in {d % 4 for d in diff}:
                raise ValueError(
                    f"EA{v} est l'entrée moins de EA{v - 4} en différentiel : ne la mettez pas dans voies"
                )
        if calibres is None:
            calibres = CAL_DEFAUT
        if np.ndim(calibres) == 0:
            calibres = [calibres] * len(voies)
        if len(calibres) != len(voies):
            raise ValueError(f"{len(calibres)} calibres pour {len(voies)} voies")
        calibres = [self._calibre(v, c) for v, c in zip(voies, calibres)]
        # le pilote range les voies dans l'ordre croissant et applique les calibres
        # dans l'ordre donné : on range les couples ensemble
        self._ordre = sorted(range(len(voies)), key=voies.__getitem__)
        voies_rangees = [voies[i] for i in self._ordre]
        calibres_ranges = [calibres[i] for i in self._ordre]
        super().config_entrees(voies_rangees, calibres_ranges, diff)
        self.voies, self.calibres, self.diff = voies, calibres, diff
        for i, c in enumerate(calibres_ranges):
            self._calibres_pilote[i] = c

    @classmethod
    def _calibre(cls, voie, valeur):
        valeur = float(np.nan_to_num(valeur, nan=CAL_DEFAUT))
        if valeur <= 0:
            return float(CAL_DEFAUT)
        if valeur > max(cls.CALIBRES):
            print(f"[SYSAM] EA{voie} : calibre {valeur:g} V ramené à 10 V, le plus grand")
        return float(cls.get_calibre(valeur))

    def config_echantillon(self, techant, nbpoints):
        """La période d'échantillonnage `techant`, en secondes, et le nombre de
        points à acquérir.

        La période est arrondie au dixième de microseconde (self.te la donne) ;
        son minimum dépend des voies (te_min). Au-delà de ce que la mémoire
        permet pour les voies configurées, la centrale rend moins de points
        que demandé : on le dit."""
        dixiemes = round(techant / self.PAS_TE)
        nbpoints = int(nbpoints)
        if self.voies and nbpoints > self.MEMOIRE // len(self.voies):
            print(
                f"[SYSAM] ATTENTION : {nbpoints} points demandés sur {len(self.voies)} voie(s), la mémoire "
                f"en permet au plus {self.MEMOIRE // len(self.voies)} : la centrale en rendra moins"
            )
        # un demi-pas de marge : pycanum passe la période en flottant 32 bits, que
        # le pilote tronque au dixième de microseconde
        super().config_echantillon((dixiemes + 0.5) / 10, nbpoints)
        self.te = dixiemes * self.PAS_TE
        self.nbpoints = nbpoints

    def config_trigger(self, voie, seuil, montant=1, pretrigger=1, pretriggerSouple=0, hysteresis=0):
        """Le déclenchement sur le passage de la voie `voie` par `seuil` (V),
        front montant (1) ou descendant (0), en gardant `pretrigger` points
        avant ; voie=-1 le désactive."""
        if voie in self.voies:
            # le pilote convertit le seuil avec le calibre rangé à la position `voie`
            lu = self._calibres_pilote[voie]
            vrai = self.calibres[self.voies.index(voie)]
            seuil = seuil * lu / vrai
        super().config_trigger(voie, seuil, montant, pretrigger, pretriggerSouple, hysteresis)

    def temps(self, reduction=1):
        """Les instants, un tableau 2D : une ligne par voie, dans l'ordre des
        voies demandées ; un point sur `reduction`."""
        return self._dans_l_ordre(super().temps(reduction))

    def entrees(self, reduction=1):
        """Les tensions, comme temps()."""
        return self._dans_l_ordre(super().entrees(reduction))

    def _dans_l_ordre(self, lignes):
        if len(self._ordre) != len(lignes):
            return lignes
        return lignes[np.argsort(self._ordre)]

    def acquerir(self, reduction=1):
        """Acquiert et rend (temps, tensions), deux tableaux 2D, une ligne par
        voie dans l'ordre des voies demandées, temps en secondes ; un point sur
        `reduction` (floor(N/reduction) points)."""
        super().acquerir()  # ne rend la main qu'à la fin
        return self.temps(reduction), self.entrees(reduction)

    def acquerir_avec_sorties(self, sortie1=None, sortie2=None):
        """Acquiert en générant en même temps les sorties S1 et S2, à la même
        période que les entrées, et rend (temps, tensions) comme acquerir.

        sortie1, sortie2 : les tensions (V) à appliquer, échantillon par
            échantillon, répétées en boucle ; None pour ne rien générer, un
            nombre pour une tension constante (l'entier 0 vaut toujours « pas
            de sortie », comme en 2026.9 : 0.0 pour une tension nulle). Au
            plus 0x1FFFF points chacune, et la mémoire est partagée : voies × N
            + points des sorties ≤ 0x3FFFF (n_max le calcule).

        La période doit être un multiple de 0,2 µs : les sorties ne connaissent
        pas d'autre cadence."""
        sorties = [_sortie(s, n) for n, s in ((1, sortie1), (2, sortie2))]
        if any(s.size for s in sorties) and self.te is not None:
            dixiemes = round(self.te / self.PAS_TE)
            if dixiemes % 2:
                raise ValueError(
                    f"période de {dixiemes / 10:.1f} µs : avec des sorties, elle doit être un multiple de "
                    "0,2 µs, sans quoi les sorties tournent à une autre cadence que les entrées"
                )
        super().acquerir_avec_sorties(*sorties)
        return self.temps(), self.entrees()


def _sortie(valeurs, numero):
    """Ce que pycanum attend d'une sortie : un tableau 1D de flottants, vide
    pour « pas de sortie »."""
    if valeurs is None:
        return np.zeros(0)
    if isinstance(valeurs, Integral) and valeurs == 0:  # l'entier 0 de 2026.9 : pas de sortie
        return np.zeros(0)
    valeurs = np.atleast_1d(np.asarray(valeurs, dtype=float))
    if valeurs.ndim != 1:
        raise ValueError(f"sortie {numero} : un tableau 1D de tensions, ou un nombre")
    return valeurs
