"""
Sysam factice : la même interface que pycanum.main, sans centrale.

tpllg.sysam s'y rabat quand pycanum n'est pas installé — sur un Mac, sur un
poste sans carte — et le dit à l'ouverture. Les acquisitions attendent la durée
demandée et rendent, dans la forme exacte de pycanum (tableaux 2D de double,
une ligne par voie, temps en secondes), un bruit de quantification.

Le simulateur refuse ce que la centrale refuse, avec les règles du pilote C de
pycanum (SysamSP5Link.c et pysysam.c), pour qu'un script essayé chez soi ne
découvre pas ses erreurs en salle de TP. Il lève ValueError, comme pycanum :

- pour un calibre au-dessus de 10 V, ou un nombre de calibres différent du
  nombre de voies ;
- pour une période d'échantillonnage sous le minimum : 0,1 µs, ou 2 µs quand
  les deux entrées d'un même module (EA0 et EA4…) sont prises en mode simple ;
- quand la mémoire de 0x3FFFF mots, partagée entre les entrées (voies × points)
  et les sorties (points de S1 + points de S2), ne suffit pas ; pour une sortie
  de plus de 0x1FFFF points ou plus rapide que 0,2 µs ;
- pour un tableau de sortie qui n'est pas un tableau 1D de flottants ;
- pour un déclenchement sur une voie non configurée.

Il en reproduit aussi les arrondis et les silences :

- la période passe en flottant 32 bits, puis est tronquée au dixième de
  microseconde : 0,7 µs demandées donnent 0,6 µs ;
- le nombre de points est arrondi au paquet de la FIFO, puis ramené en silence
  à ce que la mémoire permet pour les entrées seules ;
- les voies sont rangées dans l'ordre croissant, et les calibres appliqués
  dans cet ordre ;
- les points occupés par les sorties restent comptés jusqu'à la fermeture ;
- le seuil de déclenchement est converti avec le calibre rangé à la position
  du numéro de la voie, pas avec celui de la voie ;
- temps(r) et entrees(r) rendent floor(N/r) points.

Les fonctions qui ne simulent rien d'utile (acquisition permanente, paquets,
filtrage, lecture directe, compteur, chronomètre) lèvent NotImplementedError :
mieux vaut le savoir chez soi qu'en TP.

Écrit d'après pycanum de Frédéric Legrand :
https://www.f-legrand.fr/scidoc/docmml/sciphys/caneurosmart/interpy/interpy.html
"""

import math
import time

import numpy as np

from tpllg._interne import generateur

__all__ = ["MEMOIRE", "POINTS_SORTIE_MAX", "SYSAM_SP5", "VERBOSE", "Sysam"]

SYSAM_SP5 = 1  # le seul modèle du lycée : pas de "PCI"

VERBOSE = True

MEMOIRE = 0x3FFFF  # mots de 12 bits, entrées et sorties ensemble
POINTS_SORTIE_MAX = 0x1FFFF
TE_MIN_SORTIE_US = 0.2

_rng = generateur()

# Les codes de calibre du pilote et la tension maximale de chacun
_CALIBRE_DU_CODE = {0: 10.0, 4: 5.0, 5: 1.0, 2: 0.2}


def _code_calibre(calibre):
    """Le code que le pilote choisit pour une tension maximale : le plus petit
    calibre qui la contient. Au-dessus de 10 V, le pilote refuse."""
    if calibre > 10.0:
        raise ValueError(f"Erreur : calibre {calibre} V refusé, le plus grand est 10 V")
    if calibre > 5.0:
        return 0
    if calibre > 1.0:
        return 4
    if calibre > 0.2:
        return 5
    return 2


def _periode_us(techant):
    """La période que la centrale applique : pycanum passe `techant` en
    flottant 32 bits, le pilote compte des tops de 0,1 µs et tronque."""
    temps = float(np.float32(techant))
    sauts = max(math.floor(temps * 10.0) - 1, 0)
    if sauts > 2**40:
        raise ValueError("Erreur : periode d'echantillonnage trop grande")
    return (sauts + 1) / 10.0


def _points_par_paquet(te_us, nb_voies):
    """La FIFO se vide par paquets : 255 mots, 100 au-delà de 10 ms, 10 au-delà
    de 100 ms, une voie à la fois au-delà d'une seconde."""
    fifo = 255
    if te_us >= 1e4:
        fifo = 100
    if te_us >= 1e5:
        fifo = 10
    taille = (fifo // nb_voies) * nb_voies
    if te_us >= 1e6:
        taille = nb_voies
    return taille // nb_voies


def _non_simule(quoi):
    return NotImplementedError(f"sysam_factice ne simule pas {quoi} : essayez ce script sur la centrale")


class Sysam:
    """La centrale simulée : l'interface de pycanum.main.Sysam."""

    def __init__(self, nom):
        self.verbose = VERBOSE
        print("[SYSAM] ATTENTION : pycanum absent, centrale simulée (tpllg.sysam_factice)")
        self.ouvert = False
        self.sysamid = SYSAM_SP5 if nom == "SP5" else 0
        self.ouvrir()

    # --- ouverture
    def ouvrir(self):
        if self.ouvert:
            if self.verbose:
                print(f"[SYSAM] CAN Sysam ouvert ({self.sysamid})")
            return
        if self.sysamid == 0:
            raise ValueError("Erreur : centrale inconnue (SP5)")
        self._voies = []
        self._calibres = []
        self._codes_calibre = [0] * 8  # le tableau calibreEA du pilote, indexé par position
        self._diff = []
        self._temin_us = 0.0
        self._te_us = 0.0
        self._echantillonnage = None  # "memoire" ou "permanent"
        self._points_demandes = 0
        self._points = 0
        self._ram_entrees = 0
        self._ram_sorties = [0, 0]
        self._bits = 12
        self._donnees = None
        self.ouvert = True

    def fermer(self):
        self.ouvert = False

    def reset(self):
        self._exiger_ouvert()

    def _exiger_ouvert(self):
        if not self.ouvert:
            raise ValueError("Erreur : CAN non ouvert")

    # --- configuration
    def config_entrees(self, voies, calibres, diff=()):
        self._exiger_ouvert()
        voies = [int(v) for v in voies]
        calibres = [float(c) for c in calibres]
        diff = [int(d) % 4 for d in diff]
        codes = [_code_calibre(c) for c in calibres]
        # le pilote range les voies dans l'ordre croissant, sans les entrées
        # moins (4 à 7) des modules passés en différentiel
        rangees = [v for v in range(8) if v in voies and not (v >= 4 and v - 4 in diff)]
        if len(rangees) != len(calibres):
            raise ValueError("Erreur : le nombre de calibres ne correspond pas au nombre de voies")
        for i, code in enumerate(codes):
            self._codes_calibre[i] = code
        self._voies = rangees
        self._calibres = [_CALIBRE_DU_CODE[code] for code in codes]
        self._diff = diff
        multiplexe = any(m in rangees and m + 4 in voies and m not in diff for m in range(4))
        self._temin_us = 2.0 if multiplexe else 0.1
        if self.verbose:
            print(f"[SYSAM] Entrées EA{rangees} : calibres {self._calibres} V, différentiel {diff}")

    def config_echantillon(self, techant, nbpoints):
        """`techant` en microsecondes, comme pycanum."""
        self._configurer_echantillonnage(techant, nbpoints, "memoire")

    def config_echantillon_permanent(self, techant, nbpoints):
        self._configurer_echantillonnage(techant, nbpoints, "permanent")

    def _configurer_echantillonnage(self, techant, nbpoints, mode):
        self._exiger_ouvert()
        if self._temin_us == 0:
            raise ValueError("Erreur : entrees non configurees")
        if np.float32(techant) < self._temin_us:
            raise ValueError(
                f"Erreur : periode d'echantillonnage trop faible ({techant} µs, minimum {self._temin_us} µs)"
            )
        nbpoints = int(nbpoints)
        self._te_us = _periode_us(techant)
        nb_voies = len(self._voies)
        if mode == "memoire":
            par_paquet = _points_par_paquet(self._te_us, nb_voies)
            points = math.ceil(nbpoints / par_paquet) * par_paquet
            while points > MEMOIRE // nb_voies:
                points -= par_paquet
        else:
            points = nbpoints
        self._points_demandes = nbpoints
        self._points = points
        self._ram_entrees = points * nb_voies
        self._echantillonnage = mode
        if self.verbose:
            print(f"[SYSAM] Échantillonnage : {self._te_us:.1f} µs, {min(points, nbpoints)} points")

    def config_quantification(self, quantification):
        if 1 <= quantification <= 12:
            self._bits = int(quantification)
            if self.verbose:
                print(f"[SYSAM] Quantification sur {quantification} bits")

    def config_trigger(self, voie, seuil, montant=1, pretrigger=1, pretriggerSouple=0, hysteresis=0):
        self._exiger_ouvert()
        if voie == -1:
            if self.verbose:
                print("[SYSAM] Déclenchement désactivé")
            return
        if not self._voies:
            raise ValueError("Erreur : aucune voie configuree")
        if voie not in self._voies:
            raise ValueError("Erreur : voie de trigger non configuree")
        # le pilote convertit le seuil avec le calibre rangé à la position `voie`,
        # pas forcément celui de la voie : le seuil réellement appliqué
        calibre_lu = _CALIBRE_DU_CODE[self._codes_calibre[voie]]
        calibre_voie = self._calibres[self._voies.index(voie)]
        code = int((seuil / calibre_lu + 1.0) * 2048)
        self.seuil_applique = (code / 2048 - 1.0) * calibre_voie
        if self.verbose:
            print(
                f"[SYSAM] Déclenchement sur EA{voie}, seuil {self.seuil_applique:.3g} V, front "
                f"{'montant' if montant else 'descendant'}, {pretrigger} points avant"
                f"{', souple' if pretriggerSouple else ''}{', hystérésis' if hysteresis else ''}"
            )

    def config_trigger_externe(self, pretrigger=1, pretriggerSouple=0):
        self._exiger_ouvert()
        if self.verbose:
            souple = ", souple" if pretriggerSouple else ""
            print(f"[SYSAM] Déclenchement externe, {pretrigger} points avant{souple}")

    def config_filtre(self, listeA, listeB):
        raise _non_simule("le filtrage numérique des entrées")

    # --- acquisitions
    def acquerir(self):
        self._verifier_acquisition()
        if self._echantillonnage != "memoire":
            raise ValueError("Erreur : echantillonnage non configure")
        self._verifier_memoire()
        self._acquerir(attendre=True)

    def lancer(self):
        self._verifier_acquisition()
        self._verifier_memoire()
        self._acquerir(attendre=False)

    def acquerir_avec_sorties(self, valeurs1, valeurs2):
        self._avec_sorties(valeurs1, valeurs2, attendre=True)

    def lancer_avec_sorties(self, valeurs1, valeurs2):
        self._avec_sorties(valeurs1, valeurs2, attendre=False)

    def _avec_sorties(self, valeurs1, valeurs2, attendre):
        # pycanum remplace tout ce qui n'est pas un ndarray par « pas de sortie »
        sorties = [v if isinstance(v, np.ndarray) else np.zeros(0) for v in (valeurs1, valeurs2)]
        for n, valeurs in enumerate(sorties, start=1):
            if valeurs.dtype not in (np.float64, np.float32) or valeurs.ndim != 1:
                raise ValueError(f"erreur de format de l'echantillon {n}")
        self._verifier_acquisition()
        for n, valeurs in enumerate(sorties, start=1):
            if valeurs.size:
                self.config_sortie(n, self._te_us, valeurs, repetition=1)
        te_sorties = max(math.floor(self._te_us * 5) - 1, 0) / 5 + 0.2
        if any(v.size for v in sorties) and abs(te_sorties - self._te_us) > 1e-9:
            print(
                f"[SYSAM] ATTENTION : les sorties tournent à {te_sorties:.1f} µs, les entrées à "
                f"{self._te_us:.1f} µs : la centrale les désynchronise"
            )
        self._verifier_memoire()
        if self.verbose:
            print("[SYSAM] Génération des sorties et acquisition synchrone...")
        self._acquerir(attendre)

    def _verifier_acquisition(self):
        self._exiger_ouvert()
        if not self._voies:
            raise ValueError("Erreur : entrees non configurees")
        if self._te_us == 0:
            raise ValueError("Erreur : echantillonnage non configure")

    def _verifier_memoire(self):
        if self._ram_entrees + sum(self._ram_sorties) > MEMOIRE:
            raise ValueError(
                "Erreur : memoire insuffisante, trop de points en entree et/ou en sortie "
                f"({self._ram_entrees} pour les entrées + {sum(self._ram_sorties)} pour les sorties "
                f"> {MEMOIRE})"
            )

    def _acquerir(self, attendre):
        if attendre:
            if self.verbose:
                print("[SYSAM] Acquisition...")
            time.sleep(self._te_us * self._points * 1e-6)
        t = np.arange(self._points) * self._te_us * 1e-6
        temps = np.tile(t, (len(self._voies), 1))
        entrees = np.empty((len(self._voies), self._points))
        for i, calibre in enumerate(self._calibres):
            # un bruit gaussien d'un pas de quantification, arrondi au pas et écrêté au calibre
            pas = 2 * calibre / 2**self._bits
            bruit = _rng.normal(0, pas, self._points)
            entrees[i] = np.clip(np.round(bruit / pas) * pas, -calibre, calibre)
        self._donnees = (temps, entrees)
        if attendre and self.verbose:
            print("[SYSAM] Terminée.")

    def acquerir_permanent(self):
        raise _non_simule("l'acquisition permanente")

    def lancer_permanent(self, repetition=0):
        raise _non_simule("l'acquisition permanente")

    def stopper_acquisition(self):
        self._exiger_ouvert()

    # --- lecture des données
    def _lire(self, reduction, indice):
        if self._donnees is None:
            raise ValueError("aucune acquisition effectuée")
        points = min(self._points_demandes, self._points) // int(reduction)
        return self._donnees[indice][:, ::reduction][:, :points]

    def temps(self, reduction=1):
        return self._lire(reduction, 0)

    def entrees(self, reduction=1):
        return self._lire(reduction, 1)

    def entrees_filtrees(self, reduction=1):
        raise _non_simule("le filtrage numérique des entrées")

    def nombre_echant(self):
        return 0 if self._donnees is None else self._points

    def paquet(self, premier, reduction=1):
        raise _non_simule("la lecture par paquets")

    def paquet_filtrees(self, premier, reduction=1):
        raise _non_simule("la lecture par paquets")

    # --- sorties
    def config_sortie(self, nsortie, techant, valeurs, repetition=0):
        """`techant` en microsecondes, comme pycanum."""
        self._exiger_ouvert()
        if nsortie not in (1, 2):
            raise ValueError("Erreur : numero de sortie incorrect")
        if techant < TE_MIN_SORTIE_US:
            raise ValueError(
                "Erreur : periode d'echantillonnage de sortie trop faible (min 0.2 microsecondes)"
            )
        if len(valeurs) > POINTS_SORTIE_MAX:
            raise ValueError("Erreur : nombre de points trop grand en sortie (max 131071)")
        self._ram_sorties[nsortie - 1] = len(valeurs)

    def declencher_sorties(self, ns1, ns2):
        self._exiger_ouvert()

    def stopper_sorties(self, ns1, ns2):
        self._exiger_ouvert()

    def ecrire(self, ns1, valeur1, ns2, valeur2):
        self._exiger_ouvert()

    # --- entrées et sorties logiques, lecture directe, compteur, chronomètre
    def activer_lecture(self, voies):
        raise _non_simule("la lecture directe des entrées")

    def desactiver_lecture(self):
        raise _non_simule("la lecture directe des entrées")

    def lire(self):
        raise _non_simule("la lecture directe des entrées")

    def portC_config(self, bit, etat):
        self._exiger_ouvert()

    def portC_ecrire(self, bit, etat):
        self._exiger_ouvert()

    def portC_lire(self, bit):
        self._exiger_ouvert()
        return 0

    def portB_config(self, bit, etat):
        self._exiger_ouvert()

    def portB_ecrire(self, bit, etat):
        self._exiger_ouvert()

    def portB_lire(self, bit):
        self._exiger_ouvert()
        return 0

    def config_compteur(self, entree, front_montant, front_descend, hysteresis, duree):
        raise _non_simule("le compteur")

    def compteur(self):
        raise _non_simule("le compteur")

    def lire_compteur(self):
        raise _non_simule("le compteur")

    def config_chrono(self, entree, front_debut, front_fin, hysteresis):
        raise _non_simule("le chronomètre")

    def chrono(self):
        raise _non_simule("le chronomètre")

    def lire_chrono(self):
        raise _non_simule("le chronomètre")

    def afficher_calibrage(self):
        print(f"[SYSAM] Voies EA{self._voies} : calibres {self._calibres} V, différentiel {self._diff}")
