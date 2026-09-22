# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 13:35:03 2023

@author: a. marchand

Classe enfant de la classe pycanum pour simplifier l'accès.

Docstrings détaillées

Pour plus de détails :
https://www.f-legrand.fr/scidoc/docmml/sciphys/caneurosmart/interpy/interpy.html
"""
import numpy as np

try:
    import pycanum.main as pycan
except ImportError:  # pas de pycanum sur cette machine : centrale simulée
    from tpllg import sysam_factice as pycan

SYSAM_TYPE = "SP5"  # et non "PCI", qui n'existe pas au lycée
CAL_DEFAUT = 10  # calibre par défaut
MICROSECONDES = 1e6


class Sysam(pycan.Sysam):
    """
    Classe enfant de la classe proposée par pycanum pour régler quelques bugs
    et proposer une syntaxe un peu plus friendly et pythonique.
    Avec des détails techniques et des docstring.

    Toutes les classes existantes sont évidemment les mêmes.
    Sont changées :
        l'initialisation: on peut directement y mettre la
        configuration des voies (optionnel)
        ex: Sysam(voies=[0, 1])
        et pas besoin de mettre 'SP5', par défaut.
        
        le choix du calibre : si on ne met qu'une valeur,
        elle s'applique à toutes les voies

        les fonctions acquerir et acquerir_avec_sorties
        renvoient directement les valeurs

        On peut utiliser with, pas besoin d'ouvrir et de fermer manuellement
        Exemple minimal :

        with Sysam(voies=[0, 1]) as can:
            temps, entrees = can.acquerir()

    Caractéristiques techniques :
    Elle se compose de 4 modules de conversion analogique-numérique 12 bits
    avec un temps de conversion minimal de 100ns (10MHz), d'un module double
    sortie numérique-analogique 12 bits 200ns (5MHz), associés à une interface
    de 16 lignes d’entrées/sorties logiques.

    CAN 1 à 4 voies simples ou différentielles 10MHz
    CAN 5 à 8 voies simples ou différentielles 5MHz
    CNA 5MHz 12 bits ±10V 50mA

    2 modes de CAN : Direct (10MHz) ou multiplexé (500KHz)
    Pour travailler en mode Direct, il est nécessaire que chaque module de
    conversion travaille en mode différentiel, ou s’il travaille en mode simple,
    qu’une seule de ses entrées soit activée. En effet, si les deux entrées
    d’un des 4 modules de conversion sont activées simultanément en mode simple,
    la centrale propose alors un fonctionnement en mode multiplexé.
    MODULE0 = EA0+EA4
    MODULE1 = EA1+EA5
    etc

    RAM 512ko
    Nb de points max par acquisition = 2**18 = 262144
    
    12 bits ±1LSB, binaire naturel
    Non linéarité : ±1LSB sur la pleine échelle ±10V
    Calibre 0.1, 0.2, 1, 2, 5, 10 (ATTENTION, pycanum n'a pas accès à tous)
    Impédance entrée 1 Mohm

    """
    TE_MIN_SORTIE = 2e-7
    TE_MIN_DIRECT = 1e-7
    TE_MIN_MULTIPLEX = 2e-6
    N_MAX = 262_144
    CALIBRES = [.2, 1, 5, 10]
    MODULES_ANALOG = {0: (0, 4), 1: (1, 5), 2: (2, 6), 3: (3, 7)}

    @classmethod
    def get_calibre(cls, valeur):
        """pour info, a priori inutile c'est déjà codé en dur (lignes 226 à 235 de SysamSP5Link.c)"""
        if valeur > max(cls.CALIBRES):
            return max(cls.CALIBRES)
        for cal in cls.CALIBRES:
            if cal >= valeur:
                return cal
    
    @classmethod
    def te_min(cls, voies):
        """détermine si on fonctionne en mode direct ou multiplexé
        si les deux entrées d’un des 4 modules de conversion sont activées
        simultanément en mode simple
        """
        modules = [0]*len(cls.MODULES_ANALOG)
        for v in voies:
            for k, EA in cls.MODULES_ANALOG:
                if v in EA:
                    modules[k] += 1
        return cls.TE_MIN_DIRECT if max(modules) <= 1 else cls.TE_MIN_MULTIPLEX

    def __init__(self, voies=None, calibres=None, diff=None):
        super().__init__(SYSAM_TYPE)
        if voies is None:
            return
        self.config_entrees(voies, calibres, diff)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context related to this object.
        The parameters describe the exception that caused the context to be
        exited.
        If the context was exited without an exception, all three arguments
        will be None.
        If an exception is supplied, and the method wishes to suppress the
        exception (i.e., prevent it from being propagated), it should return
        a true value.
        Otherwise, the exception will be processed normally upon exit
        from this method.
        Note that __exit__() methods should not reraise the passed-in
        exception; this is the caller’s responsibility.
        """
        self.fermer()
        return False

    def config_entrees(self, voies, calibres=None, diff=None):
        """Sélection des entrées analogiques et configuration du calibre (gain
        de l'amplificateur d'entrée).

            voies (list): liste des entrées analogiques à sélectionner,
                numérotées de 0 à 7.

            calibres (list): liste des valeurs absolues maximales des
                tensions (en volts), une pour chaque voie sélectionnée.
            [ALT] calibres (float) : le même calibre pour chaque voie
            [DEFAUT] 10V par voie
            seuls les calibres 10, 5, 1 et 0.2 volts sont accessibles
            pour les entrées analogiques

            diff (list): argument optionnel, liste des voies en
                mode différentiel.

        Sur SP5, l faut sélectionner les entrées 0,1,2 et 3 pour bénéficier de
        la fréquence d'échantillonnage maximale (10 MHz).

        Sur SysamSP5, chaque canal (0-4,1-5,2-6,3-7) peut être placé en mode
        différentiel indépendamment des autres. Pour placer le premier et le
        deuxième canal en mode différentiel, il faut affecter [0,1]
        au dernier argument (diff).

        En principe la carte accepte les valeurs 0.1, 0.2, 1, 2, 5, 10
        Mais l'interface pycanum ne prend en compte que les valeurs 0.2, 1, 5, 10
        Si le calibre n'est pas parmi ces valeurs, la valeur immédiatement
        supérieure est sélectionnée automatiquement (avec un seuil à 10)
        """
        if diff is None:
            diff = []
        if calibres is None:
            calibres = CAL_DEFAUT
        if not hasattr(calibres, '__len__'):
            calibres = [calibres]*len(voies)
        calibres = [float(c) for c in np.nan_to_num(calibres, nan=CAL_DEFAUT)]
        calibres = [c if c > 0 else CAL_DEFAUT for c in calibres]
        super().config_entrees(voies, calibres, diff)

    def config_echantillon(self, techant, nbpoints):
        """Configuration de la période d'échantillonnage et du nombre de
        points à acquérir.

            techant: période d'échantillonnage en secondes (et non
                microsecondes)
                MIN = 1e-7 pour les entrées 1 à 4
            nbpoints: nombre de points à acquérir
        """
        return super().config_echantillon(techant*MICROSECONDES, nbpoints)

    def config_quantification(self, quantification):
        """Configuration du nombre de bits de la quantification.

            nbits (int<=12): nombre de bits utilisés pour la quantification.
        """
        return super().config_quantification(quantification)

    def acquerir(self, reduction=1):
        """Acquisition et récupération des données

            reduction (integer): facteur de réduction de la fréquence
                d'échantillonnage.

        Renvoit une fois l'acquisition finie :
            temps (ndarray): tableau ndarray (numpy). Chaque ligne du
                tableau fournit les temps (en s) échantillonnés de la
                voie correspondante.
            tableau ndaray (numpy): Chaque ligne du tableau fournit
                les tensions (en V) de la voie correspondante.
        """
        super().acquerir()  # ne sort que quand c'est fini
        return self.temps(reduction), self.entrees(reduction)
    
    def acquerir_avec_sorties(self, sortie1=0, sortie2=0):
        """Acquisition avec une utilisation simultanée et synchrone des
        sorties.
        La période d'échantillonnage des sorties est la même que pour les
        entrées.

            sortie1 (ndarray): tableau ndarray (numpy) fournissant le signal
            échantillonné à appliquer sur la sortie 1 (en volts).
            sortie2 (ndarray): idem, sortie 2
            les sorties peuvent être des entiers = valeur constante (par
            exemple 0)
            par défaut, c'est 0.

        Renvoit une fois l'acquisition finie :
            temps (ndarray): tableau ndarray (numpy). Chaque ligne du
                tableau fournit les temps (en s) échantillonnés de la
                voie correspondante.
            tableau ndaray (numpy): Chaque ligne du tableau fournit
                les tensions (en V) de la voie correspondante.
        
        Le nombre de points total utilisable pour les entrées et sorties
        est limité par la mémoire RAM du SysamSP5. Il est de 0x3FFFF,
        soit 262142.
        """
        super().acquerir_avec_sorties(sortie1, sortie2)
        return self.temps(), self.entrees()