# -*- coding: utf-8 -*-
"""
Sysam factice : la même interface que pycanum.main, sans centrale.

tpllg.sysam s'y rabat quand pycanum n'est pas installé — sur un Mac, sur un
poste sans carte — et le dit à l'ouverture. Les fonctions de configuration ne
font qu'imprimer ce qu'elles reçoivent ; les acquisitions attendent la durée
demandée et rendent, dans la forme exacte de pycanum (tableaux 2D de double,
une ligne par voie, temps en secondes), un bruit de quantification.

Écrit d'après pycanum de Frédéric Legrand :
https://www.f-legrand.fr/scidoc/docmml/sciphys/caneurosmart/interpy/interpy.html
"""
import numpy
import time


SYSAM_SP5 = 1
SYSAM_PCI = 2
ENTREE_CHRONO = 16

VERBOSE = True


class Sysam:
    def __init__(self, nom):
        print("[SYSAM] ATTENTION : pycanum absent, centrale simulée (tpllg.sysam_factice)")
        self.ouvert = False
        self.sysamid = 0
        if nom == "SP5":
            self.sysamid = SYSAM_SP5
        if nom == "PCI":
            self.sysamid = SYSAM_PCI
        self.ouvrir()
        self.verbose = VERBOSE

    def ouvrir(self):
        if self.ouvert:
            if self.verbose:
                print(f"[SYSAM] CAN Sysam ouvert ({self.sysamid})")
            return
        if self.sysamid == 0:
            if self.verbose:
                print("[SYSAM] CAN Sysam non connu")
            return
        self.reset()
        # sysam.can_ouvrir(self.sysamid)
        self.ouvert = True

    def fermer(self):
        # sysam.can_fermer()
        self.ouvert = False

    def reset(self):
        # sysam.can_reset()
        self.voies = None
        self.calibres = None
        self.diff = None
        self.techant = None
        self.nbpoints = None

    def config_entrees(self, voies, calibres, diff=[]):
        # sysam.can_config_entrees(voies, calibres, diff)
        if self.verbose:
            print("[SYSAM] Configuration entrées %s : calibres=%s, diff=%s" % (voies, calibres, diff))
        self.voies = voies
        self.calibres = calibres
        self.diff = diff

    def config_echantillon(self, techant, nbpoints):
        # sysam.can_config_echantillon(techant, nbpoints)
        if self.verbose:
            print("[SYSAM] Configuration échantillonnage techant=%.1eµs, nbpoints=%d" % (techant, nbpoints))
        self.techant = techant
        self.nbpoints = nbpoints

    def config_echantillon_permanent(self, techant, nbpoints):
        # sysam.can_config_echantillon_permanent(techant, nbpoints)
        NotImplemented

    def config_quantification(self, quantification):
        # sysam.can_config_quantification(quantification)
        if self.verbose:
            print(f"[SYSAM] Quantification sur {quantification} bits")

    def config_trigger(
        self, voie, seuil, montant=1, pretrigger=1, pretriggerSouple=0, hysteresis=0
    ):
        # sysam.can_config_trigger(
        #    0, voie, seuil, montant, pretrigger, pretriggerSouple, hysteresis
        #)
        if self.verbose:
            if voie < 0:
                print("[SYSAM] Déclenchement désactivé")
            else:
                print(f"[SYSAM] Déclenchement sur EA{voie}, seuil {seuil} V, front "
                      f"{'montant' if montant else 'descendant'}, {pretrigger} points avant"
                      f"{', souple' if pretriggerSouple else ''}"
                      f"{', hystérésis' if hysteresis else ''}")

    def config_trigger_externe(self, pretrigger=1, pretriggerSouple=0):
        # sysam.can_config_trigger(1, 0, 0, 1, pretrigger, pretriggerSouple, 0)
        if self.verbose:
            print(f"[SYSAM] Déclenchement externe, {pretrigger} points avant"
                  f"{', souple' if pretriggerSouple else ''}")

    def acquerir(self):
        if self.verbose:
            print("[SYSAM] Acquisition...")
        time.sleep(self.techant*self.nbpoints/1e6)
        # sysam.can_acquerir()
        if self.verbose:
            print("[SYSAM] Terminée.")

    def acquerir_permanent(self):
        # sysam.can_acquerir_permanent()
        NotImplemented

    def lancer(self):
        # sysam.can_lancer()
        NotImplemented

    def lancer_permanent(self, repetition=0):
        # sysam.can_lancer_permanent(repetition)
        NotImplemented

    def stopper_acquisition(self):
        # sysam.can_stopper_acquisition()
        NotImplemented

    def temps(self, reduction=1):
        # return sysam.can_temps(reduction)
        # comme pycanum : un tableau 2D de double, une ligne par voie, en secondes
        # (techant est en microsecondes), un point sur `reduction`
        t = numpy.arange(self.nbpoints)*self.techant*1e-6
        return numpy.tile(t[::reduction], (len(self.voies), 1))

    def entrees(self, reduction=1):
        # return sysam.can_entrees(reduction)
        # comme pycanum : un tableau 2D de double, une ligne par voie, en volts ;
        # ici un bruit gaussien d'un pas de quantification, arrondi au pas
        entrees = numpy.zeros((len(self.voies), self.nbpoints))
        for i, c in enumerate(self.calibres):
            pas = 2*c/2**12
            bruit = numpy.random.normal(0, pas, self.nbpoints)
            entrees[i] = numpy.round(bruit/pas)*pas
        return entrees[:, ::reduction]

    def entrees_filtrees(self, reduction=1):
        # return sysam.can_entrees_filtrees(reduction)
        NotImplemented

    def nombre_echant(self):
        # return sysam.can_nombre_echant()
        return self.nbpoints

    def paquet(self, premier, reduction=1):
        # if premier >= 0:
        #     return sysam.can_paquet(premier, reduction)
        # else:
        #     return sysam.can_paquet_circulaire(reduction)
        NotImplemented

    def paquet_filtrees(self, premier, reduction=1):
        # if premier >= 0:
        #     return sysam.can_paquet_filtrees(premier, reduction)
        # else:
        #     return sysam.can_paquet_circulaire_filtrees(reduction)
        NotImplemented

    def config_sortie(self, nsortie, techant, valeurs, repetition=0):
        # sysam.can_config_sortie(nsortie, techant, valeurs, repetition)
        NotImplemented

    def declencher_sorties(self, ns1, ns2):
        # sysam.can_declencher_sorties(ns1, ns2)
        NotImplemented

    def stopper_sorties(self, ns1, ns2):
        # sysam.can_stopper_sorties(ns1, ns2)
        NotImplemented

    def acquerir_avec_sorties(self, valeurs1, valeurs2):
        if type(valeurs1) != numpy.ndarray:
            valeurs1 = numpy.zeros(0)
        if type(valeurs2) != numpy.ndarray:
            valeurs2 = numpy.zeros(0)
        if self.verbose:
            print("[SYSAM] Génération des sorties...")
            print("[SYSAM] Acquisition synchrone...")
        time.sleep(self.techant*self.nbpoints/1e6)
        # sysam.can_acquerir_avec_sorties(valeurs1, valeurs2)
        if self.verbose:
            print("[SYSAM] Terminée.")

    def lancer_avec_sorties(self, valeurs1, valeurs2):
        if type(valeurs1) != numpy.ndarray:
            valeurs1 = numpy.zeros(0)
        if type(valeurs2) != numpy.ndarray:
            valeurs2 = numpy.zeros(0)
        # sysam.can_lancer_avec_sorties(valeurs1, valeurs2)

    def ecrire(self, ns1, valeur1, ns2, valeur2):
        # sysam.can_ecrire(ns1, valeur1, ns2, valeur2)
        NotImplemented

    def activer_lecture(self, voies):
        # sysam.can_activer_lecture(voies, 0)
        NotImplemented

    def desactiver_lecture(self):
        # sysam.can_desactiver_lecture()
        NotImplemented

    def lire(self):
        #return sysam.can_lire()
        NotImplemented

    def portC_config(self, bit, etat):
        # sysam.can_portC_config(bit, etat)
        NotImplemented

    def portC_ecrire(self, bit, etat):
        # sysam.can_portC_ecrire(bit, etat)
        NotImplemented

    def portC_lire(self, bit):
        return 0 # sysam.can_portC_lire(bit)

    def portB_config(self, bit, etat):
        # sysam.can_portB_config(bit, etat)
        NotImplemented

    def portB_ecrire(self, bit, etat):
        # sysam.can_portB_ecrire(bit, etat)
        NotImplemented


    def portB_lire(self, bit):
        #return sysam.can_portB_lire(bit)
        return 0

    def config_filtre(self, listeA, listeB):
        # sysam.can_config_filtre(
        #    numpy.array(listeA, dtype=numpy.double),
        #    numpy.array(listeB, dtype=numpy.double),
        #)
        NotImplemented

    def config_compteur(self, entree, front_montant, front_descend, hysteresis, duree):
        # sysam.can_config_compteur(
        #    entree, front_montant, front_descend, hysteresis, duree
        #)
        NotImplemented

    def compteur(self):
        NotImplemented
        # sysam.can_compteur()

    def lire_compteur(self):
        return 0
        # n = sysam.can_lire_compteur()
        # return numpy.uint64(numpy.left_shift(n[0], 32) + n[1])

    def config_chrono(self, entree, front_debut, front_fin, hysteresis):
        # sysam.can_config_chrono(entree, front_debut, front_fin, hysteresis)
        NotImplemented

    def chrono(self):
        # sysam.can_chrono()
        NotImplemented

    def lire_chrono(self):
        return 0
        # n = sysam.can_lire_chrono()
        # return numpy.uint64(2 ** 32) * numpy.uint64(n[0]) + numpy.uint64(n[1])

    def afficher_calibrage(self):
        # sysam.can_afficher_calibrage()
        print("[SYSAM] Voies : %s" % (self.voies,))
        print("[SYSAM] Calibres : %s, diff=%s" % (self.calibres, self.diff))


# def audio_table_start(ch1, ch2, f1, f2, fs=44000, fpb=256):
#     ch1 = numpy.array(ch1, dtype=numpy.float32)
#     ch2 = numpy.array(ch2, dtype=numpy.float32)
#     sysam.table_dds_start(fs, fpb, ch1, ch2, f1, f2)


# def audio_table_stop(seconds):
#     sysam.table_dds_stop(seconds)


# def harmonics_table(amp_list, phase_list, normalize, gain):
#     TABLE_SIZE = 256
#     amp = numpy.array(amp_list, dtype=numpy.float32)
#     phase = numpy.array(phase_list, dtype=numpy.float32)
#     if amp.size != phase.size:
#         raise SystemExit("amplitude et phase : tailles incompatibles")
#     t = numpy.arange(TABLE_SIZE) * 1.0 / TABLE_SIZE
#     table = numpy.zeros(TABLE_SIZE, dtype=numpy.float32)
#     for n in range(amp.size):
#         table = table + amp[n] * numpy.sin(2 * numpy.pi * (n + 1) * t + phase[n])
#     table = numpy.array(table, dtype=numpy.float32)
#     m = abs(table.min())
#     M = abs(table.max())
#     if normalize:
#         table = table / max(m, M)
#     return table * gain


# def audio_harmonics_start(
#     a1, p1, a2, p2, f1, f2, norm=True, gain=1.0, fs=44000, fpb=256
# ):
#     table_1 = harmonics_table(a1, p1, norm, gain)
#     table_2 = harmonics_table(a2, p2, norm, gain)
#     sysam.table_dds_start(fs, fpb, table_1, table_2, f1, f2)
#     return (table_1, table_2)


# def audio_harmonics_stop(seconds):
#     sysam.table_dds_stop(seconds)


# class RingBuffer:
#     def __init__(self, number_blocks_exponent=4, samples_per_block=64, Id=0):
#         self.pointer = sysam.ring_buffer_init(
#             number_blocks_exponent, samples_per_block, Id
#         )
#         self.samples_per_block = samples_per_block

#     def delete(self):
#         sysam.ring_buffer_delete(self.pointer)

#     def write(self, data):
#         data = numpy.array(data, dtype=numpy.float32)
#         sysam.ring_buffer_write(self.pointer, data)


# def output_stream_start(sample_rate, ring_buffer_1, ring_buffer_2):
#     p1 = 0
#     if ring_buffer_1:
#         p1 = ring_buffer_1.pointer
#     p2 = 0
#     if ring_buffer_2:
#         p2 = ring_buffer_2.pointer
#     sysam.output_stream_start(int(sample_rate), p1, p2)


# def output_stream_stop(seconds):
#     sysam.output_stream_stop(seconds)
