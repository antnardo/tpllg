"""Les tests passent toujours par la centrale simulée : pycanum est rendu
introuvable avant tout import de tpllg.sysam, même sur un poste qui l'a. Et
matplotlib dessine hors écran."""
import sys

import matplotlib

sys.modules["pycanum"] = None
sys.modules["pycanum.main"] = None
matplotlib.use("Agg")
