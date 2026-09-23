"""Les tests passent toujours par la centrale simulée : pycanum est rendu
introuvable avant tout import de tpllg.sysam, même sur un poste qui l'a."""
import sys

sys.modules["pycanum"] = None
sys.modules["pycanum.main"] = None
