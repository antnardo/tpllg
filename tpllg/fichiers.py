# -*- coding: utf-8 -*-
"""
Lecture de fichiers de mesures : exports Latis Pro et Regressi, et un lecteur
CSV générique qui reconnaît seul le délimiteur et la virgule décimale
(readcsv, venu de dataanalysis, 2018).

@author: a. marchand
"""
from pathlib import Path
import csv
import io
import sys

import numpy as np

REGRESSI_HEADER = 3
LATIS_HEADER = 1

def import_latispro(filename, colonnes=2, delimiter=";"):
    """
    dans latispro : Menu Fichier>Exporter>CSV
    Glisser-Déplacer des Courbes depuis Courbes disponibles dans Courbes à exporter
    """
    if colonnes <= 0:
        raise ValueError('Le nombre de colonnes doit être > 0')
    cols = [[] for _ in range(colonnes)]
    filename = Path(filename)
    if not filename.exists():
        raise ValueError(f"Le fichier {filename} n'existe pas : vérifiez le dossier d'exécution (actuellement {Path.cwd()})")
    with open(filename, encoding='iso8859') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        for i, row in enumerate(reader):
            if i >= LATIS_HEADER:
                for j, col in enumerate(cols):
                    try:
                        col.append(float(row[j].replace(',', '.')))
                    except ValueError:
                        col.append(np.nan)
    npcols = [np.array(col) for col in cols]
    for i, col in enumerate(npcols):
        if np.isnan(col).any():
            print(f"WARNING: la colonne {i} contient des valeurs non numériques")
    return npcols

def import_regressi(filename, colonnes=2, delimiter="\t"):
    """
    dans regressi : save as... csv > real csv checked
    temps en première colonne
    """
    if colonnes < 0:
        raise ValueError('Le nombre de colonnes doit être >= 0')
    t = []
    cols = [[] for _ in range(colonnes)]
    filename = Path(filename)
    if not filename.exists():
        raise ValueError(f"Le fichier {filename} n'existe pas : vérifiez le dossier d'exécution (actuellement {Path.cwd()})")
    with open(filename, encoding='utf8') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        for i, row in enumerate(reader):
            if i >= REGRESSI_HEADER:
                try:
                    t.append(float(row[0]))
                except ValueError:
                    t.append(np.nan)
                for j, col in enumerate(cols, start=1):
                    try:
                        col.append(float(row[j]))
                    except ValueError:
                        col.append(np.nan)
    t = np.array(t)
    if np.isnan(t).any():
        print("WARNING: la colonne de temps (0) contient des valeurs non numériques")
    npcols = [np.array(col) for col in cols]
    for i, col in enumerate(npcols):
        if np.isnan(col).any():
            print(f"WARNING: la colonne {i+1} contient des valeurs non numériques")
    return t, npcols


def readcsv(filename, encoding='utf8', entete=1, dtypes=None):
    """
    Lecture d'un fichier CSV.
    Args:
        filename : nom complet du fichier
        encoding : 'utf8' par défaut, sinon 'latin1' pour windows (ou 'iso8859_15') ou 'mac_roman'
        entete : nombre de ligne en début de fichier à écarter
        dtypes : liste de types exemple : [float, int, int]. Doit faire le nombre de colonnes. Par défaut ce sont des float
    Return:
        Liste de numpy array correspondant au colonnes du fichier

    Le délimiteur est reconnu automatiquement, le caractère de fin de ligne aussi.
    Si c'est une virgule qui est utilisée comme format de flottant, remplacé par un point.

    """
    print('Lecture de {:s}'.format(filename), flush=True)
    # If newline='' is not specified, newlines embedded inside quoted fields will not be interpreted correctly, and on
    # platforms that use \r\n linendings on write an extra \r will be added. It should always be safe to specify
    # newline='', since the csv module does its own (universal) newline handling.
    with io.open(filename, encoding=encoding, newline='') as csvfile:
        dialect = csv.Sniffer().sniff(csvfile.read(1024))  # find delimiter automatically

        csvfile.seek(0)  # retourne au début
        firstrow = csv.reader(csvfile, dialect).__next__()
        cols = len(firstrow)
        print('{:d} colonnes'.format(cols), flush=True)
        T = []
        for _ in range(cols):
            T.append([])
        if dtypes is None:
            dtypes = [float]*cols  # default dtypes
        else:
            assert sum([1 for dt in dtypes if dt in [float, int, str]]) == cols
        print('Formats de conversion : ', dtypes, flush=True)

        csvfile.seek(0)  # retourne au début
        reader = csv.reader(csvfile, dialect)
        j = -1
        try:
            for i, row in enumerate(reader):
                if i < entete:
                    print("Entete exclus : ", row)
                else:
                    for j, element in enumerate(row):
                        T[j].append(
                            dtypes[j](fpointformat(element, dtypes[j])))
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            raise ValueError('file {}, line {}, column {} : {}'.format(
                filename, reader.line_num, j+1, e))

    A = []
    for i in range(cols):
        A.append(np.array(T[i]))
    return A


OTHERPOINT = ','
PYTHONPOINT = '.'


def fpointformat(s, dtype):
    if dtype is not str:
        if PYTHONPOINT in s and OTHERPOINT not in s:
            return s
        elif PYTHONPOINT not in s and OTHERPOINT not in s:
            return s
        elif PYTHONPOINT not in s and OTHERPOINT in s:
            return s.replace(OTHERPOINT, PYTHONPOINT)
        else:
            # point in s and otherpoint in s
            raise ValueError
    else:
        return s

