import numpy as np

def indices_plages(freq, fondamental, delta_freq):
    """
    freq est l'array des fréquences, régulièrement espacée
    fourier (optionnel), si donné est l'array de la fft : sert à calculer le fondamental si on ne le connait pas
    fondamental (optionnel) : valeur de la fréquence du fondamental
    delta_freq (optionnel) : valeur des largeurs de frequences dans lesquelles chercher (attention aux décalages)

    soit on donne fourier, soit la valeur de la fréquence du fondamental

    Renvoit indices_bords = [(début, fin), (début, fin), ...] de la liste freq
    """
    indice_fondamental = np.argmax(freq >= fondamental)
    df = freq[1] - freq[0]
    n = len(freq)
    delta_indice = int(delta_freq/df/2)
    imax = int(n/indice_fondamental)
    return [(indice_fondamental*i-delta_indice, min(indice_fondamental*i+delta_indice, n)) for i in range(1, imax)]

def detecte_maxima_secondaires(valeurs, indices_bords, seuil=.1):
    """
    Il existe from scipy.signal import find_peaks, mais compliqué à tuner

    On cherche les max absolus dans plusieurs plages de valeurs déjà trouvées
    et données par indices_bords = [(début, fin), (début, fin), ...]
    seulement si la valeur dépasse un certain seuil

    indices_bords est renvoyé par indices_plages(freq, fondamental, delta_freq)
    
    Renvoit les indices de ces maxima secondaires
    """
    indices = []
    for debut, fin in indices_bords:
        valeurs_secondaires = valeurs[debut:fin]
        i = np.argmax(valeurs_secondaires)
        if valeurs_secondaires[i] > seuil:
            indices.append(debut+i)
    return indices

def valeurs_correspondantes(indexes1, indexes2, delta_indices):
    """
    Deux liste d'indices croissant
    Si les indices correspondent à delta_indices près, ils "correspondent"
    Renvoit les deux listes des indices qui correspondent exactement un à un, sans trou
    """
    i, j = 0, 0
    indices_final1 = []
    indices_final2 = []
    while i < len(indexes1) and j < len(indexes2):
        index1 = indexes1[i]
        index2 = indexes2[j]
        if abs(index1-index2) <= delta_indices:
            # c'est le même, on enregistre et on avance
            indices_final1.append(index1)
            indices_final2.append(index2)
            i += 1
            j += 1
        elif index1 < index2:
            # il manque un 2, on avance sur 1
            i += 1
        else:
            # il manque un 1, on avance sur 2
            j += 1
    return np.array(indices_final1), np.array(indices_final2)


def interpolation_fft(x, n_interpolation):
    """
    on fait la fft du signal, on rajoute N*n_interpolation zéros aux hautes fréquences
    on fait la fft inverse, qui contient donc le signal avec N*(ninter+1) points
    """
    N = len(x)
    tfd = np.fft.fft(x)
    N1 = N // 2
    tfd2 = np.concatenate(
        (tfd[0:N1], np.zeros(N * n_interpolation), tfd[N1:N])
    )
    y = np.real(np.fft.ifft(tfd2)) * (n_interpolation + 1)
    return y

def gain(t, e, s, freq, Np, method, **kwargs):
    """Mesure du gain selon différentes méthodes.
    Il faut a priori une idée de la période selon la méthode utilisée.

    Retourne G, phi, H avec H = G exp(j phi)
    """
    if method == 'std':
        return gain_std(t, e, s, Np, **kwargs)
    elif method == 'fit':
        return NotImplemented

def gain_std(t, e, s, Np=0, ninter=0):
    """Mesure du gain complexe H entre e et s

    Méthode due à Frédéric Legrand (f-legrand.fr, analyse fréquentielle à la
    Sysam), réécrite ici.

    Méthode : on mesure les valeurs efficaces et le déphasage par moyennage entre
    les deux signaux puisque
        <cos(omega t + phi) * exp(j omega t)> = 1/2*exp(j phi)

    Interpolation possible si ninter > 0:
        on fait la fft du signal, on rajoute N*ninter zéros aux hautes fréquences
        on fait la fft inverse, qui contient donc le signal avec N*(ninter+1) points

        ninter : facteur d'interpolation entier : multiplie le nombre de points
            (defaut: 0)

    Renvoit G, phi, H avec H = G exp(j phi)

    Il faut que le signal fasse un grand nombre de périodes
    """
    if ninter > 0:
        # lourd en calculs (O(N^2))
        e = interpolation_fft(e, ninter)
        s = interpolation_fft(s, ninter)
        # t = np.arange(len(e)) * t[len(t) - 1] / len(e)
    N = len(e)
    E, S = e.std(), s.std()
    G = S / E
    d = int(Np * (ninter + 1) / 4)  # 1/4 de période (pi/2) en nombre de points
    z = s[d:N] * (e[d:N] - 1j * e[0:N - d])  # = G * A0**2 cos(omega t + phi) * exp(j omega t)
    Z = z.mean()  # = G * A0**2/2 * exp(j phi)
    phi = np.angle(Z)
    # H = Z / E ** 2 si on veut...
    return G, phi

def choix_echantillonnage(freq, temin, Npmin, permin, Nmax, Tmax):
    """
    On veut
    un nombre minimal de points par période Npmin
    que techant ne soit pas en-dessous de temin
    que le temps total d'acquisition ne dépasse pas Tmax

    On vérifie
    un nombre minimal de périodes acquises permin
    que N ne dépasse pas Nmax

    Retourne : techant (en s) et n, le nb de points
    """
    Np = min(Npmin, 1 / (temin * freq))
    techant = int(1 / (Np * freq * temin)) * temin  # c'est de totue façon un multiple de temin
    if techant == 0:
        techant = temin
    n = min(Nmax, int(Tmax/techant))
    per = 1/(freq*techant)  # nb de périodes
    if per < permin:
        print(f'[WARNING] : nb de périodes faible {per:.1f}<{permin}')
    if n < Npmin:
        print(f"[WARNING] : nb de points d'acquisition faible {n}")
    return techant, n
