import linecache
import numpy as np


# Notacija je usklađena s radom:
#   block             - indeks bloka 3x3 (redak u blok-prikazu)
#   pos, pos_1, pos_2 - indeks pozicije unutar bloka (stupac u blok-prikazu)
#   row, col          - redak R_i i stupac S_j standardne mreze
#   free_positions    - za svaki blok lista indeksa pozicija koje su prazne


# Transformacija između standardnog prikaza sudokua (redci = redci mreže)
# i blok-prikaza (redci = 3x3 blokovi). Reshape na (3,3,3,3), transpose(0,2,1,3)
# i reshape nazad na (9,9) zamjenjuje os blokova i os pozicija unutar bloka,
# tj. preslikava (3I+i, 3J+j) u (3I+J, 3i+j).
# Ista funkcija se koristi za kodiranje i dekodiranje jer je transformacija
# sama sebi inverzna (involutorna permutacija).
def transform(instance: np.ndarray) -> np.ndarray:
    return instance.reshape(3, 3, 3, 3).transpose(0, 2, 1, 3).reshape(9, 9)


# Čita k-ti sudoku puzzle iz sudoku.csv datoteke (redak k+2, preskačemo zaglavlje).
# Vraća trojku:
#   instance       - originalna 9x9 mreža (redci = redci sudokua)
#   transf_inst    - blok-prikaz iste mreže (redci = blokovi)
#   free_positions - za svaki blok lista indeksa pozicija koje su prazne (vrijednost 0)
def get_sudoku(k: int) -> tuple[np.ndarray, np.ndarray, list[list[int]]] | None:
    line = linecache.getline('sudoku.csv', k + 2)
    if not line:
        return None
    quiz, _ = line.split(",")

    # Parsiramo 81-znakovski string u 9x9 numpy matricu
    instance = np.array([
        [int(quiz[i*9+j]) for j in range(9)]
        for i in range(9)])

    # Transformiramo u blok-prikaz koji koriste algoritmi
    transf_inst = transform(instance)

    # Bilježimo slobodne pozicije po bloku jer algoritmi smiju mijenjati
    # samo slobodne ćelije (zadane cifre su fiksirane)
    free_positions = [[pos for pos in range(9) if transf_inst[block][pos] == 0]
                      for block in range(9)]

    return (instance, transf_inst, free_positions)


# Funkcija gubitka (loss) mjeri koliko rješenje krši pravila sudokua.
# Svaki redak i svaki stupac moraju sadržavati svaki broj od 1 do 9 točno jednom.
# Za svaki broj u svakom retku/stupcu dodajemo (count - 1)^2:
#   - ako broj nedostaje:    count=0, doprinos = 1
#   - ako se pojavljuje 1x:  count=1, doprinos = 0  (ispravno)
#   - ako se pojavljuje 2x:  count=2, doprinos = 1
#   - ako se pojavljuje 3x:  count=3, doprinos = 4
# Blok-ograničenje je uvijek zadovoljeno jer generate_sol puni svaki blok
# permutacijom 1-9, pa ga nije potrebno provjeravati.
# Posljedica tog zapisa: redak mreze sijece tri bloka, a svaki blok sadrzi svaku
# vrijednost najvise jednom, pa je count <= 3 i vece brojnosti nisu dostizive.
# Po retku je stoga sum_k (c_k - 1)^2 = sum_k c_k^2 - 9 <= 27 - 9 = 18, odakle
# za devet redaka i devet stupaca slijedi 0 <= loss <= 324.
def loss_func(transf_sol: np.ndarray) -> int:
    loss = 0
    # Dekodiramo iz blok-prikaza nazad u standardnu mrežu za provjeru redaka/stupaca
    sol = transform(transf_sol)

    r_counts = [np.bincount(sol[i], minlength=10)[1:] for i in range(9)]
    s_counts = [np.bincount(sol[:,i], minlength=10)[1:] for i in range(9)]

    for i in range(9):
        loss += np.sum((r_counts[i] - 1)**2)
        loss += np.sum((s_counts[i] - 1)**2)

    return loss


# Vraća popis brojeva koji nedostaju u svakom bloku (u blok-prikazu).
# Koristimo setdiff1d da pronađemo koji od 1-9 nisu prisutni u retku bloka.
def missing_nums(transf_inst: np.ndarray) -> list[list[int]]:
    sol = transf_inst.copy()
    missing = [list(np.setdiff1d(np.arange(1, 10), sol[block])) for block in range(9)]
    return missing


# Generira nasumično početno rješenje koje zadovoljava blok-ograničenje.
# Za svaki blok uzimamo listu nedostajućih brojeva, promiješamo je i
# popunjavamo prazne ćelije tim brojevima redom.
# Time svaki blok sadrži točno jednom svaki broj 1-9 (blok-ograničenje je ispunjeno),
# a algoritmi dalje optimiziraju redak/stupac ograničenja.
def generate_sol(transf_inst) -> np.ndarray:
    sol = transf_inst.copy()
    missing = missing_nums(sol)
    for block in range(9):
        np.random.shuffle(missing[block])
        for pos in range(9):
            if not sol[block][pos]:
                sol[block][pos] = missing[block].pop()
    return sol
