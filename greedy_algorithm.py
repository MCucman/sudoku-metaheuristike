import my_parser
import numpy as np


# Racuna lokalni konflikt c(k) = r_k^2 + s_k^2 koji bi nastao postavljanjem
# vrijednosti 'value' na poziciju (block, pos) u blok-prikazu, gdje je r_k broj
# pojavljivanja te vrijednosti u odgovarajucem retku R_i standardne mreze, a
# s_k broj pojavljivanja u odgovarajucem stupcu S_j.
# Kvadriranje jace kaznjava visestruke duplikate pa pohlepni odabir preferira
# raspodjelu po razlicitim recima/stupcima.
# Vraca i stvarne koordinate (row, col) u standardnoj mrezi da izbjegnemo
# dupli izracun.
def count_conflicts(sol, block, pos, value):
    # Pretvaramo blok-indeks i poziciju unutar bloka u koordinate standardne mreže
    row = (block // 3) * 3 + (pos // 3)
    col = (block % 3) * 3 + (pos % 3)

    row_conflicts = (np.sum(sol[row, :] == value)) ** 2
    col_conflicts = (np.sum(sol[:, col] == value)) ** 2

    return row_conflicts + col_conflicts, row, col


# Pohlepni (greedy) algoritam rješava sudoku jednim prolaskom bez povratka.
# Strategija:
#   1. Sortiramo blokove po broju praznih ćelija (uzlazno). Blok s malo praznih
#      polja nudi malo kandidata za svako od njih, pa je odluka u njemu gotovo
#      prisilna i odgadanjem se ne moze poboljsati. Blokovi s mnogo praznih
#      polja imaju veliku slobodu odabira, a ta sloboda vise vrijedi kasnije,
#      kada je mreza vec popunjena i kada procjena konflikata pociva na vise
#      upisanih vrijednosti. Prisilne odluke tako dolaze prve.
#   2. Za svaku praznu ćeliju biramo broj s najmanjim lokalnim konfliktom
#      c(k) = r_k^2 + s_k^2 u retku i stupcu standardne mreže.
#   3. Odabrani broj odmah upisujemo i prelazimo na sljedeću ćeliju (bez
#      povratka), što čini algoritam determinističkim i vrlo brzim.
# Algoritam istovremeno drzi oba prikaza: standardnu mrezu 'sol', jer se
# konflikti mjere po recima i stupcima, i blok-prikaz 'block_sol', u koji se
# rjesenje upisuje i iz kojeg se citaju nedostajuce vrijednosti po bloku.
# Zbog pohlepne prirode algoritam ne garantira točno rješenje — služi kao
# brza referentna točka (baseline) za usporedbu s metaheuristikama.
def greedy_algorithm(inst: np.ndarray, transf_inst: np.ndarray) -> int:
    sol = inst.copy()               # standardna mreža, koristimo za provjeru redaka/stupaca
    block_sol = transf_inst.copy()  # blok-prikaz, ovdje upisujemo rješenje
    missing = my_parser.missing_nums(block_sol)

    # Sortiramo blokove po broju praznih ćelija (najograničeniji blok prvi)
    empty_cells = [sum(block_sol[block] == 0) for block in range(9)]
    block_order = sorted(range(9), key=lambda block: empty_cells[block])

    for block in block_order:
        for pos in range(9):
            if not block_sol[block][pos]:  # ćelija je prazna
                # Sentinel iznad svake dostizive vrijednosti. Redak standardne
                # mreze sijece tri bloka, a svaki blok sadrzi svaku vrijednost
                # najvise jednom, pa je r_k <= 3; isto vrijedi za stupac. Stoga
                # je c(k) = r_k^2 + s_k^2 <= 9 + 9 = 18.
                best_conf = 145

                for num_id, num in enumerate(missing[block]):
                    curr_conf, row, col = count_conflicts(sol, block, pos, num)

                    # Pamtimo kandidata s najmanjim konfliktom
                    if curr_conf < best_conf:
                        best_conf = curr_conf
                        best_id = num_id
                        best_row, best_col = row, col

                    # Ako je konflikt 0, bolji odabir ne postoji — odmah stajemo
                    if best_conf == 0:
                        break

                # Upisujemo odabrani broj u oba prikaza i uklanjamo ga iz liste nedostajućih
                best_num = missing[block][best_id]
                block_sol[block][pos] = best_num
                sol[best_row][best_col] = best_num
                missing[block].remove(best_num)

    return my_parser.loss_func(block_sol)
