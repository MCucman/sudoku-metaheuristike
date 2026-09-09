import my_parser
import numpy as np


# Inkrementalno računa promjenu loss funkcije uzrokovanu zamjenom dviju ćelija
# unutar istog bloka, bez ponovnog računanja cijelog loss-a.
# Zamjena (pos_1, pos_2) u bloku mijenja vrijednosti u odgovarajućim
# recima i stupcima standardne mreže — pratimo samo te promjene.
#
# Tri slučaja ovisno o relativnom položaju zamijenjenih ćelija:
#   1. isti redak mreže (row_1 == row_2): mijenjaju se samo dva stupca
#   2. isti stupac mreže (col_1 == col_2): mijenjaju se samo dva retka
#   3. opći slučaj: mijenjaju se dva retka i dva stupca
#
# Za svaki zahvaćeni redak/stupac računamo razliku (count_novo)^2 - (count_staro)^2
# jer loss koristi kvadrat broja pojavljivanja minus 1.
def count_conflicts(solution, block, pos_1, pos_2, value_1, value_2) -> int:
    # Pretvaramo blok-koordinate u koordinate standardne mreže
    row_1 = (block // 3) * 3 + (pos_1 // 3)
    col_1 = (block % 3) * 3 + (pos_1 % 3)
    row_2 = (block // 3) * 3 + (pos_2 // 3)
    col_2 = (block % 3) * 3 + (pos_2 % 3)
    diff = 0

    if row_1 == row_2:
        # Ćelije su u istom retku mreže — zamjena utječe samo na stupce col_1 i col_2
        col_value_1 = np.count_nonzero(solution[:, col_1] == value_1)
        col_value_2 = np.count_nonzero(solution[:, col_1] == value_2)
        diff += (col_value_1 - 2) ** 2 - (col_value_1 - 1) ** 2 + (col_value_2) ** 2 - (col_value_2 - 1) ** 2

        col_value_3 = np.count_nonzero(solution[:, col_2] == value_1)
        col_value_4 = np.count_nonzero(solution[:, col_2] == value_2)
        diff += (col_value_3) ** 2 - (col_value_3 - 1) ** 2 + (col_value_4 - 2) ** 2 - (col_value_4 - 1) ** 2

    elif col_1 == col_2:
        # Ćelije su u istom stupcu mreže — zamjena utječe samo na retke row_1 i row_2
        row_value_1 = np.count_nonzero(solution[row_1] == value_1)
        row_value_2 = np.count_nonzero(solution[row_1] == value_2)
        diff += (row_value_1 - 2) ** 2 - (row_value_1 - 1) ** 2 + (row_value_2) ** 2 - (row_value_2 - 1) ** 2

        row_value_3 = np.count_nonzero(solution[row_2] == value_1)
        row_value_4 = np.count_nonzero(solution[row_2] == value_2)
        diff += (row_value_3) ** 2 - (row_value_3 - 1) ** 2 + (row_value_4 - 2) ** 2 - (row_value_4 - 1) ** 2

    else:
        # Opći slučaj — zamjena utječe na retke row_1, row_2 i stupce col_1, col_2
        row_value_1 = np.count_nonzero(solution[row_1] == value_1)
        row_value_2 = np.count_nonzero(solution[row_1] == value_2)
        diff += (row_value_1 - 2) ** 2 - (row_value_1 - 1) ** 2 + (row_value_2) ** 2 - (row_value_2 - 1) ** 2

        row_value_3 = np.count_nonzero(solution[row_2] == value_1)
        row_value_4 = np.count_nonzero(solution[row_2] == value_2)
        diff += (row_value_3) ** 2 - (row_value_3 - 1) ** 2 + (row_value_4 - 2) ** 2 - (row_value_4 - 1) ** 2

        col_value_1 = np.count_nonzero(solution[:, col_1] == value_1)
        col_value_2 = np.count_nonzero(solution[:, col_1] == value_2)
        diff += (col_value_1 - 2) ** 2 - (col_value_1 - 1) ** 2 + (col_value_2) ** 2 - (col_value_2 - 1) ** 2

        col_value_3 = np.count_nonzero(solution[:, col_2] == value_1)
        col_value_4 = np.count_nonzero(solution[:, col_2] == value_2)
        diff += (col_value_3) ** 2 - (col_value_3 - 1) ** 2 + (col_value_4 - 2) ** 2 - (col_value_4 - 1) ** 2

    return diff


# Generira neighborhood_size nasumičnih zamjena unutar blokova i za svaku
# inkrementalno računa novi loss. Vraća listu procijenjenih loss vrijednosti i
# odgovarajućih poteza (block, pos_1, pos_2). Pozicije su sortirane
# (pos_1 < pos_2) kako bi isti potez u oba smjera bio predstavljen istim
# tupleom — važno za tabu listu.
# Pretraga staje ranije ako pronađe potez s loss=0 (točno rješenje).
def neighborhood(solution: np.ndarray, block_solution: np.ndarray,
                 free_positions: list[list[int]], loss_value: int,
                 neighborhood_size: int = 30) \
        -> tuple[list[int], list[tuple[int, int, int]]]:
    losses = []
    moves = []

    # Unaprijed računamo blokove s barem 2 slobodne ćelije
    swappable_blocks = [block for block in range(9) if len(free_positions[block]) >= 2]

    for _ in range(neighborhood_size):
        block = np.random.choice(swappable_blocks)

        # Sortiramo pozicije kako bi (block, a, b) i (block, b, a) bili isti
        # potez u tabu listi
        pos_1, pos_2 = sorted(np.random.choice(free_positions[block], size=2, replace=False))
        value_1, value_2 = block_solution[block, pos_1], block_solution[block, pos_2]

        # Inkrementalni izračun — brže od ponovnog poziva loss_func
        diff = count_conflicts(solution, block, pos_1, pos_2, value_1, value_2)
        temp_loss = loss_value + diff

        losses.append(temp_loss)
        moves.append((block, pos_1, pos_2))

        # Ranije završavamo ako smo pronašli točno rješenje
        if temp_loss == 0:
            break

    return losses, moves


# Tabu pretraživanje (Tabu Search) je metaheuristika koja proširuje lokalno
# pretraživanje pamćenjem nedavno izvedenih poteza u tabu listi.
#
# Ključne ideje:
#   - Svake iteracije generiramo neighborhood_size nasumičnih zamjena
#     (susjedstvo) i biramo potez s najnižim loss-om.
#   - Potez je dopušten ako NIJE u tabu listi ILI ako je bolji od dosad
#     najboljeg rješenja (aspiracijski kriterij — iznimka od tabu pravila).
#   - Izvedeni potez dodajemo u tabu listu na tabu_tenure iteracija kako
#     bismo spriječili neposredni povratak na prethodno rješenje i time
#     forsirali diversifikaciju pretrage.
#   - Kratki tenure (3) empirijski dobro radi jer sudoku ima malo slobodnih
#     ćelija po bloku, pa dulji tenure previše ograničava dostupne poteze.
#     Vrijednost nije sustavno podesena.
#
# U iteraciji u kojoj su svi generirani susjedi tabu i nijedan ne poboljsava
# best_loss ne izvodi se nista niti se sto dodaje u tabu listu, pa je duljina
# liste najvise tabu_tenure, a ne nuzno tocno toliko.
def tabu_search(transformed_instance: np.ndarray, free_positions: list[list[int]],
                max_iterations: int, tabu_tenure: int = 3,
                neighborhood_size: int = 30) -> int:
    current_solution = my_parser.generate_sol(transformed_instance)
    current_loss = my_parser.loss_func(current_solution)
    tabu_list = []
    best_loss = current_loss
    iteration = 0

    while iteration < max_iterations and best_loss > 0:
        # Dekodiramo u standardni prikaz jer count_conflicts gleda retke/stupce mreže
        solution = my_parser.transform(current_solution)
        neighbor_losses, neighbor_moves = neighborhood(
            solution, current_solution, free_positions, current_loss, neighborhood_size)

        # Prolazimo kroz susjede sortirane po loss-u (argmin petlja)
        # i tražimo prvi dopušteni potez
        for _ in range(len(neighbor_moves)):
            index = np.argmin(neighbor_losses)
            new_loss = neighbor_losses[index]
            move = neighbor_moves[index]

            # Potez je dopušten ako nije tabu ILI ako poboljšava globalni
            # minimum (aspiracijski kriterij)
            is_allowed = new_loss < best_loss or move not in tabu_list

            if is_allowed:
                # Izvršavamo potez: zamjenjujemo dvije slobodne ćelije unutar bloka
                tabu_list.append(move)
                value_1 = current_solution[move[0], move[1]]
                value_2 = current_solution[move[0], move[2]]
                current_solution[move[0], move[1]] = value_2
                current_solution[move[0], move[2]] = value_1

                current_loss = new_loss
                if current_loss < best_loss:
                    best_loss = current_loss
                break

            # Potez je tabu i ne zadovoljava aspiracijski kriterij —
            # uklanjamo ga i tražimo sljedeći
            neighbor_losses.pop(index)
            neighbor_moves.pop(index)

        # Uklanjamo najstariji potez kada tabu lista prijeđe maksimalnu duljinu
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)

        iteration += 1

    return best_loss
