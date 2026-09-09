import time

import numpy as np

import my_parser
from genetic_algorithm import genetic_algorithm
from greedy_algorithm import greedy_algorithm
from simulated_annealing import simulated_annealing
from tabu_search import tabu_search


# Pokreće zadani algoritam na n sudoku puzzlea i ispisuje statistiku.
# Parametri:
#   name             - naziv algoritma (string), koristi se za ispis
#   func             - funkcija algoritma
#   n                - broj puzzlea koje testiramo
#   attempts         - broj pokušaja po puzzleu (relevantno za stohastičke algoritme)
#   takes_both_grids - True samo za pohlepni algoritam, koji prima i originalnu i
#                      transformiranu mrezu jer istovremeno treba standardni prikaz
#                      (za racunanje konflikata) i blok-prikaz (za popunjavanje
#                      rjesenja). Ostali algoritmi rade isključivo u blok-prikazu.
#   **kwargs         - dodatni parametri koji se prosljeđuju direktno algoritmu
def run_algorithm(name, func, n, attempts=1, takes_both_grids=False, **kwargs):
    results = []       # lista (prosječni_loss, medijan_lossa) po puzzleu
    correct = 0        # broj puzzlea riješenih barem jednom
    solved_attempts = 0  # ukupan broj pokušaja koji su dali loss=0
    total_attempts = 0
    start = time.time()

    for i in range(n):
        data = my_parser.get_sudoku(i)
        if data is None:
            break  # dosegli smo kraj CSV datoteke

        inst, transf_inst, free_positions = data
        puzzle_losses = []
        puzzle_solved_attempts = 0

        for _ in range(attempts):
            if takes_both_grids:
                loss = func(inst, transf_inst)
            else:
                loss = func(transf_inst, free_positions, **kwargs)

            total_attempts += 1
            puzzle_losses.append(loss)
            if loss == 0:
                puzzle_solved_attempts += 1
                solved_attempts += 1

        # Računamo prosječni i medijan loss po puzzleu kroz sve pokušaje
        per_puzzle_avg_loss = round(sum(puzzle_losses) / len(puzzle_losses), 2) if puzzle_losses else 0
        sorted_puzzle_losses = sorted(puzzle_losses)
        mid = len(sorted_puzzle_losses) // 2
        if len(sorted_puzzle_losses) % 2 == 0:
            per_puzzle_median_loss = round(
                (sorted_puzzle_losses[mid - 1] + sorted_puzzle_losses[mid]) / 2, 2)
        else:
            per_puzzle_median_loss = round(sorted_puzzle_losses[mid], 2)

        results.append((per_puzzle_avg_loss, per_puzzle_median_loss))
        if puzzle_solved_attempts > 0:
            correct += 1

    # Agregiramo statistiku po svim puzzleima
    if results:
        avg_loss = round(sum(value for value, _ in results) / len(results), 2)
        # mean_of_medians_loss je prosjek medijana po puzzleu, nije globalni medijan
        mean_of_medians_loss = round(sum(value for _, value in results) / len(results), 2)
    else:
        avg_loss = 0
        mean_of_medians_loss = 0

    success_rate = round((solved_attempts / total_attempts) * 100, 2) if total_attempts else 0
    elapsed = round(time.time() - start, 2)

    print(f'[{name}] Average loss = {avg_loss}')
    print(f'[{name}] Mean of per-puzzle medians = {mean_of_medians_loss}')
    print(f'[{name}] Puzzles solved at least once = {correct} / {len(results)}')
    print(f'[{name}] Successful attempts = {solved_attempts} / {total_attempts} ({success_rate}%)')
    print(f'[{name}] Execution time = {elapsed} s')
    print('-' * 40)


if __name__ == "__main__":
    # --- Postavke eksperimenta -------------------------------------------
    N_PUZZLES = 10000   # broj zagonetki iz sudoku.csv
    ATTEMPTS = 30       # broj pokusaja po zagonetki za stohasticke algoritme
    SEED = 72           # seed za reproducibilnost

    # Parametri pojedinih algoritama. Sve vrijednosti odgovaraju parametrima
    # navedenima u pseudokodu u radu i mogu se mijenjati na ovom mjestu.
    TABU_PARAMS = {
        'max_iterations': 300,
        'tabu_tenure': 3,
        'neighborhood_size': 30,
    }
    SA_PARAMS = {
        'initial_temperature': 100.0,
        'cooling_rate': 0.995,
        'min_temperature': 1e-4,
        'max_iterations': 3000,
    }
    GA_PARAMS = {
        'population_size': 50,
        'generations': 55,
        'mutation_rate': 0.3,
        'elite_count': 3,
        'n_swaps': 3,
        'stagnation_limit': 6,
        'tournament_size': 3,
    }
    # ---------------------------------------------------------------------

    # Fiksiramo seed za reproducibilnost rezultata — pri velikom N_PUZZLES
    # (npr. 10000) seed nema praktičnog utjecaja na agregatne statistike
    np.random.seed(SEED)

    # Napomena o usporedivosti: pohlepni algoritam radi jedan deterministicki
    # prolaz bez ponovnih pokretanja, dok metaheuristike dobivaju ATTEMPTS
    # pokusaja i vracaju najbolju vrijednost duz cijele putanje. Metrika je ista
    # (funkcija cilja), ali racunski proracun nije, pa to treba navesti pri
    # interpretaciji rezultata.
    print('Starting all algorithms...\n')

    run_algorithm('greedy_algorithm', greedy_algorithm, N_PUZZLES,
                  takes_both_grids=True)

    run_algorithm('tabu_search', tabu_search, N_PUZZLES, attempts=ATTEMPTS,
                  **TABU_PARAMS)

    run_algorithm('simulated_annealing', simulated_annealing, N_PUZZLES,
                  attempts=ATTEMPTS, **SA_PARAMS)

    run_algorithm('genetic_algorithm', genetic_algorithm, N_PUZZLES,
                  attempts=ATTEMPTS, **GA_PARAMS)
