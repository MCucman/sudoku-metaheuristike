import my_parser
import numpy as np


# Generira susjedna rješenja zamjenom dviju slobodnih ćelija unutar istog bloka.
# Zamjena unutar bloka čuva blok-ograničenje (svaki blok i dalje sadrži 1-9),
# pa jedino redak/stupac ograničenja mogu biti narušena — što loss_func mjeri.
# Biramo nasumični blok koji ima barem 2 slobodne pozicije (swappable_blocks)
# kako bismo izbjegli beskonačnu petlju u slučaju blokova s manje od 2 slobodne ćelije.
def neighbor_solution(current_solution: np.ndarray, free_positions: list[list[int]]) \
        -> tuple[np.ndarray, tuple[int, int, int]]:
    candidate = current_solution.copy()

    # Uzimamo samo blokove s barem 2 slobodne ćelije jer zamjena zahtijeva dvije pozicije
    swappable_blocks = [block for block in range(9) if len(free_positions[block]) >= 2]
    block = np.random.choice(swappable_blocks)

    pos_1, pos_2 = np.random.choice(free_positions[block], size=2, replace=False)
    value_1, value_2 = candidate[block, pos_1], candidate[block, pos_2]
    candidate[block, pos_1], candidate[block, pos_2] = value_2, value_1

    return candidate, (block, pos_1, pos_2)


# Simulirano kaljenje (Simulated Annealing) je metaheuristika inspirirana
# fizikalnim procesom sporog hlađenja metala. Kreće od nasumičnog rješenja
# i iterativno pokušava poboljšati loss zamjenom dviju ćelija unutar bloka.
#
# Ključna ideja: za razliku od čistog lokalnog pretraživanja, SA s određenom
# vjerojatnošću prihvaća i pogoršanja kako bi izbjeglo lokalne minimume.
# Ta vjerojatnost ovisi o temperaturi i veličini pogoršanja:
#   P(prihvati) = exp(-delta / temperatura)
# S padom temperature algoritam postaje sve "pohlepniji" i konvergira prema
# (lokalnom) minimumu. Parametri hlađenja (cooling_rate, min_temperature)
# kontroliraju kompromis između istraživanja i iskorištavanja prostora rješenja.
#
# Za razliku od tabu pretrazivanja, loss kandidata racuna se punim pozivom
# loss_func, a ne inkrementalno; po iteraciji se generira samo jedan kandidat.
#
# Slucaj delta == 0 (plato) pada u probabilisticku granu, gdje je exp(0) = 1,
# pa se takvi potezi uvijek prihvacaju.
#
# Uz geometrijsko hladjenje temperatura obicno padne ispod min_temperature
# prije nego se dosegne max_iterations, pa je temperaturni kriterij taj koji u
# praksi zaustavlja izvodjenje.
def simulated_annealing(transformed_instance: np.ndarray, free_positions: list[list[int]],
                        initial_temperature: float = 100.0, cooling_rate: float = 0.995,
                        min_temperature: float = 1e-4, max_iterations: int = 3000) -> int:
    current_solution = my_parser.generate_sol(transformed_instance)
    current_loss = my_parser.loss_func(current_solution)
    best_loss = current_loss
    temperature = initial_temperature
    iteration = 0

    while iteration < max_iterations and best_loss > 0 and temperature > min_temperature:
        # Generiramo susjedno rješenje jednom zamjenom unutar bloka
        candidate_solution, _ = neighbor_solution(current_solution, free_positions)
        candidate_loss = my_parser.loss_func(candidate_solution)
        delta = candidate_loss - current_loss

        if delta < 0:
            # Poboljšanje — uvijek prihvaćamo
            current_solution = candidate_solution
            current_loss = candidate_loss
        else:
            # Pogoršanje — prihvaćamo s vjerojatnošću exp(-delta/T)
            # Velika temperatura → visoka vjerojatnost prihvata (istraživanje)
            # Mala temperatura  → mala vjerojatnost prihvata (iskorištavanje)
            acceptance_probability = np.exp(-delta / temperature)
            if np.random.random() < acceptance_probability:
                current_solution = candidate_solution
                current_loss = candidate_loss

        # Ažuriramo globalno najbolje rješenje
        if current_loss < best_loss:
            best_loss = current_loss

        # Geometrijsko hlađenje: temperatura se množi s cooling_rate svake iteracije
        temperature *= cooling_rate
        iteration += 1

    return best_loss
