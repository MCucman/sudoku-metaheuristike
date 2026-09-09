import my_parser
import numpy as np


# Turnirska selekcija: nasumično biramo tournament_size jedinki iz populacije
# i vraćamo onu s najmanjim loss-om. Veći tournament_size pojačava selekcijski
# pritisak (bolje jedinke češće pobjeđuju), manji povećava raznolikost.
def tournament_selection(population: list[np.ndarray], losses: list[int],
                         tournament_size: int = 3) -> np.ndarray:
    selected = np.random.choice(len(population), size=tournament_size, replace=False)
    best_idx = selected[np.argmin([losses[i] for i in selected])]
    return population[best_idx].copy()


# Uniformno krizanje na razini blokova: nasumicna binarna maska za svaki od
# devet blok-redaka neovisno odreduje od kojeg ga roditelja dijete nasljeduje.
# (Nije jednotockovno krizanje — nema jedne tocke reza, svaki blok se odlucuje
# zasebno.) Djeca su komplementarna: gdje maska daje blok djetetu 1 od
# roditelja 1, drugo dijete ga dobiva od roditelja 2.
# Blok na poziciji i uvijek dolazi s pozicije i jednog od roditelja, pa se
# blok-ogranicenje ne moze narusiti, a fiksna polja ostaju na svojim mjestima
# jer su u svim jedinkama na jednakim pozicijama.
def crossover(parent_1: np.ndarray, parent_2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    child_1 = parent_1.copy()
    child_2 = parent_2.copy()
    # Svaki bit maske određuje hoće li i-ti blok-redak doći od roditelja 1 ili 2
    mask = np.random.randint(0, 2, size=9, dtype=bool)

    for block in range(9):
        if not mask[block]:
            # Zamjenjujemo: dijete 1 dobiva blok od roditelja 2 i obrnuto
            child_1[block] = parent_2[block].copy()
            child_2[block] = parent_1[block].copy()

    return child_1, child_2


# Mutacija pokusava n_swaps zamjena; svaka se izvodi neovisno, s vjerojatnoscu
# mutation_rate, i to unutar jednog nasumicno odabranog bloka (blok se bira
# zasebno za svaku zamjenu). Broj stvarno izvedenih zamjena je zato slucajan,
# ~Bin(n_swaps, mutation_rate), i moze biti nula: uz n_swaps=3 i
# mutation_rate=0.3 mutacija ne mijenja nista u 0.7^3 = 34.3% slucajeva.
# Vise zamjena po mutaciji (n_swaps > 1) daje jace istrazivanje prostora
# rjesenja jer jedna zamjena unutar bloka rijetko promijeni dovoljno konflikata
# da se vidi na funkciji cilja.
# Zamjenjuju se samo slobodne pozicije, pa fiksna polja i blok-ogranicenje
# ostaju netaknuti.
def mutation(individual: np.ndarray, free_positions: list[list[int]],
             mutation_rate: float = 0.3, n_swaps: int = 3) -> np.ndarray:
    # Unaprijed filtriramo blokove s barem 2 slobodne ćelije
    swappable_blocks = [block for block in range(9) if len(free_positions[block]) >= 2]

    for _ in range(n_swaps):
        if np.random.random() < mutation_rate:
            block = np.random.choice(swappable_blocks)
            pos_1, pos_2 = np.random.choice(free_positions[block], size=2, replace=False)
            individual[block, pos_1], individual[block, pos_2] = \
                individual[block, pos_2], individual[block, pos_1]

    return individual


# Genetski algoritam (GA) održava populaciju rješenja koja evoluira kroz generacije.
# Operatori:
#   - Selekcija:  turnirska selekcija bira roditelje s tendencijom prema boljima
#   - Križanje:   uniformno na razini blok-redaka
#   - Mutacija:   nasumične zamjene slobodnih pozicija unutar bloka
#   - Elitizam:   elite_count najboljih jedinki prelazi direktno u sljedeću generaciju
#
# Problem prerane konvergencije (sve jedinke postaju slične, pretraga stagnira)
# rješavamo restartom: ako se globalni minimum ne poboljša stagnation_limit generacija,
# zadržavamo samo elite i ostatak populacije zamjenjujemo svježe generiranim jedinkama.
# Time ubrizgavamo raznolikost bez gubljenja dosad pronađenog najboljeg rješenja.
def genetic_algorithm(transformed_instance: np.ndarray, free_positions: list[list[int]],
                      population_size: int = 50, generations: int = 55,
                      mutation_rate: float = 0.3, elite_count: int = 3,
                      n_swaps: int = 3, stagnation_limit: int = 6,
                      tournament_size: int = 3) -> int:
    # Inicijalizacija: svaka jedinka je nasumično rješenje koje zadovoljava blok-ograničenje
    population = [my_parser.generate_sol(transformed_instance) for _ in range(population_size)]
    losses = [my_parser.loss_func(individual) for individual in population]
    best_loss = min(losses)
    stagnation = 0  # broj generacija bez poboljšanja globalnog minimuma

    for _ in range(generations):
        # Sortiramo populaciju po lossu da bismo lako izvukli elite
        sorted_population = [ind for _, ind in sorted(zip(losses, population), key=lambda x: x[0])]

        # Elitizam: elite_count najboljih jedinki prelazi bez promjene
        next_population = sorted_population[:elite_count]

        # Punimo ostatak populacije križanjem i mutacijom
        while len(next_population) < population_size:
            parent_1 = tournament_selection(population, losses, tournament_size)
            parent_2 = tournament_selection(population, losses, tournament_size)
            child_1, child_2 = crossover(parent_1, parent_2)
            child_1 = mutation(child_1, free_positions, mutation_rate, n_swaps)
            child_2 = mutation(child_2, free_positions, mutation_rate, n_swaps)
            next_population.append(child_1)
            if len(next_population) < population_size:
                next_population.append(child_2)

        population = next_population[:population_size]
        losses = [my_parser.loss_func(individual) for individual in population]
        gen_best = min(losses)

        # Pratimo stagnaciju: broj generacija od zadnjeg poboljšanja
        if gen_best < best_loss:
            best_loss = gen_best
            stagnation = 0
        else:
            stagnation += 1

        # Prijevremeni izlaz ako je pronađeno točno rješenje
        if best_loss == 0:
            break

        # Restart: ako nema napretka stagnation_limit generacija, zadržavamo samo
        # elite i ostatak populacije zamjenjujemo novim nasumičnim jedinkama.
        # Time izbjegavamo preranu konvergenciju i dajemo algoritmu novu priliku.
        if stagnation >= stagnation_limit:
            sorted_population = [ind for _, ind in sorted(zip(losses, population), key=lambda x: x[0])]
            keep = sorted_population[:elite_count]
            fresh = [my_parser.generate_sol(transformed_instance)
                     for _ in range(population_size - elite_count)]
            population = keep + fresh
            losses = [my_parser.loss_func(individual) for individual in population]
            stagnation = 0

    return best_loss
