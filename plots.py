"""
Generira slike za poglavlje "Eksperimentalna evaluacija".

Proizvodi tri PNG-a:
  1. greedy.png              - (a) pad lossa tijekom konstrukcije pohlepnog
                               rjesenja na jednoj instanci; (b) histogram
                               zavrsnog lossa preko vise instanci
  2. convergence_average.png - medijan i interkvartilni raspon najboljeg lossa
                               kroz iteracije/generacije, usrednjeno preko vise
                               instanci (zaseban podgraf za tabu, SA i GA)
  3. convergence_single.png  - trag najboljeg lossa triju metaheuristika na
                               jednoj instanci, na zajednickom grafu

Sve slike koriste manji uzorak instanci i sluze za ilustraciju dinamike; glavne
tablice u radu ostaju na punom eksperimentu (10000 instanci, 30 pokusaja).

"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import my_parser


# Zajednicki izgled: dovoljno velik font za citljivost u tiskanom radu.
plt.rcParams.update({
    "font.size": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

SEED = 72


# ---------------------------------------------------------------------------
# 1. Prosjecna krivulja konvergencije preko vise instanci.
#    Za svaki algoritam izvede se po jedan pokusaj na svakoj od n_instances
#    instanci, putanje se poravnaju na jednaku duljinu (zadnja vrijednost se
#    ponavlja nakon zaustavljanja), pa se crta medijan uz kvartilnu vrpcu.
#    Tabu i SA mjere se u iteracijama, GA u generacijama, pa svaki ide na
#    zaseban podgraf s vlastitom osi.
# ---------------------------------------------------------------------------
def _pad(hist, length):
    # Produljuje putanju do zadane duljine ponavljanjem zadnje vrijednosti
    # (algoritam koji je stao zadrzava svoj najbolji loss do kraja prozora).
    if len(hist) >= length:
        return np.array(hist[:length], dtype=float)
    return np.concatenate([hist, np.full(length - len(hist), hist[-1], dtype=float)])


def _collect(trace_func, n_instances, window):
    # Skuplja po jednu putanju lossa za prvih n_instances instanci i poravnava
    # ih na jednaku duljinu 'window' da bi se mogao racunati medijan po stupcu.
    curves = []
    for i in range(n_instances):
        data = my_parser.get_sudoku(i)
        if data is None:
            break
        _, transf_inst, free_positions = data
        _, hist = trace_func(transf_inst, free_positions)
        curves.append(_pad(hist, window))
    return np.vstack(curves)


def plot_convergence_average(n_instances=150):
    # Prozori su odabrani tako da pokriju tipicno trajanje svake metode:
    # tabu staje najkasnije na max_iterations=300, GA na generations=55,
    # a SA se prikazuje do 2000 iteracija (medijan zaustavljanja je nize).
    np.random.seed(SEED)
    tabu = _collect(tabu_trace, n_instances, window=300)
    np.random.seed(SEED)
    sa = _collect(sa_trace, n_instances, window=2000)
    np.random.seed(SEED)
    ga = _collect(ga_trace, n_instances, window=56)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    panels = [
        (axes[0], tabu, "Tabu pretraživanje", "Iteracija", "tab:blue"),
        (axes[1], sa, "Simulirano kaljenje", "Iteracija", "tab:orange"),
        (axes[2], ga, "Genetički algoritam", "Generacija", "tab:green"),
    ]

    for ax, curves, title, xlabel, color in panels:
        x = np.arange(curves.shape[1])
        med = np.median(curves, axis=0)
        q1 = np.percentile(curves, 25, axis=0)
        q3 = np.percentile(curves, 75, axis=0)
        ax.fill_between(x, q1, q3, color=color, alpha=0.25,
                        label="interkvartilni raspon")
        ax.plot(x, med, color=color, linewidth=1.8, label="medijan")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylim(bottom=0)
        ax.legend()

    axes[0].set_ylabel("Najbolja vrijednost $f$")

    fig.suptitle(f"Konvergencija usrednjena preko {n_instances} instanci "
                 "(medijan i interkvartilni raspon)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig("convergence_average.png", bbox_inches="tight")
    plt.close(fig)
    print("napisano: convergence_average.png  "
          f"(medijani na kraju: tabu {np.median(tabu[:, -1]):.1f}, "
          f"SA {np.median(sa[:, -1]):.1f}, GA {np.median(ga[:, -1]):.1f})")


# ---------------------------------------------------------------------------
# 2. Pohlepni algoritam.
#    (a) Proces konstrukcije na jednoj instanci: kako vrijednost funkcije cilja
#        pada kako se prazna polja jedno po jedno popunjavaju. Pohlepni gradi
#        rjesenje jednim prolaskom, pa "kroz vrijeme" znaci kroz broj upisanih
#        polja; os x ide od 0 do broja praznih polja te instance.
#    (b) Raspodjela zavrsnog lossa preko n_instances instanci: svaka instanca
#        daje jedan broj (pohlepni je determinisitcan), a visina stupca je broj
#        instanci koje su zavrsile s tom vrijednoscu f. Pokazuje da se ishodi
#        grupiraju oko tridesetak konflikata i da gotovo nijedan nije nula.
#    Logika konstrukcije istovjetna je funkciji greedy_algorithm iz rada;
#    ovdje se usput biljezi loss nakon svakog upisa.
# ---------------------------------------------------------------------------
def _greedy_run(inst, transf_inst):
    # Istovjetno greedy_algorithm iz rada, ali usput biljezi loss nakon svakog
    # upisa. Vraca (putanja_lossa, zavrsni_loss).
    from greedy_algorithm import count_conflicts
    sol = inst.copy()
    block_sol = transf_inst.copy()
    missing = my_parser.missing_nums(block_sol)

    empty_cells = [int(np.sum(block_sol[b] == 0)) for b in range(9)]
    block_order = sorted(range(9), key=lambda b: empty_cells[b])

    history = [my_parser.loss_func(block_sol)]  # loss pocetnog (nepotpunog) stanja
    for block in block_order:
        for pos in range(9):
            if not block_sol[block][pos]:
                best_conf = 145
                best_id = 0
                best_row = best_col = 0
                for num_id, num in enumerate(missing[block]):
                    curr_conf, row, col = count_conflicts(sol, block, pos, num)
                    if curr_conf < best_conf:
                        best_conf = curr_conf
                        best_id = num_id
                        best_row, best_col = row, col
                    if best_conf == 0:
                        break
                best_num = missing[block][best_id]
                block_sol[block][pos] = best_num
                sol[best_row][best_col] = best_num
                missing[block].remove(best_num)
                history.append(my_parser.loss_func(block_sol))

    return history, history[-1]


def plot_greedy(n_instances=200, example_index=1):
    # (a) Proces konstrukcije na jednoj instanci
    data = my_parser.get_sudoku(example_index)
    if data is None:
        raise SystemExit("ne mogu ucitati instancu iz sudoku.csv")
    inst, transf_inst, _ = data
    trace, _ = _greedy_run(inst, transf_inst)

    # (b) Zavrsni loss po instanci; jedan pokusaj po instanci jer je pohlepni
    # deterministican, pa vise pokusaja ne bi dalo nista novo.
    finals = []
    for i in range(n_instances):
        d = my_parser.get_sudoku(i)
        if d is None:
            break
        inst_i, transf_i, _ = d
        _, final_loss = _greedy_run(inst_i, transf_i)
        finals.append(final_loss)
    finals = np.array(finals)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(range(len(trace)), trace, color="tab:purple", linewidth=1.6)
    ax1.set_xlabel("Broj popunjenih polja")
    ax1.set_ylabel("Vrijednost $f$")
    ax1.set_title(f"(a) Konstrukcija rješenja (instanca {example_index})")
    ax1.set_ylim(bottom=0)

    # Sirina bina je 1, pa jedan stupac odgovara jednoj vrijednosti f, a
    # njegova visina broju instanci s tom zavrsnom vrijednoscu.
    ax2.hist(finals, bins=range(0, int(finals.max()) + 3),
             color="tab:purple", alpha=0.8, edgecolor="white")
    ax2.axvline(finals.mean(), color="black", linestyle="--", linewidth=1.2,
                label=f"prosjek = {finals.mean():.1f}")
    ax2.set_xlabel("Završna vrijednost $f$")
    ax2.set_ylabel("Broj instanci")
    ax2.set_title(f"(b) Raspodjela ishoda ({n_instances} instanci)")
    ax2.legend()

    fig.tight_layout()
    fig.savefig("greedy.png", bbox_inches="tight")
    plt.close(fig)
    solved = int(np.sum(finals == 0))
    print(f"napisano: greedy.png  (prosjek {finals.mean():.1f}, "
          f"rijeseno {solved}/{len(finals)})")


# ---------------------------------------------------------------------------
# Varijante algoritama koje biljeze putanju najboljeg lossa.
# Logika je istovjetna funkcijama iz rada; jedina razlika je popis 'history'
# koji se vraca uz konacni best_loss. NE koriste se u glavnom eksperimentu.
# ---------------------------------------------------------------------------
def tabu_trace(transf_inst, free_positions, max_iterations=300,
               tabu_tenure=3, neighborhood_size=30):
    from tabu_search import count_conflicts
    current = my_parser.generate_sol(transf_inst)
    current_loss = my_parser.loss_func(current)
    tabu_list = []
    best_loss = current_loss
    history = [best_loss]

    swappable = [b for b in range(9) if len(free_positions[b]) >= 2]

    it = 0
    while it < max_iterations and best_loss > 0:
        sol = my_parser.transform(current)
        losses, moves = [], []
        for _ in range(neighborhood_size):
            block = np.random.choice(swappable)
            p1, p2 = sorted(np.random.choice(free_positions[block], size=2, replace=False))
            v1, v2 = current[block, p1], current[block, p2]
            diff = count_conflicts(sol, block, p1, p2, v1, v2)
            losses.append(current_loss + diff)
            moves.append((block, p1, p2))
            if current_loss + diff == 0:
                break

        for _ in range(len(moves)):
            idx = int(np.argmin(losses))
            new_loss = losses[idx]
            move = moves[idx]
            if new_loss < best_loss or move not in tabu_list:
                tabu_list.append(move)
                v1 = current[move[0], move[1]]
                v2 = current[move[0], move[2]]
                current[move[0], move[1]] = v2
                current[move[0], move[2]] = v1
                current_loss = new_loss
                if current_loss < best_loss:
                    best_loss = current_loss
                break
            losses.pop(idx)
            moves.pop(idx)

        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)
        it += 1
        history.append(best_loss)

    return best_loss, history


def sa_trace(transf_inst, free_positions, initial_temperature=100.0,
             cooling_rate=0.995, min_temperature=1e-4, max_iterations=3000):
    from simulated_annealing import neighbor_solution
    current = my_parser.generate_sol(transf_inst)
    current_loss = my_parser.loss_func(current)
    best_loss = current_loss
    temperature = initial_temperature
    history = [best_loss]

    it = 0
    while it < max_iterations and best_loss > 0 and temperature > min_temperature:
        candidate, _ = neighbor_solution(current, free_positions)
        candidate_loss = my_parser.loss_func(candidate)
        delta = candidate_loss - current_loss
        if delta < 0:
            current, current_loss = candidate, candidate_loss
        else:
            if np.random.random() < np.exp(-delta / temperature):
                current, current_loss = candidate, candidate_loss
        if current_loss < best_loss:
            best_loss = current_loss
        temperature *= cooling_rate
        it += 1
        history.append(best_loss)

    return best_loss, history


def ga_trace(transf_inst, free_positions, population_size=50, generations=55,
             mutation_rate=0.3, elite_count=3, n_swaps=3, stagnation_limit=6,
             tournament_size=3):
    from genetic_algorithm import tournament_selection, crossover, mutation
    population = [my_parser.generate_sol(transf_inst) for _ in range(population_size)]
    losses = [my_parser.loss_func(ind) for ind in population]
    best_loss = min(losses)
    stagnation = 0
    history = [best_loss]

    for _ in range(generations):
        ordered = [ind for _, ind in sorted(zip(losses, population), key=lambda x: x[0])]
        nxt = ordered[:elite_count]
        while len(nxt) < population_size:
            p1 = tournament_selection(population, losses, tournament_size)
            p2 = tournament_selection(population, losses, tournament_size)
            c1, c2 = crossover(p1, p2)
            c1 = mutation(c1, free_positions, mutation_rate, n_swaps)
            c2 = mutation(c2, free_positions, mutation_rate, n_swaps)
            nxt.append(c1)
            if len(nxt) < population_size:
                nxt.append(c2)
        population = nxt[:population_size]
        losses = [my_parser.loss_func(ind) for ind in population]
        gen_best = min(losses)
        if gen_best < best_loss:
            best_loss = gen_best
            stagnation = 0
        else:
            stagnation += 1
        history.append(best_loss)
        if best_loss == 0:
            break
        if stagnation >= stagnation_limit:
            ordered = [ind for _, ind in sorted(zip(losses, population), key=lambda x: x[0])]
            keep = ordered[:elite_count]
            fresh = [my_parser.generate_sol(transf_inst)
                     for _ in range(population_size - elite_count)]
            population = keep + fresh
            losses = [my_parser.loss_func(ind) for ind in population]
            stagnation = 0

    return best_loss, history


# ---------------------------------------------------------------------------
# 3. Trag konvergencije na jednoj instanci.
#    Za tabu i SA os x je iteracija, za GA generacija. Kako se te tri skale
#    razlikuju redom velicine, GA se crta na zasebnoj (gornjoj) osi.
# ---------------------------------------------------------------------------
def plot_convergence_single(puzzle_index=0):
    data = my_parser.get_sudoku(puzzle_index)
    if data is None:
        raise SystemExit("ne mogu ucitati instancu iz sudoku.csv")
    _, transf_inst, free_positions = data

    # Svaki algoritam iz svog seeda kako pojedini trag ne bi ovisio o poretku.
    np.random.seed(SEED)
    _, tabu_hist = tabu_trace(transf_inst, free_positions)
    np.random.seed(SEED)
    _, sa_hist = sa_trace(transf_inst, free_positions)
    np.random.seed(SEED)
    _, ga_hist = ga_trace(transf_inst, free_positions)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(range(len(sa_hist)), sa_hist, label="Simulirano kaljenje",
            color="tab:orange", linewidth=1.3)
    ax.plot(range(len(tabu_hist)), tabu_hist, label="Tabu pretraživanje",
            color="tab:blue", linewidth=1.6)
    ax.set_xlabel("Iteracija")
    ax.set_ylabel("Najbolja vrijednost $f$")
    ax.set_title(f"Konvergencija na jednoj instanci (indeks {puzzle_index})")

    # GA na zasebnoj gornjoj osi jer mu je jedinica generacija, ne iteracija.
    ax_top = ax.twiny()
    ax_top.plot(range(len(ga_hist)), ga_hist, label="Genetički algoritam",
                color="tab:green", linewidth=1.6)
    ax_top.set_xlabel("Generacija (genetički algoritam)")
    ax_top.grid(False)

    # Zajednicka legenda
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax_top.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right")

    fig.tight_layout()
    fig.savefig("convergence_single.png", bbox_inches="tight")
    plt.close(fig)
    print("napisano: convergence_single.png  "
          f"(tabu {tabu_hist[-1]}, SA {sa_hist[-1]}, GA {ga_hist[-1]})")


if __name__ == "__main__":
    # Pohlepni: proces konstrukcije + raspodjela zavrsnog lossa.
    plot_greedy(n_instances=200, example_index=1)
    # Prosjecna konvergencija preko vise instanci (medijan + kvartilna vrpca).
    plot_convergence_average(n_instances=100)
    # Instanca na kojoj tabu i SA dosegnu f=0 (razlicitom brzinom), a GA zapne
    # s nekoliko zaostalih konflikata - pokazuje tipicnu razliku u dinamici.
    plot_convergence_single(puzzle_index=1)
    print("gotovo.")
