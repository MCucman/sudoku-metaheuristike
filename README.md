# Rješavanje Sudokua metaheuristikama

Usporedba nekoliko algoritama za rješavanje Sudoku zagonetki:

- pohlepni algoritam (`greedy_algorithm.py`)
- simulirano kaljenje (`simulated_annealing.py`)
- tabu pretraživanje (`tabu_search.py`)
- genetski algoritam (`genetic_algorithm.py`)

## Struktura

| Datoteka | Opis |
|----------|------|
| `main.py` | Pokretanje eksperimenta i usporedba algoritama |
| `my_parser.py` | Učitavanje i transformacija Sudoku instanci |
| `plots.py` | Generiranje grafova |
| `greedy_algorithm.py`, `simulated_annealing.py`, `tabu_search.py`, `genetic_algorithm.py` | Implementacije algoritama |

## Postavljanje

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Dataset

Kod koristi datoteku `sudoku.csv` (1 milijun zagonetki). Datoteka je prevelika za
repozitorij (~156 MB) pa nije uključena. Skini je s
[Kaggle: 1 million Sudoku games](https://www.kaggle.com/datasets/bryanpark/sudoku)
i smjesti u korijen projekta pod imenom `sudoku.csv`.

Format je CSV sa zaglavljem `quizzes,solutions`, gdje je svaki redak niz od 81
znamenki (0 = prazno polje).

## Pokretanje

```bash
python main.py
```
