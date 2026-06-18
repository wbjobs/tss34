from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .milp_solver import RouteInfo, _compute_route_info
from .models import (
    DeliveryPlan,
    DeliverySegment,
    DisasterPoint,
    RouteType,
    SolverResult,
    SubwayNetwork,
    SupplyType,
    Warehouse,
)


@dataclass
class Gene:
    demand_idx: int
    primary_wh_idx: int
    primary_route_idx: int
    secondary_wh_idx: int
    secondary_route_idx: int
    split_ratio: float


@dataclass
class Chromosome:
    genes: List[Gene]

    def copy(self) -> Chromosome:
        return Chromosome(genes=[copy.deepcopy(g) for g in self.genes])


@dataclass
class DemandItem:
    disaster_id: str
    supply_type: SupplyType
    quantity: int
    disaster_idx: int


def _extract_demands(disasters: List[DisasterPoint]) -> List[DemandItem]:
    items = []
    for d_idx, d in enumerate(disasters):
        for st in SupplyType:
            qty = d.demand.get(st, 0)
            if qty > 0:
                items.append(DemandItem(
                    disaster_id=d.id,
                    supply_type=st,
                    quantity=qty,
                    disaster_idx=d_idx,
                ))
    return items


def _decode_chromosome(
    chrom: Chromosome,
    demands: List[DemandItem],
    warehouses: List[Warehouse],
    disasters: List[DisasterPoint],
    routes: Dict[Tuple[str, str], List[RouteInfo]],
    subway: SubwayNetwork,
) -> Tuple[DeliveryPlan, float, Dict[str, Dict[SupplyType, int]]]:
    plan = DeliveryPlan()
    remaining_inv: Dict[str, Dict[SupplyType, int]] = {}
    for w in warehouses:
        remaining_inv[w.id] = dict(w.inventory)

    wh_by_id = {w.id: w for w in warehouses}
    dp_by_id = {d.id: d for d in disasters}

    total_cost = 0.0
    delivered: Dict[str, Dict[SupplyType, int]] = {
        d.id: {st: 0 for st in SupplyType} for d in disasters
    }
    truck_used: Dict[str, int] = {w.id: 0 for w in warehouses}
    tardiness: Dict[str, float] = {d.id: 0.0 for d in disasters}

    for gene in chrom.genes:
        if gene.demand_idx >= len(demands):
            continue
        demand = demands[gene.demand_idx]
        d = dp_by_id[demand.disaster_id]

        primary_qty = round(demand.quantity * gene.split_ratio)
        secondary_qty = demand.quantity - primary_qty

        for qty, wh_idx, route_idx in [
            (primary_qty, gene.primary_wh_idx, gene.primary_route_idx),
            (secondary_qty, gene.secondary_wh_idx, gene.secondary_route_idx),
        ]:
            if qty <= 0:
                continue
            if wh_idx < 0 or wh_idx >= len(warehouses):
                continue
            w = warehouses[wh_idx]

            route_list = routes.get((w.id, d.id), [])
            if not route_list:
                continue
            if route_idx < 0 or route_idx >= len(route_list):
                route_idx = 0
            ri = route_list[route_idx]

            available = remaining_inv[w.id].get(demand.supply_type, 0)
            actual_qty = min(qty, available)
            if actual_qty <= 0:
                continue

            remaining_inv[w.id][demand.supply_type] -= actual_qty
            delivered[d.id][demand.supply_type] += actual_qty

            segment_cost = ri.cost_per_unit * actual_qty
            total_cost += segment_cost

            truck_used[w.id] += 1

            if ri.travel_time > d.time_window.latest:
                tard = ri.travel_time - d.time_window.latest
                tardiness[d.id] = max(tardiness[d.id], tard)

            plan.add_segment(DeliverySegment(
                route_type=ri.route_type,
                from_id=w.id,
                to_id=d.id,
                supply_type=demand.supply_type,
                quantity=actual_qty,
                departure_time=0.0,
                arrival_time=ri.travel_time,
                distance=ri.distance,
                cost=segment_cost,
                line_id=ri.line_id,
                board_station_id=ri.board_station_id,
                alight_station_id=ri.alight_station_id,
            ))

    unmet: Dict[str, Dict[SupplyType, int]] = {}
    for d in disasters:
        gap = d.unmet_demand(delivered[d.id])
        if gap:
            unmet[d.id] = gap

    return plan, total_cost, unmet, tardiness, truck_used


def _fitness(
    plan: DeliveryPlan,
    total_cost: float,
    unmet: Dict[str, Dict[SupplyType, int]],
    tardiness: Dict[str, float],
    truck_used: Dict[str, int],
    warehouses: List[Warehouse],
    disasters: List[DisasterPoint],
    tardiness_penalty: float,
    unmet_penalty: float,
    truck_overload_penalty: float,
) -> float:
    cost = total_cost

    for d_id, gaps in unmet.items():
        for st, qty in gaps.items():
            cost += unmet_penalty * qty

    for d_id, tard in tardiness.items():
        cost += tardiness_penalty * tard

    for w in warehouses:
        overload = max(0, truck_used[w.id] - w.num_trucks)
        cost += truck_overload_penalty * overload

    return cost


class GASolver:
    def __init__(
        self,
        warehouses: List[Warehouse],
        disasters: List[DisasterPoint],
        subway: SubwayNetwork,
        population_size: int = 100,
        generations: int = 200,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.15,
        tournament_size: int = 5,
        elitism_count: int = 5,
        tardiness_penalty: float = 500.0,
        unmet_penalty: float = 1000.0,
        truck_overload_penalty: float = 800.0,
        max_transfer_dist: float = 15.0,
        seed: Optional[int] = None,
    ):
        self.warehouses = warehouses
        self.disasters = disasters
        self.subway = subway
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        self.tardiness_penalty = tardiness_penalty
        self.unmet_penalty = unmet_penalty
        self.truck_overload_penalty = truck_overload_penalty
        self.max_transfer_dist = max_transfer_dist

        self.rng = random.Random(seed)

        self.demands = _extract_demands(disasters)
        self.routes = _compute_route_info(warehouses, disasters, subway, max_transfer_dist)
        self.max_route_idx = max(
            len(rl) for rl in self.routes.values()
        ) if self.routes else 1

    def _random_gene(self, demand_idx: int) -> Gene:
        n_wh = len(self.warehouses)
        return Gene(
            demand_idx=demand_idx,
            primary_wh_idx=self.rng.randint(0, n_wh - 1),
            primary_route_idx=self.rng.randint(0, self.max_route_idx - 1),
            secondary_wh_idx=self.rng.randint(0, n_wh - 1),
            secondary_route_idx=self.rng.randint(0, self.max_route_idx - 1),
            split_ratio=round(self.rng.random(), 2),
        )

    def _random_chromosome(self) -> Chromosome:
        genes = [self._random_gene(i) for i in range(len(self.demands))]
        return Chromosome(genes=genes)

    def _evaluate(self, chrom: Chromosome) -> float:
        result = _decode_chromosome(
            chrom, self.demands, self.warehouses, self.disasters,
            self.routes, self.subway,
        )
        plan, total_cost, unmet, tard_dict, truck_used = result
        return _fitness(
            plan, total_cost, unmet, tard_dict, truck_used,
            self.warehouses, self.disasters,
            self.tardiness_penalty, self.unmet_penalty, self.truck_overload_penalty,
        )

    def _tournament_select(self, population: List[Chromosome], fitnesses: List[float]) -> Chromosome:
        candidates = self.rng.sample(
            list(range(len(population))),
            min(self.tournament_size, len(population)),
        )
        best_idx = min(candidates, key=lambda i: fitnesses[i])
        return population[best_idx].copy()

    def _crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        if self.rng.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        child1_genes = []
        child2_genes = []
        for g1, g2 in zip(parent1.genes, parent2.genes):
            if self.rng.random() < 0.5:
                child1_genes.append(copy.deepcopy(g1))
                child2_genes.append(copy.deepcopy(g2))
            else:
                child1_genes.append(copy.deepcopy(g2))
                child2_genes.append(copy.deepcopy(g1))

        return Chromosome(genes=child1_genes), Chromosome(genes=child2_genes)

    def _mutate(self, chrom: Chromosome) -> Chromosome:
        n_wh = len(self.warehouses)
        mutated = chrom.copy()
        for gene in mutated.genes:
            if self.rng.random() < self.mutation_rate:
                attr = self.rng.choice(["pwh", "proute", "swh", "sroute", "split"])
                if attr == "pwh":
                    gene.primary_wh_idx = self.rng.randint(0, n_wh - 1)
                elif attr == "proute":
                    gene.primary_route_idx = self.rng.randint(0, self.max_route_idx - 1)
                elif attr == "swh":
                    gene.secondary_wh_idx = self.rng.randint(0, n_wh - 1)
                elif attr == "sroute":
                    gene.secondary_route_idx = self.rng.randint(0, self.max_route_idx - 1)
                elif attr == "split":
                    gene.split_ratio = round(self.rng.random(), 2)
        return mutated

    def _local_search(self, chrom: Chromosome) -> Chromosome:
        best_chrom = chrom.copy()
        best_fit = self._evaluate(best_chrom)
        n_wh = len(self.warehouses)

        for i, gene in enumerate(best_chrom.genes):
            for new_wh in range(n_wh):
                if new_wh == gene.primary_wh_idx:
                    continue
                candidate = best_chrom.copy()
                candidate.genes[i].primary_wh_idx = new_wh
                fit = self._evaluate(candidate)
                if fit < best_fit:
                    best_fit = fit
                    best_chrom = candidate

        return best_chrom

    def solve(self, verbose: bool = False) -> SolverResult:
        start_time = time.time()

        population = [self._random_chromosome() for _ in range(self.population_size)]
        fitnesses = [self._evaluate(ch) for ch in population]

        best_idx = min(range(len(fitnesses)), key=lambda i: fitnesses[i])
        best_chrom = population[best_idx].copy()
        best_fitness = fitnesses[best_idx]

        for gen in range(self.generations):
            new_population: List[Chromosome] = []

            sorted_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])
            for i in range(min(self.elitism_count, len(population))):
                new_population.append(population[sorted_indices[i]].copy())

            while len(new_population) < self.population_size:
                parent1 = self._tournament_select(population, fitnesses)
                parent2 = self._tournament_select(population, fitnesses)
                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)

            population = new_population
            fitnesses = [self._evaluate(ch) for ch in population]

            current_best_idx = min(range(len(fitnesses)), key=lambda i: fitnesses[i])
            if fitnesses[current_best_idx] < best_fitness:
                best_fitness = fitnesses[current_best_idx]
                best_chrom = population[current_best_idx].copy()

            if verbose and (gen + 1) % 50 == 0:
                print(f"  GA Gen {gen+1}/{self.generations}: best_fitness={best_fitness:.2f}")

        if len(self.demands) <= 30:
            best_chrom = self._local_search(best_chrom)

        result = _decode_chromosome(
            best_chrom, self.demands, self.warehouses, self.disasters,
            self.routes, self.subway,
        )
        plan, total_cost, unmet, tard_dict, truck_used = result

        tardiness_cost = sum(self.tardiness_penalty * t for t in tard_dict.values())
        total_cost_with_penalty = total_cost + tardiness_cost
        for d_id, gaps in unmet.items():
            for st, qty in gaps.items():
                total_cost_with_penalty += self.unmet_penalty * qty

        solve_time = time.time() - start_time

        return SolverResult(
            plan=plan,
            total_cost=total_cost_with_penalty,
            total_time=plan.total_time(),
            solved_by="Genetic Algorithm",
            solve_time_seconds=solve_time,
            is_optimal=False,
            unmet_demands=unmet,
        )
