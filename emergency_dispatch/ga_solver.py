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

    def distance_to(self, other: Gene, n_wh: int, max_route_idx: int) -> float:
        if self.demand_idx != other.demand_idx:
            return 1.0
        dist = 0.0
        if self.primary_wh_idx != other.primary_wh_idx:
            dist += 0.3
        if self.primary_route_idx != other.primary_route_idx:
            dist += 0.15
        if self.secondary_wh_idx != other.secondary_wh_idx:
            dist += 0.3
        if self.secondary_route_idx != other.secondary_route_idx:
            dist += 0.15
        dist += abs(self.split_ratio - other.split_ratio) * 0.1
        return min(1.0, dist)


@dataclass
class Chromosome:
    genes: List[Gene]

    def copy(self) -> Chromosome:
        return Chromosome(genes=[copy.deepcopy(g) for g in self.genes])

    def genotype_distance_to(
        self,
        other: Chromosome,
        n_wh: int,
        max_route_idx: int,
    ) -> float:
        if len(self.genes) != len(other.genes):
            return 1.0
        total = 0.0
        for g1, g2 in zip(self.genes, other.genes):
            total += g1.distance_to(g2, n_wh, max_route_idx)
        return total / len(self.genes)

    def signature(self) -> Tuple:
        sig = []
        for g in self.genes:
            sig.append((
                g.primary_wh_idx, g.primary_route_idx,
                g.secondary_wh_idx, g.secondary_route_idx,
                round(g.split_ratio, 2),
            ))
        return tuple(sig)


@dataclass
class DemandItem:
    disaster_id: str
    supply_type: SupplyType
    quantity: int
    disaster_idx: int
    demand_idx: int


@dataclass
class DiversityMetrics:
    avg_distance: float
    min_distance: float
    max_distance: float
    unique_count: int
    sharing_count: int


def _extract_demands(disasters: List[DisasterPoint]) -> List[DemandItem]:
    items = []
    demand_idx = 0
    for d_idx, d in enumerate(disasters):
        for st in SupplyType:
            qty = d.demand.get(st, 0)
            if qty > 0:
                items.append(DemandItem(
                    disaster_id=d.id,
                    supply_type=st,
                    quantity=qty,
                    disaster_idx=d_idx,
                    demand_idx=demand_idx,
                ))
                demand_idx += 1
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


def _raw_fitness(
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


def _extract_phenotype_signature(
    chrom: Chromosome,
    demands: List[DemandItem],
    warehouses: List[Warehouse],
    routes: Dict[Tuple[str, str], List[RouteInfo]],
) -> Tuple:
    sig = []
    dp_by_idx = {}
    for demand in demands:
        dp_by_idx.setdefault(demand.disaster_idx, []).append(demand)

    for d_idx in sorted(dp_by_idx.keys()):
        items = dp_by_idx[d_idx]
        for demand in items:
            gene = chrom.genes[demand.demand_idx]
            w1 = warehouses[gene.primary_wh_idx]
            w2 = warehouses[gene.secondary_wh_idx]
            rlist1 = routes.get((w1.id, demands[demand.demand_idx].disaster_id), [])
            rlist2 = routes.get((w2.id, demands[demand.demand_idx].disaster_id), [])
            r1_idx = gene.primary_route_idx if rlist1 and 0 <= gene.primary_route_idx < len(rlist1) else 0
            r2_idx = gene.secondary_route_idx if rlist2 and 0 <= gene.secondary_route_idx < len(rlist2) else 0
            r1_type = rlist1[r1_idx].route_type.value if rlist1 else 'none'
            r2_type = rlist2[r2_idx].route_type.value if rlist2 else 'none'
            sig.append((
                gene.primary_wh_idx, r1_idx, r1_type,
                gene.secondary_wh_idx, r2_idx, r2_type,
                round(gene.split_ratio, 1),
            ))
    return tuple(sig)


def _phenotype_edit_distance(
    sig1: Tuple,
    sig2: Tuple,
) -> float:
    n = len(sig1)
    if n == 0:
        return 0.0
    mismatches = 0.0
    for a, b in zip(sig1, sig2):
        if a != b:
            if a[0] != b[0] or a[3] != b[3]:
                mismatches += 1.0
            elif a[1] != b[1] or a[4] != b[4]:
                mismatches += 0.5
            elif a[2] != b[2] or a[5] != b[5]:
                mismatches += 0.3
            elif abs(a[6] - b[6]) > 0.1:
                mismatches += 0.2
    return min(1.0, mismatches / n)


def _compute_pairwise_distances(
    population: List[Chromosome],
    n_wh: int,
    max_route_idx: int,
    demands: Optional[List[DemandItem]] = None,
    warehouses: Optional[List[Warehouse]] = None,
    routes: Optional[Dict[Tuple[str, str], List[RouteInfo]]] = None,
    sample_size: Optional[int] = None,
    rng: Optional[random.Random] = None,
    phenotype_cache: Optional[Dict[int, Tuple]] = None,
    use_phenotype: bool = True,
) -> List[List[float]]:
    n = len(population)

    if use_phenotype and phenotype_cache is not None and demands and warehouses and routes:
        sigs = []
        for i, ch in enumerate(population):
            ch_id = id(ch)
            if ch_id not in phenotype_cache:
                phenotype_cache[ch_id] = _extract_phenotype_signature(
                    ch, demands, warehouses, routes
                )
            sigs.append(phenotype_cache[ch_id])

    if sample_size and n > sample_size and rng:
        indices = sorted(rng.sample(range(n), sample_size))
        dist_matrix = [[0.0] * n for _ in range(n)]
        for i_idx, i in enumerate(indices):
            for j_idx, j in enumerate(indices):
                if i_idx >= j_idx:
                    continue
                if use_phenotype and sigs:
                    d = _phenotype_edit_distance(sigs[i], sigs[j])
                else:
                    d = population[i].genotype_distance_to(population[j], n_wh, max_route_idx)
                dist_matrix[i][j] = d
                dist_matrix[j][i] = d
        return dist_matrix

    dist_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if use_phenotype and sigs:
                d = _phenotype_edit_distance(sigs[i], sigs[j])
            else:
                d = population[i].genotype_distance_to(population[j], n_wh, max_route_idx)
            dist_matrix[i][j] = d
            dist_matrix[j][i] = d
    return dist_matrix


def _compute_sharing_factors(
    dist_matrix: List[List[float]],
    niche_radius: float,
    niche_alpha: float = 1.0,
) -> List[float]:
    n = len(dist_matrix)
    sharing = []
    for i in range(n):
        sh = 0.0
        for j in range(n):
            d = dist_matrix[i][j]
            if d < niche_radius:
                sh += 1.0 - (d / niche_radius) ** niche_alpha
        sharing.append(sh)
    return sharing


def _compute_diversity_metrics(
    population: List[Chromosome],
    dist_matrix: List[List[float]],
) -> DiversityMetrics:
    n = len(population)
    if n <= 1:
        return DiversityMetrics(0.0, 0.0, 0.0, 1, 0)

    all_dists = []
    for i in range(n):
        for j in range(i + 1, n):
            all_dists.append(dist_matrix[i][j])

    avg_distance = sum(all_dists) / len(all_dists)
    min_distance = min(all_dists)
    max_distance = max(all_dists)

    unique_sigs = set()
    for ch in population:
        unique_sigs.add(ch.signature())
    unique_count = len(unique_sigs)

    sharing_count = sum(1 for d in all_dists if d < 0.3)

    return DiversityMetrics(
        avg_distance=avg_distance,
        min_distance=min_distance,
        max_distance=max_distance,
        unique_count=unique_count,
        sharing_count=sharing_count,
    )


def _adaptive_mutation_rate(
    gen: int,
    total_gens: int,
    initial_rate: float = 0.35,
    final_rate: float = 0.08,
    mid_point: float = 0.5,
) -> float:
    progress = gen / max(1, total_gens - 1)
    if progress < mid_point:
        t = progress / mid_point
        return initial_rate - (initial_rate - (initial_rate + final_rate) / 2) * (t ** 0.5)
    else:
        t = (progress - mid_point) / (1.0 - mid_point)
        mid_rate = (initial_rate + final_rate) / 2
        return mid_rate - (mid_rate - final_rate) * t


class GASolver:
    def __init__(
        self,
        warehouses: List[Warehouse],
        disasters: List[DisasterPoint],
        subway: SubwayNetwork,
        population_size: int = 100,
        generations: int = 200,
        crossover_rate: float = 0.85,
        base_mutation_rate: float = 0.15,
        mutation_initial_rate: float = 0.35,
        mutation_final_rate: float = 0.08,
        tournament_size: int = 5,
        elitism_count: int = 5,
        tardiness_penalty: float = 500.0,
        unmet_penalty: float = 1000.0,
        truck_overload_penalty: float = 800.0,
        max_transfer_dist: float = 15.0,
        niche_enabled: bool = True,
        niche_radius: float = 0.25,
        niche_alpha: float = 1.0,
        niche_penalty_strength: float = 0.3,
        adaptive_mutation: bool = True,
        use_phenotype_distance: bool = True,
        diversity_sample_size: Optional[int] = 50,
        seed: Optional[int] = None,
    ):
        self.warehouses = warehouses
        self.disasters = disasters
        self.subway = subway
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.base_mutation_rate = base_mutation_rate
        self.mutation_initial_rate = mutation_initial_rate
        self.mutation_final_rate = mutation_final_rate
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        self.tardiness_penalty = tardiness_penalty
        self.unmet_penalty = unmet_penalty
        self.truck_overload_penalty = truck_overload_penalty
        self.max_transfer_dist = max_transfer_dist
        self.niche_enabled = niche_enabled
        self.niche_radius = niche_radius
        self.niche_alpha = niche_alpha
        self.niche_penalty_strength = niche_penalty_strength
        self.adaptive_mutation = adaptive_mutation
        self.use_phenotype_distance = use_phenotype_distance
        self.diversity_sample_size = diversity_sample_size

        self.rng = random.Random(seed)

        self.demands = _extract_demands(disasters)
        self.routes = _compute_route_info(warehouses, disasters, subway, max_transfer_dist)
        self.max_route_idx = max(
            len(rl) for rl in self.routes.values()
        ) if self.routes else 1
        self.n_wh = len(warehouses)

        self.phenotype_cache: Dict[int, Tuple] = {}
        self.diversity_history: List[DiversityMetrics] = []
        self.raw_fitness_history: List[float] = []
        self.shared_fitness_history: List[float] = []

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

    def _evaluate_raw(self, chrom: Chromosome) -> float:
        result = _decode_chromosome(
            chrom, self.demands, self.warehouses, self.disasters,
            self.routes, self.subway,
        )
        plan, total_cost, unmet, tard_dict, truck_used = result
        return _raw_fitness(
            plan, total_cost, unmet, tard_dict, truck_used,
            self.warehouses, self.disasters,
            self.tardiness_penalty, self.unmet_penalty, self.truck_overload_penalty,
        )

    def _evaluate_with_sharing(
        self,
        population: List[Chromosome],
        raw_fitnesses: List[float],
        dist_matrix: List[List[float]],
        gen: int = 0,
        total_gens: int = 1,
    ) -> List[float]:
        if not self.niche_enabled:
            return raw_fitnesses.copy()

        n = len(population)

        progress = gen / max(1, total_gens - 1)
        if progress < 0.2 or progress > 0.8:
            return raw_fitnesses.copy()

        elite_count = max(1, self.elitism_count)
        sorted_indices = sorted(range(n), key=lambda i: raw_fitnesses[i])
        elite_set = set(sorted_indices[:elite_count])

        neighbor_counts = [0] * n
        for i in range(n):
            for j in range(n):
                if i != j and dist_matrix[i][j] < self.niche_radius:
                    neighbor_counts[i] += 1

        max_neighbors = max(neighbor_counts) if neighbor_counts else 1
        crowd_threshold = n * 0.1

        shared_fitnesses = []
        for i, raw in enumerate(raw_fitnesses):
            if i in elite_set:
                penalty = 1.0
            elif neighbor_counts[i] <= crowd_threshold:
                penalty = 1.0
            else:
                excess = neighbor_counts[i] - crowd_threshold
                density_ratio = excess / max(1, max_neighbors - crowd_threshold)
                penalty = 1.0 / (1.0 + self.niche_penalty_strength * density_ratio)
            shared_fitnesses.append(raw * penalty)

        return shared_fitnesses

    def _tournament_select(
        self,
        population: List[Chromosome],
        fitnesses: List[float],
    ) -> Chromosome:
        candidates = self.rng.sample(
            list(range(len(population))),
            min(self.tournament_size, len(population)),
        )
        best_idx = min(candidates, key=lambda i: fitnesses[i])
        return population[best_idx].copy()

    def _crossover(
        self,
        parent1: Chromosome,
        parent2: Chromosome,
    ) -> Tuple[Chromosome, Chromosome]:
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

    def _mutate(
        self,
        chrom: Chromosome,
        mutation_rate: float,
    ) -> Chromosome:
        n_wh = len(self.warehouses)
        mutated = chrom.copy()
        for gene in mutated.genes:
            if self.rng.random() < mutation_rate:
                attr = self.rng.choice(
                    ["pwh", "proute", "swh", "sroute", "split", "swap"]
                )
                if attr == "pwh":
                    gene.primary_wh_idx = self.rng.randint(0, n_wh - 1)
                elif attr == "proute":
                    gene.primary_route_idx = self.rng.randint(0, self.max_route_idx - 1)
                elif attr == "swh":
                    gene.secondary_wh_idx = self.rng.randint(0, n_wh - 1)
                elif attr == "sroute":
                    gene.secondary_route_idx = self.rng.randint(0, self.max_route_idx - 1)
                elif attr == "split":
                    delta = self.rng.uniform(-0.3, 0.3)
                    gene.split_ratio = max(0.0, min(1.0, gene.split_ratio + delta))
                    gene.split_ratio = round(gene.split_ratio, 2)
                elif attr == "swap":
                    gene.primary_wh_idx, gene.secondary_wh_idx = (
                        gene.secondary_wh_idx, gene.primary_wh_idx
                    )
                    gene.primary_route_idx, gene.secondary_route_idx = (
                        gene.secondary_route_idx, gene.primary_route_idx
                    )
                    gene.split_ratio = round(1.0 - gene.split_ratio, 2)
        return mutated

    def _local_search(self, chrom: Chromosome) -> Chromosome:
        best_chrom = chrom.copy()
        best_fit = self._evaluate_raw(best_chrom)
        n_wh = len(self.warehouses)

        for i, gene in enumerate(best_chrom.genes):
            for new_wh in range(n_wh):
                if new_wh == gene.primary_wh_idx:
                    continue
                candidate = best_chrom.copy()
                candidate.genes[i].primary_wh_idx = new_wh
                fit = self._evaluate_raw(candidate)
                if fit < best_fit:
                    best_fit = fit
                    best_chrom = candidate

        return best_chrom

    def _inject_random_immigrants(
        self,
        population: List[Chromosome],
        fitnesses: List[float],
        num_immigrants: int,
    ) -> Tuple[List[Chromosome], List[float]]:
        if num_immigrants <= 0:
            return population, fitnesses

        sorted_idx = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])
        worst_indices = sorted_idx[-num_immigrants:]

        new_pop = population.copy()
        new_fit = fitnesses.copy()
        for idx in worst_indices:
            new_pop[idx] = self._random_chromosome()
            new_fit[idx] = self._evaluate_raw(new_pop[idx])

        return new_pop, new_fit

    def solve(self, verbose: bool = False,
              initial_population: Optional[List[Chromosome]] = None) -> SolverResult:
        start_time = time.time()

        self.diversity_history.clear()
        self.raw_fitness_history.clear()
        self.shared_fitness_history.clear()
        self._last_best_chromosome = None

        population = [self._random_chromosome() for _ in range(self.population_size)]
        if initial_population:
            for i, ch in enumerate(initial_population):
                if i < len(population) and len(ch.genes) == len(self.demands):
                    for g in ch.genes:
                        g.demand_idx = min(g.demand_idx, len(self.demands) - 1)
                    population[i] = ch
        raw_fitnesses = [self._evaluate_raw(ch) for ch in population]

        dist_matrix = _compute_pairwise_distances(
            population, self.n_wh, self.max_route_idx,
            self.demands, self.warehouses, self.routes,
            self.diversity_sample_size, self.rng,
            self.phenotype_cache, self.use_phenotype_distance,
        )
        shared_fitnesses = self._evaluate_with_sharing(
            population, raw_fitnesses, dist_matrix, 0, self.generations
        )

        best_idx = min(range(len(raw_fitnesses)), key=lambda i: raw_fitnesses[i])
        best_chrom = population[best_idx].copy()
        best_raw_fitness = raw_fitnesses[best_idx]

        diversity = _compute_diversity_metrics(population, dist_matrix)
        self.diversity_history.append(diversity)
        self.raw_fitness_history.append(best_raw_fitness)
        best_shared = min(shared_fitnesses)
        self.shared_fitness_history.append(best_shared)

        early_stuck_gens = 0
        prev_best = best_raw_fitness

        for gen in range(self.generations):
            if self.adaptive_mutation:
                current_mutation_rate = _adaptive_mutation_rate(
                    gen, self.generations,
                    self.mutation_initial_rate,
                    self.mutation_final_rate,
                )
            else:
                current_mutation_rate = self.base_mutation_rate

            new_population: List[Chromosome] = []

            for_use_fitnesses = shared_fitnesses if self.niche_enabled else raw_fitnesses

            sorted_indices = sorted(range(len(for_use_fitnesses)), key=lambda i: for_use_fitnesses[i])
            for i in range(min(self.elitism_count, len(population))):
                new_population.append(population[sorted_indices[i]].copy())

            while len(new_population) < self.population_size:
                parent1 = self._tournament_select(population, for_use_fitnesses)
                parent2 = self._tournament_select(population, for_use_fitnesses)
                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutate(child1, current_mutation_rate)
                child2 = self._mutate(child2, current_mutation_rate)
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)

            population = new_population
            raw_fitnesses = [self._evaluate_raw(ch) for ch in population]

            dist_matrix = _compute_pairwise_distances(
                population, self.n_wh, self.max_route_idx,
                self.demands, self.warehouses, self.routes,
                self.diversity_sample_size, self.rng,
                self.phenotype_cache, self.use_phenotype_distance,
            )
            shared_fitnesses = self._evaluate_with_sharing(
                population, raw_fitnesses, dist_matrix, gen, self.generations
            )

            current_best_idx = min(range(len(raw_fitnesses)), key=lambda i: raw_fitnesses[i])
            if raw_fitnesses[current_best_idx] < best_raw_fitness:
                best_raw_fitness = raw_fitnesses[current_best_idx]
                best_chrom = population[current_best_idx].copy()
                early_stuck_gens = 0
            else:
                early_stuck_gens += 1

            diversity = _compute_diversity_metrics(population, dist_matrix)
            self.diversity_history.append(diversity)
            self.raw_fitness_history.append(best_raw_fitness)
            best_shared = min(shared_fitnesses)
            self.shared_fitness_history.append(best_shared)

            if self.niche_enabled and diversity.unique_count < self.population_size * 0.1:
                num_immigrants = max(5, int(self.population_size * 0.15))
                population, raw_fitnesses = self._inject_random_immigrants(
                    population, raw_fitnesses, num_immigrants
                )
                dist_matrix = _compute_pairwise_distances(
                    population, self.n_wh, self.max_route_idx,
                    self.demands, self.warehouses, self.routes,
                    self.diversity_sample_size, self.rng,
                    self.phenotype_cache, self.use_phenotype_distance,
                )
                shared_fitnesses = self._evaluate_with_sharing(
                    population, raw_fitnesses, dist_matrix, gen, self.generations
                )
                if verbose:
                    print(f"  [Diversity Rescue] Gen {gen+1}: 注入 {num_immigrants} 个随机移民")

            if verbose and (gen + 1) % 50 == 0:
                print(
                    f"  GA Gen {gen+1}/{self.generations}: "
                    f"best_raw={best_raw_fitness:.2f}, "
                    f"best_shared={best_shared:.2f}, "
                    f"mut_rate={current_mutation_rate:.3f}, "
                    f"avg_dist={diversity.avg_distance:.3f}, "
                    f"unique={diversity.unique_count}"
                )

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
        self._last_best_chromosome = best_chrom

        return SolverResult(
            plan=plan,
            total_cost=total_cost_with_penalty,
            total_time=plan.total_time(),
            solved_by="Genetic Algorithm (with Niche & Adaptive Mutation)",
            solve_time_seconds=solve_time,
            is_optimal=False,
            unmet_demands=unmet,
        )

    def _best_chromosome_from_result(self, result: SolverResult) -> Optional[Chromosome]:
        return getattr(self, '_last_best_chromosome', None)
