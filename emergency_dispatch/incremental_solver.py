from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import pulp

from .dynamic_env import DynamicEnvironment
from .ga_solver import (
    Chromosome,
    DemandItem,
    GASolver,
    Gene,
)
from .milp_solver import (
    RouteInfo,
    _compute_route_info,
)
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
class WarmStartValues:
    x_values: Dict[Tuple, int] = field(default_factory=dict)
    y_values: Dict[Tuple, int] = field(default_factory=dict)
    tardiness_values: Dict[str, float] = field(default_factory=dict)


@dataclass
class ReplannerContext:
    last_solver_result: Optional[SolverResult] = None
    last_chromosome: Optional[Chromosome] = None
    warm_start: Optional[WarmStartValues] = None
    replan_count: int = 0


class IncrementalMILPSolver:
    def __init__(
        self,
        warehouses: List[Warehouse],
        subway: SubwayNetwork,
        env: DynamicEnvironment,
        tardiness_penalty: float = 500.0,
        unmet_penalty: float = 1000.0,
        max_transfer_dist: float = 15.0,
        time_limit: Optional[float] = 60.0,
    ):
        self.warehouses = warehouses
        self.subway = subway
        self.env = env
        self.tardiness_penalty = tardiness_penalty
        self.unmet_penalty = unmet_penalty
        self.max_transfer_dist = max_transfer_dist
        self.time_limit = time_limit

    def _get_adjusted_disasters(
        self,
        current_time: float,
        time_window_relaxation: float = 1.5,
    ) -> List[DisasterPoint]:
        adjusted = []
        for dp in self.env.disasters():
            state = self.env.states[dp.id]
            remaining = state.remaining_demand()
            if not remaining:
                continue
            new_dp = DisasterPoint(
                id=dp.id,
                location=dp.location,
                demand=remaining,
                time_window=state.disaster.time_window,
                priority=dp.priority,
            )
            remaining_time = new_dp.time_window.latest - current_time
            if remaining_time < 0:
                relaxed_latest = new_dp.time_window.latest + abs(remaining_time) * 0.5
                new_dp.time_window = type(new_dp.time_window)(
                    earliest=current_time,
                    latest=max(current_time + 0.5, relaxed_latest),
                )
            else:
                new_dp.time_window = type(new_dp.time_window)(
                    earliest=current_time,
                    latest=new_dp.time_window.latest,
                )
            adjusted.append(new_dp)
        return adjusted

    def _extract_warm_start_from_result(
        self,
        result: SolverResult,
        routes: Dict[Tuple[str, str], List[RouteInfo]],
    ) -> WarmStartValues:
        ws = WarmStartValues()
        for seg in result.plan.segments:
            w_id = seg.from_id
            d_id = seg.to_id
            route_list = routes.get((w_id, d_id), [])
            matched_r_idx = -1
            for r_idx, ri in enumerate(route_list):
                if ri.route_type == seg.route_type:
                    if seg.line_id and ri.line_id == seg.line_id:
                        matched_r_idx = r_idx
                        break
                    elif not seg.line_id:
                        matched_r_idx = r_idx
                        break
            if matched_r_idx >= 0:
                key = (w_id, d_id, seg.supply_type, matched_r_idx)
                ws.x_values[key] = ws.x_values.get(key, 0) + seg.quantity
                y_key = (w_id, d_id, matched_r_idx)
                ws.y_values[y_key] = 1
        return ws

    def _get_available_inventory(
        self,
    ) -> Dict[Tuple[str, SupplyType], int]:
        inv = {}
        for w in self.warehouses:
            for st in SupplyType:
                avail = self.env.execution.available_warehouse_inventory(w, st)
                if avail > 0:
                    inv[(w.id, st)] = avail
        return inv

    def solve(
        self,
        current_time: float,
        warm_start: Optional[WarmStartValues] = None,
        affected_ids: Optional[Set[str]] = None,
        time_window_relaxation: float = 1.5,
    ) -> Tuple[SolverResult, WarmStartValues]:
        start_time = time.time()
        disasters = self._get_adjusted_disasters(current_time, time_window_relaxation)

        if not disasters:
            return SolverResult(
                plan=DeliveryPlan(),
                total_cost=0.0,
                total_time=0.0,
                solved_by="Incremental MILP (no unserved)",
                solve_time_seconds=time.time() - start_time,
                is_optimal=True,
            ), WarmStartValues()

        available_inv = self._get_available_inventory()
        routes = _compute_route_info(
            self.warehouses, disasters, self.subway, self.max_transfer_dist
        )

        prob = pulp.LpProblem("IncrementalDispatch", pulp.LpMinimize)
        supply_types = list(SupplyType)
        route_keys: List[Tuple[str, str, SupplyType, int]] = []
        x = {}
        y = {}

        for w in self.warehouses:
            for d in disasters:
                route_list = routes.get((w.id, d.id), [])
                for r_idx, ri in enumerate(route_list):
                    for st in supply_types:
                        key = (w.id, d.id, st, r_idx)
                        route_keys.append(key)
                        x[key] = pulp.LpVariable(
                            f"x_{w.id}_{d.id}_{st.value}_{r_idx}",
                            lowBound=0, cat=pulp.LpInteger,
                        )
                    y[(w.id, d.id, r_idx)] = pulp.LpVariable(
                        f"y_{w.id}_{d.id}_{r_idx}", cat=pulp.LpBinary,
                    )

        tardiness = {}
        unmet = {}
        for d in disasters:
            tardiness[d.id] = pulp.LpVariable(
                f"tard_{d.id}", lowBound=0, cat=pulp.LpContinuous,
            )
            for st in supply_types:
                if st.value in [v.value for v in d.demand]:
                    unmet[(d.id, st)] = pulp.LpVariable(
                        f"unmet_{d.id}_{st.value}", lowBound=0, cat=pulp.LpInteger,
                    )

        M = 1000
        obj_terms = []
        for key in route_keys:
            w_id, d_id, st, r_idx = key
            ri = routes[(w_id, d_id)][r_idx]
            obj_terms.append(ri.cost_per_unit * x[key])
        for d in disasters:
            obj_terms.append(self.tardiness_penalty * d.priority * tardiness[d.id])
        for (d_id, st), var in unmet.items():
            dp = next(d for d in disasters if d.id == d_id)
            obj_terms.append(self.unmet_penalty * dp.priority * var)
        prob += pulp.lpSum(obj_terms)

        for d in disasters:
            for st in supply_types:
                req = d.demand.get(st, 0)
                if req <= 0:
                    continue
                key_um = (d.id, st)
                prob += (
                    pulp.lpSum(
                        x[(w.id, d.id, st, r_idx)]
                        for w in self.warehouses
                        for r_idx in range(len(routes.get((w.id, d.id), [])))
                    ) + unmet.get(key_um, 0) >= req,
                    f"demand_{d.id}_{st.value}",
                )

        for w in self.warehouses:
            for st in supply_types:
                avail = available_inv.get((w.id, st), 0)
                if avail <= 0:
                    for d in disasters:
                        for r_idx in range(len(routes.get((w.id, d.id), []))):
                            key = (w.id, d.id, st, r_idx)
                            if key in x:
                                prob += x[key] == 0
                    continue
                prob += (
                    pulp.lpSum(
                        x[(w.id, d.id, st, r_idx)]
                        for d in disasters
                        for r_idx in range(len(routes.get((w.id, d.id), [])))
                    ) <= avail,
                    f"inv_{w.id}_{st.value}",
                )

        for key in route_keys:
            w_id, d_id, st, r_idx = key
            prob += x[key] <= M * y[(w_id, d_id, r_idx)]

        for w in self.warehouses:
            rc_terms = []
            for d in disasters:
                for r_idx in range(len(routes.get((w.id, d.id), []))):
                    rc_terms.append(y[(w.id, d.id, r_idx)])
            prob += pulp.lpSum(rc_terms) <= w.num_trucks

        for d in disasters:
            for w in self.warehouses:
                route_list = routes.get((w.id, d.id), [])
                for r_idx, ri in enumerate(route_list):
                    arr = current_time + ri.travel_time
                    deadline = d.time_window.latest
                    prob += (
                        arr - deadline
                        <= tardiness[d.id] + M * (1 - y[(w.id, d.id, r_idx)])
                    )

        for line in self.subway.lines.values():
            terms = []
            for w in self.warehouses:
                for d in disasters:
                    rl = routes.get((w.id, d.id), [])
                    for r_idx, ri in enumerate(rl):
                        if ri.route_type == RouteType.MULTIMODAL and ri.line_id == line.id:
                            for st in supply_types:
                                terms.append(x[(w.id, d.id, st, r_idx)])
            if terms:
                rem_hours = max(0.5, 24.0 - current_time)
                max_trips = int(rem_hours / max(0.05, line.trip_interval))
                prob += pulp.lpSum(terms) <= line.capacity_per_trip * max_trips

        if warm_start:
            for key, val in warm_start.x_values.items():
                if key in x and val > 0:
                    try:
                        x[key].setInitialValue(min(val, x[key].upBound or 10000))
                    except Exception:
                        pass
            for key, val in warm_start.y_values.items():
                if key in y:
                    try:
                        y[key].setInitialValue(val)
                    except Exception:
                        pass

        solver = pulp.PULP_CBC_CMD(
            timeLimit=self.time_limit, msg=0,
            warmStart=True if warm_start else False,
            keepFiles=True if warm_start else False,
        )
        status = prob.solve(solver)
        solve_time = time.time() - start_time
        is_optimal = pulp.LpStatus[status] == "Optimal"

        plan = DeliveryPlan()
        total_cost = 0.0
        new_ws = WarmStartValues()

        for key in route_keys:
            w_id, d_id, st, r_idx = key
            val = int(round(pulp.value(x[key]) or 0))
            if val <= 0:
                continue
            ri = routes[(w_id, d_id)][r_idx]
            seg_cost = ri.cost_per_unit * val
            total_cost += seg_cost
            plan.add_segment(DeliverySegment(
                route_type=ri.route_type, from_id=w_id, to_id=d_id,
                supply_type=st, quantity=val,
                departure_time=current_time,
                arrival_time=current_time + ri.travel_time,
                distance=ri.distance, cost=seg_cost,
                line_id=ri.line_id,
                board_station_id=ri.board_station_id,
                alight_station_id=ri.alight_station_id,
            ))
            new_ws.x_values[key] = val
            new_ws.y_values[(w_id, d_id, r_idx)] = 1

        for d in disasters:
            t_val = pulp.value(tardiness[d.id]) or 0.0
            total_cost += self.tardiness_penalty * d.priority * max(0, t_val)
            new_ws.tardiness_values[d.id] = t_val
        for (d_id, st), var in unmet.items():
            v = int(round(pulp.value(var) or 0))
            if v > 0:
                dp = next(d for d in disasters if d.id == d_id)
                total_cost += self.unmet_penalty * dp.priority * v

        unmet_map: Dict[str, Dict[SupplyType, int]] = {}
        for d in disasters:
            gap = plan.unmet_demand(d)
            if gap:
                unmet_map[d.id] = gap

        return SolverResult(
            plan=plan, total_cost=total_cost,
            total_time=plan.total_time(),
            solved_by=f"Incremental MILP (warm={'yes' if warm_start else 'no'})",
            solve_time_seconds=solve_time, is_optimal=is_optimal,
            unmet_demands=unmet_map,
        ), new_ws


class IncrementalGASolver:
    def __init__(
        self,
        warehouses: List[Warehouse],
        subway: SubwayNetwork,
        env: DynamicEnvironment,
        population_size: int = 80,
        generations: int = 80,
        tardiness_penalty: float = 500.0,
        unmet_penalty: float = 1000.0,
        max_transfer_dist: float = 15.0,
        niche_enabled: bool = True,
        adaptive_mutation: bool = True,
        use_phenotype_distance: bool = True,
        seed: Optional[int] = None,
    ):
        self.warehouses = warehouses
        self.subway = subway
        self.env = env
        self.population_size = population_size
        self.generations = generations
        self.tardiness_penalty = tardiness_penalty
        self.unmet_penalty = unmet_penalty
        self.max_transfer_dist = max_transfer_dist
        self.niche_enabled = niche_enabled
        self.adaptive_mutation = adaptive_mutation
        self.use_phenotype_distance = use_phenotype_distance
        self.seed = seed

    def _build_adjusted_problem(
        self,
        current_time: float,
        affected_ids: Optional[Set[str]] = None,
    ) -> Tuple[List[Warehouse], List[DisasterPoint], Dict[int, int]]:
        disasters = []
        idx_map = {}
        for dp in self.env.disasters():
            state = self.env.states[dp.id]
            remaining = state.remaining_demand()
            if not remaining:
                continue
            new_tw_earliest = current_time
            new_tw_latest = max(current_time + 0.5, state.disaster.time_window.latest)
            from .models import TimeWindow
            adj_dp = DisasterPoint(
                id=dp.id, location=dp.location, demand=remaining,
                time_window=TimeWindow(earliest=new_tw_earliest, latest=new_tw_latest),
                priority=dp.priority,
            )
            idx_map[len(disasters)] = dp.id
            disasters.append(adj_dp)

        adj_warehouses = []
        for w in self.warehouses:
            from dataclasses import replace
            new_inv = {}
            for st in SupplyType:
                avail = self.env.execution.available_warehouse_inventory(w, st)
                if avail > 0:
                    new_inv[st] = avail
            if new_inv:
                nw = replace(w, inventory=new_inv)
                adj_warehouses.append(nw)

        return adj_warehouses, disasters, idx_map

    def _seed_with_previous(
        self,
        solver: GASolver,
        last_chrom: Optional[Chromosome],
        last_demand_keys: List[Tuple[str, SupplyType]],
    ) -> List[Chromosome]:
        seeded = []
        if last_chrom is None or len(last_chrom.genes) != len(last_demand_keys):
            return seeded

        old_gene_by_key: Dict[Tuple[str, SupplyType], Gene] = {}
        for i, g in enumerate(last_chrom.genes):
            if i < len(last_demand_keys):
                old_gene_by_key[last_demand_keys[i]] = g

        new_demands = solver.demands
        new_dp_ids = [d.id for d in solver.disasters]

        seeded_genes = []
        matched = 0
        for demand in new_demands:
            dp_id = new_dp_ids[demand.disaster_idx]
            key = (dp_id, demand.supply_type)
            if key in old_gene_by_key:
                og = old_gene_by_key[key]
                max_routes = max(1, solver.max_route_idx - 1)
                nw = len(self.warehouses)
                seeded_genes.append(Gene(
                    demand_idx=demand.demand_idx,
                    primary_wh_idx=min(og.primary_wh_idx, nw - 1),
                    primary_route_idx=min(og.primary_route_idx, max_routes),
                    secondary_wh_idx=min(og.secondary_wh_idx, nw - 1),
                    secondary_route_idx=min(og.secondary_route_idx, max_routes),
                    split_ratio=og.split_ratio,
                ))
                matched += 1
            else:
                seeded_genes.append(solver._random_gene(demand.demand_idx))

        if matched > 0:
            seeded.append(Chromosome(genes=seeded_genes))
            n_mut = min(8, self.population_size // 5)
            for _ in range(n_mut):
                mutated = Chromosome(
                    genes=[copy.deepcopy(g) for g in seeded_genes]
                )
                mutated = solver._mutate(mutated, 0.25)
                seeded.append(mutated)
        return seeded

    def solve(
        self,
        current_time: float,
        last_result: Optional[SolverResult] = None,
        last_chromosome: Optional[Chromosome] = None,
        last_demand_keys: Optional[List[Tuple[str, SupplyType]]] = None,
        affected_ids: Optional[Set[str]] = None,
    ) -> Tuple[SolverResult, Chromosome, List[Tuple[str, SupplyType]]]:
        adj_whs, adj_dps, idx_map = self._build_adjusted_problem(
            current_time, affected_ids
        )
        new_dp_ids = [d.id for d in adj_dps]

        if not adj_dps or not adj_whs:
            return SolverResult(
                plan=DeliveryPlan(), total_cost=0.0, total_time=0.0,
                solved_by="Incremental GA (no unserved)",
                solve_time_seconds=0.0, is_optimal=True,
            ), Chromosome(genes=[]), []

        solver = GASolver(
            warehouses=adj_whs, disasters=adj_dps, subway=self.subway,
            population_size=self.population_size,
            generations=self.generations,
            seed=self.seed,
            niche_enabled=self.niche_enabled,
            adaptive_mutation=self.adaptive_mutation,
            use_phenotype_distance=self.use_phenotype_distance,
            tardiness_penalty=self.tardiness_penalty,
            unmet_penalty=self.unmet_penalty,
            max_transfer_dist=self.max_transfer_dist,
        )

        new_demand_keys = [
            (new_dp_ids[d.disaster_idx], d.supply_type)
            for d in solver.demands
        ]

        seeded = self._seed_with_previous(
            solver, last_chromosome, last_demand_keys or []
        )

        result = solver.solve(verbose=False, initial_population=seeded if seeded else None)
        best_chrom = solver._best_chromosome_from_result(result)
        if best_chrom is None:
            best_chrom = Chromosome(genes=[])

        adjusted_segments = []
        for seg in result.plan.segments:
            adj_seg = copy.deepcopy(seg)
            adj_seg.departure_time = seg.departure_time + current_time
            adj_seg.arrival_time = seg.arrival_time + current_time
            adjusted_segments.append(adj_seg)
        result.plan.segments = adjusted_segments

        return result, best_chrom, new_demand_keys


def run_dynamic_simulation(
    env: DynamicEnvironment,
    solver_type: str = "ga",
    event_interval_minutes: float = 10.0,
    milp_time_limit: float = 30.0,
    ga_generations: int = 80,
    ga_population: int = 80,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> Dict:
    history: List[Dict] = []
    total_cost = 0.0
    total_segments = 0
    ctx = ReplannerContext()
    last_demand_keys: List[Tuple[str, SupplyType]] = []

    events = env.generate_events()
    if verbose:
        print(f"[动态模拟] 生成 {len(events)} 个事件，总时长 {env.total_hours:.1f}h")
        for e in events[:5]:
            print(f"   t={e.event_time:.2f}h: {e.description()}")
        if len(events) > 5:
            print(f"   ... 还有 {len(events)-5} 个事件")
        print()

    current_time = 0.0
    next_event_idx = 0

    while current_time < env.total_hours:
        next_time = min(env.total_hours, current_time + env.event_interval)
        triggered = []
        while next_event_idx < len(events) and events[next_event_idx].event_time <= next_time:
            triggered.append(events[next_event_idx])
            next_event_idx += 1

        affected: Set[str] = set()
        for event in triggered:
            if verbose:
                print(f"[t={current_time:.2f}h] {event.description()}")
            env.apply_event(event)
            for dp_id in env.get_affected_disaster_ids(event):
                affected.add(dp_id)

        if triggered or ctx.replan_count == 0:
            if verbose:
                print(f"  -> 重规划 #{ctx.replan_count + 1} "
                      f"(受影响 {len(affected)} 个灾情点)")

            if solver_type == "milp":
                milp = IncrementalMILPSolver(
                    env.warehouses, env.subway, env,
                    time_limit=milp_time_limit,
                )
                result, ws = milp.solve(
                    current_time,
                    warm_start=ctx.warm_start,
                    affected_ids=affected,
                )
                ctx.warm_start = ws
            else:
                ga = IncrementalGASolver(
                    env.warehouses, env.subway, env,
                    generations=ga_generations, population_size=ga_population,
                    seed=seed,
                )
                result, chrom, new_keys = ga.solve(
                    current_time,
                    last_result=ctx.last_solver_result,
                    last_chromosome=ctx.last_chromosome,
                    last_demand_keys=last_demand_keys,
                    affected_ids=affected,
                )
                ctx.last_chromosome = chrom
                last_demand_keys = new_keys

            ctx.last_solver_result = result
            ctx.replan_count += 1
            total_cost += result.total_cost
            total_segments += len(result.plan.segments)

            new_segs = [s for s in result.plan.segments
                        if s.departure_time >= current_time - 0.001]
            env.advance_execution(next_time, new_segs)

            unserved = env.get_unserved_disasters()
            if verbose:
                print(f"  -> 本批成本={result.total_cost:.1f}, "
                      f"新配送段={len(new_segs)}, "
                      f"未满足灾情点={len(unserved)}/{len(env.states)}")

        entry = {
            "time": current_time,
            "triggered_events": len(triggered),
            "affected_count": len(affected),
            "replan_cost": result.total_cost if triggered or ctx.replan_count == 1 else 0,
            "segments_added": len(new_segs) if (triggered or ctx.replan_count == 1) else 0,
            "unserved_count": len(unserved),
            "solve_time": result.solve_time_seconds if (triggered or ctx.replan_count == 1) else 0,
        }
        history.append(entry)

        current_time = next_time

    if verbose:
        print()
        print("=" * 60)
        print("  动态模拟汇总")
        print("=" * 60)
        print(f"  总重规划次数    : {ctx.replan_count}")
        print(f"  总配送成本      : {total_cost:.2f}")
        print(f"  总配送段数      : {total_segments}")
        print(f"  灾情点总数      : {len(env.states)}")
        fully = sum(1 for s in env.states.values() if s.is_fully_served())
        print(f"  完全满足的灾情点: {fully}/{len(env.states)} ({100*fully/len(env.states):.1f}%)")
        pending_total = 0
        for s in env.states.values():
            pending_total += sum(s.remaining_demand().values())
        print(f"  剩余未满足物资  : {pending_total}")
        print("=" * 60)

    return {
        "history": history,
        "total_cost": total_cost,
        "total_segments": total_segments,
        "replan_count": ctx.replan_count,
        "disaster_count": len(env.states),
        "fully_served": sum(1 for s in env.states.values() if s.is_fully_served()),
    }
