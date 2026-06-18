from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import pulp

from .models import (
    DeliveryPlan,
    DeliverySegment,
    DisasterPoint,
    Location,
    MultimodalRoute,
    RouteType,
    SolverResult,
    SubwayLine,
    SubwayNetwork,
    SupplyType,
    TimeWindow,
    Warehouse,
)


class RouteInfo:
    def __init__(
        self,
        route_type: RouteType,
        from_id: str,
        to_id: str,
        travel_time: float,
        distance: float,
        cost_per_unit: float,
        line_id: Optional[str] = None,
        board_station_id: Optional[str] = None,
        alight_station_id: Optional[str] = None,
    ):
        self.route_type = route_type
        self.from_id = from_id
        self.to_id = to_id
        self.travel_time = travel_time
        self.distance = distance
        self.cost_per_unit = cost_per_unit
        self.line_id = line_id
        self.board_station_id = board_station_id
        self.alight_station_id = alight_station_id


def _compute_route_info(
    warehouses: List[Warehouse],
    disasters: List[DisasterPoint],
    subway: SubwayNetwork,
    max_transfer_dist: float = 15.0,
) -> Dict[Tuple[str, str], List[RouteInfo]]:
    routes: Dict[Tuple[str, str], List[RouteInfo]] = {}

    for w in warehouses:
        for d in disasters:
            key = (w.id, d.id)
            route_list: List[RouteInfo] = []

            direct_dist = w.location.distance_to(d.location)
            direct_time = w.loading_time + direct_dist / w.truck_speed
            direct_cost = direct_dist * w.truck_cost_per_km
            route_list.append(RouteInfo(
                route_type=RouteType.DIRECT_TRUCK,
                from_id=w.id,
                to_id=d.id,
                travel_time=direct_time,
                distance=direct_dist,
                cost_per_unit=direct_cost,
            ))

            for line in subway.lines.values():
                transfer_stations = [s for s in line.stations if s.transfer_available]
                best_board = None
                best_alight = None
                best_total_time = float('inf')
                best_truck1_dist = 0.0
                best_truck2_dist = 0.0
                best_subway_time = 0.0

                for ts_board in transfer_stations:
                    truck1_dist = w.location.distance_to(ts_board.location)
                    if truck1_dist > max_transfer_dist:
                        continue
                    for ts_alight in transfer_stations:
                        if ts_board.id == ts_alight.id:
                            continue
                        truck2_dist = ts_alight.location.distance_to(d.location)
                        if truck2_dist > max_transfer_dist:
                            continue
                        subway_time = line.get_travel_time(ts_board.id, ts_alight.id)
                        if subway_time is None:
                            continue
                        total_time = (
                            w.loading_time
                            + truck1_dist / w.truck_speed
                            + subway_time
                            + w.loading_time
                            + truck2_dist / w.truck_speed
                        )
                        if total_time < best_total_time:
                            best_total_time = total_time
                            best_board = ts_board
                            best_alight = ts_alight
                            best_truck1_dist = truck1_dist
                            best_truck2_dist = truck2_dist
                            best_subway_time = subway_time

                if best_board and best_alight:
                    total_dist = best_truck1_dist + best_truck2_dist
                    num_stations = len(line.stations_between(best_board.id, best_alight.id))
                    subway_cost = num_stations * line.cost_per_unit_per_station
                    total_cost = (
                        best_truck1_dist * w.truck_cost_per_km
                        + subway_cost
                        + best_truck2_dist * w.truck_cost_per_km
                    )
                    route_list.append(RouteInfo(
                        route_type=RouteType.MULTIMODAL,
                        from_id=w.id,
                        to_id=d.id,
                        travel_time=best_total_time,
                        distance=total_dist,
                        cost_per_unit=total_cost,
                        line_id=line.id,
                        board_station_id=best_board.id,
                        alight_station_id=best_alight.id,
                    ))

            routes[key] = route_list

    return routes


class MILPSolver:
    def __init__(
        self,
        warehouses: List[Warehouse],
        disasters: List[DisasterPoint],
        subway: SubwayNetwork,
        tardiness_penalty: float = 500.0,
        max_transfer_dist: float = 15.0,
        time_limit: Optional[float] = 120.0,
    ):
        self.warehouses = warehouses
        self.disasters = disasters
        self.subway = subway
        self.tardiness_penalty = tardiness_penalty
        self.max_transfer_dist = max_transfer_dist
        self.time_limit = time_limit

        self.wh_by_id = {w.id: w for w in warehouses}
        self.dp_by_id = {d.id: d for d in disasters}

        self.routes = _compute_route_info(warehouses, disasters, subway, max_transfer_dist)

    def solve(self) -> SolverResult:
        start_time = time.time()

        prob = pulp.LpProblem("EmergencyDispatch", pulp.LpMinimize)

        supply_types = list(SupplyType)
        route_keys: List[Tuple[str, str, SupplyType, int]] = []
        x = {}
        y = {}

        for w in self.warehouses:
            for d in self.disasters:
                route_list = self.routes.get((w.id, d.id), [])
                for r_idx, ri in enumerate(route_list):
                    for st in supply_types:
                        key = (w.id, d.id, st, r_idx)
                        route_keys.append(key)
                        x[key] = pulp.LpVariable(
                            f"x_{w.id}_{d.id}_{st.value}_{r_idx}",
                            lowBound=0,
                            cat=pulp.LpInteger,
                        )
                    y[(w.id, d.id, r_idx)] = pulp.LpVariable(
                        f"y_{w.id}_{d.id}_{r_idx}",
                        cat=pulp.LpBinary,
                    )

        tardiness = {}
        for d in self.disasters:
            tardiness[d.id] = pulp.LpVariable(
                f"tardiness_{d.id}", lowBound=0, cat=pulp.LpContinuous
            )

        M = 1000

        obj_terms = []
        for key in route_keys:
            w_id, d_id, st, r_idx = key
            ri = self.routes[(w_id, d_id)][r_idx]
            obj_terms.append(ri.cost_per_unit * x[key])
        for d in self.disasters:
            obj_terms.append(self.tardiness_penalty * tardiness[d.id])
        prob += pulp.lpSum(obj_terms)

        for d in self.disasters:
            for st in supply_types:
                prob += (
                    pulp.lpSum(
                        x[(w.id, d.id, st, r_idx)]
                        for w in self.warehouses
                        for r_idx in range(len(self.routes.get((w.id, d.id), [])))
                    ) >= d.demand.get(st, 0),
                    f"demand_{d.id}_{st.value}",
                )

        for w in self.warehouses:
            for st in supply_types:
                prob += (
                    pulp.lpSum(
                        x[(w.id, d.id, st, r_idx)]
                        for d in self.disasters
                        for r_idx in range(len(self.routes.get((w.id, d.id), [])))
                    ) <= w.inventory.get(st, 0),
                    f"inventory_{w.id}_{st.value}",
                )

        for key in route_keys:
            w_id, d_id, st, r_idx = key
            prob += (
                x[key] <= M * y[(w_id, d_id, r_idx)],
                f"bigM_{w_id}_{d_id}_{st.value}_{r_idx}",
            )

        for w in self.warehouses:
            route_count_terms = []
            for d in self.disasters:
                for r_idx in range(len(self.routes.get((w.id, d.id), []))):
                    route_count_terms.append(y[(w.id, d.id, r_idx)])
            prob += (
                pulp.lpSum(route_count_terms) <= w.num_trucks,
                f"trucks_{w.id}",
            )

        for d in self.disasters:
            for w in self.warehouses:
                route_list = self.routes.get((w.id, d.id), [])
                for r_idx, ri in enumerate(route_list):
                    prob += (
                        ri.travel_time - d.time_window.latest
                        <= tardiness[d.id] + M * (1 - y[(w.id, d.id, r_idx)]),
                        f"tardiness_{w.id}_{d.id}_{r_idx}",
                    )

        for line in self.subway.lines.values():
            subway_x_terms = []
            for w in self.warehouses:
                for d in self.disasters:
                    route_list = self.routes.get((w.id, d.id), [])
                    for r_idx, ri in enumerate(route_list):
                        if ri.route_type == RouteType.MULTIMODAL and ri.line_id == line.id:
                            for st in supply_types:
                                subway_x_terms.append(x[(w.id, d.id, st, r_idx)])
            if subway_x_terms:
                max_trips = int(24.0 / line.trip_interval)
                total_capacity = line.capacity_per_trip * max_trips
                prob += (
                    pulp.lpSum(subway_x_terms) <= total_capacity,
                    f"subway_cap_{line.id}",
                )

        solver = pulp.PULP_CBC_CMD(
            timeLimit=self.time_limit,
            msg=0,
        )
        status = prob.solve(solver)

        solve_time = time.time() - start_time
        is_optimal = pulp.LpStatus[status] == "Optimal"

        plan = DeliveryPlan()
        total_cost = 0.0

        for key in route_keys:
            w_id, d_id, st, r_idx = key
            val = int(round(pulp.value(x[key]) or 0))
            if val <= 0:
                continue
            ri = self.routes[(w_id, d_id)][r_idx]
            segment_cost = ri.cost_per_unit * val
            total_cost += segment_cost

            dep_time = 0.0
            arr_time = ri.travel_time

            plan.add_segment(DeliverySegment(
                route_type=ri.route_type,
                from_id=w_id,
                to_id=d_id,
                supply_type=st,
                quantity=val,
                departure_time=dep_time,
                arrival_time=arr_time,
                distance=ri.distance,
                cost=segment_cost,
                line_id=ri.line_id,
                board_station_id=ri.board_station_id,
                alight_station_id=ri.alight_station_id,
            ))

        unmet: Dict[str, Dict[SupplyType, int]] = {}
        for d in self.disasters:
            gap = plan.unmet_demand(d)
            if gap:
                unmet[d.id] = gap

        tardiness_cost = 0.0
        for d in self.disasters:
            t_val = pulp.value(tardiness[d.id]) or 0.0
            tardiness_cost += self.tardiness_penalty * max(0, t_val)
        total_cost += tardiness_cost

        return SolverResult(
            plan=plan,
            total_cost=total_cost,
            total_time=plan.total_time(),
            solved_by="MILP (PuLP/CBC)",
            solve_time_seconds=solve_time,
            is_optimal=is_optimal,
            unmet_demands=unmet,
        )
