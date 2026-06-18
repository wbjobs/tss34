from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class SupplyType(Enum):
    WATER = "water"
    FOOD = "food"
    TENT = "tent"


class RouteType(Enum):
    DIRECT_TRUCK = "direct_truck"
    MULTIMODAL = "multimodal"


@dataclass
class Location:
    x: float
    y: float

    def distance_to(self, other: Location) -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def __hash__(self):
        return hash((round(self.x, 6), round(self.y, 6)))


@dataclass
class TimeWindow:
    earliest: float
    latest: float

    def is_within(self, time: float) -> bool:
        return self.earliest <= time <= self.latest

    def penalty(self, arrival_time: float) -> float:
        if arrival_time <= self.latest:
            return 0.0
        return (arrival_time - self.latest) * 100.0


@dataclass
class Warehouse:
    id: str
    location: Location
    inventory: Dict[SupplyType, int]
    num_trucks: int = 3
    truck_speed: float = 40.0
    truck_capacity: float = 50.0
    truck_cost_per_km: float = 5.0
    loading_time: float = 0.25

    def available_inventory(self, supply_type: SupplyType) -> int:
        return self.inventory.get(supply_type, 0)

    def total_inventory(self) -> int:
        return sum(self.inventory.values())


@dataclass
class DisasterPoint:
    id: str
    location: Location
    demand: Dict[SupplyType, int]
    time_window: TimeWindow
    priority: float = 1.0

    def total_demand(self) -> int:
        return sum(self.demand.values())

    def unmet_demand(self, delivered: Dict[SupplyType, int]) -> Dict[SupplyType, int]:
        result = {}
        for st, qty in self.demand.items():
            gap = qty - delivered.get(st, 0)
            if gap > 0:
                result[st] = gap
        return result


@dataclass
class SubwayStation:
    id: str
    location: Location
    line_id: str
    transfer_available: bool = False

    def distance_to(self, other: SubwayStation) -> float:
        return self.location.distance_to(other.location)


@dataclass
class SubwayLine:
    id: str
    stations: List[SubwayStation]
    travel_time_between: Dict[Tuple[str, str], float]
    capacity_per_trip: float = 80.0
    trip_interval: float = 0.2
    operating_hours: Tuple[float, float] = (0.0, 24.0)
    cost_per_unit_per_station: float = 1.0

    def get_travel_time(self, from_station_id: str, to_station_id: str) -> Optional[float]:
        key = (from_station_id, to_station_id)
        if key in self.travel_time_between:
            return self.travel_time_between[key]
        if (to_station_id, from_station_id) in self.travel_time_between:
            return self.travel_time_between[(to_station_id, from_station_id)]
        return None

    def station_by_id(self, station_id: str) -> Optional[SubwayStation]:
        for s in self.stations:
            if s.id == station_id:
                return s
        return None

    def stations_between(self, from_id: str, to_id: str) -> List[SubwayStation]:
        from_idx = None
        to_idx = None
        for i, s in enumerate(self.stations):
            if s.id == from_id:
                from_idx = i
            if s.id == to_id:
                to_idx = i
        if from_idx is None or to_idx is None:
            return []
        if from_idx <= to_idx:
            return self.stations[from_idx:to_idx + 1]
        else:
            return list(reversed(self.stations[to_idx:from_idx + 1]))


@dataclass
class SubwayNetwork:
    lines: Dict[str, SubwayLine] = field(default_factory=dict)

    def add_line(self, line: SubwayLine):
        self.lines[line.id] = line

    def all_transfer_stations(self) -> List[SubwayStation]:
        result = []
        for line in self.lines.values():
            for s in line.stations:
                if s.transfer_available:
                    result.append(s)
        return result

    def nearest_transfer_station(self, loc: Location, line_id: Optional[str] = None) -> Optional[Tuple[SubwayStation, float]]:
        best = None
        best_dist = float('inf')
        for line in self.lines.values():
            if line_id and line.id != line_id:
                continue
            for s in line.stations:
                if s.transfer_available:
                    d = s.location.distance_to(loc)
                    if d < best_dist:
                        best_dist = d
                        best = s
        if best:
            return (best, best_dist)
        return None

    def find_multimodal_route(self, from_loc: Location, to_loc: Location) -> Optional[MultimodalRoute]:
        best_route = None
        best_total_time = float('inf')

        for line in self.lines.values():
            transfer_stations = [s for s in line.stations if s.transfer_available]
            for ts_from in transfer_stations:
                for ts_to in transfer_stations:
                    if ts_from.id == ts_to.id:
                        continue
                    travel_time = line.get_travel_time(ts_from.id, ts_to.id)
                    if travel_time is None:
                        continue
                    total_time = travel_time
                    if total_time < best_total_time:
                        best_total_time = total_time
                        best_route = MultimodalRoute(
                            board_station=ts_from,
                            alight_station=ts_to,
                            line=line,
                            subway_travel_time=travel_time,
                        )

        return best_route


@dataclass
class MultimodalRoute:
    board_station: SubwayStation
    alight_station: SubwayStation
    line: SubwayLine
    subway_travel_time: float


@dataclass
class DeliverySegment:
    route_type: RouteType
    from_id: str
    to_id: str
    supply_type: SupplyType
    quantity: int
    departure_time: float
    arrival_time: float
    distance: float
    cost: float
    line_id: Optional[str] = None
    board_station_id: Optional[str] = None
    alight_station_id: Optional[str] = None


@dataclass
class DeliveryPlan:
    segments: List[DeliverySegment] = field(default_factory=list)

    def add_segment(self, seg: DeliverySegment):
        self.segments.append(seg)

    def total_cost(self) -> float:
        return sum(seg.cost for seg in self.segments)

    def total_time(self) -> float:
        if not self.segments:
            return 0.0
        return max(seg.arrival_time for seg in self.segments)

    def delivered_to(self, disaster_id: str) -> Dict[SupplyType, int]:
        result: Dict[SupplyType, int] = {}
        for seg in self.segments:
            if seg.to_id == disaster_id:
                result[seg.supply_type] = result.get(seg.supply_type, 0) + seg.quantity
        return result

    def unmet_demand(self, disaster: DisasterPoint) -> Dict[SupplyType, int]:
        delivered = self.delivered_to(disaster.id)
        return disaster.unmet_demand(delivered)


@dataclass
class SolverResult:
    plan: DeliveryPlan
    total_cost: float
    total_time: float
    solved_by: str
    solve_time_seconds: float = 0.0
    is_optimal: bool = False
    unmet_demands: Dict[str, Dict[SupplyType, int]] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"{'='*60}",
            f"  城市应急物资调度 - 求解结果报告",
            f"{'='*60}",
            f"  求解方法: {self.solved_by}",
            f"  是否最优: {'是' if self.is_optimal else '否'}",
            f"  求解耗时: {self.solve_time_seconds:.2f} 秒",
            f"  总成本  : {self.total_cost:.2f}",
            f"  总时间  : {self.total_time:.2f} 小时",
            f"{'-'*60}",
        ]
        for seg in self.plan.segments:
            rt = "直达卡车" if seg.route_type == RouteType.DIRECT_TRUCK else "多模态"
            line_info = f" [地铁线{seg.line_id}]" if seg.line_id else ""
            lines.append(
                f"  {rt}{line_info}: {seg.from_id} -> {seg.to_id} | "
                f"{seg.supply_type.value} x{seg.quantity} | "
                f"出发 {seg.departure_time:.2f}h -> 到达 {seg.arrival_time:.2f}h | "
                f"距离 {seg.distance:.1f}km | 费用 {seg.cost:.1f}"
            )
        lines.append(f"{'-'*60}")
        if self.unmet_demands:
            lines.append("  未满足需求:")
            for dp_id, unmet in self.unmet_demands.items():
                for st, qty in unmet.items():
                    lines.append(f"    {dp_id}: {st.value} 缺 {qty}")
        else:
            lines.append("  所有需求已满足 [OK]")
        lines.append(f"{'='*60}")
        return "\n".join(lines)
