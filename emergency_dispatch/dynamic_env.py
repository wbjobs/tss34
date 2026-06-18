from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from .models import (
    DeliverySegment,
    DisasterPoint,
    Location,
    SupplyType,
    TimeWindow,
    Warehouse,
)


class EventType(Enum):
    NEW_DISASTER = "new_disaster"
    DEMAND_INCREASE = "demand_increase"


@dataclass
class DisasterEvent:
    event_type: EventType
    event_time: float
    disaster_id: str
    details: Dict = field(default_factory=dict)

    def description(self) -> str:
        if self.event_type == EventType.NEW_DISASTER:
            pos = self.details.get("location", Location(0, 0))
            return f"[新增灾情] {self.disaster_id} 位置({pos.x:.1f},{pos.y:.1f})"
        else:
            inc = self.details.get("increase", {})
            inc_str = ", ".join(f"{st.value}+{qty}" for st, qty in inc.items())
            return f"[需求增加] {self.disaster_id}: {inc_str}"


@dataclass
class DisasterState:
    disaster: DisasterPoint
    created_at: float
    original_demand: Dict[SupplyType, int]
    delivered: Dict[SupplyType, int] = field(default_factory=dict)
    in_transit: Dict[SupplyType, int] = field(default_factory=dict)

    def remaining_demand(self) -> Dict[SupplyType, int]:
        result = {}
        for st, qty in self.disaster.demand.items():
            done = self.delivered.get(st, 0) + self.in_transit.get(st, 0)
            gap = qty - done
            if gap > 0:
                result[st] = gap
        return result

    def is_fully_served(self) -> bool:
        return len(self.remaining_demand()) == 0


@dataclass
class ExecutionState:
    current_time: float = 0.0
    completed_segments: List[DeliverySegment] = field(default_factory=list)
    in_transit_segments: List[DeliverySegment] = field(default_factory=list)
    committed_warehouse_usage: Dict[Tuple[str, SupplyType], int] = field(default_factory=dict)

    def advance_time(self, to_time: float):
        still_in_transit = []
        for seg in self.in_transit_segments:
            if seg.arrival_time <= to_time:
                self.completed_segments.append(seg)
            else:
                still_in_transit.append(seg)
        self.in_transit_segments = still_in_transit
        self.current_time = to_time

    def commit_segments(self, segments: List[DeliverySegment], warehouses: List[Warehouse]):
        wh_by_id = {w.id: w for w in warehouses}
        for seg in segments:
            self.in_transit_segments.append(seg)
            key = (seg.from_id, seg.supply_type)
            self.committed_warehouse_usage[key] = (
                self.committed_warehouse_usage.get(key, 0) + seg.quantity
            )

    def available_warehouse_inventory(
        self, warehouse: Warehouse, supply_type: SupplyType
    ) -> int:
        base = warehouse.available_inventory(supply_type)
        used = self.committed_warehouse_usage.get((warehouse.id, supply_type), 0)
        return max(0, base - used)


class DynamicEnvironment:
    def __init__(
        self,
        initial_warehouses: List[Warehouse],
        initial_disasters: List[DisasterPoint],
        subway_network: SubwayNetwork = None,
        event_interval_minutes: float = 10.0,
        total_simulation_hours: float = 6.0,
        new_disaster_probability: float = 0.5,
        map_bounds: Tuple[float, float, float, float] = (0.0, 0.0, 25.0, 25.0),
        seed: Optional[int] = None,
    ):
        self.warehouses = [w for w in initial_warehouses]
        self.subway = subway_network
        self.states: Dict[str, DisasterState] = {}
        self.events: List[DisasterEvent] = []
        self.event_interval = event_interval_minutes / 60.0
        self.total_hours = total_simulation_hours
        self.new_disaster_prob = new_disaster_probability
        self.map_bounds = map_bounds
        self.rng = random.Random(seed)
        self.next_disaster_counter = 100
        self.execution = ExecutionState()

        for dp in initial_disasters:
            self.states[dp.id] = DisasterState(
                disaster=dp,
                created_at=0.0,
                original_demand=dict(dp.demand),
            )

    def disasters(self) -> List[DisasterPoint]:
        return [s.disaster for s in self.states.values()]

    def _generate_random_location(self) -> Location:
        x0, y0, x1, y1 = self.map_bounds
        return Location(
            x=self.rng.uniform(x0, x1),
            y=self.rng.uniform(y0, y1),
        )

    def _generate_random_demand(self) -> Dict[SupplyType, int]:
        demand = {}
        for st in SupplyType:
            if self.rng.random() < 0.85:
                demand[st] = self.rng.randint(5, 40)
        return demand

    def _generate_time_window(self, event_time: float) -> TimeWindow:
        earliest = event_time
        latest = event_time + self.rng.uniform(3.0, 6.0)
        return TimeWindow(earliest=earliest, latest=latest)

    def generate_events(self) -> List[DisasterEvent]:
        self.events = []
        t = self.event_interval
        while t < self.total_hours:
            if self.rng.random() < self.new_disaster_prob:
                self._generate_new_disaster_event(t)
            else:
                self._generate_demand_increase_event(t)
            t += self.event_interval
        return self.events

    def _generate_new_disaster_event(self, t: float):
        dp_id = f"D{self.next_disaster_counter}"
        self.next_disaster_counter += 1
        location = self._generate_random_location()
        demand = self._generate_random_demand()
        tw = self._generate_time_window(t)
        priority = round(self.rng.uniform(0.8, 2.5), 1)
        event = DisasterEvent(
            event_type=EventType.NEW_DISASTER,
            event_time=t,
            disaster_id=dp_id,
            details={
                "location": location,
                "demand": demand,
                "time_window": tw,
                "priority": priority,
            },
        )
        self.events.append(event)

    def _generate_demand_increase_event(self, t: float):
        existing = list(self.states.keys())
        if not existing:
            self._generate_new_disaster_event(t)
            return
        dp_id = self.rng.choice(existing)
        increase = {}
        for st in SupplyType:
            if self.rng.random() < 0.4:
                increase[st] = self.rng.randint(3, 20)
        if not increase:
            st = self.rng.choice(list(SupplyType))
            increase[st] = self.rng.randint(3, 20)
        event = DisasterEvent(
            event_type=EventType.DEMAND_INCREASE,
            event_time=t,
            disaster_id=dp_id,
            details={"increase": increase},
        )
        self.events.append(event)

    def apply_event(self, event: DisasterEvent) -> DisasterPoint:
        if event.event_type == EventType.NEW_DISASTER:
            dp = DisasterPoint(
                id=event.disaster_id,
                location=event.details["location"],
                demand=dict(event.details["demand"]),
                time_window=event.details["time_window"],
                priority=event.details["priority"],
            )
            self.states[event.disaster_id] = DisasterState(
                disaster=dp,
                created_at=event.event_time,
                original_demand=dict(dp.demand),
            )
            return dp
        else:
            state = self.states[event.disaster_id]
            increase: Dict[SupplyType, int] = event.details["increase"]
            for st, qty in increase.items():
                state.disaster.demand[st] = state.disaster.demand.get(st, 0) + qty
            return state.disaster

    def get_unserved_disasters(self) -> List[DisasterPoint]:
        result = []
        for state in self.states.values():
            if not state.is_fully_served():
                result.append(state.disaster)
        return result

    def pending_events(self, from_time: float, to_time: float) -> List[DisasterEvent]:
        return [e for e in self.events if from_time < e.event_time <= to_time]

    def get_affected_disaster_ids(
        self, event: DisasterEvent, radius_km: float = 15.0
    ) -> List[str]:
        affected = [event.disaster_id]
        source_state = self.states.get(event.disaster_id)
        if source_state is None:
            return affected
        source_loc = source_state.disaster.location
        for dp_id, state in self.states.items():
            if dp_id == event.disaster_id:
                continue
            if source_loc.distance_to(state.disaster.location) <= radius_km:
                affected.append(dp_id)
        return affected

    def advance_execution(self, to_time: float, new_segments: List[DeliverySegment]):
        self.execution.commit_segments(new_segments, self.warehouses)
        self.execution.advance_time(to_time)

        for seg in self.execution.completed_segments:
            if seg.to_id in self.states:
                self.states[seg.to_id].delivered[seg.supply_type] = (
                    self.states[seg.to_id].delivered.get(seg.supply_type, 0) + seg.quantity
                )
            key = (seg.from_id, seg.supply_type)
            if key in self.execution.committed_warehouse_usage:
                self.execution.committed_warehouse_usage[key] = max(
                    0,
                    self.execution.committed_warehouse_usage[key] - seg.quantity,
                )

        remaining_completed = []
        for seg in self.execution.completed_segments:
            for st in self.states.values():
                pass
            remaining_completed.append(seg)
        self.execution.completed_segments = remaining_completed[-500:]
