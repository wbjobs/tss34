from __future__ import annotations

import io
from typing import List, Optional

from .models import (
    DisasterPoint,
    RouteType,
    SolverResult,
    SubwayNetwork,
    SupplyType,
    Warehouse,
)


def generate_report(
    result: SolverResult,
    warehouses: List[Warehouse],
    disasters: List[DisasterPoint],
    subway: SubwayNetwork,
    title: Optional[str] = None,
) -> str:
    buf = io.StringIO()

    if title is None:
        title = "城市应急物资调度 - 求解结果报告"

    buf.write(f"{'='*70}\n")
    buf.write(f"  {title}\n")
    buf.write(f"{'='*70}\n\n")

    buf.write(f"  求解方法      : {result.solved_by}\n")
    buf.write(f"  是否最优      : {'是' if result.is_optimal else '否'}\n")
    buf.write(f"  求解耗时      : {result.solve_time_seconds:.2f} 秒\n")
    buf.write(f"  总成本(含罚金) : {result.total_cost:.2f}\n")
    buf.write(f"  最晚到达时间  : {result.total_time:.2f} 小时\n\n")

    buf.write(f"{'-'*70}\n")
    buf.write(f"  一、问题规模\n")
    buf.write(f"{'-'*70}\n")
    buf.write(f"  仓库数量      : {len(warehouses)}\n")
    buf.write(f"  灾情点数量    : {len(disasters)}\n")
    buf.write(f"  地铁线路数量  : {len(subway.lines)}\n")

    total_demand = sum(d.total_demand() for d in disasters)
    total_inventory = sum(w.total_inventory() for w in warehouses)
    buf.write(f"  总需求量      : {total_demand}\n")
    buf.write(f"  总库存量      : {total_inventory}\n\n")

    buf.write(f"{'-'*70}\n")
    buf.write(f"  二、仓库信息\n")
    buf.write(f"{'-'*70}\n")
    for w in warehouses:
        inv_str = ", ".join(f"{st.value}={w.inventory.get(st, 0)}" for st in SupplyType)
        buf.write(f"  {w.id}: 位置({w.location.x:.1f}, {w.location.y:.1f}) "
                  f"库存[{inv_str}] 卡车数={w.num_trucks}\n")
    buf.write("\n")

    buf.write(f"{'-'*70}\n")
    buf.write(f"  三、灾情点信息\n")
    buf.write(f"{'-'*70}\n")
    for d in disasters:
        dem_str = ", ".join(f"{st.value}={d.demand.get(st, 0)}" for st in SupplyType)
        buf.write(f"  {d.id}: 位置({d.location.x:.1f}, {d.location.y:.1f}) "
                  f"需求[{dem_str}] 时间窗[{d.time_window.earliest:.1f}h-{d.time_window.latest:.1f}h] "
                  f"优先级={d.priority:.1f}\n")
    buf.write("\n")

    buf.write(f"{'-'*70}\n")
    buf.write(f"  四、地铁网络\n")
    buf.write(f"{'-'*70}\n")
    for line in subway.lines.values():
        stations_str = " -> ".join(s.id for s in line.stations)
        transfer_str = ", ".join(s.id for s in line.stations if s.transfer_available)
        buf.write(f"  线路 {line.id}: {stations_str}\n")
        buf.write(f"    换乘站: {transfer_str}\n")
        buf.write(f"    每班容量: {line.capacity_per_trip}, 班次间隔: {line.trip_interval}h\n")
    buf.write("\n")

    buf.write(f"{'-'*70}\n")
    buf.write(f"  五、配送路线详情\n")
    buf.write(f"{'-'*70}\n")

    if not result.plan.segments:
        buf.write("  (无配送路线)\n")
    else:
        direct_segs = [s for s in result.plan.segments if s.route_type == RouteType.DIRECT_TRUCK]
        multi_segs = [s for s in result.plan.segments if s.route_type == RouteType.MULTIMODAL]

        if direct_segs:
            buf.write("\n  [直达卡车配送]\n")
            buf.write(f"  {'序号':<4} {'起点':<8} {'终点':<8} {'物资':<6} {'数量':<4} "
                      f"{'出发时间':<10} {'到达时间':<10} {'距离km':<8} {'费用':<8}\n")
            buf.write(f"  {'-'*70}\n")
            for i, seg in enumerate(direct_segs, 1):
                buf.write(f"  {i:<4} {seg.from_id:<8} {seg.to_id:<8} "
                          f"{seg.supply_type.value:<6} {seg.quantity:<4} "
                          f"{seg.departure_time:<10.2f} {seg.arrival_time:<10.2f} "
                          f"{seg.distance:<8.1f} {seg.cost:<8.1f}\n")

        if multi_segs:
            buf.write("\n  [多模态配送 (卡车+地铁+卡车)]\n")
            buf.write(f"  {'序号':<4} {'起点':<8} {'终点':<8} {'物资':<6} {'数量':<4} "
                      f"{'地铁线':<8} {'上车站':<10} {'下车站':<10} "
                      f"{'出发时间':<10} {'到达时间':<10} {'费用':<8}\n")
            buf.write(f"  {'-'*90}\n")
            for i, seg in enumerate(multi_segs, 1):
                buf.write(f"  {i:<4} {seg.from_id:<8} {seg.to_id:<8} "
                          f"{seg.supply_type.value:<6} {seg.quantity:<4} "
                          f"{seg.line_id or '-':<8} {seg.board_station_id or '-':<10} "
                          f"{seg.alight_station_id or '-':<10} "
                          f"{seg.departure_time:<10.2f} {seg.arrival_time:<10.2f} "
                          f"{seg.cost:<8.1f}\n")

    buf.write("\n")

    buf.write(f"{'-'*70}\n")
    buf.write(f"  六、成本汇总\n")
    buf.write(f"{'-'*70}\n")
    transport_cost = sum(seg.cost for seg in result.plan.segments)
    penalty_cost = max(0.0, result.total_cost - transport_cost)
    buf.write(f"  运输成本      : {transport_cost:.2f}\n")
    buf.write(f"  罚金(迟到+未满足) : {penalty_cost:.2f}\n")
    buf.write(f"  总成本        : {result.total_cost:.2f}\n\n")

    buf.write(f"{'-'*70}\n")
    buf.write(f"  七、需求满足情况\n")
    buf.write(f"{'-'*70}\n")
    all_satisfied = True
    for d in disasters:
        delivered = result.plan.delivered_to(d.id)
        dem_str_parts = []
        for st in SupplyType:
            need = d.demand.get(st, 0)
            got = delivered.get(st, 0)
            if need > 0:
                status = f"{got}/{need}"
                if got >= need:
                    status += " [OK]"
                else:
                    status += " [X]"
                    all_satisfied = False
                dem_str_parts.append(f"{st.value}={status}")
        buf.write(f"  {d.id}: {', '.join(dem_str_parts)}\n")

    if all_satisfied:
        buf.write("\n  [*] 所有灾情点需求已满足!\n")
    else:
        buf.write("\n  [X] 部分需求未满足，建议增加库存或调整时间窗。\n")

    buf.write(f"\n{'='*70}\n")

    return buf.getvalue()
