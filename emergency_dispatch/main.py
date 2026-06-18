from __future__ import annotations

import argparse
import sys
import random
from typing import List, Tuple

from .models import (
    DisasterPoint,
    Location,
    SubwayLine,
    SubwayNetwork,
    SubwayStation,
    SupplyType,
    TimeWindow,
    Warehouse,
)
from .milp_solver import MILPSolver
from .ga_solver import GASolver
from .report import generate_report


def build_small_problem() -> Tuple[List[Warehouse], List[DisasterPoint], SubwayNetwork]:
    warehouses = [
        Warehouse(
            id="W1",
            location=Location(x=2.0, y=3.0),
            inventory={SupplyType.WATER: 60, SupplyType.FOOD: 50, SupplyType.TENT: 20},
            num_trucks=3,
            truck_speed=40.0,
            truck_capacity=50.0,
            truck_cost_per_km=5.0,
            loading_time=0.25,
        ),
        Warehouse(
            id="W2",
            location=Location(x=15.0, y=12.0),
            inventory={SupplyType.WATER: 40, SupplyType.FOOD: 60, SupplyType.TENT: 15},
            num_trucks=3,
            truck_speed=40.0,
            truck_capacity=50.0,
            truck_cost_per_km=5.0,
            loading_time=0.25,
        ),
        Warehouse(
            id="W3",
            location=Location(x=8.0, y=18.0),
            inventory={SupplyType.WATER: 50, SupplyType.FOOD: 30, SupplyType.TENT: 25},
            num_trucks=2,
            truck_speed=40.0,
            truck_capacity=50.0,
            truck_cost_per_km=5.0,
            loading_time=0.25,
        ),
    ]

    disasters = [
        DisasterPoint(
            id="D1",
            location=Location(x=5.0, y=8.0),
            demand={SupplyType.WATER: 30, SupplyType.FOOD: 20, SupplyType.TENT: 5},
            time_window=TimeWindow(earliest=0.0, latest=4.0),
            priority=1.5,
        ),
        DisasterPoint(
            id="D2",
            location=Location(x=12.0, y=5.0),
            demand={SupplyType.WATER: 20, SupplyType.FOOD: 15, SupplyType.TENT: 8},
            time_window=TimeWindow(earliest=0.0, latest=3.5),
            priority=1.2,
        ),
        DisasterPoint(
            id="D3",
            location=Location(x=18.0, y=15.0),
            demand={SupplyType.WATER: 25, SupplyType.FOOD: 30, SupplyType.TENT: 10},
            time_window=TimeWindow(earliest=0.0, latest=4.5),
            priority=1.0,
        ),
        DisasterPoint(
            id="D4",
            location=Location(x=6.0, y=16.0),
            demand={SupplyType.WATER: 15, SupplyType.FOOD: 10, SupplyType.TENT: 3},
            time_window=TimeWindow(earliest=0.0, latest=3.0),
            priority=2.0,
        ),
        DisasterPoint(
            id="D5",
            location=Location(x=10.0, y=10.0),
            demand={SupplyType.WATER: 20, SupplyType.FOOD: 25, SupplyType.TENT: 7},
            time_window=TimeWindow(earliest=0.0, latest=4.0),
            priority=1.8,
        ),
    ]

    line1_stations = [
        SubwayStation("L1_S1", Location(x=3.0, y=5.0), "L1", transfer_available=True),
        SubwayStation("L1_S2", Location(x=6.0, y=7.0), "L1", transfer_available=False),
        SubwayStation("L1_S3", Location(x=9.0, y=9.0), "L1", transfer_available=True),
        SubwayStation("L1_S4", Location(x=12.0, y=11.0), "L1", transfer_available=False),
        SubwayStation("L1_S5", Location(x=15.0, y=13.0), "L1", transfer_available=True),
    ]

    line1_travel = {}
    for i in range(len(line1_stations)):
        for j in range(len(line1_stations)):
            if i != j:
                d = line1_stations[i].location.distance_to(line1_stations[j].location)
                line1_travel[(line1_stations[i].id, line1_stations[j].id)] = d / 60.0

    line1 = SubwayLine(
        id="L1",
        stations=line1_stations,
        travel_time_between=line1_travel,
        capacity_per_trip=80.0,
        trip_interval=0.2,
        operating_hours=(0.0, 24.0),
        cost_per_unit_per_station=1.0,
    )

    line2_stations = [
        SubwayStation("L2_S1", Location(x=7.0, y=3.0), "L2", transfer_available=True),
        SubwayStation("L2_S2", Location(x=8.0, y=6.0), "L2", transfer_available=False),
        SubwayStation("L2_S3", Location(x=9.0, y=9.0), "L2", transfer_available=True),
        SubwayStation("L2_S4", Location(x=10.0, y=12.0), "L2", transfer_available=False),
        SubwayStation("L2_S5", Location(x=11.0, y=15.0), "L2", transfer_available=True),
    ]

    line2_travel = {}
    for i in range(len(line2_stations)):
        for j in range(len(line2_stations)):
            if i != j:
                d = line2_stations[i].location.distance_to(line2_stations[j].location)
                line2_travel[(line2_stations[i].id, line2_stations[j].id)] = d / 60.0

    line2 = SubwayLine(
        id="L2",
        stations=line2_stations,
        travel_time_between=line2_travel,
        capacity_per_trip=60.0,
        trip_interval=0.25,
        operating_hours=(0.0, 24.0),
        cost_per_unit_per_station=1.5,
    )

    subway = SubwayNetwork()
    subway.add_line(line1)
    subway.add_line(line2)

    return warehouses, disasters, subway


def build_large_problem(
    num_warehouses: int = 5,
    num_disasters: int = 12,
    num_subway_lines: int = 3,
    seed: int = 42,
) -> Tuple[List[Warehouse], List[DisasterPoint], SubwayNetwork]:
    rng = random.Random(seed)

    warehouses = []
    for i in range(num_warehouses):
        w = Warehouse(
            id=f"W{i+1}",
            location=Location(x=rng.uniform(1, 25), y=rng.uniform(1, 25)),
            inventory={
                SupplyType.WATER: rng.randint(30, 80),
                SupplyType.FOOD: rng.randint(30, 80),
                SupplyType.TENT: rng.randint(10, 30),
            },
            num_trucks=rng.randint(2, 5),
            truck_speed=40.0,
            truck_capacity=50.0,
            truck_cost_per_km=5.0,
            loading_time=0.25,
        )
        warehouses.append(w)

    disasters = []
    for i in range(num_disasters):
        d = DisasterPoint(
            id=f"D{i+1}",
            location=Location(x=rng.uniform(1, 25), y=rng.uniform(1, 25)),
            demand={
                SupplyType.WATER: rng.randint(10, 40),
                SupplyType.FOOD: rng.randint(10, 35),
                SupplyType.TENT: rng.randint(3, 12),
            },
            time_window=TimeWindow(
                earliest=0.0,
                latest=rng.uniform(3.0, 6.0),
            ),
            priority=round(rng.uniform(0.8, 2.5), 1),
        )
        disasters.append(d)

    subway = SubwayNetwork()
    for li in range(num_subway_lines):
        num_stations = rng.randint(4, 7)
        stations = []
        angle_offset = li * (2 * 3.14159 / num_subway_lines)
        for si in range(num_stations):
            t = si / (num_stations - 1)
            cx = 12 + 10 * t * math_cos(angle_offset + t * 1.5)
            cy = 12 + 10 * t * math_sin(angle_offset + t * 1.5)
            is_transfer = (si == 0 or si == num_stations - 1 or si == num_stations // 2)
            stations.append(SubwayStation(
                f"L{li+1}_S{si+1}",
                Location(x=cx, y=cy),
                f"L{li+1}",
                transfer_available=is_transfer,
            ))

        travel = {}
        for a in range(len(stations)):
            for b in range(len(stations)):
                if a != b:
                    d = stations[a].location.distance_to(stations[b].location)
                    travel[(stations[a].id, stations[b].id)] = d / 60.0

        line = SubwayLine(
            id=f"L{li+1}",
            stations=stations,
            travel_time_between=travel,
            capacity_per_trip=rng.uniform(50, 100),
            trip_interval=rng.uniform(0.15, 0.3),
            operating_hours=(0.0, 24.0),
            cost_per_unit_per_station=rng.uniform(0.8, 2.0),
        )
        subway.add_line(line)

    return warehouses, disasters, subway


def math_cos(x: float) -> float:
    import math
    return math.cos(x)


def math_sin(x: float) -> float:
    import math
    return math.sin(x)


def main():
    parser = argparse.ArgumentParser(
        description="城市应急物资调度优化求解器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m emergency_dispatch --scale small --solver milp
  python -m emergency_dispatch --scale large --solver ga --generations 300
  python -m emergency_dispatch --scale small --solver both
        """,
    )
    parser.add_argument(
        "--scale",
        choices=["small", "large"],
        default="small",
        help="问题规模: small=3仓库5灾情点2地铁线, large=5仓库12灾情点3地铁线",
    )
    parser.add_argument(
        "--solver",
        choices=["milp", "ga", "both", "compare-ga"],
        default="milp",
        help="求解方法: milp=精确求解, ga=遗传算法, both=两者对比, compare-ga=新旧GA对比",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=200,
        help="遗传算法代数 (默认200)",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=100,
        help="遗传算法种群大小 (默认100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (默认42)",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=120.0,
        help="MILP求解时间限制(秒) (默认120)",
    )
    parser.add_argument(
        "--no-niche",
        action="store_true",
        help="关闭小生境技术 (用于对比测试)",
    )
    parser.add_argument(
        "--no-adaptive-mutation",
        action="store_true",
        help="关闭自适应变异 (用于对比测试)",
    )
    parser.add_argument(
        "--no-phenotype-distance",
        action="store_true",
        help="关闭表现型距离，改用基因型距离 (用于对比测试)",
    )
    parser.add_argument(
        "--niche-radius",
        type=float,
        default=0.25,
        help="小生境半径 (默认0.25，只惩罚高度相似的个体)",
    )
    parser.add_argument(
        "--niche-strength",
        type=float,
        default=0.3,
        help="小生境惩罚强度 (默认0.3，值越大惩罚越重)",
    )
    parser.add_argument(
        "--mutation-init",
        type=float,
        default=0.35,
        help="自适应变异初始率 (默认0.35)",
    )
    parser.add_argument(
        "--mutation-final",
        type=float,
        default=0.08,
        help="自适应变异最终率 (默认0.08)",
    )

    args = parser.parse_args()

    if args.scale == "small":
        print("[*] 构建小规模问题 (3仓库, 5灾情点, 2地铁线)...")
        warehouses, disasters, subway = build_small_problem()
    else:
        print("[*] 构建大规模问题 (5仓库, 12灾情点, 3地铁线)...")
        warehouses, disasters, subway = build_large_problem(seed=args.seed)

    total_demand = sum(d.total_demand() for d in disasters)
    total_inv = sum(w.total_inventory() for w in warehouses)
    print(f"   总需求: {total_demand}, 总库存: {total_inv}")
    print()

    results = []

    if args.solver in ("milp", "both"):
        if args.scale == "large":
            print("[!] 大规模问题使用 MILP 求解可能耗时很长，请耐心等待...")
        print("[MILP] 精确求解中...")
        milp_solver = MILPSolver(
            warehouses=warehouses,
            disasters=disasters,
            subway=subway,
            time_limit=args.time_limit,
        )
        milp_result = milp_solver.solve()
        results.append(("MILP", milp_result))
        print(f"   求解完成! 耗时 {milp_result.solve_time_seconds:.2f}s, "
              f"成本 {milp_result.total_cost:.2f}, "
              f"最优={'是' if milp_result.is_optimal else '否'}")
        print()

    if args.solver in ("ga", "both"):
        niche_enabled = not args.no_niche
        adaptive_enabled = not args.no_adaptive_mutation
        phenotype_enabled = not args.no_phenotype_distance
        mode_desc = []
        if niche_enabled:
            mode_desc.append("小生境开")
        else:
            mode_desc.append("小生境关")
        if adaptive_enabled:
            mode_desc.append("自适应变异开")
        else:
            mode_desc.append("自适应变异关")
        if phenotype_enabled:
            mode_desc.append("表现型距离")
        else:
            mode_desc.append("基因型距离")
        print(f"[GA] 遗传算法求解中 (种群={args.population}, 代数={args.generations}, {', '.join(mode_desc)})...")
        ga_solver = GASolver(
            warehouses=warehouses,
            disasters=disasters,
            subway=subway,
            population_size=args.population,
            generations=args.generations,
            seed=args.seed,
            niche_enabled=niche_enabled,
            niche_radius=args.niche_radius,
            niche_penalty_strength=args.niche_strength,
            adaptive_mutation=adaptive_enabled,
            use_phenotype_distance=phenotype_enabled,
            mutation_initial_rate=args.mutation_init,
            mutation_final_rate=args.mutation_final,
        )
        ga_result = ga_solver.solve(verbose=True)
        results.append(("GA", ga_result))
        print(f"   求解完成! 耗时 {ga_result.solve_time_seconds:.2f}s, "
              f"成本 {ga_result.total_cost:.2f}")
        if niche_enabled or adaptive_enabled:
            if ga_solver.diversity_history:
                final_div = ga_solver.diversity_history[-1]
                print(f"   最终种群: 平均距离={final_div.avg_distance:.3f}, "
                      f"唯一染色体={final_div.unique_count}")
        print()

    if args.solver == "compare-ga":
        print("=" * 70)
        print("  新旧 GA 对比测试 (相同种子和问题)")
        print("=" * 70)
        print()

        configs = [
            ("GA(旧版-无改进)", False, False, False, 0.15),
            ("GA(仅小生境)", True, False, True, 0.15),
            ("GA(仅自适应变异)", False, True, True, 0.35),
            ("GA(新版-两者都开)", True, True, True, 0.35),
        ]

        all_results = []
        all_solvers = []

        for name, niche_on, adapt_on, pheno_on, mut_rate in configs:
            print(f"[{name}] 求解中...")
            solver = GASolver(
                warehouses=warehouses,
                disasters=disasters,
                subway=subway,
                population_size=args.population,
                generations=args.generations,
                seed=args.seed,
                niche_enabled=niche_on,
                adaptive_mutation=adapt_on,
                use_phenotype_distance=pheno_on,
                base_mutation_rate=mut_rate,
                mutation_initial_rate=args.mutation_init,
                mutation_final_rate=args.mutation_final,
                niche_radius=args.niche_radius,
                niche_penalty_strength=args.niche_strength,
            )
            result = solver.solve(verbose=True)
            all_results.append((name, result, solver))
            print(f"   完成! 成本={result.total_cost:.2f}, 耗时={result.solve_time_seconds:.2f}s")
            if solver.diversity_history:
                init_div = solver.diversity_history[0]
                final_div = solver.diversity_history[-1]
                print(f"   多样性: 初始 avg_dist={init_div.avg_distance:.3f}, unique={init_div.unique_count}")
                print(f"            最终 avg_dist={final_div.avg_distance:.3f}, unique={final_div.unique_count}")
            print()

        print("=" * 70)
        print("  汇总对比")
        print("=" * 70)
        print(f"  {'方法':<22} {'总成本':>12} {'总时间(h)':>12} {'求解耗时(s)':>12} "
              f"{'初始多样性':>12} {'最终多样性':>12}")
        print(f"  {'-'*84}")
        best_cost = min(r[1].total_cost for r in all_results)
        for name, result, solver in all_results:
            init_avg = solver.diversity_history[0].avg_distance if solver.diversity_history else 0
            final_avg = solver.diversity_history[-1].avg_distance if solver.diversity_history else 0
            cost = result.total_cost
            marker = "  *" if abs(cost - best_cost) < 0.01 else ""
            print(f"  {name:<22} {cost:>12.2f}{marker} {result.total_time:>12.2f} "
                  f"{result.solve_time_seconds:>12.2f} {init_avg:>12.3f} {final_avg:>12.3f}")

        best_name, best_result, _ = min(all_results, key=lambda x: x[1].total_cost)
        worst_name, worst_result, _ = max(all_results, key=lambda x: x[1].total_cost)
        improvement = (worst_result.total_cost - best_result.total_cost) / worst_result.total_cost * 100
        print(f"\n  最佳方法: {best_name} (成本 {best_result.total_cost:.2f})")
        print(f"  最差方法: {worst_name} (成本 {worst_result.total_cost:.2f})")
        print(f"  改进幅度: {improvement:+.1f}%")
        print(f"\n  多样性分析:")
        for name, result, solver in all_results:
            if solver.diversity_history:
                init_unique = solver.diversity_history[0].unique_count
                final_unique = solver.diversity_history[-1].unique_count
                drop = (init_unique - final_unique) / init_unique * 100
                print(f"    {name}: 唯一染色体数 {init_unique} -> {final_unique} (下降 {drop:.0f}%)")
        print("=" * 70)
        print()

        _, best_result, best_solver = min(all_results, key=lambda x: x[1].total_cost)
        results.append(("Best GA", best_result))

    for name, result in results:
        title = f"{'='*20} {name} 求解结果 {'='*20}"
        report = generate_report(result, warehouses, disasters, subway, title=name)
        print(report)

    if len(results) == 2:
        print("[对比] 两种求解方法对比:")
        print(f"   {'方法':<8} {'总成本':>12} {'总时间(h)':>12} {'求解耗时(s)':>12} {'最优':>6}")
        print(f"   {'-'*52}")
        for name, result in results:
            opt = "是" if result.is_optimal else "否"
            print(f"   {name:<8} {result.total_cost:>12.2f} {result.total_time:>12.2f} "
                  f"{result.solve_time_seconds:>12.2f} {opt:>6}")
        milp_r = results[0][1]
        ga_r = results[1][1]
        gap = (ga_r.total_cost - milp_r.total_cost) / milp_r.total_cost * 100
        print(f"\n   GA 相比 MILP 的成本差距: {gap:+.1f}%")


if __name__ == "__main__":
    main()
