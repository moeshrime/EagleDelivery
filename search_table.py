"""Run BFS and A* on a list of (map, start, goal) tasks and save one CSV table.

Usage:
    python3 search_table.py                      # the six required Part A tasks
    python3 search_table.py --map maps/my_campus.txt --tasks SW --output results/search/custom_search.csv
"""
import argparse
import csv
from pathlib import Path

import agents
from campus import CampusMap


def main():
    here = Path(__file__).parent
    parser = argparse.ArgumentParser(description="BFS vs A* comparison table")
    parser.add_argument("--map", nargs="+",
                        default=[str(here / "maps" / "emu_campus.txt"),
                                 str(here / "maps" / "emu_detour.txt")])
    parser.add_argument("--tasks", nargs="+", default=["SW", "LD", "HP"])
    parser.add_argument("--output", default=str(here / "results" / "search" / "search_results.csv"))
    args = parser.parse_args()

    rows = []
    for map_file in args.map:
        campus = CampusMap(map_file)
        for task in args.tasks:
            start, goal = campus.location(task[0]), campus.location(task[1])
            bfs_result = agents.bfs(campus, start, goal)
            astar_result = agents.astar(campus, start, goal)
            rows.append({
                "map": Path(map_file).name,
                "task": task[0] + "-" + task[1],
                "bfs_length": len(bfs_result.path) - 1 if bfs_result.path else "NA",
                "astar_length": len(astar_result.path) - 1 if astar_result.path else "NA",
                "bfs_expanded": bfs_result.expanded,
                "astar_expanded": astar_result.expanded,
            })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print("Saved", output)


if __name__ == "__main__":
    main()
