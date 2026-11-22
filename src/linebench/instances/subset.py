"""
Instance subset builder for LINEBENCH.
Implements the pre-declared selection rules from docs/protocol.md:
  1. Classical Stratum (Scholl 1993):
     - All instances with n <= 100 tasks.
     - Stratified subsample of 15 instances from large graphs (n > 100: Arcus 2, Bartholdi, Barthol2, Scholl).
  2. Generated Stratum (Otto et al. 2013 - SALBPGen):
     - 3 instances per cell across Small (n=20), Medium (n=50), Large (n=100).
  3. Simulation Subset:
     - 30 instances stratified across small (10), medium (10), and large (10).
Outputs configs/instances.yaml and provides reproducible subset loaders.
"""

import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from linebench.instances.loader import Instance, load_alb, load_in2


@dataclass(frozen=True)
class InstanceEntry:
    """Represents a selected benchmark instance metadata entry."""

    id: str
    stratum: str  # 'classical' or 'generated'
    source_file: str
    num_tasks: int
    cycle_time: int
    optimal_stations: int | None
    best_known_stations: int | None
    is_optimum_proven: bool
    cell: str | None = None  # e.g., 'BN_0.2_bimodal' for generated


def parse_scholl_master_xlsx(
    xlsx_path: str = "data/raw/scholl1993/SALBP data sets.xlsx",
) -> list[dict[str, object]]:
    """Parses Sheet 1 (graphs) and Sheet 2 (SALBP-1) from Scholl master spreadsheet."""
    path = Path(xlsx_path)
    if not path.is_file():
        raise FileNotFoundError(f"Scholl master xlsx not found: {path}")

    with zipfile.ZipFile(path) as z:
        shared_strings = []
        if "xl/sharedStrings.xml" in z.namelist():
            ss_root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in ss_root.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}si"):
                t = si.find("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")
                shared_strings.append(t.text if t is not None else "")

        # Sheet 1: Precedence graphs
        s1_root = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
        graphs_meta = {}
        for r in s1_root.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row"):
            cells = []
            for c in r.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c"):
                v = c.find("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v")
                val = v.text if v is not None else ""
                if c.attrib.get("t") == "s" and val != "":
                    val = shared_strings[int(val)]
                cells.append(val)
            if len(cells) >= 3 and cells[0].endswith(".IN2"):
                fname, name, n_tasks = cells[0], cells[1], int(cells[2])
                graphs_meta[name.lower()] = {
                    "file": f"data/raw/scholl1993/precedence graphs/{fname}",
                    "name": name,
                    "n": n_tasks,
                }

        # Sheet 2: SALBP-1 instances
        s2_root = ET.fromstring(z.read("xl/worksheets/sheet2.xml"))
        instances = []
        for r in s2_root.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row"):
            cells = []
            for c in r.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c"):
                v = c.find("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v")
                val = v.text if v is not None else ""
                if c.attrib.get("t") == "s" and val != "":
                    val = shared_strings[int(val)]
                cells.append(val)
            if len(cells) >= 4 and cells[0] and cells[1] and cells[1].isdigit():
                g_name, cycle_time, m_star_val = cells[0], int(cells[1]), cells[2]
                opt_proven = cells[7] == "1" if len(cells) > 7 else False

                # Handle ranges like '[32,33]' or clean integer
                if m_star_val.isdigit():
                    m_star = int(m_star_val)
                    opt = m_star if opt_proven else None
                    bk = m_star
                else:
                    opt = None
                    bk = 32  # Wee-mag lower end

                meta = graphs_meta.get(g_name.lower())
                if meta:
                    inst_id = f"scholl_{meta['name'].lower()}_c{cycle_time}"
                    instances.append(
                        {
                            "id": inst_id,
                            "stratum": "classical",
                            "source_file": meta["file"],
                            "num_tasks": meta["n"],
                            "cycle_time": cycle_time,
                            "optimal_stations": opt,
                            "best_known_stations": bk,
                            "is_optimum_proven": opt_proven,
                            "cell": None,
                        }
                    )
        return instances


def parse_otto_details_xlsx(
    size_label: str, folder_name: str, is_medium: bool = False
) -> list[dict[str, object]]:
    """Parses details spreadsheet for an Otto 2013 size tier."""
    xlsx_path = Path(f"data/raw/otto2013/Details of the {size_label}.xlsx")
    if not xlsx_path.is_file():
        raise FileNotFoundError(f"Otto details xlsx not found: {xlsx_path}")

    with zipfile.ZipFile(xlsx_path) as z:
        shared_strings = []
        if "xl/sharedStrings.xml" in z.namelist():
            ss_root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in ss_root.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}si"):
                t = si.find("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")
                shared_strings.append(t.text if t is not None else "")

        sheet_xml = z.read("xl/worksheets/sheet1.xml")
        s_root = ET.fromstring(sheet_xml)
        rows = []
        for r in s_root.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row"):
            row_vals = []
            for c in r.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c"):
                v = c.find("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v")
                val = v.text if v is not None else ""
                if c.attrib.get("t") == "s" and val != "":
                    val = shared_strings[int(val)]
                row_vals.append(val)
            if any(row_vals):
                rows.append(row_vals)

        instances = []
        for r in rows[2:]:
            if not r or not r[0].isdigit():
                continue
            if is_medium:
                perm_no = r[1].strip()
                if perm_no != "1":
                    continue  # Only take unpermuted instance
                fname = r[2].strip()
                graph = r[3].strip()
                os_val = r[4].strip()
                dist = r[5].strip()
                m_opt_str = r[7].strip() if len(r) > 7 else ""
            else:
                fname = r[1].strip()
                graph = r[2].strip()
                os_val = r[3].strip()
                dist = r[4].strip()
                m_opt_str = r[6].strip() if len(r) > 6 else ""

            cell_key = f"{graph}_{os_val}_{dist}"
            opt_proven = m_opt_str.isdigit()
            opt_val = int(m_opt_str) if opt_proven else None

            alb_file = f"data/raw/otto2013/{folder_name}/{folder_name}/{fname}.alb"
            instances.append(
                {
                    "id": fname,
                    "stratum": "generated",
                    "source_file": alb_file,
                    "num_tasks": int(folder_name.split("=")[-1]),
                    "cycle_time": 1000,
                    "optimal_stations": opt_val,
                    "best_known_stations": opt_val,
                    "is_optimum_proven": opt_proven,
                    "cell": cell_key,
                }
            )
        return instances


def build_instance_subsets(
    seed: int = 42,
    simulation_seed: int = 2026,
    scholl_large_sample_size: int = 15,
    generated_instances_per_cell: int = 3,
) -> dict[str, object]:
    """Constructs the reproducible benchmark and simulation subsets."""
    rng = np.random.default_rng(seed)

    # 1. Classical Stratum
    scholl_all = parse_scholl_master_xlsx()
    scholl_small_med = [inst for inst in scholl_all if inst["num_tasks"] <= 100]
    scholl_large = [inst for inst in scholl_all if inst["num_tasks"] > 100]

    # Sample 15 instances from large graphs
    large_indices = rng.choice(
        len(scholl_large), size=min(scholl_large_sample_size, len(scholl_large)), replace=False
    )
    scholl_large_sample = [scholl_large[i] for i in sorted(large_indices)]
    classical_subset = scholl_small_med + scholl_large_sample

    # 2. Generated Stratum
    otto_tiers = [
        ("small data set (n=20)", "small data set_n=20", False),
        ("medium data set (n=50 and n=50permuted)", "medium data set_n=50", True),
        ("large data set (n=100)", "large data set_n=100", False),
    ]

    generated_subset = []
    for size_label, folder, is_m in otto_tiers:
        tier_instances = parse_otto_details_xlsx(size_label, folder, is_medium=is_m)
        cells: dict[str, list[dict[str, object]]] = {}
        for inst in tier_instances:
            cells.setdefault(inst["cell"], []).append(inst)

        for cell_key, inst_list in sorted(cells.items()):
            n_sample = min(generated_instances_per_cell, len(inst_list))
            chosen_indices = rng.choice(len(inst_list), size=n_sample, replace=False)
            for idx in sorted(chosen_indices):
                generated_subset.append(inst_list[idx])

    benchmark_subset = classical_subset + generated_subset

    # 3. Simulation Subset (30 instances: 10 small, 10 medium, 10 large)
    sim_rng = np.random.default_rng(simulation_seed)
    small_pool = [inst for inst in benchmark_subset if inst["num_tasks"] <= 25]
    med_pool = [inst for inst in benchmark_subset if 30 <= inst["num_tasks"] <= 60]
    large_pool = [inst for inst in benchmark_subset if inst["num_tasks"] >= 70]

    sim_small = [
        small_pool[i]
        for i in sorted(sim_rng.choice(len(small_pool), size=min(10, len(small_pool)), replace=False))
    ]
    sim_med = [
        med_pool[i]
        for i in sorted(sim_rng.choice(len(med_pool), size=min(10, len(med_pool)), replace=False))
    ]
    sim_large = [
        large_pool[i]
        for i in sorted(sim_rng.choice(len(large_pool), size=min(10, len(large_pool)), replace=False))
    ]
    simulation_subset = sim_small + sim_med + sim_large

    return {
        "metadata": {
            "master_seed": seed,
            "simulation_seed": simulation_seed,
            "total_benchmark_instances": len(benchmark_subset),
            "classical_instances_count": len(classical_subset),
            "generated_instances_count": len(generated_subset),
            "simulation_instances_count": len(simulation_subset),
        },
        "benchmark_instances": benchmark_subset,
        "simulation_instances": simulation_subset,
    }


def freeze_instances_yaml(output_path: str = "configs/instances.yaml") -> dict[str, object]:
    """Generates and writes configs/instances.yaml."""
    data = build_instance_subsets(
        seed=42,
        simulation_seed=2026,
        scholl_large_sample_size=15,
        generated_instances_per_cell=3,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False, indent=2)
    return data


def load_subset_instance(entry: dict[str, object], project_root: str | Path | None = None) -> Instance:
    """Instantiates an Instance dataclass from an instance metadata entry.

    Resolves relative source_file paths against project_root or by searching
    current and parent directories.
    """
    src_str = str(entry["source_file"])
    c = int(entry["cycle_time"])
    name = str(entry["id"])

    src_path = Path(src_str)
    if not src_path.is_file():
        # Try resolving against explicit project_root or looking upward
        if project_root is not None:
            candidate = Path(project_root) / src_path
            if candidate.is_file():
                src_path = candidate
        else:
            # Check ../data if running from notebooks/ or similar subdirectories
            for parent in [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]:
                candidate = parent / src_path
                if candidate.is_file():
                    src_path = candidate
                    break

    if src_str.endswith(".IN2"):
        return load_in2(src_path, cycle_time=c, name=name)
    else:
        return load_alb(src_path)
