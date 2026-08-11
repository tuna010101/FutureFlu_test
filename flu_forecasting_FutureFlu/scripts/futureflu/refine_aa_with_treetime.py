"""Refine amino-acid trees with TreeTime.

English: Produces time-scaled trees and node metadata from AA alignments.
中文：从氨基酸比对生成时间树和节点 metadata。
"""

from __future__ import annotations

import argparse
import sys

from Bio import Phylo
from augur.utils import InvalidTreeError, get_numerical_dates, read_metadata, read_tree, write_json
from treetime import TreeTime


def collect_node_data(tree, attributes):
    data = {}
    for node in tree.find_clades():
        data[node.name] = {attribute: getattr(node, attribute) for attribute in attributes if hasattr(node, attribute)}
    return data


def read_tree_with_raw_dates(tree_path, metadata):
    tree = read_tree(tree_path)
    for node in tree.get_terminals():
        if node.name in metadata and "date" in metadata[node.name]:
            node.raw_date = metadata[node.name]["date"]
    return tree


def run_treetime(args, metadata, dates, root, marginal, vary_rate, coalescent, use_covariance):
    # TreeTime mutates the tree in-place, so each fallback attempt needs a fresh tree.
    tree = read_tree_with_raw_dates(args.tree, metadata)
    tt = TreeTime(tree=tree, aln=args.alignment, dates=dates, verbose=1, gtr=args.gtr_model)

    if args.clock_filter_iqd:
        tt.clock_filter(reroot=root, n_iqd=args.clock_filter_iqd, plot=False)
        leaves = [node for node in tt.tree.get_terminals()]
        for node in leaves:
            if node.bad_branch:
                tt.tree.prune(node)
                print(f"pruning leaf {node.name}")
        tt.prepare_tree()

    tt.run(
        infer_gtr=False,
        root=root,
        Tc=coalescent,
        time_marginal=marginal,
        branch_length_mode=args.branch_length_inference,
        resolve_polytomies=(not args.keep_polytomies),
        max_iter=args.max_iter,
        fixed_clock_rate=args.clock_rate,
        vary_rate=vary_rate,
        use_covariation=use_covariance,
    )
    return tt


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AA-aware replacement for augur refine using TreeTime with an amino-acid model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--alignment", "-a", required=True, help="AA alignment in fasta format")
    parser.add_argument("--tree", "-t", required=True, help="prebuilt Newick tree")
    parser.add_argument("--metadata", required=True, help="tsv/csv table with metadata including date")
    parser.add_argument("--output-tree", required=True, help="output timetree in Newick format")
    parser.add_argument("--output-node-data", required=True, help="output node data JSON")
    parser.add_argument("--timetree", action="store_true", default=True, help="kept for interface compatibility")
    parser.add_argument("--coalescent", default="const", help="TreeTime coalescent mode")
    parser.add_argument("--clock-rate", type=float, help="fixed clock rate")
    parser.add_argument("--clock-std-dev", type=float, help="standard deviation of the fixed clock rate")
    parser.add_argument("--root", nargs="+", default=["least-squares"], help="rooting method or outgroup node(s)")
    parser.add_argument("--keep-root", action="store_true", help="keep the input root")
    parser.add_argument("--covariance", dest="covariance", action="store_true", help="use covariance-aware rate estimation")
    parser.add_argument("--no-covariance", dest="covariance", action="store_false")
    parser.add_argument("--keep-polytomies", action="store_true", help="do not resolve polytomies")
    parser.add_argument("--date-format", default="%Y-%m-%d", help="date format")
    parser.add_argument("--date-confidence", action="store_true", help="calculate confidence intervals for node dates")
    parser.add_argument("--date-inference", default="marginal", choices=["joint", "marginal"], help="node date assignment mode")
    parser.add_argument(
        "--branch-length-inference",
        default="auto",
        choices=["auto", "joint", "marginal", "input"],
        help="TreeTime branch length mode",
    )
    parser.add_argument("--clock-filter-iqd", type=float, help="clock-filter threshold in IQD units")
    parser.add_argument("--year-bounds", type=int, nargs="+", help="bounds for ambiguous years")
    parser.add_argument("--gtr-model", default="JTT92", help="TreeTime substitution model")
    parser.add_argument("--max-iter", type=int, default=2, help="maximum TreeTime iterations")
    parser.set_defaults(covariance=True)
    args = parser.parse_args()

    sys.setrecursionlimit(max(sys.getrecursionlimit(), 100000))

    metadata, columns = read_metadata(args.metadata)
    if args.year_bounds:
        args.year_bounds.sort()
    dates = get_numerical_dates(metadata, fmt=args.date_format, min_max_year=args.year_bounds)

    root = None if args.keep_root else args.root
    if root and len(root) == 1:
        root = root[0]

    confidence = args.date_confidence
    use_marginal = args.date_inference == "marginal"
    if confidence and use_marginal:
        marginal = "assign"
    else:
        marginal = confidence

    if confidence and args.clock_std_dev:
        vary_rate = args.clock_std_dev
    elif confidence and args.covariance:
        vary_rate = True
    else:
        vary_rate = False

    attempts = [
        {
            "label": "requested",
            "root": root,
            "coalescent": args.coalescent,
            "marginal": marginal,
            "vary_rate": vary_rate,
            "use_covariance": args.covariance,
        },
        {
            "label": "fallback_no_coalescent",
            "root": root,
            "coalescent": None,
            "marginal": marginal,
            "vary_rate": vary_rate,
            "use_covariance": args.covariance,
        },
        {
            "label": "fallback_input_root_no_coalescent",
            "root": None,
            "coalescent": None,
            "marginal": marginal,
            "vary_rate": vary_rate,
            "use_covariance": args.covariance,
        },
    ]

    tt = None
    successful_attempt = None
    for attempt in attempts:
        try:
            tt = run_treetime(
                args=args,
                metadata=metadata,
                dates=dates,
                root=attempt["root"],
                marginal=attempt["marginal"],
                vary_rate=attempt["vary_rate"],
                coalescent=attempt["coalescent"],
                use_covariance=attempt["use_covariance"],
            )
            successful_attempt = attempt
            if attempt["label"] != "requested":
                print(f"TreeTime fallback attempt succeeded: {attempt['label']}")
            break
        except (FileNotFoundError, InvalidTreeError) as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 1
        except Exception as error:
            print(
                f"WARNING: TreeTime attempt failed ({attempt['label']}): "
                f"{type(error).__name__}: {error}",
                file=sys.stderr,
            )

    if tt is None:
        print("ERROR: all TreeTime attempts failed", file=sys.stderr)
        return 1

    if confidence:
        for node in tt.tree.find_clades():
            node.num_date_confidence = list(tt.get_max_posterior_region(node, 0.9))

    node_data = {
        "alignment": args.alignment,
        "input_tree": args.tree,
        "clock": {
            "rate": tt.date2dist.clock_rate,
            "intercept": tt.date2dist.intercept,
            "rtt_Tmrca": -tt.date2dist.intercept / tt.date2dist.clock_rate,
        },
        "timetree_inference": {
            "attempt": successful_attempt["label"],
            "coalescent": successful_attempt["coalescent"],
            "root": successful_attempt["root"],
            "date_inference": args.date_inference,
            "date_confidence": args.date_confidence,
            "covariance": successful_attempt["use_covariance"],
        },
    }
    attributes = ["branch_length", "numdate", "clock_length", "mutation_length", "raw_date", "date"]
    if confidence:
        attributes.append("num_date_confidence")
    node_data["nodes"] = collect_node_data(tt.tree, attributes)

    Phylo.write(tt.tree, args.output_tree, "newick", format_branch_length="%1.8f")
    write_json(node_data, args.output_node_data)
    print(f"updated tree written to {args.output_tree}")
    print(f"node attributes written to {args.output_node_data}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
