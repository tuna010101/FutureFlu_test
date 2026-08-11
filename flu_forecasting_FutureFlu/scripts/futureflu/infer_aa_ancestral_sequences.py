"""Infer ancestral amino-acid sequences for tree nodes.

English: Uses TreeTime to export node-level amino-acid sequences and mutations.
中文：使用 TreeTime 导出节点级氨基酸序列和突变。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from augur.utils import read_tree, write_json
from treetime import TreeAnc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Infer ancestral amino-acid sequences with TreeTime's AA model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tree", required=True, help="Newick tree with named internal nodes")
    parser.add_argument("--alignment", required=True, help="amino-acid FASTA alignment for one gene")
    parser.add_argument("--output-sequences", required=True, help="FASTA with tip and internal AA sequences")
    parser.add_argument("--output-node-data", help="optional JSON node data with inferred AA sequences")
    parser.add_argument("--gtr-model", default="JTT92", help="TreeTime amino-acid substitution model")
    parser.add_argument("--inference", choices=["joint", "marginal"], default="joint")
    args = parser.parse_args()

    tree = read_tree(args.tree)
    tree_anc = TreeAnc(tree=tree, aln=args.alignment, gtr=args.gtr_model, verbose=1)
    tree_anc.infer_ancestral_sequences(
        infer_gtr=False,
        marginal=(args.inference == "marginal"),
    )

    records = []
    node_data = {"nodes": {}}
    for node in tree_anc.tree.find_clades():
        sequence = getattr(node, "sequence", None)
        if sequence is None:
            continue

        sequence_string = "".join(sequence)
        records.append(SeqRecord(Seq(sequence_string), id=node.name, description=""))
        node_data["nodes"][node.name] = {"sequence": sequence_string}

        mutations = getattr(node, "mutations", None)
        if mutations is not None:
            node_data["nodes"][node.name]["muts"] = [
                f"{ancestral}{int(position) + 1}{derived}"
                for ancestral, position, derived in mutations
            ]

    output_sequences = Path(args.output_sequences)
    output_sequences.parent.mkdir(parents=True, exist_ok=True)
    SeqIO.write(records, output_sequences, "fasta")

    if args.output_node_data:
        write_json(node_data, args.output_node_data)

    print(f"wrote {len(records)} sequences to {output_sequences}")
    if args.output_node_data:
        print(f"wrote node data to {args.output_node_data}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
