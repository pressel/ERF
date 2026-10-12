#!/usr/bin/env python3
"""Compare composite leaf inventories for identical SBM qualification decks."""

import math
import re
import sys


def read_inventory(path):
    values = {}
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            match = re.search(
                r"SBM_AMR_QUAL_INVENTORY component=(\d+) initial=([^ ]+) "
                r"final=([^ ]+).*?abs_error=([^ ]+).*?tolerance=([^ ]+)",
                line,
            )
            if match:
                component = int(match.group(1))
                values[component] = tuple(float(match.group(i)) for i in range(2, 6))
    return values


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: CompareAMRQualificationLogs.py REF TEST NCOMP")
    reference, candidate, ncomp = sys.argv[1], sys.argv[2], int(sys.argv[3])
    ref = read_inventory(reference)
    got = read_inventory(candidate)
    if set(ref) != set(range(ncomp)) or set(got) != set(range(ncomp)):
        raise SystemExit(
            f"expected exactly {ncomp} inventory components; "
            f"reference={sorted(ref)}, candidate={sorted(got)}"
        )

    # ERF's qualified leaf diagnostic sums N positive cell contributions in
    # serial FAB order, then combines boxes, levels, and the MPI reduction.
    # For these 16^3/32^3 periodic layouts N=18,432 leaf cells.  Add the
    # maximum 40 FAB partial sums, two level merges, the MPI tree depth, and
    # final subtraction to the conservative sequential-sum operation count.
    eps = sys.float_info.epsilon
    operation_count = 18_432 + 40 + 2 + 1 + 1
    q = operation_count * eps
    gamma = q / (1.0 - q)
    for component in range(ncomp):
        ref_initial, ref_final, _, _ = ref[component]
        got_initial, got_final, _, _ = got[component]
        if not math.isclose(ref_initial, got_initial, rel_tol=gamma, abs_tol=0.0):
            raise SystemExit(
                f"component {component} initial inventory differs: "
                f"{ref_initial:.17g} vs {got_initial:.17g}, gamma={gamma:.3g}"
            )
        bound = gamma * (abs(ref_final) + abs(got_final))
        error = abs(ref_final - got_final)
        if error > bound:
            raise SystemExit(
                f"component {component} final inventory differs: "
                f"error={error:.17g}, bound={bound:.17g}, "
                f"reference={ref_final:.17g}, candidate={got_final:.17g}"
            )
    print(
        f"component inventories match within gamma_{operation_count}={gamma:.3g}"
    )


if __name__ == "__main__":
    main()
