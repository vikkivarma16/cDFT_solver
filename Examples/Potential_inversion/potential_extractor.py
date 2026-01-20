import json
from pathlib import Path


def export_pairs_to_txt(
    json_path,
    output_dir="pair_txt",
    r_key="r",
    u_key="u_r",   # change to "U" if your JSON uses that
):
    """
    Export each pair potential from JSON into a separate two-column text file.

    Output files:
        aa.txt, ab.txt, bb.txt, ...

    Columns:
        r   u(r)
    """

    json_path = Path(json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Load JSON
    # -----------------------------
    with open(json_path, "r") as f:
        data = json.load(f)

    if "pairs" not in data:
        raise KeyError("JSON must contain a top-level 'pairs' key")

    # -----------------------------
    # Write one file per pair
    # -----------------------------
    for pair, pdata in data["pairs"].items():

        if r_key not in pdata or u_key not in pdata:
            raise KeyError(f"Pair '{pair}' missing '{r_key}' or '{u_key}'")

        r = pdata[r_key]
        u = pdata[u_key]

        if len(r) != len(u):
            raise ValueError(f"Length mismatch in pair '{pair}'")

        out_file = output_dir / f"{pair}.txt"

        with open(out_file, "w") as f:
            f.write("# r    u(r)\n")
            for ri, ui in zip(r, u):
                f.write(f"{ri:16.8e} {ui:16.8e}\n")

        print(f"✅ wrote {out_file}")


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    export_pairs_to_txt(
        json_path="multistate_potential.json",  # your JSON file
        output_dir="pair_potentials_txt",
        u_key="u_r",  # or "U" if that's your key
    )

