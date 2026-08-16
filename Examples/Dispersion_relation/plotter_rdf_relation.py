import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_all_rdf_in_folder():
    """
    Finds RDF JSON files in the current folder and plots g(r), c(r), gamma(r)
    """

    here = Path(__file__).parent

    json_files = list(here.glob("*_rdf.json"))

    if not json_files:
        print("❌ No RDF JSON files found in this folder.")
        return

    for json_path in json_files:
        print(f"\n📂 Processing: {json_path.name}")

        with open(json_path, "r") as f:
            data = json.load(f)

        species = data["metadata"]["species"]
        pairs = data["pairs"]
        N = len(species)

        quantities = ["g_r", "c_r", "gamma_r"]

        for quantity in quantities:

            fig, axes = plt.subplots(N, N, figsize=(4*N, 4*N), sharex=True)

            if N == 1:
                axes = np.array([[axes]])

            for i, si in enumerate(species):
                for j, sj in enumerate(species):

                    pair_key = f"{si}{sj}"

                    if pair_key not in pairs:
                        continue

                    r = np.array(pairs[pair_key]["r"])
                    y = np.array(pairs[pair_key][quantity])

                    ax = axes[i, j]
                    ax.plot(r, y, linewidth=2)

                    ax.set_title(f"{si}-{sj}")
                    ax.grid(True)

                    if i == N - 1:
                        ax.set_xlabel("r")

                    if j == 0:
                        ax.set_ylabel(quantity)

            plt.tight_layout()

            out_file = here / f"{json_path.stem}_{quantity}.png"
            plt.savefig(out_file, dpi=300)
            plt.close()

            print(f"✅ Saved: {out_file.name}")


# -------------------------------------------------
# Run directly
# -------------------------------------------------
if __name__ == "__main__":
    plot_all_rdf_in_folder()
