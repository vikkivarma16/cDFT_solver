import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def find_json_file():
    """
    Automatically find RDF JSON file in current directory.
    """
    files = list(Path(".").glob("*dispersion.json"))

    if len(files) == 0:
        raise FileNotFoundError("❌ No *_rdf.json file found in current directory")

    if len(files) > 1:
        print("⚠️ Multiple JSON files found. Using first one:")
        for f in files:
            print("   ", f)

    return files[0]


def load_dispersion_data(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    if "dispersion_relation" not in data:
        raise KeyError("❌ No dispersion_relation found in JSON")

    disp = data["dispersion_relation"]

    k = np.array(disp["k"])
    omega = np.array(disp["omega"])

    return k, omega


def plot_dispersion(k, omega):
    Nk, Nmodes = omega.shape

    plt.figure(figsize=(6, 4))

    for mode in range(Nmodes):
        plt.plot(k, omega[:, mode], label=f"mode {mode}")

    plt.axhline(0.0, linestyle="--")
    plt.xlim(0.0, 20.0)
    plt.ylim(-500.0, 0.0)

    plt.xlabel("k")
    plt.ylabel("ω(k)")
    plt.title("Dispersion Relation")

    if Nmodes > 1:
        plt.legend()

    plt.tight_layout()
    plt.savefig("dispersion_all_modes.png", dpi=300)
    print("✅ Saved → dispersion_all_modes.png")
    plt.close()


def plot_fastest_mode(k, omega):
    omega_max = np.max(omega, axis=1)

    plt.figure(figsize=(6, 4))

    plt.plot(k, omega_max, label="fastest mode")

    plt.axhline(0.0, linestyle="--")
    plt.xlim(0.0,20)

    plt.xlabel("k")
    plt.ylabel("ω_max(k)")
    plt.title("Fastest Growing Mode")

    plt.legend()
    plt.tight_layout()
    plt.savefig("dispersion_fastest_mode.png", dpi=300)
    print("✅ Saved → dispersion_fastest_mode.png")
    plt.close()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    json_file = find_json_file()
    print(f"📂 Using file: {json_file}")

    k, omega = load_dispersion_data(json_file)

    plot_dispersion(k, omega)
    plot_fastest_mode(k, omega)
