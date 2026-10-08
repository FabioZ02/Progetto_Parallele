#!/usr/bin/env python3
# ======================================================================
# benchmark_cuda.py — confronto tra le 5 versioni CUDA (F1..F5) del kernel FORZE del GSA.
#
# Per ogni N in N_LIST e per ogni variante:
#   1. compila gsa_cuda.cu con i flag della variante e -DPROFILE;
#   2. lo esegue REPS volte e legge dal breakdown la riga
#      "k_forces  totale = ... ms (X ms/iter)", cioe' il tempo del SOLO kernel forze
#      misurato con eventi CUDA;
#   3. tiene la mediana delle ripetizioni.
# Alla fine disegna il tempo per iterazione del kernel forze al variare di N
# (fig_forze_sweep.pdf / .png). Solo confronti CUDA contro CUDA.
#
# L'algoritmo GSA gira sempre per intero (fitness, best/worst, masse, Kbest,
# forze, update). Varia SOLO il kernel forze; la fitness resta la versione
# baseline (nessun flag FIT_), cosi' tra le versioni cambia una cosa sola.
#
# Uso:  make sweep            (oppure: python3 benchmark_cuda.py)
# Architettura GPU: variabile d'ambiente ARCH (default sm_75 = Tesla T4;
# il Makefile passa la sua ARCH, per esempio "native").
# Tutte le misure vanno fatte nella stessa sessione: i tempi assoluti variano
# tra sessioni e macchine, il confronto tra varianti resta stabile.
# ======================================================================
import os, subprocess, re, statistics, sys

# ---- parametri dello sweep (istanza della relazione, si varia solo N) ----
N_LIST   = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]
K, NC, DEG = 64, 512, 3
MAX_ITER = 100          # iterazioni GSA per esecuzione
REPS     = 2            # ripetizioni per punto, si tiene la mediana
SRC      = "gsa_cuda.cu"
ARCH     = os.environ.get("ARCH", "sm_75")   # Tesla T4 se non specificato

# ---- le 5 versioni del kernel FORZE (una leva in piu' a ogni passo) ----
FOR_VARIANTS = [
    ("F1 baseline (global)",   []),                       # thread/agente, sorgenti da global
    ("F2 tiled shared",        ["-DTILED"]),              # + sorgenti a tile in shared
    ("F3 warp+shuffle",        ["-DFORCE_WARP"]),         # warp/agente, R una volta, riduzione shuffle
    ("F4 warp+shuffle+tiled",  ["-DFORCE_WARP_TILED"]),   # F3 + tile dei sorgenti in shared
    ("F5 warp lane-dim+tiled", ["-DFORCE_WARP_DIM"]),     # warp/agente, lane sulle dimensioni, tile con padding
]

# riga del breakdown stampata da gsa_cuda.cu con -DPROFILE; si cattura il valore ms/iter
RE_FOR = re.compile(r"k_forces\s+totale\s*=\s*[\d.]+ ms\s*\(([\d.]+) ms/iter\)")


def check_source():
    """Verifica che gsa_cuda.cu contenga il profiling e tutte le varianti forze,
    altrimenti il breakdown non viene stampato e la regex non trova k_forces."""
    try:
        src = open(SRC).read()
    except FileNotFoundError:
        print(f"{SRC} non trovato nella cartella corrente."); sys.exit(1)
    needed = ["#ifdef PROFILE", "k_forces  totale", "k_forces_warp_tiled", "k_forces_warp(", "k_forces_warp_dim"]
    missing = [s for s in needed if s not in src]
    if missing:
        print(f"{SRC} e' una versione VECCHIA (manca: {missing})."); sys.exit(1)


def compile_variant(binname, flags):
    """Compila una variante con l'istanza della relazione e il profiling attivo."""
    cmd = ["nvcc", "-O2", f"-arch={ARCH}", "-DPROFILE",
           f"-DK={K}", f"-DNC={NC}", f"-DDEG={DEG}", f"-DMAX_ITER={MAX_ITER}",
           *flags, "-o", binname, SRC]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("ERRORE di compilazione:", " ".join(cmd)); print(r.stderr); sys.exit(1)


def run_forces(binname):
    """Esegue il binario e restituisce il tempo del kernel forze in ms/iterazione."""
    r = subprocess.run([f"./{binname}"], capture_output=True, text=True)
    if r.returncode != 0:
        print("ERRORE di esecuzione:", binname); print(r.stdout, r.stderr); sys.exit(1)
    if "Profiling: ON" not in r.stdout:
        print(f"Il binario non e' compilato con PROFILE."); print(r.stdout); sys.exit(1)
    m = RE_FOR.search(r.stdout)
    if not m:
        print("Non trovo k_forces nel breakdown per", binname); print(r.stdout); sys.exit(1)
    return float(m.group(1))


def sweep():
    """Ritorna { nome_variante: [tempo(N) per ogni N in N_LIST] }."""
    out = {name: [] for name, _ in FOR_VARIANTS}
    for n in N_LIST:
        for name, flags in FOR_VARIANTS:
            binname = "bench_bin"                         # N e' una costante di compilazione: si ricompila ogni volta
            compile_variant(binname, flags + [f"-DN={n}"])
            reps = [run_forces(binname) for _ in range(REPS)]
            t = statistics.median(reps)
            out[name].append(t)
            print(f"N={n:5d}  {name:26s}  k_forces = {t:8.3f} ms/iter", flush=True)
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")                                 # nessuna finestra: salva solo su file
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 130})
    MARK = ["o", "s", "^", "D", "v"]

    print(f"=== Sweep kernel FORZE (5 versioni), N = {N_LIST[0]} -> {N_LIST[-1]}, arch {ARCH} ===")
    check_source()
    forc = sweep()

    # una curva per variante: tempo del kernel forze per iterazione al variare di N
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for i, (name, _) in enumerate(FOR_VARIANTS):
        ax.plot(N_LIST, forc[name], marker=MARK[i], label=name, linewidth=1.8)
    ax.set_xlabel("N (numero di agenti)")
    ax.set_ylabel("tempo kernel forze (ms / iterazione)")
    ax.set_title(f"Kernel forze: 5 versioni CUDA al variare di N — K={K} NC={NC} DEG={DEG}")
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig("fig_forze_sweep.pdf"); fig.savefig("fig_forze_sweep.png")

    if os.path.exists("bench_bin"):                       # il binario temporaneo non serve piu'
        os.remove("bench_bin")
    print("\nFatto: fig_forze_sweep.{pdf,png}")


if __name__ == "__main__":
    main()
