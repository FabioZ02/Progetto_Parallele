#!/usr/bin/env python3
# ======================================================================
# scaling_cuda.py — i due test di scalabilita' di Lex01 (slide 28-32) sulla versione F5 di GSA.
#
# Slide 28: speedup S = T1/TP (Ti = tempo di computazione con i processori),
#           efficienza = S/P, caso ideale S = P.
# Slide 29-30, STRONG scaling (legge di Amdahl): dimensione del problema FISSA (N = N_MAX),
#           tempo di computazione vs numero di processori, scala log. Ideale: T(P) = T1/P.
# Slide 31-32, WEAK scaling (legge di Gustafson): lavoro per processore costante, la
#           dimensione del problema cresce con P. Tempo di computazione vs numero di
#           processori. Ideale: T(P) = T1 (retta orizzontale).
#           Il costo di un'iterazione GSA cresce come N^2 (forze), quindi N cresce come
#           sqrt(P) per tenere costante il lavoro per processore.
#
# PROCESSORE = 1 SM (streaming multiprocessor) della GPU.
# Per usare solo P SM, gsa_cuda.cu si compila con -DNBLK=P: i kernel fitness, forze F5 e
# update si lanciano con esattamente P blocchi, lo scheduler li assegna a P SM distinti
# e gli altri restano inattivi. Tutti gli N agenti sono comunque coperti (ciclo grid-stride).
# Non serve CUDA MPS: funziona anche sotto WSL e su qualsiasi GPU.
#
# Tempo misurato = "Tempo di calcolo" stampato da gsa_cuda.cu: wall time dell'intero ciclo
# GSA (ITER iterazioni), parte host compresa, come il "tempo di computazione" delle slide.
#
# Uso:
#   make scaling          (python3 scaling_cuda.py)          misura, salva scaling_F5.csv e le figure
#   make scaling-csv      (python3 scaling_cuda.py --da-csv) rifa' le figure dal CSV, senza GPU
# Architettura GPU: variabile d'ambiente ARCH (default "native" = GPU presente).
# ======================================================================
import subprocess, re, statistics, sys, os, math, csv

CUDA_SRC   = "gsa_cuda.cu"
ARCH       = os.environ.get("ARCH", "native")
K, NC, DEG = 64, 512, 3            # istanza della relazione
FLAGS_F5   = ["-DFORCE_WARP_DIM"]  # si misura solo la versione F5
N_MAX      = 12800                 # N fisso dello strong scaling = N del weak con tutti gli SM
ITER       = 10                    # iterazioni GSA per esecuzione
REPS       = 2                     # ripetizioni, si tiene la mediana
CSV_FILE   = "scaling_F5.csv"

RE_TOT = re.compile(r"Tempo di calcolo\s*=\s*([\d.]+)\s*ms")
RE_SM  = re.compile(r"\|\s*SM\s+(\d+)")


# ---------------------------------------------------------------- compilazione ed esecuzione
def compila(binname, n, extra):
    """Compila F5 con N = n, l'istanza della relazione e i flag aggiuntivi."""
    if os.path.exists(binname):                # gia' compilato in questa misura
        return
    cmd = ["nvcc", "-O2", f"-arch={ARCH}", f"-DN={n}", f"-DK={K}", f"-DNC={NC}",
           f"-DDEG={DEG}", f"-DMAX_ITER={ITER}", *FLAGS_F5, *extra, "-o", binname, CUDA_SRC]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("ERRORE di compilazione:", " ".join(cmd)); print(r.stderr); sys.exit(1)


def esegui(binname):
    """Esegue il binario e restituisce l'output (si ferma in caso di errore)."""
    r = subprocess.run([f"./{binname}"], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"ERRORE eseguendo {binname}:"); print(r.stdout, r.stderr); sys.exit(1)
    return r.stdout


def numero_sm():
    """Legge il numero di SM della GPU dalla riga 'Device: ... | SM n' di gsa_cuda.cu."""
    binname = "bin_F5_probe"
    compila(binname, 64, [])
    m = RE_SM.search(esegui(binname))
    if not m:
        print("Non riesco a leggere il numero di SM dall'output di gsa_cuda.cu."); sys.exit(1)
    return int(m.group(1))


def tempo(n, p):
    """Tempo di computazione (s) di F5 con N = n su P = p SM: mediana di REPS esecuzioni."""
    binname = f"bin_F5_{n}_p{p}"
    compila(binname, n, [f"-DNBLK={p}"])       # P blocchi fissi = P SM usati
    t = []
    for _ in range(REPS):
        m = RE_TOT.search(esegui(binname))
        if not m:
            print(f"Non trovo 'Tempo di calcolo' nell'output di {binname}."); sys.exit(1)
        t.append(float(m.group(1)) / 1000)
    return statistics.median(t)


# ---------------------------------------------------------------- misura
def misura():
    """Esegue strong e weak scaling per ogni P e salva i risultati in CSV_FILE."""
    if not os.path.exists(CUDA_SRC):
        print(f"Manca {CUDA_SRC} nella cartella corrente."); sys.exit(1)
    sm = numero_sm()
    # P = 1, 2, 4, ... fino al numero di SM, che si include sempre come ultimo punto
    procs = [p for p in (2 ** e for e in range(10)) if p < sm] + [sm]
    # weak: N proporzionale a sqrt(P) (lavoro ~ N^2), arrotondato a multipli di 64;
    # con tutti gli SM si arriva a N_MAX, lo stesso N dello strong scaling
    n_weak = [max(64, int(round(N_MAX * math.sqrt(p / sm) / 64) * 64)) for p in procs]
    print(f"GPU con {sm} SM   P = {procs}   strong: N = {N_MAX}   weak: N = {n_weak}", flush=True)

    righe = []
    for p, nw in zip(procs, n_weak):
        ts, tw = tempo(N_MAX, p), tempo(nw, p)
        righe += [("strong", p, N_MAX, ts), ("weak", p, nw, tw)]
        print(f"  P={p:2d}  strong {ts:8.3f} s   weak (N={nw:5d}) {tw:7.3f} s", flush=True)

    # controllo: se i P blocchi non finissero su P SM distinti, T(1)/T(2) sarebbe circa 1
    if len(procs) > 1 and righe[0][3] / righe[2][3] < 1.5:
        print(f"ATTENZIONE: T(1)/T(2) = {righe[0][3] / righe[2][3]:.2f}, atteso circa 2.")
    with open(CSV_FILE, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["scaling", "P", "N", "tempo_s"]); w.writerows(righe)
    return righe


def leggi_csv():
    """Rilegge i risultati di una misura precedente (per rifare le figure senza GPU)."""
    if not os.path.exists(CSV_FILE):
        print(f"Manca {CSV_FILE}: lancia prima la misura (make scaling)."); sys.exit(1)
    with open(CSV_FILE) as f:
        return [(d["scaling"], int(d["P"]), int(d["N"]), float(d["tempo_s"]))
                for d in csv.DictReader(f)]


# ---------------------------------------------------------------- figure (slide 30 e 32)
def figura(righe, scaling):
    """Stampa la tabella speedup/efficienza e disegna la figura di strong o weak scaling."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = sorted((p, n, t) for s, p, n, t in righe if s == scaling)
    P = [x[0] for x in pts]; N = [x[1] for x in pts]; T = [x[2] for x in pts]

    # slide 28: S = T1/TP, E = S/P. Nel weak con P processori si fa (N/N1)^2 volte il lavoro
    # di T1, quindi lo speedup e' quello "scalato" di Gustafson (slide 31).
    S = [T[0] / t for t in T] if scaling == "strong" else \
        [(n * n) / (N[0] * N[0]) * T[0] / t for n, t in zip(N, T)]
    print(f"\n{scaling.upper()} scaling — F5")
    print(f"  {'P':>3} {'N':>6} {'T (s)':>9} {'S':>7} {'E = S/P':>8}")
    for p, n, t, s in zip(P, N, T, S):
        print(f"  {p:>3} {n:>6} {t:>9.3f} {s:>7.2f} {s / p:>8.2f}")

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    if scaling == "strong":
        # ideale: T1/P, in scala log-log e' una retta (slide 30)
        ax.plot(P, [T[0] / p for p in P], "--", color="tab:green", zorder=3, label="ideal strong scaling")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_title(f"Strong scaling — F5, N = {N[0]} fisso")
    else:
        # ideale: tempo costante T1 (slide 32)
        ax.plot(P, [T[0]] * len(P), "--", color="tab:green", zorder=3, label="ideal weak scaling")
        ax.set_xscale("log", base=2); ax.set_ylim(0, max(T) * 1.4)
        ax.set_title("Weak scaling — F5, lavoro per processore costante")
        for p, n, t in zip(P, N, T):
            ax.annotate(f"N={n}", (p, t), xytext=(0, 9), textcoords="offset points",
                        ha="center", fontsize=8)
    ax.plot(P, T, "o-", color="tab:orange", linewidth=1.8, label="F5 misurato")
    ax.set_xticks(P); ax.set_xticklabels([str(p) for p in P])
    ax.set_xlabel("numero di processori P (SM)")
    ax.set_ylabel(f"tempo di computazione (s), {ITER} iterazioni")
    ax.grid(alpha=.25); ax.legend(frameon=False)
    fig.tight_layout()
    nome = f"fig_{scaling}_scaling_F5"
    fig.savefig(nome + ".pdf"); fig.savefig(nome + ".png", dpi=130)
    return nome + ".png"


def main():
    righe = leggi_csv() if "--da-csv" in sys.argv else misura()
    figs = [figura(righe, "strong"), figura(righe, "weak")]
    print("\nFatto:", ", ".join(f.replace(".png", ".{pdf,png}") for f in figs))


if __name__ == "__main__":
    main()
