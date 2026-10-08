# ======================================================================
# Makefile — progetto GSA (Gravitational Search Algorithm)
#            versione seriale in C + 5 versioni CUDA del kernel forze (F1..F5)
# ----------------------------------------------------------------------
# Target principali:
#   make               compila il seriale e le 5 versioni CUDA sull'istanza della relazione
#   make demo          lancia tutto in versione rapida e produce tutti i grafici (*_demo.png/pdf):
#                        1. tabella seriale / F1 / F4 / F5 (N piccolo)
#                        2. sweep delle 5 versioni F1..F5 (N = 100 -> 3200)  -> fig_forze_sweep_demo.*
#                        3. strong e weak scaling di F5                      -> fig_*_scaling_F5_demo.*
#                           (senza MPS, es. WSL: ridisegna da scaling_F5.csv se presente)
#   make clear         cancella eseguibili, log e file temporanei (alias: make clean)
#
# I grafici richiedono python3 + matplotlib:
#   sudo apt install -y --no-install-recommends python3-matplotlib && sudo apt clean
#
# Target per riprodurre le figure della relazione (versione completa, lunghi):
#   make sweep         tempo del kernel forze delle 5 versioni al variare di N  -> fig_forze_sweep.*
#   make scaling       strong e weak scaling di F5 (Lex01, slide 28-32): processore = 1 SM,
#                      P SM usati lanciando P blocchi (-DNBLK=P)       -> fig_*_scaling_F5.*
#   make scaling-csv   rifa' le figure di scaling da scaling_F5.csv, senza GPU
#   make relazione     sweep + scaling
#   make clear-risultati  cancella figure e CSV prodotti dai benchmark
#
# Parametri modificabili da riga di comando, per esempio:
#   make N=3200 MAX_ITER=200        (dopo `make clear`, perche' N e' fissato in compilazione)
#   make demo ARCH=sm_75            (architettura esplicita invece di "native")
# ======================================================================

# ---- compilatori e opzioni ----
CC      := gcc
NVCC    := nvcc
PYTHON  ?= python3
CFLAGS  := -O2 -Wall -Wno-misleading-indentation   # gli if su una riga sono voluti
LDLIBS  := -lm
ARCH    ?= native              # "native" = architettura della GPU presente (nvcc >= 11.5); T4 = sm_75
NVFLAGS := -O2 -arch=$(strip $(ARCH))

# ---- istanza della relazione (stessi valori degli script di benchmark) ----
N        ?= 1600               # numero di agenti
K        ?= 64                 # numero di variabili (dimensioni)
NC       ?= 512                # numero di weak constraint
DEG      ?= 3                  # grado dei monomi
MAX_ITER ?= 100                # iterazioni GSA
PARAMS   := -DN=$(strip $(N)) -DK=$(strip $(K)) -DNC=$(strip $(NC)) \
            -DDEG=$(strip $(DEG)) -DMAX_ITER=$(strip $(MAX_ITER))

# ---- istanza della demo: stesso problema, N e iterazioni ridotti per durare pochi secondi ----
DEMO_N    ?= 200
DEMO_ITER ?= 50
DEMO_PARAMS := -DN=$(strip $(DEMO_N)) -DK=$(strip $(K)) -DNC=$(strip $(NC)) \
               -DDEG=$(strip $(DEG)) -DMAX_ITER=$(strip $(DEMO_ITER))

# ---- flag che selezionano la versione del kernel forze in gsa_cuda.cu ----
FLAGS_F1 :=                    # baseline: un thread per agente, global memory
FLAGS_F2 := -DTILED            # + sorgenti a tile in shared memory
FLAGS_F3 := -DFORCE_WARP       # un warp per agente, riduzione con shuffle
FLAGS_F4 := -DFORCE_WARP_TILED # F3 + tile in shared
FLAGS_F5 := -DFORCE_WARP_DIM   # warp per agente, lane sulle dimensioni, padding

VERSIONI := F1 F2 F3 F4 F5
CUDA_BIN := $(addprefix gsa_cuda_,$(VERSIONI))   # gsa_cuda_F1 ... gsa_cuda_F5
DEMO_BIN := demo_serial demo_F1 demo_F4 demo_F5

# target che non corrispondono a file
.PHONY: all demo tabella check-python sweep scaling scaling-csv relazione clear clean clear-risultati

# ======================================================================
# Compilazione
# ======================================================================

# target di default: seriale + 5 versioni CUDA
all: gsa_serial $(CUDA_BIN)

# versione seriale (C puro)
gsa_serial: gsa_serial.c
	$(CC) $(CFLAGS) $(PARAMS) -o $@ $< $(LDLIBS)

# regola a pattern: gsa_cuda_F3 -> $* = F3 -> usa i flag FLAGS_F3
gsa_cuda_%: gsa_cuda.cu
	$(NVCC) $(NVFLAGS) $(PARAMS) $(FLAGS_$*) -o $@ $<

# binari della demo (istanza ridotta, nomi separati per non mescolarli con quelli di `make`)
demo_serial: gsa_serial.c
	$(CC) $(CFLAGS) $(DEMO_PARAMS) -o $@ $< $(LDLIBS)

# demo_F5 -> $* = 5 -> usa i flag FLAGS_F5
demo_F%: gsa_cuda.cu
	$(NVCC) $(NVFLAGS) $(DEMO_PARAMS) $(FLAGS_F$*) -o $@ $<

# ======================================================================
# Demo: tabella dei tempi + tutti i grafici in versione rapida
# ======================================================================
# QUICK=1 dice agli script di usare meno N e meno iterazioni e di salvare i file con
# suffisso _demo, cosi' le figure complete della relazione non vengono sovrascritte.
demo: check-python tabella
	@echo ""
	@echo "=== Grafico 1: sweep delle 5 versioni del kernel forze ==="
	ARCH=$(strip $(ARCH)) QUICK=1 $(PYTHON) benchmark_cuda.py
	@echo ""
	@echo "=== Grafici 2-3: strong e weak scaling di F5 ==="
	ARCH=$(strip $(ARCH)) QUICK=1 $(PYTHON) scaling_cuda.py --auto
	@echo ""
	@echo "=== Grafici prodotti ==="
	@ls -1 fig_*_demo.png fig_*_demo.pdf 2>/dev/null || echo "(nessuno)"

# controlla che matplotlib sia installato prima di partire (evita di scoprirlo dopo minuti)
check-python:
	@$(PYTHON) -c "import matplotlib" 2>/dev/null || { \
	    echo "Manca matplotlib. Installalo con:"; \
	    echo "  sudo apt install -y --no-install-recommends python3-matplotlib && sudo apt clean"; \
	    exit 1; }

# Tabella: esegue seriale, F1, F4, F5 sulla stessa istanza e confronta i tempi.
# Ogni esecuzione salva l'output completo in <binario>.log; a video si stampano la
# fitness trovata e il tempo. Lo speedup e' calcolato rispetto al seriale.
tabella: $(DEMO_BIN)
	@echo ""
	@echo "=== DEMO: N=$(strip $(DEMO_N)) K=$(strip $(K)) NC=$(strip $(NC)) DEG=$(strip $(DEG)) iterazioni=$(strip $(DEMO_ITER)) ==="
	@for b in $(DEMO_BIN); do \
	    ./$$b > $$b.log || { echo "ERRORE eseguendo $$b (vedi $$b.log)"; exit 1; }; \
	done
	@printf "%-12s %-34s %12s %10s\n" "versione" "miglior fitness" "tempo (ms)" "speedup"
	@t0=$$(sed -n 's/^Tempo di calcolo= *\([0-9.]*\) ms.*/\1/p' demo_serial.log); \
	for b in $(DEMO_BIN); do \
	    t=$$(sed -n 's/^Tempo di calcolo= *\([0-9.]*\) ms.*/\1/p' $$b.log); \
	    f=$$(sed -n 's/^Miglior fitness = *\(.*\)/\1/p' $$b.log); \
	    awk -v n="$${b#demo_}" -v f="$$f" -v a="$$t0" -v t="$$t" \
	        'BEGIN{printf "%-12s %-34s %12.1f %9.1fx\n", n, f, t, a/t}'; \
	done
	@echo "(output completo di ogni versione nei file demo_*.log)"

# ======================================================================
# Riproduzione delle figure della relazione
# ======================================================================
# ARCH viene passata agli script come variabile d'ambiente
sweep:
	ARCH=$(strip $(ARCH)) $(PYTHON) benchmark_cuda.py

scaling:
	ARCH=$(strip $(ARCH)) $(PYTHON) scaling_cuda.py

scaling-csv:
	$(PYTHON) scaling_cuda.py --da-csv

relazione: sweep scaling

# ======================================================================
# Pulizia
# ======================================================================
# clear: eseguibili, log della demo, binari temporanei degli script
clear:
	rm -f gsa_serial $(CUDA_BIN) $(DEMO_BIN) demo_*.log bench_bin bin_F5_*

clean: clear

# figure e CSV dei benchmark (tenuti separati: rimisurarli richiede minuti di GPU)
clear-risultati:
	rm -f fig_forze_sweep.pdf fig_forze_sweep.png \
	      fig_strong_scaling_F5.pdf fig_strong_scaling_F5.png \
	      fig_weak_scaling_F5.pdf fig_weak_scaling_F5.png scaling_F5.csv
