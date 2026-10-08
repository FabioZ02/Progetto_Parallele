/* ======================================================================
 * gsa_cuda.cu  —  Gravitational Search Algorithm, versione CUDA
 * ----------------------------------------------------------------------
 * Stesso problema e stesso algoritmo del seriale (max f - penalita', variabili
 * intere, hard constraint a clamp, weak constraint a penalita'), stessi semi e
 * stesso ordine dei numeri casuali.
 *
 * DIVISIONE DEL LAVORO HOST / DEVICE (per iterazione):
 *   device: (1) fitness, (5) forze, (6) update + clamp
 *   host  : best/worst, G(t), masse, ordinamento Kbest, numeri casuali (O(N), O(N log N))
 *
 * Kernel FORZE (parte O(N^2*K), collo di bottiglia): 5 versioni confrontate nella relazione
 *   F1  (default)            : un thread per agente, sorgenti letti dalla GLOBAL memory
 *   F2  -DTILED              : come F1, ma i sorgenti si caricano a TILE in SHARED e si riusano
 *   F3  -DFORCE_WARP         : UN WARP PER agente, le 32 lane si spartiscono i sorgenti,
 *                              contributi ridotti con __shfl_down_sync
 *   F4  -DFORCE_WARP_TILED   : F3 + sorgenti a tile in shared, riusati dai warp del blocco
 *   F5  -DFORCE_WARP_DIM     : un warp per agente + tile in shared, ma nell'accumulo le lane
 *                              si spartiscono le DIMENSIONI: letture contigue, padding K+1
 *                              contro i bank conflict, accumulatore in registri
 *   (-DFORCE_TD: variante scartata, un thread per (agente,dim), lasciata come risultato negativo)
 *
 * Kernel FITNESS (parte O(N*NC*NT*DEG)): nei benchmark resta la BASELINE, cosi' tra le
 * versioni cambia solo il kernel forze. Varianti studiate (appendice):
 *   default        : un thread per agente
 *   -DFIT_LDG      : tabelle vincoli via read-only cache (__ldg)
 *   -DFIT_WARP     : un warp per agente, lane sui vincoli, riduzione con shuffle
 *   -DFIT_WARP_SH  : come FIT_WARP, coordinate intere in shared (una copia per warp)
 *   -DFIT_WARP_SHT : come FIT_WARP_SH, tabelle vincoli trasposte (accessi coalescenti)
 *
 * -DPROFILE: misura con eventi CUDA il tempo dei kernel fitness e forze e stampa
 *            "k_forces  totale = ... ms (... ms/iter)", letto da benchmark_cuda.py.
 *
 * -DNBLK=P (solo con F5): scalabilita' strong/weak. I kernel fitness, forze F5 e update
 *            si lanciano con esattamente P blocchi: lo scheduler li assegna a P SM
 *            distinti, gli altri restano inattivi -> si usano P "processori".
 *            I kernel coprono comunque tutti gli N agenti con un ciclo grid-stride.
 *            Usato da scaling_cuda.py (make scaling).
 *
 * COMPILAZIONE (di solito tramite il Makefile: `make`, `make demo`):
 *   nvcc -O2 -arch=native -DN=1600 -DK=64 -DNC=512 -DDEG=3 -DMAX_ITER=100 \
 *        -DFORCE_WARP_DIM -o gsa_cuda_F5 gsa_cuda.cu
 * La valutazione e' intera: le fitness non dipendono dalla variante del kernel fitness.
 * Le forze sono in double: F3-F5 sommano in ordine diverso, quindi possono differire
 * da F1 di pochi ULP.
 * ====================================================================== */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* Gestione errori CUDA (dalle slide del corso) */
static void HandleError(cudaError_t e,const char*f,int l){
    if(e!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(e),f,l);exit(EXIT_FAILURE);}
}
#define HANDLE_ERROR(e) (HandleError((e),__FILE__,__LINE__))
/* Da chiamare dopo ogni lancio di kernel: i lanci non restituiscono un errore direttamente. */
static void checkCUDAError(const char*m){
    cudaError_t e=cudaGetLastError();
    if(e!=cudaSuccess){fprintf(stderr,"ERRORE CUDA >%s<: >%s<\n",m,cudaGetErrorString(e));exit(-1);}
}

/* ---- taglia dell'istanza (override con -D...) ---- */
#ifndef N
#define N 50              /* agenti */
#endif
#ifndef K
#define K 30              /* variabili */
#endif
#ifndef MAX_ITER
#define MAX_ITER 1000
#endif
#ifndef BLOCK
#define BLOCK 256         /* thread per blocco (multiplo di 32 per le varianti warp) */
#endif
#ifndef TILE
#define TILE 32           /* sorgenti caricati per tile (varianti shared) */
#endif
#ifndef WARP
#define WARP 32           /* lane per warp */
#endif
/* ---- problema: polinomi a coefficienti interi ---- */
#ifndef DEG
#define DEG 2
#endif
#ifndef NT
#define NT K
#endif
#ifndef NC
#define NC 100            /* numero di weak constraint */
#endif
#define LO (-5)
#define HI ( 5)
#define CSEED 12345u
#define G0 100.0
#define ALPHA 20.0
#define EPS 1e-9
#define SEED 42u

/* Numero di blocchi dei kernel: quello necessario a coprire N (default) oppure
 * fissato a NBLK per gli esperimenti di scalabilita' (P blocchi = P SM usati). */
#ifdef NBLK
  #if !defined(FORCE_WARP_DIM)
    #error "NBLK (scalabilita') e' previsto solo per la versione F5 (-DFORCE_WARP_DIM)"
  #endif
  #if defined(FIT_LDG) || defined(FIT_WARP) || defined(FIT_WARP_SH) || defined(FIT_WARP_SHT)
    #error "NBLK va usato con la fitness baseline (nessun flag FIT_)"
  #endif
  #define GRID(nb) (NBLK)
#else
  #define GRID(nb) (nb)
#endif

/* FIT_LDG: letture dei vincoli via read-only data-cache (__ldg).
 * Degrada a lettura normale sul path host e senza il flag. */
#if defined(FIT_LDG) && defined(__CUDA_ARCH__)
  #define RLD(e) __ldg(&(e))
#else
  #define RLD(e) (e)
#endif

/* Tabelle delle funzioni su host: generate come nel seriale, poi copiate sul device. */
static int h_cf[NT], h_vf[NT*DEG], h_cg[NC*NT], h_vg[NC*NT*DEG], h_Pj[NC];

static int randc(void){ int c; do { c=rand()%7-3; } while(c==0); return c; }
static void gen_functions(void){
    srand(CSEED);
    for(int t=0;t<NT;++t){ h_cf[t]=randc(); for(int d=0;d<DEG;++d) h_vf[t*DEG+d]=rand()%K; }
    for(int j=0;j<NC;++j){
        for(int t=0;t<NT;++t){ h_cg[j*NT+t]=randc(); for(int d=0;d<DEG;++d) h_vg[(j*NT+t)*DEG+d]=rand()%K; }
        h_Pj[j]=1+rand()%10;
    }
}

/* __host__ __device__: stesse funzioni usate sia su CPU sia su GPU -> stessi numeri. */
__host__ __device__ static int to_int(double v){          /* variabile intera: round + clamp nel dominio */
    int r=(int)(v>=0 ? v+0.5 : v-0.5);
    if(r<LO)r=LO; if(r>HI)r=HI; return r;
}
__host__ __device__ static long long poly_eval(const int*xi,const int*coeff,const int*vars,int nt){
    long long s=0;                                          /* somma di monomi coeff * prod(x), interi */
    for(int t=0;t<nt;++t){
        long long term=RLD(coeff[t]);                       /* read-only cache su coeff (device, con FIT_LDG) */
        for(int d=0;d<DEG;++d) term *= (long long) xi[ RLD(vars[t*DEG+d]) ];  /* e su vars */
        s+=term;
    }
    return s;
}

/* KERNEL 1a - FITNESS BASELINE: un thread per agente; valuta f e gli NC vincoli in interi.
 * Costo O(NC*NT*DEG) per thread: la riduzione della penalita' e' SERIALE dentro il thread.
 * Ciclo grid-stride: con la griglia normale (blocchi sufficienti a coprire N) ogni thread
 * fa una sola passata, come prima; con NBLK blocchi fissi ogni thread gestisce piu' agenti. */
__global__ void k_fitness(const double*x,double*fit,
                          const int*cf,const int*vf,const int*cg,const int*vg,const int*Pj){
    for(int i=blockIdx.x*blockDim.x+threadIdx.x; i<N; i+=gridDim.x*blockDim.x){   /* 1 thread <-> 1 agente */
        int xi[K];
        for(int d=0;d<K;++d) xi[d]=to_int(x[i*K+d]);
        long long f=poly_eval(xi,cf,vf,NT);
        long long pen=0;
        for(int j=0;j<NC;++j){ long long g=poly_eval(xi, cg+j*NT, vg+(size_t)j*NT*DEG, NT);
            if(!(g>0)) pen+=Pj[j]; }                       /* vincolo violato -> penalita' */
        fit[i]=(double)(f-pen);
    }
}

/* KERNEL 1b - FITNESS WARP: UN WARP PER AGENTE.
 * Le 32 lane si spartiscono gli NC vincoli (lane l fa j=l, l+32, l+64, ...),
 * ognuna accumula la sua penalita' parziale, poi la somma si riduce sul warp con
 * __shfl_down_sync. La somma e' intera -> il risultato e' IDENTICO al
 * baseline (addizione associativa: l'ordine non cambia il valore). */
__global__ void k_fitness_warp(const double*x,double*fit,
                               const int*cf,const int*vf,const int*cg,const int*vg,const int*Pj){
    int i    = (blockIdx.x*blockDim.x+threadIdx.x)/WARP;   /* warp id = agente */
    int lane = threadIdx.x % WARP;
    if(i>=N) return;                                       /* uniforme sul warp: tutte 32 le lane concordano */
    /* ogni lane si carica le coordinate intere dell'agente (ridondante ma economico
     * rispetto ai ~NC*NT*DEG prodotti dei vincoli) */
    int xi[K];
    for(int d=0;d<K;++d) xi[d]=to_int(x[i*K+d]);
    /* le lane si spartiscono i vincoli */
    long long pen=0;
    for(int j=lane;j<NC;j+=WARP){
        long long g=poly_eval(xi, cg+j*NT, vg+(size_t)j*NT*DEG, NT);
        if(!(g>0)) pen+=Pj[j];
    }
    /* riduzione della penalita' sul warp: a ogni passo la lane l somma il valore della lane l+off */
    for(int off=WARP/2; off>0; off>>=1)
        pen += __shfl_down_sync(0xffffffff, pen, off);
    /* lane 0 aggiunge f (una volta) e scrive */
    if(lane==0){
        long long f=poly_eval(xi,cf,vf,NT);
        fit[i]=(double)(f-pen);
    }
}

/* KERNEL 1c - FITNESS WARP + SHARED xi: come 1b, ma le coordinate intere
 * dell'agente stanno UNA sola volta per warp in shared memory (non replicate
 * per lane in local memory). Le 32 lane caricano xi cooperativamente, poi lo
 * condividono per valutare i vincoli. Risultato ancora bit-identico (aritmetica intera). */
__global__ void k_fitness_warp_sh(const double*x,double*fit,
                                  const int*cf,const int*vf,const int*cg,const int*vg,const int*Pj){
    extern __shared__ int s_xi[];                          /* wpb*K int: un blocco xi per warp del blocco */
    int wpb  = blockDim.x / WARP;                          /* warp per blocco */
    int wl   = threadIdx.x / WARP;                         /* indice del warp dentro il blocco */
    int lane = threadIdx.x % WARP;
    int i    = blockIdx.x*wpb + wl;                        /* warp id globale = agente */
    int *xi  = s_xi + wl*K;                                /* la fetta di shared di questo warp */
    if(i<N)                                                /* caricamento COOPERATIVO di xi (una copia sola) */
        for(int d=lane; d<K; d+=WARP) xi[d]=to_int(x[i*K+d]);
    __syncwarp();                                          /* Volta+: serve sync esplicita intra-warp */
    if(i>=N) return;                                       /* uniforme sul warp */
    long long pen=0;
    for(int j=lane;j<NC;j+=WARP){                          /* le lane si spartiscono i vincoli */
        long long g=poly_eval(xi, cg+j*NT, vg+(size_t)j*NT*DEG, NT);
        if(!(g>0)) pen+=Pj[j];
    }
    for(int off=WARP/2; off>0; off>>=1)                    /* riduzione penalita' sul warp */
        pen += __shfl_down_sync(0xffffffff, pen, off);
    if(lane==0){ long long f=poly_eval(xi,cf,vf,NT); fit[i]=(double)(f-pen); }
}

/* poly_eval su tabelle TRASPOSTE: coeff[t*NC+j] e vars[(t*NC+j)*DEG+d] a j fisso.
 * Passo NC tra monomi consecutivi -> a monomio fisso le lane (j contigui) coalescono. */
__device__ static long long poly_eval_T(const int*xi,const int*coeffT,const int*varsT,int j,int nt){
    long long s=0;
    for(int t=0;t<nt;++t){
        long long term=coeffT[(size_t)t*NC+j];
        for(int d=0;d<DEG;++d) term *= (long long) xi[ varsT[((size_t)t*NC+j)*DEG+d] ];
        s+=term;
    }
    return s;
}

/* KERNEL 1d - FITNESS WARP + SHARED xi + tabelle TRASPOSTE: come 1c, ma
 * i vincoli si leggono da cgT/vgT trasposte, cosi' a monomio fisso le 32 lane del warp
 * accedono a indici j contigui -> letture coalescenti. f resta su layout normale
 * (una sola valutazione, lane 0). Risultato ancora bit-identico. */
__global__ void k_fitness_warp_sht(const double*x,double*fit,
                                   const int*cf,const int*vf,const int*cgT,const int*vgT,const int*Pj){
    extern __shared__ int s_xi[];
    int wpb  = blockDim.x / WARP;
    int wl   = threadIdx.x / WARP;
    int lane = threadIdx.x % WARP;
    int i    = blockIdx.x*wpb + wl;
    int *xi  = s_xi + wl*K;
    if(i<N) for(int d=lane; d<K; d+=WARP) xi[d]=to_int(x[i*K+d]);
    __syncwarp();
    if(i>=N) return;
    long long pen=0;
    for(int j=lane;j<NC;j+=WARP){
        long long g=poly_eval_T(xi, cgT, vgT, j, NT);
        if(!(g>0)) pen+=Pj[j];
    }
    for(int off=WARP/2; off>0; off>>=1)
        pen += __shfl_down_sync(0xffffffff, pen, off);
    if(lane==0){ long long f=poly_eval(xi,cf,vf,NT); fit[i]=(double)(f-pen); }
}

/* KERNEL 2a - FORZE F1, BASELINE: un thread per agente i; ogni thread rilegge le
 * posizioni dei Kbest sorgenti dalla GLOBAL memory e accumula le K componenti
 * dell'accelerazione in acc_l[K] (array locale del thread). */
__global__ void k_forces(const double*x,const double*M,const int*src,int kbest,
                         const double*randF,double G,double*acc){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N) return;
    double acc_l[K]; for(int d=0;d<K;++d) acc_l[d]=0;
    for(int s=0;s<kbest;++s){ int j=src[s]; if(j==i)continue;               /* sorgente j (un agente non attrae se stesso) */
        double R=0; for(int d=0;d<K;++d){double df=x[j*K+d]-x[i*K+d];R+=df*df;} R=sqrt(R);   /* distanza R_ij */
        double factor=G*M[j]/(R+EPS),r=randF[s];
        for(int d=0;d<K;++d) acc_l[d]+=r*factor*(x[j*K+d]-x[i*K+d]);       /* attrazione verso j */
    }
    for(int d=0;d<K;++d) acc[i*K+d]=acc_l[d];
}

/* KERNEL 2b - FORZE F2, TILED: i sorgenti si caricano una volta in SHARED memory e
 * si riusano da tutti i thread del blocco. Stesso ordine di accumulo del baseline
 * -> risultato identico, ma molte meno letture dalla global memory. */
__global__ void k_forces_tiled(const double*x,const double*M,const int*src,int kbest,
                               const double*randF,double G,double*acc){
    __shared__ double s_x[TILE*K];   /* posizioni dei sorgenti del tile (in shared) */
    __shared__ double s_M[TILE];     /* loro masse */
    __shared__ double s_r[TILE];     /* loro numeri casuali */
    __shared__ int    s_j[TILE];     /* loro indici (per saltare j==i) */
    int i=blockIdx.x*blockDim.x+threadIdx.x; int active=(i<N);   /* i thread in eccesso caricano ma non calcolano */
    double x_i[K],acc_l[K];
    if(active) for(int d=0;d<K;++d){ x_i[d]=x[i*K+d]; acc_l[d]=0; }   /* la propria posizione sta nei registri */
    for(int s0=0;s0<kbest;s0+=TILE){                       /* scorre i sorgenti a blocchi di TILE */
        int cnt=kbest-s0; if(cnt>TILE)cnt=TILE;            /* l'ultimo tile puo' essere incompleto */
        for(int t=threadIdx.x;t<cnt;t+=blockDim.x){        /* caricamento COOPERATIVO del tile */
            int j=src[s0+t]; s_j[t]=j; s_M[t]=M[j]; s_r[t]=randF[s0+t];
            for(int d=0;d<K;++d) s_x[t*K+d]=x[j*K+d];
        }
        __syncthreads();                                   /* BARRIERA: il tile e' pronto per tutti */
        if(active) for(int t=0;t<cnt;++t){ int j=s_j[t]; if(j==i)continue;   /* RIUSO dalla shared, non dalla global */
            double R=0; for(int d=0;d<K;++d){double df=s_x[t*K+d]-x_i[d];R+=df*df;} R=sqrt(R);
            double factor=G*s_M[t]/(R+EPS),r=s_r[t];
            for(int d=0;d<K;++d) acc_l[d]+=r*factor*(s_x[t*K+d]-x_i[d]);
        }
        __syncthreads();                                   /* prima di sovrascrivere il tile successivo */
    }
    if(active) for(int d=0;d<K;++d) acc[i*K+d]=acc_l[d];
}

/* KERNEL 2c - FORZE, THREAD (agente,dim) (variante scartata): un thread per coppia
 * (agente, dimensione) -> N*K thread. L'accumulatore e' uno SCALARE (niente acc_l[K]),
 * ma la distanza R_ij viene RICALCOLATA da ognuna delle K dimensioni (spreco ~K x).
 * Stesso ordine di accumulo del baseline -> risultato identico. */
__global__ void k_forces_td(const double*x,const double*M,const int*src,int kbest,
                            const double*randF,double G,double*acc){
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    int i=tid/K, d=tid%K; if(i>=N) return;                /* (i,d): agente e dimensione gestiti */
    double xi_d=x[i*K+d];                                  /* la propria coordinata in dim d */
    double a=0.0;
    for(int s=0;s<kbest;++s){ int j=src[s]; if(j==i)continue;
        double R=0; for(int e=0;e<K;++e){double df=x[j*K+e]-x[i*K+e];R+=df*df;} R=sqrt(R);  /* R ricalcolata */
        double factor=G*M[j]/(R+EPS),r=randF[s];
        a+=r*factor*(x[j*K+d]-xi_d);
    }
    acc[i*K+d]=a;
}

/* KERNEL 2d - FORZE F3, WARP per agente: UN WARP PER AGENTE, le 32 lane si spartiscono
 * i SORGENTI (lane l fa s=l, l+32, ...). La distanza R_ij si calcola UNA volta per
 * sorgente, i K contributi si accumulano in un acc_l[K] parziale per lane; alla fine
 * ogni componente si riduce sul warp con __shfl_down_sync.
 * NB: accumulo in double con ordine diverso dal baseline -> possibili differenze di
 * pochi ULP (non e' intero). */
__global__ void k_forces_warp(const double*x,const double*M,const int*src,int kbest,
                             const double*randF,double G,double*acc){
    int i    = blockIdx.x*(blockDim.x/WARP) + threadIdx.x/WARP;   /* warp globale = agente */
    int lane = threadIdx.x % WARP;
    if(i>=N) return;                                              /* uniforme sul warp */
    double acc_l[K]; for(int d=0;d<K;++d) acc_l[d]=0.0;           /* accumulatore parziale della lane */
    for(int s=lane;s<kbest;s+=WARP){ int j=src[s]; if(j==i)continue;
        double R=0; for(int d=0;d<K;++d){double df=x[j*K+d]-x[i*K+d];R+=df*df;} R=sqrt(R);  /* R: una volta per sorgente */
        double factor=G*M[j]/(R+EPS),r=randF[s];
        for(int d=0;d<K;++d) acc_l[d]+=r*factor*(x[j*K+d]-x[i*K+d]);
    }
    /* riduzione sul warp, una componente per volta (K riduzioni ad albero, log2(32)=5 passi ciascuna) */
    for(int d=0;d<K;++d){
        double v=acc_l[d];
        for(int off=WARP/2; off>0; off>>=1) v += __shfl_down_sync(0xffffffff, v, off);
        if(lane==0) acc[i*K+d]=v;                                 /* il totale finisce nella lane 0 */
    }
}

/* KERNEL 2e - FORZE F4, WARP + TILING shared: unisce le due leve. UN WARP PER AGENTE
 * (R una volta per sorgente, riduzione shuffle) E i sorgenti caricati a TILE in SHARED
 * una volta per blocco: gli wpb warp del blocco sono agenti diversi ma attraggono verso
 * gli STESSI Kbest sorgenti, quindi riusano lo stesso tile invece di rileggerlo da global.
 * NB: accumulo in double con ordine diverso dal baseline -> possibili differenze di pochi ULP. */
__global__ void k_forces_warp_tiled(const double*x,const double*M,const int*src,int kbest,
                                    const double*randF,double G,double*acc){
    __shared__ double s_x[TILE*K];   /* un tile di sorgenti per BLOCCO, condiviso da tutti i warp */
    __shared__ double s_M[TILE];
    __shared__ double s_r[TILE];
    __shared__ int    s_j[TILE];
    int wpb  = blockDim.x / WARP;                         /* warp per blocco */
    int wl   = threadIdx.x / WARP;                        /* warp dentro il blocco */
    int lane = threadIdx.x % WARP;
    int i    = blockIdx.x*wpb + wl;                       /* warp = agente */
    int active = (i<N);                                   /* uniforme sul warp (i non dipende da lane) */
    double acc_l[K]; if(active) for(int d=0;d<K;++d) acc_l[d]=0.0;
    for(int s0=0;s0<kbest;s0+=TILE){
        int cnt=kbest-s0; if(cnt>TILE)cnt=TILE;
        for(int t=threadIdx.x;t<cnt;t+=blockDim.x){       /* caricamento cooperativo del tile (tutto il blocco) */
            int j=src[s0+t]; s_j[t]=j; s_M[t]=M[j]; s_r[t]=randF[s0+t];
            for(int d=0;d<K;++d) s_x[t*K+d]=x[j*K+d];
        }
        __syncthreads();                                  /* tile pronto per tutti i warp */
        if(active) for(int t=lane;t<cnt;t+=WARP){         /* le lane del warp si spartiscono i sorgenti del tile */
            int j=s_j[t]; if(j==i) continue;
            double R=0; for(int d=0;d<K;++d){double df=s_x[t*K+d]-x[i*K+d];R+=df*df;} R=sqrt(R);
            double factor=G*s_M[t]/(R+EPS), r=s_r[t];
            for(int d=0;d<K;++d) acc_l[d]+=r*factor*(s_x[t*K+d]-x[i*K+d]);
        }
        __syncthreads();                                  /* prima di sovrascrivere il tile */
    }
    if(active) for(int d=0;d<K;++d){                      /* riduzione shuffle di ogni componente sul warp */
        double v=acc_l[d];
        for(int off=WARP/2; off>0; off>>=1) v += __shfl_down_sync(0xffffffff, v, off);
        if(lane==0) acc[i*K+d]=v;
    }
}

/* KERNEL 2f - FORZE F5, WARP per agente con LANE SULLE DIMENSIONI.
 * Unisce la decomposizione di F3/F4 (un warp per agente -> tutti gli SM) con l'accesso
 * alla memoria di F2 (tutte le lane sullo stesso sorgente). Per ogni tile in shared:
 *  - caricamento cooperativo COALESCENTE: thread consecutivi leggono d consecutivi;
 *  - fase A (lane <-> sorgente): la lane t calcola R_t e il coefficiente r*G*M/(R+EPS)
 *    UNA volta per coppia (agente, sorgente); s_x ha passo K+1 (padding), cosi' le lane
 *    che leggono righe diverse cadono su banchi diversi (niente bank conflict);
 *  - fase B (lane <-> dimensione): il warp scorre i sorgenti in ordine, la lane l
 *    aggiorna le dimensioni l, l+32, ...: letture contigue in shared, il coefficiente
 *    e' letto in broadcast, l'accumulatore sta in DPL=K/32 registri (niente acc_l[K]).
 * Stesso ordine di accumulo e stesse espressioni del baseline. */
#define DPL ((K+WARP-1)/WARP)                             /* dimensioni per lane (K=64 -> 2) */
__global__ void k_forces_warp_dim(const double*x,const double*M,const int*src,int kbest,
                                  const double*randF,double G,double*acc){
    __shared__ double s_x[TILE*(K+1)];                    /* tile sorgenti, passo K+1 (padding) */
    __shared__ double s_M[TILE];
    __shared__ double s_r[TILE];
    __shared__ int    s_j[TILE];
    __shared__ double s_own[(BLOCK/WARP)*K];              /* riga dell'agente di ogni warp */
    __shared__ double s_c[(BLOCK/WARP)*TILE];             /* coefficiente per (warp, sorgente del tile) */
    int wpb  = blockDim.x / WARP;
    int wl   = threadIdx.x / WARP;
    int lane = threadIdx.x % WARP;
    double *own = s_own + wl*K, *cf = s_c + wl*TILE;      /* fette di shared di questo warp */
    /* Ciclo grid-stride sui gruppi di wpb agenti: con la griglia normale ogni blocco fa una
     * sola passata (come prima); con NBLK blocchi fissi ne fa piu' d'una. La condizione
     * dipende solo da blockIdx, quindi tutti i warp del blocco fanno le stesse passate e
     * i __syncthreads() restano raggiunti da tutto il blocco. */
    for(int base=blockIdx.x*wpb; base<N; base+=gridDim.x*wpb){
        int i      = base + wl;                           /* warp = agente */
        int active = (i<N);                               /* uniforme sul warp */
        double xd[DPL], a[DPL];                           /* proprie coordinate e accumulatore: in registri */
        if(active) for(int d=lane; d<K; d+=WARP) own[d]=x[i*K+d];   /* lettura coalescente della propria riga */
        __syncwarp();
        for(int c=0;c<DPL;++c){ int d=lane+c*WARP; xd[c]=(active&&d<K)?own[d]:0.0; a[c]=0.0; }
        for(int s0=0;s0<kbest;s0+=TILE){
            int cnt=kbest-s0; if(cnt>TILE)cnt=TILE;
            for(int t=threadIdx.x;t<cnt;t+=blockDim.x){   /* metadati del tile */
                int j=src[s0+t]; s_j[t]=j; s_M[t]=M[j]; s_r[t]=randF[s0+t];
            }
            for(int e=threadIdx.x;e<cnt*K;e+=blockDim.x){ /* posizioni: thread consecutivi -> d consecutivi */
                int t=e/K, d=e%K; s_x[t*(K+1)+d]=x[src[s0+t]*K+d];
            }
            __syncthreads();                              /* tile pronto per tutti i warp */
            if(active){
                /* fase A: lane <-> sorgente, R e coefficiente una volta per coppia */
                for(int t=lane;t<cnt;t+=WARP){
                    double R=0; for(int d=0;d<K;++d){double df=s_x[t*(K+1)+d]-own[d];R+=df*df;} R=sqrt(R);
                    double factor=G*s_M[t]/(R+EPS), r=s_r[t];
                    cf[t]=r*factor;
                }
                __syncwarp();                             /* coefficienti visibili a tutto il warp */
                /* fase B: lane <-> dimensione, sorgenti in ordine (come il baseline) */
                for(int t=0;t<cnt;++t){ if(s_j[t]==i) continue;   /* uniforme sul warp */
                    double coef=cf[t];                    /* broadcast: tutte le lane leggono lo stesso indirizzo */
                    for(int c=0;c<DPL;++c){ int d=lane+c*WARP;
                        if(d<K) a[c]+=coef*(s_x[t*(K+1)+d]-xd[c]); }
                }
            }
            __syncthreads();                              /* prima di sovrascrivere il tile */
        }
        if(active) for(int c=0;c<DPL;++c){ int d=lane+c*WARP; if(d<K) acc[i*K+d]=a[c]; }  /* scrittura coalescente */
    }
}

/* KERNEL 3 - UPDATE: nuova velocita' e posizione, poi CLAMP nel dominio (hard constraint).
 * Un thread per agente, comune a tutte le versioni; ciclo grid-stride come k_fitness. */
__global__ void k_update(double*x,double*v,const double*acc,const double*randV,
                         const double*LB,const double*UB){
    for(int i=blockIdx.x*blockDim.x+threadIdx.x; i<N; i+=gridDim.x*blockDim.x)
        for(int d=0;d<K;++d){ int id=i*K+d;
            v[id]=randV[id]*v[id]+acc[id]; double xv=x[id]+v[id];
            if(xv<LB[d])xv=LB[d]; if(xv>UB[d])xv=UB[d]; x[id]=xv; }
}

/* ===== host ===== */
static double urand(void){ return (double)rand()/((double)RAND_MAX+1.0); }   /* uniforme in [0,1) */
static const double*g_mass_ptr;                                             /* masse per l'ordinamento */
static int cmp_mass_desc(const void*a,const void*b){                        /* ordina indici per massa decrescente */
    double ma=g_mass_ptr[*(const int*)a],mb=g_mass_ptr[*(const int*)b];
    return (ma<mb)-(ma>mb);
}

int main(void){
    gen_functions(); srand(SEED);                          /* stesse funzioni e stessa dinamica del seriale */

    cudaDeviceProp prop; HANDLE_ERROR(cudaGetDeviceProperties(&prop,0));   /* info device (dalle slide) */
    printf("Device: %s | cc %d.%d | SM %d\n",prop.name,prop.major,prop.minor,prop.multiProcessorCount);
#ifdef NBLK
    printf("Scalabilita': griglia fissa di %d blocchi (= SM usati)\n",NBLK);
#endif
    /* stampa quale variante e' stata compilata (letto anche dagli script) */
#if defined(FORCE_WARP_DIM)
    printf("Forze:   WARP per agente, lane sulle DIMENSIONI + TILING shared (F5)\n");
#elif defined(FORCE_WARP_TILED)
    printf("Forze:   WARP + shuffle + TILING shared (F4)\n");
#elif defined(FORCE_WARP)
    printf("Forze:   WARP per agente + shuffle sui sorgenti (F3)\n");
#elif defined(FORCE_TD)
    printf("Forze:   THREAD (agente,dim), R ricalcolata\n");
#elif defined(TILED)
    printf("Forze:   TILED (shared), TILE=%d (F2)\n",TILE);
#else
    printf("Forze:   BASELINE (global) (F1)\n");
#endif
#if defined(FIT_WARP_SHT)
    printf("Fitness: WARP+SHUFFLE, xi in SHARED, tabelle TRASPOSTE (1 warp/agente)\n");
#elif defined(FIT_WARP_SH)
    printf("Fitness: WARP+SHUFFLE, xi in SHARED (1 warp/agente)\n");
#elif defined(FIT_WARP)
    printf("Fitness: WARP+SHUFFLE, xi in local (1 warp/agente)\n");
#elif defined(FIT_LDG)
    printf("Fitness: LDG (read-only cache)\n");
#else
    printf("Fitness: BASELINE (1 thread/agente, global)\n");
#endif
#ifdef PROFILE
    printf("Profiling: ON (tempi per-kernel con eventi CUDA)\n");
#endif
    size_t tab_bytes=sizeof(h_cf)+sizeof(h_vf)+sizeof(h_cg)+sizeof(h_vg)+sizeof(h_Pj);
    printf("Tabelle funzioni: %.1f KB (NC=%d, NT=%d, DEG=%d)\n\n",tab_bytes/1024.0,NC,NT,DEG);

    const int nB=(N+BLOCK-1)/BLOCK;                        /* blocchi per coprire N agenti (1 thread/agente) */
    const size_t szXK=(size_t)N*K*sizeof(double), szN=(size_t)N*sizeof(double);
    double h_LB[K],h_UB[K]; for(int d=0;d<K;++d){h_LB[d]=LO;h_UB[d]=HI;}

    /* buffer host */
    double *h_x=(double*)malloc(szXK),*h_fit=(double*)malloc(szN),*h_M=(double*)malloc(szN);
    double *h_randF=(double*)malloc(szN),*h_randV=(double*)malloc(szXK);
    int *idx=(int*)malloc((size_t)N*sizeof(int)),*h_src=(int*)malloc((size_t)N*sizeof(int));
    double gbest_x[K];

    /* buffer device: allocazione dinamica in global memory */
    double *d_x,*d_v,*d_acc,*d_fit,*d_M,*d_randF,*d_randV,*d_LB,*d_UB; int *d_src;
    int *d_cf,*d_vf,*d_cg,*d_vg,*d_Pj;
    HANDLE_ERROR(cudaMalloc(&d_x,szXK));   HANDLE_ERROR(cudaMalloc(&d_v,szXK));
    HANDLE_ERROR(cudaMalloc(&d_acc,szXK)); HANDLE_ERROR(cudaMalloc(&d_fit,szN));
    HANDLE_ERROR(cudaMalloc(&d_M,szN));    HANDLE_ERROR(cudaMalloc(&d_randF,szN));
    HANDLE_ERROR(cudaMalloc(&d_randV,szXK));
    HANDLE_ERROR(cudaMalloc(&d_LB,K*sizeof(double))); HANDLE_ERROR(cudaMalloc(&d_UB,K*sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_src,(size_t)N*sizeof(int)));
    HANDLE_ERROR(cudaMalloc(&d_cf,sizeof(h_cf))); HANDLE_ERROR(cudaMalloc(&d_vf,sizeof(h_vf)));
    HANDLE_ERROR(cudaMalloc(&d_cg,sizeof(h_cg))); HANDLE_ERROR(cudaMalloc(&d_vg,sizeof(h_vg)));
    HANDLE_ERROR(cudaMalloc(&d_Pj,sizeof(h_Pj)));

    /* posizioni iniziali generate su host (stessa sequenza del seriale), velocita' nulle */
    for(int i=0;i<N;++i) for(int d=0;d<K;++d) h_x[i*K+d]=h_LB[d]+urand()*(h_UB[d]-h_LB[d]);
    HANDLE_ERROR(cudaMemcpy(d_x,h_x,szXK,cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemset(d_v,0,szXK));
    HANDLE_ERROR(cudaMemcpy(d_LB,h_LB,K*sizeof(double),cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_UB,h_UB,K*sizeof(double),cudaMemcpyHostToDevice));
    /* le tabelle delle funzioni si copiano UNA VOLTA sola (non cambiano) */
    HANDLE_ERROR(cudaMemcpy(d_cf,h_cf,sizeof(h_cf),cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_vf,h_vf,sizeof(h_vf),cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_cg,h_cg,sizeof(h_cg),cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_vg,h_vg,sizeof(h_vg),cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_Pj,h_Pj,sizeof(h_Pj),cudaMemcpyHostToDevice));

#ifdef FIT_WARP_SHT
    /* FIT_WARP_SHT: tabelle vincoli trasposte cgT[t*NC+j], vgT[(t*NC+j)*DEG+d].
     * Trasposizione UNA volta sola su host, poi copia sul device. */
    int *h_cgT=(int*)malloc(sizeof(h_cg)), *h_vgT=(int*)malloc(sizeof(h_vg));
    for(int j=0;j<NC;++j) for(int t=0;t<NT;++t){
        h_cgT[(size_t)t*NC+j]=h_cg[j*NT+t];
        for(int d=0;d<DEG;++d) h_vgT[((size_t)t*NC+j)*DEG+d]=h_vg[((size_t)j*NT+t)*DEG+d];
    }
    int *d_cgT,*d_vgT;
    HANDLE_ERROR(cudaMalloc(&d_cgT,sizeof(h_cg))); HANDLE_ERROR(cudaMalloc(&d_vgT,sizeof(h_vg)));
    HANDLE_ERROR(cudaMemcpy(d_cgT,h_cgT,sizeof(h_cg),cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_vgT,h_vgT,sizeof(h_vg),cudaMemcpyHostToDevice));
#endif

#ifdef PROFILE
    /* eventi CUDA per cronometrare i soli kernel fitness e forze */
    cudaEvent_t evA,evB,evC; float ms_tmp;
    HANDLE_ERROR(cudaEventCreate(&evA)); HANDLE_ERROR(cudaEventCreate(&evB)); HANDLE_ERROR(cudaEventCreate(&evC));
    double t_fit=0.0, t_for=0.0;
#endif

    double gbest_fit=-INFINITY;
    struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);   /* inizio misura: solo il ciclo GSA */

    /* ===================== CICLO PRINCIPALE ===================== */
    for(int t=0;t<MAX_ITER;++t){
        /* (1) fitness sul device */
#ifdef PROFILE
        cudaEventRecord(evA);
#endif
#if defined(FIT_WARP_SHT)
        { int wpb=BLOCK/WARP; int nBw=(N+wpb-1)/wpb; size_t shm=(size_t)wpb*K*sizeof(int);
          k_fitness_warp_sht<<<nBw,BLOCK,shm>>>(d_x,d_fit,d_cf,d_vf,d_cgT,d_vgT,d_Pj); }
        checkCUDAError("k_fitness_warp_sht");
#elif defined(FIT_WARP_SH)
        { int wpb=BLOCK/WARP; int nBw=(N+wpb-1)/wpb; size_t shm=(size_t)wpb*K*sizeof(int);
          k_fitness_warp_sh<<<nBw,BLOCK,shm>>>(d_x,d_fit,d_cf,d_vf,d_cg,d_vg,d_Pj); }
        checkCUDAError("k_fitness_warp_sh");
#elif defined(FIT_WARP)
        { int nthr=N*WARP; int nBw=(nthr+BLOCK-1)/BLOCK;
          k_fitness_warp<<<nBw,BLOCK>>>(d_x,d_fit,d_cf,d_vf,d_cg,d_vg,d_Pj); }
        checkCUDAError("k_fitness_warp");
#else
        k_fitness<<<GRID(nB),BLOCK>>>(d_x,d_fit,d_cf,d_vf,d_cg,d_vg,d_Pj); checkCUDAError("k_fitness");
#endif
#ifdef PROFILE
        cudaEventRecord(evB); cudaEventSynchronize(evB);
        cudaEventElapsedTime(&ms_tmp,evA,evB); t_fit+=ms_tmp;
#endif
        /* le fitness tornano sull'host (cudaMemcpy e' sincrona: attende la fine del kernel) */
        HANDLE_ERROR(cudaMemcpy(h_fit,d_fit,szN,cudaMemcpyDeviceToHost));

        /* (2) best/worst su HOST (costo O(N), trascurabile) */
        double best=-INFINITY,worst=INFINITY; int ibest=0;
        for(int i=0;i<N;++i){ if(h_fit[i]>best){best=h_fit[i];ibest=i;} if(h_fit[i]<worst)worst=h_fit[i]; }
        if(best>gbest_fit){ gbest_fit=best;              /* salva la miglior soluzione: copia solo la sua riga */
            HANDLE_ERROR(cudaMemcpy(gbest_x,d_x+(size_t)ibest*K,K*sizeof(double),cudaMemcpyDeviceToHost)); }

        double G=G0*exp(-ALPHA*(double)t/(double)MAX_ITER);   /* costante gravitazionale decrescente */
        /* (3) masse normalizzate su host */
        double sum_m=0; for(int i=0;i<N;++i){h_M[i]=(h_fit[i]-worst)/(best-worst+EPS);sum_m+=h_M[i];}
        for(int i=0;i<N;++i) h_M[i]/=(sum_m+EPS);

        /* (4) Kbest su host (ordinamento) + numeri casuali su host (no cuRAND, stessa sequenza del seriale) */
        int kbest=(int)round(N-(N-1)*((double)t/(double)(MAX_ITER-1))); if(kbest<1)kbest=1;
        for(int i=0;i<N;++i) idx[i]=i; g_mass_ptr=h_M; qsort(idx,N,sizeof(int),cmp_mass_desc);
        for(int s=0;s<kbest;++s) h_src[s]=idx[s];
        for(int s=0;s<N;++s) h_randF[s]=urand();
        for(int s=0;s<N*K;++s) h_randV[s]=urand();

        /* copia su device dei dati che servono ai kernel di questa iterazione */
        HANDLE_ERROR(cudaMemcpy(d_M,h_M,szN,cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_src,h_src,(size_t)kbest*sizeof(int),cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_randF,h_randF,szN,cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_randV,h_randV,szXK,cudaMemcpyHostToDevice));

        /* (5) forze sul device: la variante e' scelta a compile time.
         *     Le varianti "un warp per agente" lanciano N warp: N*32 thread, wpb agenti per blocco. */
#ifdef PROFILE
        cudaEventRecord(evA);
#endif
#if defined(FORCE_WARP_DIM)
        { int wpb=BLOCK/WARP; int nBw=(N+wpb-1)/wpb;
          k_forces_warp_dim<<<GRID(nBw),BLOCK>>>(d_x,d_M,d_src,kbest,d_randF,G,d_acc); }
        checkCUDAError("k_forces_warp_dim");
#elif defined(FORCE_WARP_TILED)
        { int wpb=BLOCK/WARP; int nBw=(N+wpb-1)/wpb;
          k_forces_warp_tiled<<<nBw,BLOCK>>>(d_x,d_M,d_src,kbest,d_randF,G,d_acc); }
        checkCUDAError("k_forces_warp_tiled");
#elif defined(FORCE_WARP)
        { int wpb=BLOCK/WARP; int nBw=(N+wpb-1)/wpb;
          k_forces_warp<<<nBw,BLOCK>>>(d_x,d_M,d_src,kbest,d_randF,G,d_acc); }
        checkCUDAError("k_forces_warp");
#elif defined(FORCE_TD)
        { int nthr=N*K; int nBtd=(nthr+BLOCK-1)/BLOCK;
          k_forces_td<<<nBtd,BLOCK>>>(d_x,d_M,d_src,kbest,d_randF,G,d_acc); }
        checkCUDAError("k_forces_td");
#elif defined(TILED)
        k_forces_tiled<<<nB,BLOCK>>>(d_x,d_M,d_src,kbest,d_randF,G,d_acc); checkCUDAError("k_forces_tiled");
#else
        k_forces<<<nB,BLOCK>>>(d_x,d_M,d_src,kbest,d_randF,G,d_acc); checkCUDAError("k_forces");
#endif
#ifdef PROFILE
        cudaEventRecord(evC); cudaEventSynchronize(evC);
        cudaEventElapsedTime(&ms_tmp,evA,evC); t_for+=ms_tmp;
#endif
        /* (6) update + clamp sul device (stesso stream: parte dopo la fine del kernel forze) */
        k_update<<<GRID(nB),BLOCK>>>(d_x,d_v,d_acc,d_randV,d_LB,d_UB); checkCUDAError("k_update");

        if(t%100==0) printf("iter %5d  best-so-far fitness = %.0f\n",t,gbest_fit);
    }
    HANDLE_ERROR(cudaDeviceSynchronize());                 /* attende la fine dei kernel prima di misurare */
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double secs=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;

    /* report finale (rivaluta la migliore soluzione su host, stessa aritmetica intera) */
    int xi[K]; for(int d=0;d<K;++d) xi[d]=to_int(gbest_x[d]);
    long long f_b=poly_eval(xi,h_cf,h_vf,NT), pen_b=0; int viol_b=0;
    for(int j=0;j<NC;++j){ long long g=poly_eval(xi,h_cg+j*NT,h_vg+(size_t)j*NT*DEG,NT); if(!(g>0)){pen_b+=h_Pj[j];viol_b++;} }
    printf("\n=== Risultato (CUDA, interi) ===\n");
    printf("N=%d K=%d NC=%d DEG=%d NT=%d\n",N,K,NC,DEG,NT);
    printf("Miglior fitness = %lld  (f=%lld - penalita'=%lld)\n",f_b-pen_b,f_b,pen_b);
    printf("Vincoli violati = %d / %d\n",viol_b,NC);
    printf("x* (interi)     ="); for(int d=0;d<K&&d<8;++d) printf(" %d",xi[d]);
    printf(" %s\n",K>8?"...":"");
    printf("Tempo di calcolo= %.3f ms\n",secs*1000.0);
#ifdef PROFILE
    printf("\n--- Breakdown per-kernel (somma su %d iterazioni) ---\n",MAX_ITER);
    printf("k_fitness totale = %.3f ms  (%.3f ms/iter)\n", t_fit, t_fit/MAX_ITER);
    printf("k_forces  totale = %.3f ms  (%.3f ms/iter)\n", t_for, t_for/MAX_ITER);
    printf("NB: nel build PROFILE il tempo di calcolo totale e' gonfiato dalle sync per-iterazione;\n");
    printf("    leggi il breakdown, non il totale.\n");
    cudaEventDestroy(evA); cudaEventDestroy(evB); cudaEventDestroy(evC);
#endif

    /* rilascio della memoria device e host */
    cudaFree(d_x);cudaFree(d_v);cudaFree(d_acc);cudaFree(d_fit);cudaFree(d_M);
    cudaFree(d_randF);cudaFree(d_randV);cudaFree(d_LB);cudaFree(d_UB);cudaFree(d_src);
    cudaFree(d_cf);cudaFree(d_vf);cudaFree(d_cg);cudaFree(d_vg);cudaFree(d_Pj);
#ifdef FIT_WARP_SHT
    cudaFree(d_cgT);cudaFree(d_vgT); free(h_cgT);free(h_vgT);
#endif
    free(h_x);free(h_fit);free(h_M);free(h_randF);free(h_randV);free(idx);free(h_src);
    return 0;
}
