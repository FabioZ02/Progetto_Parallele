/* ======================================================================
 * gsa_serial.c  —  Gravitational Search Algorithm, versione SERIALE (C)
 * ----------------------------------------------------------------------
 * PROBLEMA:  massimizzare  f(x_1..x_K)
 *   - x_i INTERE nel dominio D_i = [LO,HI]   -> HARD constraint (mai violato)
 *   - NC "weak constraint" g_j(x) > 0        -> se violato costa una penalita' P_j
 *   Si massimizza quindi   fitness = f(x) - somma(P_j dei vincoli violati).
 *
 * f e i g_j sono POLINOMI A COEFFICIENTI INTERI; le variabili sono INTERE,
 * percio' la valutazione e' tutta in aritmetica intera. Questa e' la versione
 * di riferimento: gsa_cuda.cu esegue lo stesso identico algoritmo, con gli
 * stessi semi, le stesse tabelle e lo stesso ordine dei numeri casuali.
 *
 * COMPILAZIONE (di solito tramite il Makefile: `make`, `make demo`):
 *   gcc -O2 -DN=1600 -DK=64 -DNC=512 -DDEG=3 -DMAX_ITER=100 \
 *       -o gsa_serial gsa_serial.c -lm
 * Tutti i parametri dell'istanza (N, K, NC, DEG, NT, MAX_ITER) si possono
 * cambiare con -D senza modificare il file.
 *
 * OUTPUT: andamento della fitness ogni 100 iterazioni, soluzione migliore e
 * la riga "Tempo di calcolo= ... ms" (solo il ciclo GSA, senza la generazione
 * delle funzioni), letta dagli script di benchmark.
 * ====================================================================== */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* ---- taglia dell'istanza: si cambia da riga di comando con -D... ---- */
#ifndef N
#define N        50       /* numero di agenti GSA */
#endif
#ifndef K
#define K        30       /* numero di variabili (dimensioni) */
#endif
#ifndef MAX_ITER
#define MAX_ITER 1000     /* iterazioni del ciclo GSA */
#endif
/* ---- definizione del problema (polinomi a coefficienti interi) ---- */
#ifndef DEG
#define DEG 2             /* grado dei monomi */
#endif
#ifndef NT
#define NT  K             /* numero di monomi per polinomio */
#endif
#ifndef NC
#define NC  100           /* numero di weak constraint g_j */
#endif
#define LO (-5)           /* estremi del dominio intero D_i = [LO,HI] */
#define HI ( 5)
#define CSEED 12345u      /* seme delle FUNZIONI (fisso: f e g_j non cambiano con N) */
#define G0    100.0       /* costante gravitazionale iniziale */
#define ALPHA 20.0        /* velocita' di decadimento di G */
#define EPS   1e-9        /* evita divisioni per zero */
#define SEED  42u         /* seme della dinamica GSA */

/* Tabelle che DEFINISCONO le funzioni: coefficienti e indici dei monomi.
 * Il monomio t di f vale  cf[t] * x[vf[t*DEG+0]] * ... * x[vf[t*DEG+DEG-1]].
 * Uguali identiche nella versione CUDA (stesso seme, stessa generazione). */
static int cf[NT];            /* coeff. di f              */
static int vf[NT*DEG];        /* indici variabili di f    */
static int cg[NC*NT];         /* coeff. dei g_j           */
static int vg[NC*NT*DEG];     /* indici variabili dei g_j */
static int Pj[NC];            /* penalita' P_j            */

/* Coefficiente intero casuale in [-3,3], mai nullo (un monomio nullo sarebbe inutile). */
static int randc(void){ int c; do { c = rand()%7 - 3; } while (c==0); return c; }

/* Genera una volta sola f e gli NC vincoli, con coefficienti/indici casuali interi.
 * Usa il seme CSEED, separato da quello della dinamica: cambiando N il problema resta lo stesso. */
static void gen_functions(void){
    srand(CSEED);
    for (int t=0;t<NT;++t){ cf[t]=randc(); for(int d=0;d<DEG;++d) vf[t*DEG+d]=rand()%K; }
    for (int j=0;j<NC;++j){
        for (int t=0;t<NT;++t){ cg[j*NT+t]=randc(); for(int d=0;d<DEG;++d) vg[(j*NT+t)*DEG+d]=rand()%K; }
        Pj[j]=1+rand()%10;                                   /* penalita' in [1,10] */
    }
}

/* Variabili INTERE: arrotonda la posizione continua e la riporta nel dominio (hard constraint). */
static int to_int(double v){
    int r = (int)(v>=0 ? v+0.5 : v-0.5);                     /* arrotondamento all'intero piu' vicino */
    if (r<LO) r=LO; if (r>HI) r=HI; return r;               /* clamp in [LO,HI] */
}

/* Valuta un polinomio: somma di NT monomi, ogni monomio = coeff * prodotto di DEG variabili.
 * Tutto in long long: risultato esatto, identico su CPU e GPU. */
static long long poly_eval(const int *xi,const int *coeff,const int *vars,int nt){
    long long s=0;
    for(int t=0;t<nt;++t){
        long long term=coeff[t];
        for(int d=0;d<DEG;++d) term *= (long long) xi[ vars[t*DEG+d] ];
        s+=term;
    }
    return s;
}

/* FITNESS = obiettivo meno le penalita' dei vincoli violati.
 * E' la funzione che il GSA massimizza (da qui derivano le masse).
 * I puntatori of/op/ov (opzionali) restituiscono f, penalita' e numero di vincoli violati. */
static double fitness(const double *x,long long *of,long long *op,int *ov){
    int xi[K]; for(int d=0;d<K;++d) xi[d]=to_int(x[d]);      /* posizione intera */
    long long f = poly_eval(xi, cf, vf, NT);                /* obiettivo f(x) */
    long long pen=0; int viol=0;
    for(int j=0;j<NC;++j){                                   /* scorre i weak constraint */
        long long g = poly_eval(xi, cg+j*NT, vg+(size_t)j*NT*DEG, NT);
        if(!(g>0)){ pen += Pj[j]; viol++; }                 /* g_j <= 0 -> violato -> +P_j */
    }
    if(of)*of=f; if(op)*op=pen; if(ov)*ov=viol;
    return (double)(f - pen);
}

/* Numero casuale uniforme in [0,1). */
static double urand(void){ return (double)rand()/((double)RAND_MAX+1.0); }

/* Confronto per qsort: ordina gli indici degli agenti per massa DECRESCENTE (servono i Kbest). */
static const double *g_mass_ptr;
static int cmp_mass_desc(const void*a,const void*b){
    double ma=g_mass_ptr[*(const int*)a],mb=g_mass_ptr[*(const int*)b];
    return (ma<mb)-(ma>mb);
}

int main(void){
    gen_functions();                 /* prima le funzioni (seme CSEED) */
    srand(SEED);                     /* poi la dinamica GSA (seme SEED) */

    double LB[K],UB[K];              /* limiti del dominio per ogni dimensione */
    for(int d=0;d<K;++d){LB[d]=LO;UB[d]=HI;}

    /* Dati in array PIATTI N*K (row-major): la coord. d dell'agente i sta in i*K+d.
     * Stesso layout della versione CUDA, per un confronto equo. */
    double *x=malloc((size_t)N*K*sizeof(double)),*v=malloc((size_t)N*K*sizeof(double));     /* posizioni, velocita' */
    double *acc=malloc((size_t)N*K*sizeof(double)),*fit=malloc((size_t)N*sizeof(double));   /* accelerazioni, fitness */
    double *M=malloc((size_t)N*sizeof(double)); int *idx=malloc((size_t)N*sizeof(int));     /* masse, indici ordinati */
    double *randF=malloc((size_t)N*sizeof(double)),*randV=malloc((size_t)N*K*sizeof(double)); /* casuali forze/update */
    double gbest_x[K];               /* miglior soluzione trovata finora */

    for(int i=0;i<N;++i) for(int d=0;d<K;++d) x[i*K+d]=LB[d]+urand()*(UB[d]-LB[d]);  /* posizioni iniziali casuali */
    for(int q=0;q<N*K;++q) v[q]=0.0;                                                 /* velocita' iniziali nulle */
    double gbest_fit=-INFINITY;

    struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);   /* inizio misura: solo il ciclo GSA */

    /* ===================== CICLO PRINCIPALE GSA ===================== */
    for(int t=0;t<MAX_ITER;++t){

        /* (1) fitness di ogni agente: costo O(N * NC * NT * DEG) */
        for(int i=0;i<N;++i) fit[i]=fitness(&x[i*K],NULL,NULL,NULL);

        /* migliore e peggiore della popolazione (massimizzazione) */
        double best=-INFINITY,worst=INFINITY; int ibest=0;
        for(int i=0;i<N;++i){ if(fit[i]>best){best=fit[i];ibest=i;} if(fit[i]<worst)worst=fit[i]; }
        if(best>gbest_fit){ gbest_fit=best; for(int d=0;d<K;++d) gbest_x[d]=x[ibest*K+d]; }  /* salva la migliore soluzione */

        /* (2) costante gravitazionale, decrescente nel tempo: G(t) = G0 * exp(-ALPHA*t/T) */
        double G=G0*exp(-ALPHA*(double)t/(double)MAX_ITER);

        /* (3) masse normalizzate dalla fitness (peggiore -> 0, migliore -> massima), somma 1 */
        double sum_m=0; for(int i=0;i<N;++i){M[i]=(fit[i]-worst)/(best-worst+EPS);sum_m+=M[i];}
        for(int i=0;i<N;++i) M[i]/=(sum_m+EPS);

        /* (4) Kbest: solo i piu' pesanti esercitano forza; il loro numero cala linearmente da N a 1 */
        int kbest=(int)round(N-(N-1)*((double)t/(double)(MAX_ITER-1))); if(kbest<1)kbest=1;
        for(int i=0;i<N;++i) idx[i]=i; g_mass_ptr=M; qsort(idx,N,sizeof(int),cmp_mass_desc);
        for(int s=0;s<N;++s) randF[s]=urand();       /* casuali per forze e update (stesso ordine della CUDA) */
        for(int s=0;s<N*K;++s) randV[s]=urand();

        /* (5) FORZE e accelerazioni: doppio ciclo O(N^2 * K) -> e' il COLLO DI BOTTIGLIA,
         *     ed e' la parte che le versioni CUDA F1..F5 parallelizzano in modi diversi */
        for(int i=0;i<N;++i){                                         /* agente che subisce la forza */
            double acc_l[K]; for(int d=0;d<K;++d) acc_l[d]=0;
            for(int s=0;s<kbest;++s){ int j=idx[s]; if(j==i)continue; /* sorgente j tra i Kbest */
                double R=0; for(int d=0;d<K;++d){double df=x[j*K+d]-x[i*K+d];R+=df*df;} R=sqrt(R);  /* distanza euclidea */
                double factor=G*M[j]/(R+EPS),r=randF[s];
                for(int d=0;d<K;++d) acc_l[d]+=r*factor*(x[j*K+d]-x[i*K+d]);   /* attrazione verso j */
            }
            for(int d=0;d<K;++d) acc[i*K+d]=acc_l[d];
        }
        /* (6) UPDATE: nuova velocita' e posizione, poi CLAMP nel dominio (hard constraint) */
        for(int i=0;i<N;++i) for(int d=0;d<K;++d){ int id=i*K+d;
            v[id]=randV[id]*v[id]+acc[id]; double xv=x[id]+v[id];
            if(xv<LB[d])xv=LB[d]; if(xv>UB[d])xv=UB[d]; x[id]=xv; }

        if(t%100==0) printf("iter %5d  best-so-far fitness = %.0f\n",t,gbest_fit);
    }
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double secs=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;

    /* report finale: rivaluta la soluzione migliore per stampare f, penalita' e vincoli violati */
    long long f_b,pen_b; int viol_b; double check=fitness(gbest_x,&f_b,&pen_b,&viol_b);
    printf("\n=== Risultato (seriale, interi) ===\n");
    printf("N=%d K=%d NC=%d DEG=%d NT=%d\n",N,K,NC,DEG,NT);
    printf("Miglior fitness = %.0f  (f=%lld - penalita'=%lld)\n",check,f_b,pen_b);
    printf("Vincoli violati = %d / %d\n",viol_b,NC);
    printf("x* (interi)     ="); for(int d=0;d<K&&d<8;++d) printf(" %d",to_int(gbest_x[d]));
    printf(" %s\n",K>8?"...":"");
    printf("Tempo di calcolo= %.3f ms\n",secs*1000.0);
    free(x);free(v);free(acc);free(fit);free(M);free(idx);free(randF);free(randV);
    return 0;
}
