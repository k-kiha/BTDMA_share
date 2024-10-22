#include<mpi.h> 

struct tdma_plan{
    int test;
    double *rd_a, *tr_a;
    double *rd_b, *tr_b;
    double *rd_c, *tr_c;
    double *rd_d, *tr_d;

    double *send;
    double *recv;
    
    int (*countsA)[3],(*distA)[3];
    int (*countsB)[3],(*distB)[3];
    int sizeA[3],sizeB[3];

    int *a2a_count_s,*a2a_dist_s;
    int *a2a_count_r,*a2a_dist_r;

    MPI_Comm comm;
    int myrank,nprocs;

};

typedef struct tdma_plan TDMA_PLAN;

struct btdma_plan{
    int test;
    int nsys_sub;
    double *rd_a, *tr_a;
    double *rd_b, *tr_b;
    double *rd_c, *tr_c;
    double *rd_d, *tr_d;

    double *sendM,*sendV;
    double *recvM,*recvV;
    
    int (*countsAM)[3],(*distAM)[3];
    int (*countsBM)[3],(*distBM)[3];
    int sizeAM[3],sizeBM[3];

    int (*countsAV)[3],(*distAV)[3];
    int (*countsBV)[3],(*distBV)[3];
    int sizeAV[3],sizeBV[3];

    int *a2a_count_sM,*a2a_dist_sM;
    int *a2a_count_rM,*a2a_dist_rM;

    int *a2a_count_sV,*a2a_dist_sV;
    int *a2a_count_rV,*a2a_dist_rV;

    MPI_Comm comm;
    int myrank,nprocs;

};
typedef struct btdma_plan BTDMA_PLAN;

int tdma(int, double*, double*, double*, double*);
int tdma_many(int, int, double*, double*, double*, double*);
void btdma(int, int, double *, double *, double *, double *);
void btdma_many(int, int, int, double *, double *, double *, double *);

void tdma_makeplan(int,int, TDMA_PLAN*, MPI_Comm);
void tdma_cleanplan(TDMA_PLAN*);
void tdma_many_mpi(int, int, double*, double*, double*, double*, TDMA_PLAN*);

void btdma_makeplan(int,int,int, BTDMA_PLAN*, MPI_Comm);
void btdma_cleanplan(BTDMA_PLAN*);
void btdma_many_mpi(int, int, int, double *, double *, double *, double *, BTDMA_PLAN*);
