#include<stdio.h>
#include<stdlib.h>

#include<mpi.h> 
#include "mpiutil.h"

#include"tdma.h"

#include <lapacke.h>
#include <cblas.h>

int arraysum1d(int a[],int n) {
    int sum=0;
    for (int i = 0; i <= n; i++) sum=sum+a[i];
    return sum;
}

void tdma_many_modi(int,int,double*,double*,double*,double*,double*,double*,double*,double*);
void tdma_many_a2av_forward(TDMA_PLAN*);
void tdma_many_a2av_backward(TDMA_PLAN*);
void tdma_many_update(int,int,double*,double*,double*,double*,double*);

void btdma_many_modi(int,int,int,double*,double*,double*,double*,double*,double*,double*,double*,int);
void btdma_many_a2av_forward(BTDMA_PLAN*);
void btdma_many_a2av_backward(BTDMA_PLAN*);
void btdma_many_update(int,int,int,double*,double*,double*,double*,double*);

int tdma(int n, double *a, double *b, double *c, double *d){

    double tmp;

    d[0]=d[0]/b[0];
    c[0]=c[0]/b[0];

    for (int i = 1; i < n; i++)
    {
        tmp  = 1. /(b[i]-a[i]*c[i-1]);
        d[i] = tmp*(d[i]-a[i]*d[i-1]);
        c[i] = tmp*c[i];
    }

    for (int i = n-2; i >= 0; i--)
    {
        d[i] = d[i]-c[i]*d[i+1];
    }

    return 0;
}
int tdma_many(int n, int nsys, double *a, double *b, double *c, double *d){
    int ij=0,ijp=0,ijm=0;
    double tmp;

    for (int j = 0; j < nsys; j++)
    {
        d[j + 0*nsys]=d[j + 0*nsys]/b[j + 0*nsys];
        c[j + 0*nsys]=c[j + 0*nsys]/b[j + 0*nsys];
    }
    

    for (int i = 1; i < n; i++)
    {
        for (int j = 0; j < nsys; j++)
        {
            ij = j + (i  )*nsys;
            ijm= j + (i-1)*nsys;
            tmp = 1. /(b[ij]-a[ij]*c[ijm]);
            d[ij] = tmp*(d[ij]-a[ij]*d[ijm]);
            c[ij] = tmp*c[ij];
        }
    }

    for (int i = n-2; i >= 0; i--)
    {
        for (int j = 0; j < nsys; j++)
        {
            ij = j + (i  )*nsys;
            ijp= j + (i+1)*nsys;
            d[ij] = d[ij]-c[ij]*d[ijp];
        }
        
    }

    return 0;
}

void btdma(int n, int m, double *a, double *b, double *c, double *d){
    // LAPACK_COL_MAJOR
    double Sol[m*(m+1)];             //todo: 나중에 동적할당으로 수정 필요
    double RR[m*m],AkCkm[m*m];    //todo: 나중에 동적할당으로 수정 필요
    double AkDkm[m],CkDkp[m];       //todo: 나중에 동적할당으로 수정 필요
    int ij,ijp;
    double alpha = 1.0, beta = 0.0;

    int info,ipiv[m];

    for(int i = 0; i < m; i++){
        Sol[i] = d[i+0*(m)];
    }

    for(int j = 0; j < m; j++){
        for(int i = 0; i < m; i++){
            ij = i + m*j;
            ijp= i + m*(j+1);
            Sol[ijp] = c[ij + 0*(m*m)];
            RR[ij]   = b[ij + 0*(m*m)];
        }
    }

    info=LAPACKE_dgesv(LAPACK_COL_MAJOR, m, m+1, RR, m, ipiv, Sol, m);
   
    for(int i = 0; i < m; i++){
        d[i+0*(m)] = Sol[i];
    }

    for(int j = 0; j < m; j++){
        for(int i = 0; i < m; i++){
            ij = i + m*j;
            ijp= i + m*(j+1);
            c[ij + 0*(m*m)] = Sol[ijp] ;
        }
    }

    for (int q = 1; q < n; q++){
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, m, m, 1., &a[0 + q*(m*m)], m, &c[0 + (q-1)*(m*m)], m, 0., AkCkm, m);
        cblas_dgemv(CblasColMajor, CblasNoTrans              , m, m   , 1., &a[0 + q*(m*m)], m, &d[0 + (q-1)*(m*m)], 1, 0., AkDkm, 1);

        for(int i = 0; i < m; i++){
            Sol[i] = d[i+q*(m)]-AkDkm[i];
        }

        for(int j = 0; j < m; j++){
            for(int i = 0; i < m; i++){
                ij = i + m*j;
                ijp= i + m*(j+1);
                Sol[ijp] = c[ij + q*(m*m)];
                RR[ij]   = b[ij + q*(m*m)] - AkCkm[ij];
            }
        }

        info=LAPACKE_dgesv(LAPACK_COL_MAJOR, m, m+1, RR, m, ipiv, Sol, m);

        for(int i = 0; i < m; i++){
            d[i+q*(m)] = Sol[i];
        }

        for(int j = 0; j < m; j++){
            for(int i = 0; i < m; i++){
                ij = i + m*j;
                ijp= i + m*(j+1);
                c[ij + q*(m*m)] = Sol[ijp] ;
            }
        }        
    }


    for (int q = n-2; q > -1; q--){
        cblas_dgemv(CblasColMajor, CblasNoTrans, m, m, 1., &c[0 + q*(m*m)], m, &d[0 + (q+1)*(m  )], 1, 0., CkDkp, 1);

        for(int i = 0; i < m; i++){
            d[i+q*(m)] = d[i+q*(m)] - CkDkp[i];
        }
    }    
}
void btdma_many(int n, int nsys, int m, double *a, double *b, double *c, double *d){

    // LAPACK_COL_MAJOR
    double Sol[m*(m+1)];             //todo: 나중에 동적할당으로 수정 필요
    double RR[m*m],AkCkm[m*m];    //todo: 나중에 동적할당으로 수정 필요
    double AkDkm[m],CkDkp[m];       //todo: 나중에 동적할당으로 수정 필요
    int ij,ijp;
    double alpha = 1.0, beta = 0.0;

    int info,ipiv[m];
    for(int sys = 0; sys < nsys; sys++){
        for(int i = 0; i < m; i++){
            Sol[i] = d[i + sys*(m)+ 0*(m*nsys)];
        }

        for(int j = 0; j < m; j++){
            for(int i = 0; i < m; i++){
                ij = i + m*j;
                ijp= i + m*(j+1);
                Sol[ijp] = c[ij + sys*(m*m)+ 0*(m*m*nsys)];
                RR[ij]   = b[ij + sys*(m*m)+ 0*(m*m*nsys)];
            }
        }

        info=LAPACKE_dgesv(LAPACK_COL_MAJOR, m, m+1, RR, m, ipiv, Sol, m);
    
        for(int i = 0; i < m; i++){
            d[i + sys*(m)+ 0*(m*nsys)] = Sol[i];
        }

        for(int j = 0; j < m; j++){
            for(int i = 0; i < m; i++){
                ij = i + m*j;
                ijp= i + m*(j+1);
                c[ij + sys*(m*m)+ 0*(m*m*nsys)] = Sol[ijp] ;
            }
        }
    }

    for (int q = 1; q < n; q++){
        for(int sys = 0; sys < nsys; sys++){
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, m, m, 1., &a[0 + sys*(m*m)+ q*(m*m*nsys)], m, &c[0 + sys*(m*m)+ (q-1)*(m*m*nsys)], m, 0., AkCkm, m);
            cblas_dgemv(CblasColMajor, CblasNoTrans              , m, m   , 1., &a[0 + sys*(m*m)+ q*(m*m*nsys)], m, &d[0 + sys*(m)+ (q-1)*(m*nsys)], 1, 0., AkDkm, 1);

            for(int i = 0; i < m; i++){
                Sol[i] = d[i + sys*(m)+ q*(m*nsys)]-AkDkm[i];
            }

            for(int j = 0; j < m; j++){
                for(int i = 0; i < m; i++){
                    ij = i + m*j;
                    ijp= i + m*(j+1);
                    Sol[ijp] = c[ij + sys*(m*m)+ q*(m*m*nsys)];
                    RR[ij]   = b[ij + sys*(m*m)+ q*(m*m*nsys)] - AkCkm[ij];
                }
            }

            info=LAPACKE_dgesv(LAPACK_COL_MAJOR, m, m+1, RR, m, ipiv, Sol, m);

            for(int i = 0; i < m; i++){
                d[i + sys*(m)+ q*(m*nsys)] = Sol[i];
            }

            for(int j = 0; j < m; j++){
                for(int i = 0; i < m; i++){
                    ij = i + m*j;
                    ijp= i + m*(j+1);
                    c[ij + sys*(m*m)+ q*(m*m*nsys)] = Sol[ijp] ;
                }
            }        
        }
    }

    for (int q = n-2; q > -1; q--){
        for(int sys = 0; sys < nsys; sys++){
            cblas_dgemv(CblasColMajor, CblasNoTrans, m, m, 1., &c[0 + sys*(m*m)+ q*(m*m*nsys)], m, &d[0 + sys*(m)+ (q+1)*(m*nsys)], 1, 0., CkDkp, 1);

            for(int i = 0; i < m; i++){
                d[i + sys*(m)+ q*(m*nsys)] = d[i + sys*(m)+ q*(m*nsys)] - CkDkp[i];
            }
        }
    }    
}

void tdma_makeplan(int nrow_sub,int nsys, TDMA_PLAN *plan, MPI_Comm comm){
    int nprocs,myrank;
    int nsys_sub;
    int indx_tmpa,indx_tmpb;
    MPI_Comm_size(comm, &nprocs); 
    MPI_Comm_rank(comm, &myrank);

    plan->myrank = myrank;
    plan->nprocs = nprocs;
    plan->comm = comm;

    nsys_sub = mpiutil_para(0,nsys-1,myrank,nprocs, &indx_tmpa, &indx_tmpb);

    plan->sizeA[0]=2         ;plan->sizeA[1]=nsys    ;plan->sizeA[2]=1;
    plan->sizeB[0]=(2*nprocs);plan->sizeB[1]=nsys_sub;plan->sizeB[2]=1;

    plan->send = (double *)malloc(2*nsys*sizeof(double));
    plan->recv = (double *)malloc((2*nprocs)*nsys_sub*sizeof(double));

    plan->rd_a = (double *)malloc(2*nsys*sizeof(double));
    plan->rd_b = (double *)malloc(2*nsys*sizeof(double));
    plan->rd_c = (double *)malloc(2*nsys*sizeof(double));
    plan->rd_d = (double *)malloc(2*nsys*sizeof(double));

    plan->tr_a = (double *)malloc((2*nprocs)*nsys_sub*sizeof(double));
    plan->tr_b = (double *)malloc((2*nprocs)*nsys_sub*sizeof(double));
    plan->tr_c = (double *)malloc((2*nprocs)*nsys_sub*sizeof(double));
    plan->tr_d = (double *)malloc((2*nprocs)*nsys_sub*sizeof(double));

    plan->countsA = (int (*)[3])malloc(nprocs * sizeof(int[3]));
    plan->countsB = (int (*)[3])malloc(nprocs * sizeof(int[3]));
    plan->distA   = (int (*)[3])malloc(nprocs * sizeof(int[3]));
    plan->distB   = (int (*)[3])malloc(nprocs * sizeof(int[3]));

    plan->a2a_count_s = (int *)malloc(nprocs * sizeof(int));
    plan->a2a_count_r = (int *)malloc(nprocs * sizeof(int));
    plan->a2a_dist_s = (int *)malloc(nprocs * sizeof(int));
    plan->a2a_dist_r = (int *)malloc(nprocs * sizeof(int));

    mpiutil_pack_info(2,nsys,1,plan->countsA,plan->distA,comm);
    mpiutil_unpack_info((2*nprocs),nsys,1,plan->countsB,plan->distB,comm);


    for (int i = 0; i < nprocs; i++){
        plan->a2a_count_s[i]=plan->countsA[i][0]*plan->countsA[i][1]*plan->countsA[i][2];
        plan->a2a_count_r[i]=plan->countsB[i][0]*plan->countsB[i][1]*plan->countsB[i][2];

        plan->a2a_dist_s[i]= arraysum1d(plan->a2a_count_s,i) - plan->a2a_count_s[i];
        plan->a2a_dist_r[i]= arraysum1d(plan->a2a_count_r,i) - plan->a2a_count_r[i];
    }
}
void tdma_cleanplan(TDMA_PLAN *plan){

    free(plan->send);
    free(plan->recv);
    free(plan->rd_a);
    free(plan->rd_b);
    free(plan->rd_c);
    free(plan->rd_d);
    free(plan->tr_a);
    free(plan->tr_b);
    free(plan->tr_c);
    free(plan->tr_d);
    free(plan->countsA);
    free(plan->countsB);
    free(plan->distA  );
    free(plan->distB  );
    free(plan->a2a_count_s);
    free(plan->a2a_count_r);
    free(plan->a2a_dist_s );
    free(plan->a2a_dist_r );
}
void tdma_many_mpi(int nrow_sub, int nsys, double* a, double* b, double* c, double* d, TDMA_PLAN* plan){
    
        tdma_many_modi(nrow_sub,nsys,a,b,c,d,plan->rd_a,plan->rd_b,plan->rd_c,plan->rd_d);

        tdma_many_a2av_forward(plan);

        tdma_many(plan->sizeB[0], plan->sizeB[1], plan->tr_a,plan->tr_b,plan->tr_c,plan->tr_d);

        tdma_many_a2av_backward(plan);

        tdma_many_update(nrow_sub,nsys,a,b,c,d,plan->rd_d);

}
void tdma_many_update(int nrow_sub,int nsys,double *a,double *b,double *c,double *d,double *rd_d){
    int i0j,ipj,imj,iej,isj;
    double r;

    for (int j = 0; j < nsys; j++){
        i0j=j+(0  )*nsys;
        ipj=j+(0+1)*nsys;
        iej=j+(nrow_sub-1)*nsys;

        d[i0j] = rd_d[i0j];
        d[iej] = rd_d[ipj];
    }
    
    for (int i = 1; i < nrow_sub-1; i++){
        for (int j = 0; j < nsys; j++){
            i0j=j+(i)*nsys;
            isj=j+(0)*nsys;
            iej=j+(nrow_sub-1)*nsys;
            
            d[i0j] =  d[i0j]-a[i0j]*d[isj]-c[i0j]*d[iej];
        }
    }


}
void tdma_many_a2av_backward(TDMA_PLAN* plan){
    
    mpiutil_pack_d(plan->tr_d,plan->recv, plan->countsB,plan->distB,plan->sizeB, plan->comm);
    MPI_Alltoallv((void *)(plan->recv),plan->a2a_count_r,plan->a2a_dist_r,MPI_DOUBLE, 
                  (void *)(plan->send),plan->a2a_count_s,plan->a2a_dist_s,MPI_DOUBLE, plan->comm);
    mpiutil_unpack_d(plan->send, plan->rd_d, plan->countsA,plan->distA,plan->sizeA, plan->comm);
}
void tdma_many_a2av_forward(TDMA_PLAN* plan){
    
    mpiutil_pack_d(plan->rd_a,plan->send, plan->countsA,plan->distA,plan->sizeA, plan->comm);
    MPI_Alltoallv((void *)(plan->send),plan->a2a_count_s,plan->a2a_dist_s,MPI_DOUBLE, 
                  (void *)(plan->recv),plan->a2a_count_r,plan->a2a_dist_r,MPI_DOUBLE, plan->comm);
    mpiutil_unpack_d(plan->recv,plan->tr_a, plan->countsB,plan->distB,plan->sizeB, plan->comm);

    mpiutil_pack_d(plan->rd_b,plan->send, plan->countsA,plan->distA,plan->sizeA, plan->comm);
    MPI_Alltoallv((void *)(plan->send),plan->a2a_count_s,plan->a2a_dist_s,MPI_DOUBLE, 
                  (void *)(plan->recv),plan->a2a_count_r,plan->a2a_dist_r,MPI_DOUBLE, plan->comm);
    mpiutil_unpack_d(plan->recv,plan->tr_b, plan->countsB,plan->distB,plan->sizeB, plan->comm);

    mpiutil_pack_d(plan->rd_c,plan->send, plan->countsA,plan->distA,plan->sizeA, plan->comm);
    MPI_Alltoallv((void *)(plan->send),plan->a2a_count_s,plan->a2a_dist_s,MPI_DOUBLE, 
                  (void *)(plan->recv),plan->a2a_count_r,plan->a2a_dist_r,MPI_DOUBLE, plan->comm);
    mpiutil_unpack_d(plan->recv,plan->tr_c, plan->countsB,plan->distB,plan->sizeB, plan->comm);

    mpiutil_pack_d(plan->rd_d,plan->send, plan->countsA,plan->distA,plan->sizeA, plan->comm);
    MPI_Alltoallv((void *)(plan->send),plan->a2a_count_s,plan->a2a_dist_s,MPI_DOUBLE, 
                  (void *)(plan->recv),plan->a2a_count_r,plan->a2a_dist_r,MPI_DOUBLE, plan->comm);
    mpiutil_unpack_d(plan->recv,plan->tr_d, plan->countsB,plan->distB,plan->sizeB, plan->comm);

}
void tdma_many_modi(int nrow_sub,int nsys,double *a,double *b,double *c,double *d,double *rd_a,double *rd_b,double *rd_c,double *rd_d){
    int i0j,ipj,imj,iej;
    double r;

    for (int j = 0; j < nsys; j++){
        i0j=j+(0)*nsys;
        ipj=j+(1)*nsys;
        
        a[i0j] = a[i0j]/b[i0j];
        d[i0j] = d[i0j]/b[i0j];
        c[i0j] = c[i0j]/b[i0j];

        a[ipj] = a[ipj]/b[ipj];
        d[ipj] = d[ipj]/b[ipj];
        c[ipj] = c[ipj]/b[ipj];
    }
    
    for (int i = 2; i < nrow_sub; i++){
        for (int j = 0; j < nsys; j++){
            i0j=j+(i  )*nsys;
            ipj=j+(i+1)*nsys;
            imj=j+(i-1)*nsys;
            
            r      =  1./(b[i0j]-a[i0j]*c[imj]);
            d[i0j] =  r*(d[i0j]-a[i0j]*d[imj]);
            c[i0j] =  r*c[i0j];
            a[i0j] = -r*a[i0j]*a[imj];
        }
    }

    for (int i = nrow_sub-3; i > 0; i--){
        for (int j = 0; j < nsys; j++){
            i0j=j+(i  )*nsys;
            ipj=j+(i+1)*nsys;
            imj=j+(i-1)*nsys;
    
            d[i0j] = d[i0j]-c[i0j]*d[ipj];
            a[i0j] = a[i0j]-c[i0j]*a[ipj];
            c[i0j] =-c[i0j]*c[ipj];
        }
    }

    for (int j = 0; j < nsys; j++){
        i0j=j+(0  )*nsys;
        ipj=j+(0+1)*nsys;
        iej=j+(nrow_sub-1)*nsys;

        r = 1./(1.-a[ipj]*c[i0j]);
        d[i0j] =  r*(d[i0j]-c[i0j]*d[ipj]);
        a[i0j] =  r*a[i0j];
        c[i0j] = -r*c[i0j]*c[ipj];

        rd_a[i0j] = a[i0j]; rd_a[ipj] = a[iej];
        rd_b[i0j] = 1.  ; rd_b[ipj] = 1.;
        rd_c[i0j] = c[i0j]; rd_c[ipj] = c[iej];
        rd_d[i0j] = d[i0j]; rd_d[ipj] = d[iej];
    }
}

void btdma_makeplan(int nrow_sub,int nsys, int m, BTDMA_PLAN *plan, MPI_Comm comm){
    int nprocs,myrank;
    int nsys_sub;
    int indx_tmpa,indx_tmpb;
    MPI_Comm_size(comm, &nprocs); 
    MPI_Comm_rank(comm, &myrank);

    plan->myrank = myrank;
    plan->nprocs = nprocs;
    plan->comm = comm;

    nsys_sub = mpiutil_para(0,(nsys-1),myrank,nprocs, &indx_tmpa, &indx_tmpb);
    plan->nsys_sub = nsys_sub;
    //Matrix
    plan->sizeAM[0]=2         ;plan->sizeAM[1]=nsys    ;plan->sizeAM[2]=(m*m);
    plan->sizeBM[0]=(2*nprocs);plan->sizeBM[1]=nsys_sub;plan->sizeBM[2]=(m*m);

    plan->sendM = (double *)malloc(2*nsys*(m*m)*sizeof(double));
    plan->recvM = (double *)malloc((2*nprocs)*nsys_sub*(m*m)*sizeof(double));

    plan->rd_a = (double *)malloc(2*nsys*(m*m)*sizeof(double));
    plan->rd_b = (double *)malloc(2*nsys*(m*m)*sizeof(double));
    plan->rd_c = (double *)malloc(2*nsys*(m*m)*sizeof(double));

    plan->tr_a = (double *)malloc((2*nprocs)*nsys_sub*(m*m)*sizeof(double));
    plan->tr_b = (double *)malloc((2*nprocs)*nsys_sub*(m*m)*sizeof(double));
    plan->tr_c = (double *)malloc((2*nprocs)*nsys_sub*(m*m)*sizeof(double));

    plan->countsAM = (int (*)[3])malloc(nprocs * sizeof(int[3]));
    plan->countsBM = (int (*)[3])malloc(nprocs * sizeof(int[3]));
    plan->distAM   = (int (*)[3])malloc(nprocs * sizeof(int[3]));
    plan->distBM   = (int (*)[3])malloc(nprocs * sizeof(int[3]));

    plan->a2a_count_sM = (int *)malloc(nprocs * sizeof(int));
    plan->a2a_count_rM = (int *)malloc(nprocs * sizeof(int));
    plan->a2a_dist_sM = (int *)malloc(nprocs * sizeof(int));
    plan->a2a_dist_rM = (int *)malloc(nprocs * sizeof(int));

    mpiutil_pack_info(2,nsys,(m*m),plan->countsAM,plan->distAM,comm);
    mpiutil_unpack_info((2*nprocs),nsys,(m*m),plan->countsBM,plan->distBM,comm);


    for (int i = 0; i < nprocs; i++){
        plan->a2a_count_sM[i]=plan->countsAM[i][0]*plan->countsAM[i][1]*plan->countsAM[i][2];
        plan->a2a_count_rM[i]=plan->countsBM[i][0]*plan->countsBM[i][1]*plan->countsBM[i][2];

        plan->a2a_dist_sM[i]= arraysum1d(plan->a2a_count_sM,i) - plan->a2a_count_sM[i];
        plan->a2a_dist_rM[i]= arraysum1d(plan->a2a_count_rM,i) - plan->a2a_count_rM[i];
    }

    //Vector
    plan->sizeAV[0]=2         ;plan->sizeAV[1]=nsys    ;plan->sizeAV[2]=(m);
    plan->sizeBV[0]=(2*nprocs);plan->sizeBV[1]=nsys_sub;plan->sizeBV[2]=(m);

    plan->sendV = (double *)malloc(2*nsys*(m  )*sizeof(double));
    plan->recvV = (double *)malloc((2*nprocs)*nsys_sub*(m  )*sizeof(double));

    plan->rd_d = (double *)malloc(2*nsys*(m  )*sizeof(double));

    plan->tr_d = (double *)malloc((2*nprocs)*nsys_sub*(m  )*sizeof(double));

    plan->countsAV = (int (*)[3])malloc(nprocs * sizeof(int[3]));
    plan->countsBV = (int (*)[3])malloc(nprocs * sizeof(int[3]));
    plan->distAV   = (int (*)[3])malloc(nprocs * sizeof(int[3]));
    plan->distBV   = (int (*)[3])malloc(nprocs * sizeof(int[3]));

    plan->a2a_count_sV = (int *)malloc(nprocs * sizeof(int));
    plan->a2a_count_rV = (int *)malloc(nprocs * sizeof(int));
    plan->a2a_dist_sV = (int *)malloc(nprocs * sizeof(int));
    plan->a2a_dist_rV = (int *)malloc(nprocs * sizeof(int));

    mpiutil_pack_info(2,nsys,(m  ),plan->countsAV,plan->distAV,comm);
    mpiutil_unpack_info((2*nprocs),nsys,(m  ),plan->countsBV,plan->distBV,comm);


    for (int i = 0; i < nprocs; i++){
        plan->a2a_count_sV[i]=plan->countsAV[i][0]*plan->countsAV[i][1]*plan->countsAV[i][2];
        plan->a2a_count_rV[i]=plan->countsBV[i][0]*plan->countsBV[i][1]*plan->countsBV[i][2];

        plan->a2a_dist_sV[i]= arraysum1d(plan->a2a_count_sV,i) - plan->a2a_count_sV[i];
        plan->a2a_dist_rV[i]= arraysum1d(plan->a2a_count_rV,i) - plan->a2a_count_rV[i];
    }
    
}
void btdma_cleanplan(BTDMA_PLAN *plan){
    free(plan->sendM);
    free(plan->recvM);
    free(plan->rd_a);
    free(plan->rd_b);
    free(plan->rd_c);
    free(plan->tr_a);
    free(plan->tr_b);
    free(plan->tr_c);
    free(plan->countsAM);
    free(plan->countsBM);
    free(plan->distAM);
    free(plan->distBM);
    free(plan->a2a_count_sM);
    free(plan->a2a_count_rM);
    free(plan->a2a_dist_sM);
    free(plan->a2a_dist_rM);
    free(plan->sendV);
    free(plan->recvV);
    free(plan->rd_d);
    free(plan->tr_d);
    free(plan->countsAV);
    free(plan->countsBV);
    free(plan->distAV);
    free(plan->distBV);
    free(plan->a2a_count_sV);
    free(plan->a2a_count_rV);
    free(plan->a2a_dist_sV);
    free(plan->a2a_dist_rV);
}
void btdma_many_mpi(int nrow_sub, int nsys, int m, double *a, double *b, double *c, double *d, BTDMA_PLAN *plan){
    btdma_many_modi(nrow_sub,nsys,m,a,b,c,d,plan->rd_a,plan->rd_b,plan->rd_c,plan->rd_d,plan->myrank);

    btdma_many_a2av_forward(plan);

    btdma_many(plan->sizeBM[0], plan->sizeBM[1], m, plan->tr_a,plan->tr_b,plan->tr_c,plan->tr_d);

    btdma_many_a2av_backward(plan);

    btdma_many_update(nrow_sub,nsys,m,a,b,c,d,plan->rd_d);

}
void btdma_many_modi(int nrow_sub,int nsys,int m,double *a,double *b,double *c,double *d,double *rd_a,double *rd_b,double *rd_c,double *rd_d,int myrank){
    int ij,ijp;
    double alpha = 1.0, beta = 0.0;
    int info,ipiv[m];        //todo: m 나중에 동적할당으로 수정 필요

    // LAPACK_COL_MAJOR
    double Sol[m*(1+2*m)];          //todo: m 나중에 동적할당으로 수정 필요
    double RR[m*m];                 //todo: m 나중에 동적할당으로 수정 필요
    double AqDqm[m],CqDqp[m];       //todo: m 나중에 동적할당으로 수정 필요
    double AqAqm[m*m],AqCqm[m*m];   //todo: m 나중에 동적할당으로 수정 필요
    double CqAqp[m*m],CqCqp[m*m];   //todo: m 나중에 동적할당으로 수정 필요
    double AqpCq[m*m];
    double delta_ij;


    for (int sys = 0; sys < nsys; sys++){
        // d[sys+(0)*nsys] = d[sys+(0)*nsys]/b[sys+(0)*nsys];
        // a[sys+(0)*nsys] = a[sys+(0)*nsys]/b[sys+(0)*nsys];
        // c[sys+(0)*nsys] = c[sys+(0)*nsys]/b[sys+(0)*nsys];
        for(int i = 0; i < m; i++){
            Sol[i] = d[i + sys*(m)+ 0*(m*nsys)];
        }
        for(int j = 0; j < m; j++){
            for(int i = 0; i < m; i++){
                ij  = i + m*j;
                ijp = i + m*(j+1);

                Sol[ijp    ] = a[ij + sys*(m*m)+ 0*(m*m*nsys)];
                Sol[ijp+m*m] = c[ij + sys*(m*m)+ 0*(m*m*nsys)];
                RR [ij]      = b[ij + sys*(m*m)+ 0*(m*m*nsys)];
            }
        }
        info=LAPACKE_dgesv(LAPACK_COL_MAJOR, m, 1+2*m, RR, m, ipiv, Sol, m);
        for(int i = 0; i < m; i++){
            d[i + sys*(m)+ 0*(m*nsys)] = Sol[i];
        }
        for(int j = 0; j < m; j++){
            for(int i = 0; i < m; i++){
                ij = i + m*j;
                ijp= i + m*(j+1);

                a[ij + sys*(m*m)+ 0*(m*m*nsys)] = Sol[ijp    ];
                c[ij + sys*(m*m)+ 0*(m*m*nsys)] = Sol[ijp+m*m];
            }
        }


        // a[sys+(1)*nsys] = a[sys+(1)*nsys]/b[sys+(1)*nsys];
        // d[sys+(1)*nsys] = d[sys+(1)*nsys]/b[sys+(1)*nsys];
        // c[sys+(1)*nsys] = c[sys+(1)*nsys]/b[sys+(1)*nsys];
        for(int i = 0; i < m; i++){
            Sol[i] = d[i + sys*(m)+ 1*(m*nsys)];
        }
        for(int j = 0; j < m; j++){
            for(int i = 0; i < m; i++){
                ij  = i + m*j;
                ijp = i + m*(j+1);

                Sol[ijp    ] = a[ij + sys*(m*m)+ 1*(m*m*nsys)];
                Sol[ijp+m*m] = c[ij + sys*(m*m)+ 1*(m*m*nsys)];
                RR [ij]      = b[ij + sys*(m*m)+ 1*(m*m*nsys)];
            }
        }
        info=LAPACKE_dgesv(LAPACK_COL_MAJOR, m, 1+2*m, RR, m, ipiv, Sol, m);


        for(int i = 0; i < m; i++){
            d[i + sys*(m)+ 1*(m*nsys)] = Sol[i];
        }
        for(int j = 0; j < m; j++){
            for(int i = 0; i < m; i++){
                ij = i + m*j;
                ijp= i + m*(j+1);

                a[ij + sys*(m*m)+ 1*(m*m*nsys)] = Sol[ijp    ];
                c[ij + sys*(m*m)+ 1*(m*m*nsys)] = Sol[ijp+m*m];
            }
        }
    }
    

    for (int q = 2; q < nrow_sub; q++){
        for (int sys = 0; sys < nsys; sys++){
            // r      =  1./(b[sys+(q  )*nsys]-a[sys+(q  )*nsys]*c[sys+(q-1)*nsys]);
            // d[sys+(q  )*nsys] =  r*(d[sys+(q  )*nsys]-a[sys+(q  )*nsys]*d[sys+(q-1)*nsys]);
            // a[sys+(q  )*nsys] = -r*a[sys+(q  )*nsys]*a[sys+(q-1)*nsys];
            // c[sys+(q  )*nsys] =  r*c[sys+(q  )*nsys];

            // --- d[:] = d[:] - AqDqm[:]
            // --- a[:,:] = -AqAqm[:,:]
            // --- c[:,:] = c[:,:]
            // --- RR[:,:] = b[:,:]-AqCqm[:,:]

            cblas_dgemv(CblasColMajor, CblasNoTrans              , m, m   , 1., &a[0 + sys*(m*m)+ q*(m*m*nsys)], m, &d[0 + sys*(m  )+ (q-1)*(m*  nsys)], 1, 0., AqDqm, 1);
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, m, m, 1., &a[0 + sys*(m*m)+ q*(m*m*nsys)], m, &a[0 + sys*(m*m)+ (q-1)*(m*m*nsys)], m, 0., AqAqm, m);
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, m, m, 1., &a[0 + sys*(m*m)+ q*(m*m*nsys)], m, &c[0 + sys*(m*m)+ (q-1)*(m*m*nsys)], m, 0., AqCqm, m);
            
            for(int i = 0; i < m; i++){
                Sol[i] = d[i + sys*(m)+ q*(m*nsys)]-AqDqm[i];
            }
            for(int j = 0; j < m; j++){
                for(int i = 0; i < m; i++){
                    ij  = i + m*j;
                    ijp = i + m*(j+1);

                    Sol[ijp    ] =-AqAqm[ij];
                    Sol[ijp+m*m] = c[ij + sys*(m*m)+ q*(m*m*nsys)];
                    RR [ij]      = b[ij + sys*(m*m)+ q*(m*m*nsys)]-AqCqm[ij];
                }
            }
            
            info=LAPACKE_dgesv(LAPACK_COL_MAJOR, m, 1+2*m, RR, m, ipiv, Sol, m);
            for(int i = 0; i < m; i++){
                d[i + sys*(m)+ q*(m*nsys)] = Sol[i];
            }
            for(int j = 0; j < m; j++){
                for(int i = 0; i < m; i++){
                    ij = i + m*j;
                    ijp= i + m*(j+1);

                    a[ij + sys*(m*m)+ q*(m*m*nsys)] = Sol[ijp    ];
                    c[ij + sys*(m*m)+ q*(m*m*nsys)] = Sol[ijp+m*m];
                }
            }
        }

    }

    for (int q = nrow_sub-3; q > 0; q--){
        for (int sys = 0; sys < nsys; sys++){
            // d[sys+(q  )*nsys] = d[sys+(q  )*nsys]-c[sys+(q  )*nsys]*d[sys+(q+1)*nsys];
            // a[sys+(q  )*nsys] = a[sys+(q  )*nsys]-c[sys+(q  )*nsys]*a[sys+(q+1)*nsys];
            // c[sys+(q  )*nsys] =-c[sys+(q  )*nsys]*c[sys+(q+1)*nsys];

            // --- d[:]= d[:]-CqDqp[:]
            // --- a[:,:]= a[:,:]-CqAqp[:,:]
            // --- c[:,:]=-CqCqp[:,:]

            cblas_dgemv(CblasColMajor, CblasNoTrans              , m, m   , 1., &c[0 + sys*(m*m)+ q*(m*m*nsys)], m, &d[0 + sys*(m  )+ (q+1)*(m*  nsys)], 1, 0., CqDqp, 1);
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, m, m, 1., &c[0 + sys*(m*m)+ q*(m*m*nsys)], m, &a[0 + sys*(m*m)+ (q+1)*(m*m*nsys)], m, 0., CqAqp, m);
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, m, m, 1., &c[0 + sys*(m*m)+ q*(m*m*nsys)], m, &c[0 + sys*(m*m)+ (q+1)*(m*m*nsys)], m, 0., CqCqp, m);

            for(int i = 0; i < m; i++){
                d[i + sys*(m)+ q*(m*nsys)] = d[i + sys*(m)+ q*(m*nsys)]-CqDqp[i];
            }
            for(int j = 0; j < m; j++){
                for(int i = 0; i < m; i++){
                    ij  = i + m*j;
                    ijp = i + m*(j+1);

                    a[ij + sys*(m*m)+ q*(m*m*nsys)] = a[ij + sys*(m*m)+ q*(m*m*nsys)]-CqAqp[ij];
                    c[ij + sys*(m*m)+ q*(m*m*nsys)] =-CqCqp[ij];
                }
            }

            
        }
    }


    for (int sys = 0; sys < nsys; sys++){
        // r = 1./(1.-a[sys+(0+1)*nsys]*c[sys+(0  )*nsys]);
        // d[sys+(0  )*nsys] =  r*(d[sys+(0  )*nsys]-c[sys+(0  )*nsys]*d[sys+(0+1)*nsys]);
        // a[sys+(0  )*nsys] =  r*a[sys+(0  )*nsys];
        // c[sys+(0  )*nsys] = -r*c[sys+(0  )*nsys]*c[sys+(0+1)*nsys];

        // --- d[:]= d[:]-CqDqp[:]
        // --- a[:,:]= a[:,:]
        // --- c[:,:]=-CqCqp[:,:]
        // --- RR[:,:]=1.-AqpCq[:,:]

        cblas_dgemv(CblasColMajor, CblasNoTrans              , m, m   , 1., &c[0 + sys*(m*m)+ 0*(m*m*nsys)], m, &d[0 + sys*(m  )+ 1*(m*  nsys)], 1, 0., CqDqp, 1);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, m, m, 1., &c[0 + sys*(m*m)+ 0*(m*m*nsys)], m, &c[0 + sys*(m*m)+ 1*(m*m*nsys)], m, 0., CqCqp, m);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, m, m, 1., &a[0 + sys*(m*m)+ 1*(m*m*nsys)], m, &c[0 + sys*(m*m)+ 0*(m*m*nsys)], m, 0., AqpCq, m);

        for(int i = 0; i < m; i++){
            Sol[i] = d[i + sys*(m  )+ 0*(m*  nsys)]-CqDqp[i];
        }
        for(int j = 0; j < m; j++){
            for(int i = 0; i < m; i++){
                ij  = i + m*j;
                ijp = i + m*(j+1);

                delta_ij =0.;
                if(i==j) delta_ij=1.;

                Sol[ijp    ] = a[ij + sys*(m*m)+ 0*(m*m*nsys)];
                Sol[ijp+m*m] =-CqCqp[ij];
                RR [ij]      = delta_ij-AqpCq[ij];
            }
        }
        info=LAPACKE_dgesv(LAPACK_COL_MAJOR, m, 1+2*m, RR, m, ipiv, Sol, m);


        for(int i = 0; i < m; i++){
            d[i + sys*(m)+ 0*(m*nsys)] = Sol[i];
        }
        for(int j = 0; j < m; j++){
            for(int i = 0; i < m; i++){
                ij = i + m*j;
                ijp= i + m*(j+1);

                a[ij + sys*(m*m)+ 0*(m*m*nsys)] = Sol[ijp    ];
                c[ij + sys*(m*m)+ 0*(m*m*nsys)] = Sol[ijp+m*m];
            }
        }

        // rd_a[sys+(0  )*nsys] = a[sys+(0  )*nsys]; rd_a[sys+(0+1)*nsys] = a[sys+(nrow_sub-1)*nsys];
        // rd_b[sys+(0  )*nsys] = 1.  ; rd_b[sys+(0+1)*nsys] = 1.;
        // rd_c[sys+(0  )*nsys] = c[sys+(0  )*nsys]; rd_c[sys+(0+1)*nsys] = c[sys+(nrow_sub-1)*nsys];
        // rd_d[sys+(0  )*nsys] = d[sys+(0  )*nsys]; rd_d[sys+(0+1)*nsys] = d[sys+(nrow_sub-1)*nsys];

        for(int i = 0; i < m; i++){
            rd_d[i + sys*(m  )+ 0*(m  *nsys)] = d[i + sys*(m  )+ 0           *(m  *nsys)];
            rd_d[i + sys*(m  )+ 1*(m  *nsys)] = d[i + sys*(m  )+ (nrow_sub-1)*(m  *nsys)];
        }

        
        for(int j = 0; j < m; j++){
            for(int i = 0; i < m; i++){
                ij = i + m*j;

                delta_ij =0.;
                if(i==j) delta_ij=1.;

                rd_a[ij + sys*(m*m)+ 0*(m*m*nsys)] = a[ij + sys*(m*m)+ 0           *(m*m*nsys)];
                rd_a[ij + sys*(m*m)+ 1*(m*m*nsys)] = a[ij + sys*(m*m)+ (nrow_sub-1)*(m*m*nsys)];

                rd_b[ij + sys*(m*m)+ 0*(m*m*nsys)] = delta_ij;
                rd_b[ij + sys*(m*m)+ 1*(m*m*nsys)] = delta_ij;

                rd_c[ij + sys*(m*m)+ 0*(m*m*nsys)] = c[ij + sys*(m*m)+ 0           *(m*m*nsys)];
                rd_c[ij + sys*(m*m)+ 1*(m*m*nsys)] = c[ij + sys*(m*m)+ (nrow_sub-1)*(m*m*nsys)];
            }
        }
    }


    // if (info == 0) {
    //     // 해를 출력, B는 이제 해를 담고 있음
    //     printf("Solution\n");
    // } else {
    //     printf("An error occurred: %d,%d\n", sys,q);
    //     break;
    // }

}
void btdma_many_a2av_forward(BTDMA_PLAN* plan){
    
    mpiutil_pack_d(plan->rd_a,plan->sendM, plan->countsAM,plan->distAM,plan->sizeAM, plan->comm);
    MPI_Alltoallv((void *)(plan->sendM),plan->a2a_count_sM,plan->a2a_dist_sM,MPI_DOUBLE, 
                  (void *)(plan->recvM),plan->a2a_count_rM,plan->a2a_dist_rM,MPI_DOUBLE, plan->comm);
    mpiutil_unpack_d(plan->recvM,plan->tr_a, plan->countsBM,plan->distBM,plan->sizeBM, plan->comm);

    mpiutil_pack_d(plan->rd_b,plan->sendM, plan->countsAM,plan->distAM,plan->sizeAM, plan->comm);
    MPI_Alltoallv((void *)(plan->sendM),plan->a2a_count_sM,plan->a2a_dist_sM,MPI_DOUBLE, 
                  (void *)(plan->recvM),plan->a2a_count_rM,plan->a2a_dist_rM,MPI_DOUBLE, plan->comm);
    mpiutil_unpack_d(plan->recvM,plan->tr_b, plan->countsBM,plan->distBM,plan->sizeBM, plan->comm);

    mpiutil_pack_d(plan->rd_c,plan->sendM, plan->countsAM,plan->distAM,plan->sizeAM, plan->comm);
    MPI_Alltoallv((void *)(plan->sendM),plan->a2a_count_sM,plan->a2a_dist_sM,MPI_DOUBLE, 
                  (void *)(plan->recvM),plan->a2a_count_rM,plan->a2a_dist_rM,MPI_DOUBLE, plan->comm);
    mpiutil_unpack_d(plan->recvM,plan->tr_c, plan->countsBM,plan->distBM,plan->sizeBM, plan->comm);

    mpiutil_pack_d(plan->rd_d,plan->sendV, plan->countsAV,plan->distAV,plan->sizeAV, plan->comm);
    MPI_Alltoallv((void *)(plan->sendV),plan->a2a_count_sV,plan->a2a_dist_sV,MPI_DOUBLE, 
                  (void *)(plan->recvV),plan->a2a_count_rV,plan->a2a_dist_rV,MPI_DOUBLE, plan->comm);
    mpiutil_unpack_d(plan->recvV,plan->tr_d, plan->countsBV,plan->distBV,plan->sizeBV, plan->comm);

}
void btdma_many_a2av_backward(BTDMA_PLAN* plan){
    
    mpiutil_pack_d(plan->tr_d,plan->recvV, plan->countsBV,plan->distBV,plan->sizeBV, plan->comm);
    MPI_Alltoallv((void *)(plan->recvV),plan->a2a_count_rV,plan->a2a_dist_rV,MPI_DOUBLE, 
                  (void *)(plan->sendV),plan->a2a_count_sV,plan->a2a_dist_sV,MPI_DOUBLE, plan->comm);
    mpiutil_unpack_d(plan->sendV, plan->rd_d, plan->countsAV,plan->distAV,plan->sizeAV, plan->comm);
}
void btdma_many_update(int nrow_sub,int nsys,int m,double *a,double *b,double *c,double *d,double *rd_d){
    double AqD0[m*m],CqDn[m*m]; //todo: 나중에 동적할당으로 수정 필요

    for (int sys = 0; sys < nsys; sys++){
        for (int i = 0; i < m; i++)
        {
            d[i + sys*(m)+ 0*(m*nsys)] = rd_d[i + sys*(m)+ 0*(m*nsys)];
            d[i + sys*(m)+ (nrow_sub-1)*(m*nsys)] = rd_d[i + sys*(m)+ 1*(m*nsys)];
        }
    }
    
    for (int q = 1; q < nrow_sub-1; q++){
        for (int sys = 0; sys < nsys; sys++){
            cblas_dgemv(CblasColMajor, CblasNoTrans, m, m   , 1., &a[0 + sys*(m*m)+ q*(m*m*nsys)], m, &d[0 + sys*(m  )+ 0           *(m*  nsys)], 1, 0., AqD0, 1);
            cblas_dgemv(CblasColMajor, CblasNoTrans, m, m   , 1., &c[0 + sys*(m*m)+ q*(m*m*nsys)], m, &d[0 + sys*(m  )+ (nrow_sub-1)*(m*  nsys)], 1, 0., CqDn, 1);
            for (int i = 0; i < m; i++){
                d[i + sys*(m)+ q*(m*nsys)] =  d[i + sys*(m)+ q*(m*nsys)] -AqD0[i]-CqDn[i];                               
            }
            
        }
    }


}
