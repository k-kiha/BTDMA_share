#include <stdio.h>
#include "mpiutil.h"


int arraysum(int a[][3],int n,int fix) {
    int sum=0;

    for (int i = 0; i <= n; i++) sum=sum+a[i][fix];

    return sum;
}

int mpiutil_para(int sta_g, int end_g, int myrank, int nprocs, int *indx_a, int *indx_b){
    int n = end_g-sta_g+1;
    int tmp1=(n/nprocs);
    int tmp2=(n%nprocs);

    *indx_a = myrank*tmp1 + ((myrank<tmp2)?myrank:tmp2) + sta_g;
    *indx_b = *indx_a + tmp1+((myrank<tmp2)?1:0) -1;
    
    return tmp1+((myrank<tmp2)?1:0);
}

void mpiutil_pack_d(double *sorce,double *buf, int counts[][3],int dist[][3],int size[3], MPI_Comm comm){
    int ijk,ijk_pack;
    int nprocs;

    MPI_Comm_size(comm, &nprocs);

    ijk_pack=0;
    for (int rank = 0; rank < nprocs; rank++){
        for (int i = 0; i < counts[rank][0]; i++){
            for (int j = 0; j < counts[rank][1]; j++){       
                for (int k = 0; k < counts[rank][2]; k++){
                    ijk= (k+dist[rank][2])
                        +(j+dist[rank][1])*size[2]
                        +(i+dist[rank][0])*size[2]*size[1];
                    buf[ijk_pack]=sorce[ijk];
                    ijk_pack++;
                }
            }
        }
    }
}

void mpiutil_unpack_d(double *sorce,double *buf, int counts[][3],int dist[][3],int size[3], MPI_Comm comm){
    int ijk,ijk_pack;
    int nprocs;

    MPI_Comm_size(comm, &nprocs);

    ijk_pack=0;
    for (int rank = 0; rank < nprocs; rank++){
        for (int i = 0; i < counts[rank][0]; i++){
            for (int j = 0; j < counts[rank][1]; j++){       
                for (int k = 0; k < counts[rank][2]; k++){
                    ijk= (k+dist[rank][2])
                        +(j+dist[rank][1])*size[2]
                        +(i+dist[rank][0])*size[2]*size[1];
                    buf[ijk]=sorce[ijk_pack];
                    ijk_pack++;
                }
            }
        }
    }
}

void mpiutil_pack_info(int n1sub, int n2sub, int n3sub, int counts[][3],int dist[][3], MPI_Comm comm){
    int nprocs;
    int indx_tmpa,indx_tmpb;
    MPI_Comm_size(comm, &nprocs); 

    for (int rank = 0; rank < nprocs; rank++)
    {
        counts[rank][0] = n1sub;
        counts[rank][1] = mpiutil_para(0,n2sub-1,rank,nprocs, &indx_tmpa, &indx_tmpb);
        counts[rank][2] = n3sub;

        dist[rank][0] = 0;
        dist[rank][1] = arraysum(counts,rank,1)-counts[rank][1];
        dist[rank][2] = 0;
    }    
    
}

void mpiutil_unpack_info(int n1sub, int n2sub, int n3sub, int counts[][3],int dist[][3], MPI_Comm comm){
    int nprocs,myrank;
    int indx_tmpa,indx_tmpb;
    MPI_Comm_size(comm, &nprocs); 
    MPI_Comm_rank(comm, &myrank);

    for (int rank = 0; rank < nprocs; rank++)
    {
        counts[rank][0] = mpiutil_para(0,n1sub-1,rank,nprocs, &indx_tmpa, &indx_tmpb);;
        counts[rank][1] = mpiutil_para(0,n2sub-1,myrank,nprocs, &indx_tmpa, &indx_tmpb);
        counts[rank][2] = n3sub;

        dist[rank][0] = arraysum(counts,rank,0)-counts[rank][0];
        dist[rank][1] = 0;
        dist[rank][2] = 0;
    }    
    
}

void mpiutil_filewrite(int n1, int n2, int n3, double *a, double *b, double *c, double *d, int rrr, int fixx) {
    char filename[20];
    // rrr 값을 기준으로 파일 이름 설정
    sprintf(filename, "out%03d.csv", rrr);
    
    // CSV 파일 열기
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error opening file!\n");
        return;
    }
    
    // 데이터 출력 (여기선 예시로 배열의 값들 출력)
    int ijk;
    if (fixx==1)
    {
        for (int i = 0; i < n1; i++)
        {
            fprintf(file, "[i=%d]--->k \n",i);
            for (int j = 0; j < n2; j++)
            {
                fprintf(file, "j=%3d|::> ",j);
                for (int k = 0; k < n3; k++)
                {
                    ijk = k + j*n3 + i*n3*n2;
                    fprintf(file, "%3d,%3d,%3d:%3d|::",(int)a[ijk],(int)b[ijk],(int)c[ijk],(int)d[ijk]);
                }
                fprintf(file, "\n");    
            }
        }
            
    }else if (fixx==2)
    {
        for (int j = 0; j < n2; j++)
        {
            fprintf(file, "[j=%d]--->k \n",j);
            for (int i = 0; i < n1; i++)
            {
                fprintf(file, "i=%3d|::> ",i);
                for (int k = 0; k < n3; k++)
                {
                    ijk = k + j*n3 + i*n3*n2;
                    fprintf(file, "%3d,%3d,%3d:%3d|::",(int)a[ijk],(int)b[ijk],(int)c[ijk],(int)d[ijk]);
                }
                fprintf(file, "\n");    
            }
        }
    }else if (fixx==3)
    {
        for (int k = 0; k < n3; k++)
        {
            fprintf(file, "[k=%d]--->j \n",k);
            for (int i = 0; i < n1; i++)
            {
                fprintf(file, "i=%3d|::> ",i);
                for (int j = 0; j < n2; j++)
                {
                    ijk = k + j*n3 + i*n3*n2;
                    fprintf(file, "%3d,%3d,%3d:%3d|::",(int)a[ijk],(int)b[ijk],(int)c[ijk],(int)d[ijk]);
                }
                fprintf(file, "\n");    
            }
        }
    }
    
    // 파일 닫기
    fclose(file);
}

void mpiutil_alltoallv(double *sendbuf,int sendcounts[],int sdispls[], double *recvbuf,int recvcounts[],int rdispls[], MPI_Comm comm){
    MPI_Request request;
    MPI_Status status;

    //미완성

}