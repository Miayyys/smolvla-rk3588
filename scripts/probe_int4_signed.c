#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rknn_matmul_api.h"
int main(){int M=16,K=128,N=128; rknn_matmul_ctx ctx=0; rknn_matmul_info info={0};rknn_matmul_io_attr at={0};info.M=M;info.K=K;info.N=N;info.type=RKNN_INT4_MM_INT4_TO_INT16;info.B_layout=1;info.AC_layout=1;
if(rknn_matmul_create(&ctx,&info,&at))return 2;
rknn_matmul_set_core_mask(ctx,RKNN_NPU_CORE_0);
rknn_tensor_mem *a=rknn_create_mem(ctx,at.A.size),*b=rknn_create_mem(ctx,at.B.size),*c=rknn_create_mem(ctx,at.C.size);
int8_t A[16*128],B[128*128];unsigned char packed[128*128/2];
for(int i=0;i<M*K;i++)A[i]=(i*7+i/17)%15-7;
for(int i=0;i<K*N;i++)B[i]=(i*3+i/13)%15-7;
memset(a->virt_addr,0,at.A.size);int ak=at.A.dims[2];
for(int m=0;m<M;m++)for(int k=0;k<K;k++){int idx=(k/ak*M+m)*ak+k%ak;((unsigned char*)a->virt_addr)[idx/2]|=(A[m*K+k]&15)<<((idx%2)*4);}
for(int i=0;i<K*N;i+=2)packed[i/2]=(B[i]&15)|((B[i+1]&15)<<4);
rknn_B_normal_layout_to_native_layout(packed,b->virt_addr,K,N,&info);
rknn_matmul_set_io_mem(ctx,a,&at.A);rknn_matmul_set_io_mem(ctx,b,&at.B);rknn_matmul_set_io_mem(ctx,c,&at.C);
int ret=0;for(int i=0;i<10;i++)ret=rknn_matmul_run(ctx);
int bad=0,maxerr=0;int cn=at.C.dims[2];
for(int m=0;m<M;m++)for(int n=0;n<N;n++){int ref=0;for(int k=0;k<K;k++)ref+=A[m*K+k]*B[k*N+n];int idx=(n/cn*M+m)*cn+n%cn;int got=((int16_t*)c->virt_addr)[idx];int err=abs(got-ref);if(err){bad++;if(err>maxerr)maxerr=err;}}
printf("run=%d signed_INT4 mismatches=%d/%d max_abs_error=%d\n",ret,bad,M*N,maxerr);
rknn_destroy_mem(ctx,a);rknn_destroy_mem(ctx,b);rknn_destroy_mem(ctx,c);rknn_matmul_destroy(ctx);return (ret||bad)?1:0;}
