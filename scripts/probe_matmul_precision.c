#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "rknn_matmul_api.h"
int main(int argc,char**argv){
 int type=atoi(argv[1]); rknn_matmul_ctx ctx=0;
 rknn_matmul_info info={0};rknn_matmul_io_attr attr={0};
 info.B_layout=1;info.AC_layout=1;info.M=16;info.K=128;info.N=128;info.type=type;
 int ret=rknn_matmul_create(&ctx,&info,&attr);
 printf("type=%d create=%d\n",type,ret);fflush(stdout);if(ret)return 2;
 rknn_matmul_set_core_mask(ctx,RKNN_NPU_CORE_0);
 rknn_tensor_mem *a=rknn_create_mem(ctx,attr.A.size),*b=rknn_create_mem(ctx,attr.B.size),*c=rknn_create_mem(ctx,attr.C.size);
 if(!a||!b||!c)return 3;
 void *ap=(char*)a->virt_addr+a->offset,*bp=(char*)b->virt_addr+b->offset,*cp=(char*)c->virt_addr+c->offset;
 // Constant inputs A=1, B=1, reference C=128, independent of layout.
 if(type==1||type==4||type==5||type==6||type==7||type==8||type==12){unsigned short *v=ap;for(unsigned i=0;i<attr.A.size/2;i++)v[i]=0x3c00;}
 else memset(ap,type==10?0x11:1,attr.A.size);
 if(type==1||type==4){unsigned short *v=bp;for(unsigned i=0;i<attr.B.size/2;i++)v[i]=0x3c00;}
 else memset(bp,(type==7||type==8||type==10||type==11||type==12||type==15)?0x11:1,attr.B.size);
 memset(cp,0,attr.C.size);
 ret=rknn_matmul_set_io_mem(ctx,a,&attr.A);if(!ret)ret=rknn_matmul_set_io_mem(ctx,b,&attr.B);if(!ret)ret=rknn_matmul_set_io_mem(ctx,c,&attr.C);
 printf("bind=%d Atype=%d Btype=%d Ctype=%d\n",ret,attr.A.type,attr.B.type,attr.C.type);
 if(!ret)ret=rknn_mem_sync(ctx,a,RKNN_MEMORY_SYNC_TO_DEVICE);
 if(!ret)ret=rknn_mem_sync(ctx,b,RKNN_MEMORY_SYNC_TO_DEVICE);
 for(int repeat=0;repeat<3 && !ret;repeat++)ret=rknn_matmul_run(ctx);
 if(!ret)ret=rknn_mem_sync(ctx,c,RKNN_MEMORY_SYNC_FROM_DEVICE);
 printf("run=%d\n",ret);
 if(!ret){int bad=0;double first=0;for(int i=0;i<16*128;i++){double v=0;
 switch(attr.C.type){case RKNN_TENSOR_FLOAT32:v=((float*)cp)[i];break;case RKNN_TENSOR_INT32:v=((int*)cp)[i];break;case RKNN_TENSOR_INT16:v=((short*)cp)[i];break;case RKNN_TENSOR_FLOAT16:v=((_Float16*)cp)[i];break;default:bad++;continue;}
 if(i==0)first=v;if(!isfinite(v)||fabs(v-128)>0.01)bad++;}
 printf("first=%g mismatches=%d/2048 expected=128\n",first,bad);if(bad)ret=-99;}
 rknn_destroy_mem(ctx,a);rknn_destroy_mem(ctx,b);rknn_destroy_mem(ctx,c);rknn_matmul_destroy(ctx);return ret?4:0;
}
