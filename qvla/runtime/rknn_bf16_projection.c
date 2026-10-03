/* Native BF16 input adapter for a single, contiguous 3D projection.
 * RKNN 2.3.2 can expose an undefined logical input dtype for a BF16 graph.
 * Bind the validated native BF16 input instead of using that broken converter.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "rknn_api.h"
typedef struct {rknn_context ctx;rknn_tensor_mem *input;uint32_t ni,no;} projection;
void qvla_bf16_close(void *handle) {
 projection *p=handle;if(!p)return;
 if(p->input)rknn_destroy_mem(p->ctx,p->input);
 if(p->ctx)rknn_destroy(p->ctx);
 free(p);
}
void *qvla_bf16_open(const char *path,uint32_t ni,uint32_t no,int *status) {
 projection *p=calloc(1,sizeof(*p));if(!p){*status=-100;return NULL;}
 *status=rknn_init(&p->ctx,(void*)path,0,0,NULL);if(*status)goto fail;
 *status=rknn_set_core_mask(p->ctx,RKNN_NPU_CORE_0);if(*status)goto fail;
 rknn_input_output_num io={0};
 *status=rknn_query(p->ctx,RKNN_QUERY_IN_OUT_NUM,&io,sizeof(io));if(*status)goto fail;
 if(io.n_input!=1||io.n_output!=1){*status=-101;goto fail;}
 rknn_tensor_attr a={0},b={0};
 *status=rknn_query(p->ctx,RKNN_QUERY_NATIVE_INPUT_ATTR,&a,sizeof(a));if(*status)goto fail;
 *status=rknn_query(p->ctx,RKNN_QUERY_OUTPUT_ATTR,&b,sizeof(b));if(*status)goto fail;
 if(a.type!=RKNN_TENSOR_BFLOAT16||b.type!=RKNN_TENSOR_BFLOAT16||a.n_elems!=ni||b.n_elems!=no||a.size!=ni*2||a.size_with_stride!=ni*2){*status=-102;goto fail;}
 p->ni=ni;p->no=no;p->input=rknn_create_mem(p->ctx,a.size_with_stride);
 if(!p->input){*status=-103;goto fail;}
 a.pass_through=1;
 *status=rknn_set_io_mem(p->ctx,p->input,&a);if(*status)goto fail;
 return p;
 fail:qvla_bf16_close(p);return NULL;
}
int qvla_bf16_run(void *handle,const float *input,float *output) {
 projection *p=handle;uint16_t *buf=(uint16_t*)((char*)p->input->virt_addr+p->input->offset);
 for(uint32_t i=0;i<p->ni;i++){
  uint32_t bits;memcpy(&bits,input+i,4);bits+=0x7fff+((bits>>16)&1);buf[i]=bits>>16;
 }
 int ret=rknn_mem_sync(p->ctx,p->input,RKNN_MEMORY_SYNC_TO_DEVICE);if(ret)return ret;
 ret=rknn_run(p->ctx,NULL);if(ret)return ret;
 rknn_output out={0};out.want_float=1;
 ret=rknn_outputs_get(p->ctx,1,&out,NULL);if(ret)return ret;
 if(out.size!=p->no*4||!out.buf)ret=-104;
 else memcpy(output,out.buf,out.size);
 int released=rknn_outputs_release(p->ctx,1,&out);return ret?ret:released;
}
