/* Isolated standard C-API replay of frozen logical inputs; no model patching. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
#include "rknn_api.h"
typedef struct { rknn_context ctx; uint32_t ni,no; } probe;
void qvla_probe_close(void *handle) {
    probe *p=handle; if(!p)return;
    if(p->ctx)rknn_destroy(p->ctx);
    free(p);
}
void *qvla_probe_open(const char *path,int core,uint32_t flags,int *status) {
    probe *p=calloc(1,sizeof(*p));if(!p){*status=-1000;return NULL;}
    *status=rknn_init(&p->ctx,(void*)path,0,flags,NULL);if(*status)goto fail;
    *status=rknn_set_core_mask(p->ctx,(rknn_core_mask)core);if(*status)goto fail;
    rknn_input_output_num n={0};*status=rknn_query(p->ctx,RKNN_QUERY_IN_OUT_NUM,&n,sizeof(n));if(*status)goto fail;
    p->ni=n.n_input;p->no=n.n_output;return p;
fail: qvla_probe_close(p);return NULL;
}
int qvla_probe_input_format(void *handle,uint32_t index) {
    probe *p=handle;rknn_tensor_attr a={0};a.index=index;
    int rc=rknn_query(p->ctx,RKNN_QUERY_INPUT_ATTR,&a,sizeof(a));return rc?-1004:(int)a.fmt;
}
int qvla_probe_run(void *handle,void **buffers,uint32_t *sizes,int *types,uint32_t ni,
                   void **targets,uint32_t *target_sizes,uint32_t no,int delay_ms) {
    probe *p=handle;if(ni!=p->ni||no!=p->no)return -1001;
    rknn_input *inputs=calloc(ni,sizeof(*inputs));rknn_output *outputs=calloc(no,sizeof(*outputs));
    if(!inputs||!outputs){free(inputs);free(outputs);return -1002;}
    for(uint32_t i=0;i<ni;i++){
        inputs[i].index=i;inputs[i].buf=buffers[i];inputs[i].size=sizes[i];
        inputs[i].type=(rknn_tensor_type)types[i];
        inputs[i].fmt=(rknn_tensor_format)qvla_probe_input_format(handle,i);
    }
    int rc=rknn_inputs_set(p->ctx,ni,inputs);if(rc)goto done;
    rc=rknn_run(p->ctx,NULL);if(rc)goto done;
    if(delay_ms>0){struct timespec t={delay_ms/1000,(delay_ms%1000)*1000000L};nanosleep(&t,NULL);}
    for(uint32_t i=0;i<no;i++){outputs[i].index=i;outputs[i].want_float=1;}
    rc=rknn_outputs_get(p->ctx,no,outputs,NULL);if(rc)goto done;
    for(uint32_t i=0;i<no;i++){
        if(outputs[i].size!=target_sizes[i]||!outputs[i].buf){rc=-1003;break;}
        memcpy(targets[i],outputs[i].buf,outputs[i].size);
    }
    {int released=rknn_outputs_release(p->ctx,no,outputs);if(!rc)rc=released;}
done: free(inputs);free(outputs);return rc;
}

int qvla_probe_perf(void *handle,const char *path) {
    probe *p=handle;rknn_perf_detail detail={0};
    int rc=rknn_query(p->ctx,RKNN_QUERY_PERF_DETAIL,&detail,sizeof(detail));if(rc)return rc;
    FILE *f=fopen(path,"wb");if(!f)return -1005;
    fwrite(detail.perf_data,1,detail.data_len,f);fclose(f);return 0;
}
