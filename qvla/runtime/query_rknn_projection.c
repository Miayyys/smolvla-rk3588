/* Inspect logical/native interfaces before attempting any buffer conversion. */
#include <stdio.h>
#include "rknn_api.h"
int main(int argc,char **argv) {
 if(argc!=2)return 2;
 rknn_context ctx=0;int ret=rknn_init(&ctx,argv[1],0,0,NULL);
 if(ret){fprintf(stderr,"init=%d\n",ret);return 3;}
 rknn_input_output_num io={0};
 ret=rknn_query(ctx,RKNN_QUERY_IN_OUT_NUM,&io,sizeof(io));
 if(ret)return 4;
 int commands[]={RKNN_QUERY_INPUT_ATTR,RKNN_QUERY_OUTPUT_ATTR,RKNN_QUERY_NATIVE_INPUT_ATTR,RKNN_QUERY_NATIVE_OUTPUT_ATTR};
 for(int j=0;j<4;j++)for(unsigned i=0;i<(j%2?io.n_output:io.n_input);i++){
  rknn_tensor_attr a={0};a.index=i;
  ret=rknn_query(ctx,commands[j],&a,sizeof(a));
  printf("query=%d index=%u ret=%d name=%s type=%d(%s) fmt=%d size=%u stride_size=%u dims=",commands[j],i,ret,a.name,a.type,get_type_string(a.type),a.fmt,a.size,a.size_with_stride);
  for(unsigned d=0;d<a.n_dims;d++)printf("%u,",a.dims[d]);puts("");
 }
 rknn_destroy(ctx);return 0;
}
