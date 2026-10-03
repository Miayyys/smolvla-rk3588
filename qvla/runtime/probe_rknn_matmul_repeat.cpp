// Isolate public FP16 matmul repetition from private RKLLM scheduling/packing.
#include "rknn_matmul_api.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void check(int rc) { if (rc < 0) { std::fprintf(stderr,"RKNN failure %d\n",rc); std::exit(1); } }
int main(int argc,char** argv) {
  if(argc!=1 && argc!=4)return 3;
  for (int layout : {0, 1}) for (int t : {64, 96, 160}) for (int kind : {0, 1}) {
    if(argc==4 && (t!=96 || kind!=0))continue;
    int M=argc==4?3*t:t, K=kind?t:64, N=kind?64:t;
    rknn_matmul_info info={}; info.M=M;info.K=K;info.N=N;
    info.type=RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
    info.AC_layout=layout;info.B_layout=layout;
    rknn_matmul_io_attr attr={};rknn_matmul_ctx ctx=0;
    check(rknn_matmul_create(&ctx,&info,&attr));
    check(rknn_matmul_set_core_mask(ctx,RKNN_NPU_CORE_0));
    auto *A=rknn_create_mem(ctx,attr.A.size), *B=rknn_create_mem(ctx,attr.B.size), *C=rknn_create_mem(ctx,attr.C.size);
    if(!A||!B||!C)return 2;
    std::memset(A->virt_addr,0,attr.A.size);std::memset(B->virt_addr,0,attr.B.size);
    auto* av=static_cast<__fp16*>(A->virt_addr);auto* bv=static_cast<__fp16*>(B->virt_addr);
    std::vector<__fp16> normal_a(M*K),normal_b(K*N);
    for(int i=0;i<M*K;++i)normal_a[i]=((i*37%101)-50)/64.f;
    for(int i=0;i<K*N;++i)normal_b[i]=((i*19%97)-48)/64.f;
    if(argc==4){
      FILE* f=std::fopen(argv[1],"rb");if(!f||std::fread(normal_a.data(),2,M*K,f)!=(size_t)(M*K))return 4;std::fclose(f);
      std::vector<__fp16> feature(K*N);f=std::fopen(argv[2],"rb");if(!f||std::fread(feature.data(),2,K*N,f)!=(size_t)(K*N))return 4;std::fclose(f);
      for(int k=0;k<K;++k)for(int n=0;n<N;++n)normal_b[k*N+n]=feature[n*K+k];
    }
    if(layout){
      int sub=attr.A.dims[2];
      for(int m=0;m<M;++m)for(int k=0;k<K;++k)av[(k/sub*M+m)*sub+k%sub]=normal_a[m*K+k];
      check(rknn_B_normal_layout_to_native_layout(normal_b.data(),bv,K,N,&info));
    }else{std::copy(normal_a.begin(),normal_a.end(),av);std::copy(normal_b.begin(),normal_b.end(),bv);}
    check(rknn_matmul_set_io_mem(ctx,A,&attr.A));check(rknn_matmul_set_io_mem(ctx,B,&attr.B));check(rknn_matmul_set_io_mem(ctx,C,&attr.C));
    std::vector<float> first(M*N),reference(M*N,0);
    for(int m=0;m<M;++m)for(int n=0;n<N;++n)for(int k=0;k<K;++k)
      reference[m*N+n]+=static_cast<float>(normal_a[m*K+k])*static_cast<float>(normal_b[k*N+n]);
    float repeat_max=0,ref_max=0;
    for(int r=0;r<20;++r) {
      check(rknn_mem_sync(ctx,A,RKNN_MEMORY_SYNC_TO_DEVICE));check(rknn_mem_sync(ctx,B,RKNN_MEMORY_SYNC_TO_DEVICE));
      check(rknn_matmul_run(ctx));check(rknn_mem_sync(ctx,C,RKNN_MEMORY_SYNC_FROM_DEVICE));
      auto* packed=static_cast<float*>(C->virt_addr);
      std::vector<float> normal_c(M*N);
      if(layout){int sub=attr.C.dims[2];for(int m=0;m<M;++m)for(int n=0;n<N;++n)normal_c[m*N+n]=packed[(n/sub*M+m)*sub+n%sub];}
      else std::copy(packed,packed+M*N,normal_c.begin());
      auto* cv=normal_c.data();
      if(r==0)std::copy(cv,cv+M*N,first.begin());
      for(int i=0;i<M*N;++i){repeat_max=std::max(repeat_max,std::fabs(cv[i]-first[i]));ref_max=std::max(ref_max,std::fabs(cv[i]-reference[i]));}
    }
    std::printf("{\"layout\":%d,\"M\":%d,\"K\":%d,\"N\":%d,\"repeats\":20,\"repeat_max\":%.9g,\"fp32_ref_max\":%.9g}\n",layout,M,K,N,repeat_max,ref_max);
    if(argc==4){std::vector<float> bad(M*N);FILE* f=std::fopen(argv[3],"rb");if(!f||std::fread(bad.data(),4,M*N,f)!=(size_t)(M*N))return 4;std::fclose(f);float diff=0;for(int i=0;i<M*N;++i)diff=std::max(diff,std::fabs(bad[i]-reference[i]));std::printf("{\"saved_rkllm_vs_reference_max\":%.9g}\n",diff);}
    rknn_destroy_mem(ctx,A);rknn_destroy_mem(ctx,B);rknn_destroy_mem(ctx,C);rknn_matmul_destroy(ctx);
  }
}
