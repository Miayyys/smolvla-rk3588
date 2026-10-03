// Persistent native language worker. Experimental ABI pinned to RKLLM 1.3.1.
#include <cstddef>
#include <cstdlib>
#include "rkllm.h"
#include <vector>
#include <dlfcn.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <cstring>
#include <algorithm>
#include "rkllm_matmul_trace.h"
struct Input { std::vector<float> values; int tokens=0; bool error=false; std::string hidden_path; };
static int embed(void* data,int32_t* ids,uint64_t n,void* out,uint64_t bytes) {
 auto& x=*static_cast<Input*>(data);
 if(bytes<n*960*sizeof(float))return -1;
 for(uint64_t i=0;i<n;++i){if(ids[i]<100||ids[i]>=100+x.tokens)return -1;
  std::memcpy(static_cast<float*>(out)+i*960,x.values.data()+(ids[i]-100)*960,960*sizeof(float));}
 return 0;
}
static int result(RKLLMResult* output,void* data,LLMCallState state){
 if(state==RKLLM_RUN_NORMAL && output && output->last_hidden_layer.hidden_states && output->last_hidden_layer.num_tokens>0){
  auto& h=output->last_hidden_layer;std::ofstream f(static_cast<Input*>(data)->hidden_path,std::ios::binary);
  f.write(reinterpret_cast<const char*>(h.hidden_states),h.num_tokens*h.embd_size*sizeof(float));
 }
 if(state==RKLLM_RUN_ERROR)static_cast<Input*>(data)->error=true;return 0;
}
int main(int argc,char** argv){
 if(argc!=4 && argc!=5)return 2;
 uint64_t mask_trace[5]={};
 Input x;x.hidden_path=std::string(argv[3])+".hidden";LLMHandle handle=nullptr;auto p=rkllm_createDefaultParam();
 p.model_path=argv[1];p.max_context_len=256;p.max_new_tokens=1;
 p.top_k=1;p.top_p=1;p.temperature=0;p.is_async=false;
 int threads=argc==5?std::atoi(argv[4]):3;
 if(threads!=1 && threads!=3)return 11;
 p.extend_param.enabled_cpus_num=threads;p.extend_param.enabled_cpus_mask=threads==1?0x10:0x70;
 RKLLMCallback cb={};cb.result_callback=result;cb.embed_callback=embed;cb.embed_userdata=&x;
 if(rkllm_init(&handle,&p,&cb))return 3;
 rkllm_set_chat_template(handle,"","","");
 std::cout<<"RKLLM_REPLY ready"<<std::endl;
 int actual;
 while(std::cin>>actual){
  if(actual<2||actual>160)break;
  if(!std::getenv("QVLA_RKLLM_STOCK")){
  // Version-pinned patch data: count of real tokens, excluding alignment tail.
  Dl_info info={};if(!dladdr(reinterpret_cast<void*>(rkllm_init),&info))return 8;
  auto* count=reinterpret_cast<int*>(static_cast<char*>(info.dli_fbase)+0x7137f0);
  auto page=reinterpret_cast<uintptr_t>(count)&~(static_cast<uintptr_t>(getpagesize())-1);
  if(mprotect(reinterpret_cast<void*>(page),getpagesize(),PROT_READ|PROT_WRITE|PROT_EXEC))return 9;
  *count=actual;
  *reinterpret_cast<uint64_t*>(static_cast<char*>(info.dli_fbase)+0x7137e0)=reinterpret_cast<uint64_t>(mask_trace);
  if(std::getenv("QVLA_RKLLM_MATMUL_TRACE")){
    // Same page; used only with the separately hashed trace library.
    *reinterpret_cast<uint64_t*>(static_cast<char*>(info.dli_fbase)+0x7138f0)=reinterpret_cast<uint64_t>(matmul_trace);
  }
  if(mprotect(reinterpret_cast<void*>(page),getpagesize(),PROT_READ|PROT_EXEC))return 10;
  }
  const char* token_override=std::getenv("QVLA_RKLLM_TOKENS");
  x.tokens=token_override?std::atoi(token_override):160;
  if(x.tokens<actual || x.tokens>160)return 12;
  x.error=false;x.values.assign(x.tokens*960,0.f);
  std::ifstream f(argv[2],std::ios::binary);f.read(reinterpret_cast<char*>(x.values.data()),actual*960*sizeof(float));
  if(f.gcount()!=static_cast<std::streamsize>(actual*960*sizeof(float)))return 4;
  if(rkllm_clear_kv_cache(handle,0,nullptr,nullptr))return 5;
  int size=-1;if(rkllm_get_kv_cache_size(handle,&size)||size!=0)return 6;
  std::vector<int32_t> ids(x.tokens);for(int i=0;i<x.tokens;++i)ids[i]=100+i;
  RKLLMInput input={};input.input_type=RKLLM_INPUT_TOKEN;input.token_input.input_ids=ids.data();input.token_input.n_tokens=x.tokens;input.role="user";
  RKLLMPromptCacheParam cache={};cache.save_prompt_cache=1;cache.prompt_cache_path=argv[3];
  RKLLMInferParam infer={};infer.mode=std::getenv("QVLA_RKLLM_HIDDEN")?RKLLM_INFER_GET_LAST_HIDDEN_LAYER:RKLLM_INFER_GENERATE;infer.keep_history=1;infer.prompt_cache_params=&cache;
  trace_op=0;
  int rc=rkllm_run(handle,&input,&infer,&x);
  ++trace_round;
  std::cout<<"MASK_SHAPE "<<mask_trace[0]<<" "<<mask_trace[1]<<" "<<mask_trace[2]<<" "<<mask_trace[3]<<" "<<mask_trace[4]<<std::endl;
  if(rc||x.error||rkllm_get_kv_cache_size(handle,&size)||size!=x.tokens)return 7;
  std::cout<<"RKLLM_REPLY done "<<size<<std::endl;
 }
 rkllm_destroy(handle);return 0;
}
