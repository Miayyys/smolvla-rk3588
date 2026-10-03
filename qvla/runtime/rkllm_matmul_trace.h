// Private ABI audit only. Do not use for normal inference or performance claims.
#include <cstdio>
#include <cstdint>
static unsigned trace_round=0,trace_op=0;
static void matmul_trace(void* context,uint64_t id,uint64_t offset,uint64_t size,void* tensor,void* shape,int input) {
  auto read=[](void* p,size_t off){uint64_t v;std::memcpy(&v,static_cast<char*>(p)+off,8);return v;};
  void* node=reinterpret_cast<void*>(read(context,0x58));
  for(int depth=0;node && depth<64;++depth){
    auto key=read(node,0x20);
    if(key==id)break;
    node=reinterpret_cast<void*>(read(node,key>id?0x10:0x18));
  }
  if(!node || read(node,0x20)!=id)return;
  auto* descriptor=static_cast<char*>(node)+0x28;
  auto capacity=read(descriptor,0x28);
  auto address=read(descriptor,0x20);
  auto* table=static_cast<char*>(tensor);int32_t back;std::memcpy(&back,table,4);
  auto* vt=table-back;uint16_t field;std::memcpy(&field,vt+14,2);
  uint32_t relative;std::memcpy(&relative,table+field,4);auto* name_blob=table+field+relative;
  uint32_t name_size;std::memcpy(&name_size,name_blob,4);auto* name=name_blob+4;
  auto* dims=static_cast<int*>(shape);
  if(input && std::getenv("QVLA_RKLLM_FULL_INPUT_SYNC")){
    uint16_t size_field;std::memcpy(&size_field,vt+16,2);uint32_t plan_size;std::memcpy(&plan_size,table+size_field,4);
    if(offset>capacity || plan_size>capacity-offset)std::abort();
    Dl_info lib={};if(!dladdr(reinterpret_cast<void*>(rkllm_init),&lib))std::abort();
    auto sync=reinterpret_cast<int(*)(void*,uint64_t,uint64_t,uint64_t)>(static_cast<char*>(lib.dli_fbase)+0x3407d0);
    if(sync(context,id,offset,plan_size))std::abort();
  }
  const char* dir=std::getenv("QVLA_RKLLM_TRACE_DIR");
  if(dir)
  std::fprintf(stderr,"MATMUL_SYNC round=%u index=%u input=%d tensor=%.*s M=%d K=%d N=%d offset=%llu bytes=%llu capacity=%llu\n",
    trace_round,trace_op,input,(int)std::min<uint32_t>(name_size,128),name,dims[0],dims[1],dims[2],
    (unsigned long long)offset,(unsigned long long)size,(unsigned long long)capacity);
  if(dir && trace_op<40 && address && offset<=capacity && size<=capacity-offset){
    char path[512];std::snprintf(path,sizeof(path),"%s/round%u_op%03u_%s_%.*s.bin",dir,trace_round,trace_op,input?"input":"output",(int)std::min<uint32_t>(name_size,128),name);
    FILE* f=std::fopen(path,"wb");if(!f)std::abort();
    if(std::fwrite(reinterpret_cast<void*>(address+offset),1,size,f)!=size)std::abort();std::fclose(f);
    if(!input && std::getenv("QVLA_RKLLM_DOUBLE_OUTPUT_SYNC")){
      Dl_info lib={};if(!dladdr(reinterpret_cast<void*>(rkllm_init),&lib))std::abort();
      auto sync=reinterpret_cast<int(*)(void*,uint64_t,uint64_t,uint64_t)>(static_cast<char*>(lib.dli_fbase)+0x3406d4);
      if(sync(context,id,offset,size))std::abort();
      std::snprintf(path,sizeof(path),"%s/round%u_op%03u_second_output_%.*s.bin",dir,trace_round,trace_op,(int)std::min<uint32_t>(name_size,128),name);
      f=std::fopen(path,"wb");if(!f)std::abort();if(std::fwrite(reinterpret_cast<void*>(address+offset),1,size,f)!=size)std::abort();std::fclose(f);
    }
    if(input && trace_op==3){
      std::snprintf(path,sizeof(path),"%s/round%u_qk_pool.bin",dir,trace_round);f=std::fopen(path,"wb");if(!f)std::abort();
      if(std::fwrite(reinterpret_cast<void*>(address),1,capacity,f)!=capacity)std::abort();std::fclose(f);
    }
    if(!input && std::getenv("QVLA_RKLLM_TRACE_INTERNAL_REGION") && std::strncmp(name,"matmul_qk_C",name_size)==0){
      // Candidate compiled PlanOffset's CPU view. Its live runtime allocation
      // and layout are NOT established; never label it a verified native B.
      Dl_info lib={};if(!dladdr(reinterpret_cast<void*>(rkllm_init),&lib))std::abort();
      auto sync=reinterpret_cast<int(*)(void*,uint64_t,uint64_t,uint64_t)>(static_cast<char*>(lib.dli_fbase)+0x3406d4);
      const char* configured=std::getenv("QVLA_RKLLM_INTERNAL_OFFSET");if(!configured)std::abort();
      uint64_t b_offset=std::strtoull(configured,nullptr,0);
      uint64_t b_size=static_cast<uint64_t>(dims[1])*dims[2]*2;
      if(b_offset>capacity || b_size>capacity-b_offset || sync(context,id,b_offset,b_size))std::abort();
      std::snprintf(path,sizeof(path),"%s/round%u_op%03u_internal_region.bin",dir,trace_round,trace_op);f=std::fopen(path,"wb");if(!f)std::abort();
      if(std::fwrite(reinterpret_cast<void*>(address+b_offset),1,b_size,f)!=b_size)std::abort();std::fclose(f);
    }
  }
  if(!input)++trace_op;
}
