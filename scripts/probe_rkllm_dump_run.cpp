#include <cstddef>
#include "rkllm.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <cstring>

static int get_embed(void* userdata, int32_t* tokens, uint64_t count, void* output, uint64_t bytes) {
    auto* values=static_cast<std::vector<float>*>(userdata);
    if(bytes<count*256*sizeof(float))return -1;
    auto* dst=static_cast<float*>(output);
    for(uint64_t i=0;i<count;i++) {
        if(tokens[i]<100 || tokens[i]>=108)return -1;
        std::memcpy(dst+i*256,values->data()+(tokens[i]-100)*256,256*sizeof(float));
    }
    std::printf("EMBED_CALLBACK count=%llu\n",static_cast<unsigned long long>(count));
    return 0;
}

static int callback(RKLLMResult* result, void*, LLMCallState state) {
    if (state == RKLLM_RUN_ERROR) std::fprintf(stderr,"CALLBACK_ERROR\n");
    if (state == RKLLM_RUN_NORMAL && result && result->last_hidden_layer.hidden_states &&
        result->last_hidden_layer.num_tokens && result->last_hidden_layer.embd_size) {
        auto& h=result->last_hidden_layer;
        std::ofstream out("last_hidden.bin",std::ios::binary);
        out.write(reinterpret_cast<const char*>(h.hidden_states),
                  h.num_tokens*h.embd_size*sizeof(float));
        std::printf("HIDDEN tokens=%d width=%d\n",h.num_tokens,h.embd_size);
    }
    return 0;
}
int main(int argc,char** argv) {
    if(argc<4 || argc>6) {std::fprintf(stderr,"model embeds infer_mode(0/1) [max_new_tokens] [embed|token|callback]\n");return 2;}
    std::vector<float> embeds(8*256);
    std::ifstream in(argv[2],std::ios::binary);
    in.read(reinterpret_cast<char*>(embeds.data()),embeds.size()*sizeof(float));
    if(in.gcount()!=static_cast<std::streamsize>(embeds.size()*sizeof(float)))return 3;
    LLMHandle handle=nullptr;
    RKLLMParam param=rkllm_createDefaultParam();
    param.model_path=argv[1];param.max_context_len=128;
    param.max_new_tokens=argc>4?std::atoi(argv[4]):1;
    param.top_k=1;param.top_p=1;param.temperature=0;param.is_async=false;
    param.extend_param.enabled_cpus_num=4;
    param.extend_param.enabled_cpus_mask=0xf0;
    RKLLMCallback cb={};cb.result_callback=callback;
    if(argc>5 && std::string(argv[5])=="callback") {
        cb.embed_callback=get_embed;cb.embed_userdata=&embeds;
    }
    int rc=rkllm_init(&handle,&param,&cb);std::printf("INIT=%d\n",rc);
    if(rc)return 4;
    rkllm_set_chat_template(handle,"","","");
    RKLLMInput input={};input.input_type=RKLLM_INPUT_EMBED;
    input.embed_input.embed=embeds.data();input.embed_input.n_tokens=8;
    input.role="user";
    int32_t tokens[8]={100,101,102,103,104,105,106,107};
    if(argc>5 && (std::string(argv[5])=="token" || std::string(argv[5])=="callback")) {
        input.input_type=RKLLM_INPUT_TOKEN;
        input.token_input.input_ids=tokens;input.token_input.n_tokens=8;
    }
    RKLLMPromptCacheParam cache={};cache.save_prompt_cache=1;
    cache.prompt_cache_path="prompt_cache.bin";
    RKLLMInferParam infer={};infer.mode=static_cast<RKLLMInferMode>(std::atoi(argv[3]));
    infer.keep_history=1;infer.prompt_cache_params=&cache;
    rc=rkllm_run(handle,&input,&infer,nullptr);std::printf("RUN=%d\n",rc);
    int size=0;int cache_rc=rkllm_get_kv_cache_size(handle,&size);
    std::printf("CACHE_RC=%d TOKENS=%d\n",cache_rc,size);
    rkllm_destroy(handle);return rc?5:0;
}
