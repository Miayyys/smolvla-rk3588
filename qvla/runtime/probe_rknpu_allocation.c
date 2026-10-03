/* Process-local allocation diagnostic. ABI from LZAMP's rknpu_ioctl.h. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>

struct mem_create {uint32_t handle,flags;uint64_t size,obj_addr,dma_addr,sram_size;int32_t domain;uint32_t core_mask;};
struct mem_sync {uint32_t flags,reserved;uint64_t obj_addr,offset,size;};
static uint64_t uncached_objects[512];static unsigned count;
int ioctl(int fd,unsigned long request,...) {
  static int (*real_ioctl)(int,unsigned long,...);
  if(!real_ioctl)real_ioctl=dlsym(RTLD_NEXT,"ioctl");
  va_list ap;va_start(ap,request);void* arg=va_arg(ap,void*);va_end(ap);
  const char* mode=getenv("QVLA_RKNPU_ALLOC_MODE");
  unsigned type=_IOC_TYPE(request),nr=_IOC_NR(request),bytes=_IOC_SIZE(request);
  int create=(type=='d'&&nr==0x42)||(type=='r'&&nr==2);
  int sync=(type=='d'&&nr==0x45)||(type=='r'&&nr==5);
  if(sync && bytes==sizeof(struct mem_sync) && mode && !strcmp(mode,"wc")){
    struct mem_sync* m=arg;
    for(unsigned i=0;i<count;++i)if(uncached_objects[i]==m->obj_addr)return 0; // no CPU cache on this mapping
  }
  if(create && bytes==sizeof(struct mem_create)){
    struct mem_create* m=arg;uint32_t original=m->flags;
    const char* size_filter=getenv("QVLA_RKNPU_WC_SIZE");
    int changed=mode && !strcmp(mode,"wc") && (m->flags&2) && (!size_filter || m->size==strtoull(size_filter,NULL,0));
    if(changed)m->flags=(m->flags&~2u)|4u;
    if(mode && !strcmp(mode,"no_sram")){m->flags&=~256u;m->sram_size=0;}
    int rc=real_ioctl(fd,request,arg);
    if(!rc && changed){
      if(count>=512)abort();uncached_objects[count++]=m->obj_addr;
    }
    fprintf(stderr,"NPU_ALLOC original=%x actual=%x size=%llu sram=%llu rc=%d\n",original,m->flags,(unsigned long long)m->size,(unsigned long long)m->sram_size,rc);
    return rc;
  }
  if((type=='d'&&nr==0x41)||(type=='r'&&nr==1)){
    uint32_t* w=arg;
    if(getenv("QVLA_RKNPU_SUBMIT_TRACE"))fprintf(stderr,"NPU_SUBMIT flags=%x tasks=%u request_bytes=%u\n",w[0],w[3],bytes);
  }
  return real_ioctl(fd,request,arg);
}
