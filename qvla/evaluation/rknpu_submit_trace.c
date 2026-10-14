/* Read-only per-process observation of the pinned RKNPU submit ABI. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <time.h>
struct subtask {uint32_t start,count;};
struct submit {uint32_t flags,timeout,start,count,counter;int32_t priority;uint64_t object;uint32_t domain,reserved;uint64_t base;int64_t hw_us;uint32_t mask;int32_t fence;struct subtask sub[5];};
int ioctl(int fd,unsigned long request,...) {
    static int (*original)(int,unsigned long,...);
    if(!original)original=dlsym(RTLD_NEXT,"ioctl");
    va_list ap;va_start(ap,request);void *arg=va_arg(ap,void*);va_end(ap);
    int trace=((_IOC_TYPE(request)=='r' && _IOC_NR(request)==1)||(_IOC_TYPE(request)=='d' && _IOC_NR(request)==0x41)) && _IOC_SIZE(request)==sizeof(struct submit);
    struct submit before;if(trace)before=*(struct submit*)arg;
    int rc=original(fd,request,arg);
    if(trace){
        struct submit *s=arg;
        fprintf(stderr,"SUBMIT_TRACE flags=%x mask=%x start=%u count=%u sub=%u:%u,%u:%u,%u:%u,%u:%u,%u:%u rc=%d counter=%u hw_us=%lld\n",before.flags,before.mask,before.start,before.count,
            before.sub[0].start,before.sub[0].count,before.sub[1].start,before.sub[1].count,before.sub[2].start,before.sub[2].count,before.sub[3].start,before.sub[3].count,before.sub[4].start,before.sub[4].count,
            rc,s->counter,(long long)s->hw_us);
    }
    return rc;
}
