.section .probe_mask,"ax"
stp x0,x1,[sp,#-16]!
ldr x0, trace_pointer
cbz x0, 3f
ldr x1,[x25,#0x10]
str x1,[x0,#0]
ldr x1,[x25,#0x18]
str x1,[x0,#8]
ldr x1,[x25,#0x38]
str x1,[x0,#16]
str x28,[x0,#24]
str x26,[x0,#32]
3:
ldp x0,x1,[sp],#16
ldr w0, actual_tokens
sub w0,w0,#1
cmp x5,x0
b.lt 1f
b.gt 2f
cmp x12,x0
b.eq 1f
2:
fmov s0,s1
1:
str s0,[x7,x5,lsl #2]
b mask_return
.section .probe_context,"ax"
stp x0,x1,[sp,#-16]!
ldr w0,[x24,#4]
str w0,[x24,#8]
strb wzr,[x24,#57]
ldp x0,x1,[sp],#16
b context_return

.section .probe_count,"ax"
.global actual_tokens
actual_tokens: .word 151

.section .probe_trace,"ax"
trace_pointer: .quad 0
