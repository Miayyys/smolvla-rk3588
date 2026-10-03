.section .probe_mask,"ax"
sub x0,x28,#1
cmp x5,x0
b.ne 1f
cmp x12,x0
b.eq 1f
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
