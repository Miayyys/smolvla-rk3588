// Diagnostic call-site hook: original output sync, then a worker callback.
// The original call can clobber caller-saved registers; only its return is kept.
.section .probe_matmul,"ax"
sub sp,sp,#96
stp x0,x1,[sp,#0]
stp x2,x3,[sp,#16]
stp x29,x30,[sp,#32]
mov x4,x24 // current output Tensor flatbuffer
str x4,[sp,#48]
ldr x5,[x26]
str x5,[sp,#56]
bl original_sync
str x0,[sp,#64]
ldr x16,callback_pointer
cbz x16,1f
ldp x0,x1,[sp,#0]
ldp x2,x3,[sp,#16]
ldp x4,x5,[sp,#48]
mov w6,#0
blr x16
1:
ldr x0,[sp,#64]
ldp x29,x30,[sp,#32]
add sp,sp,#96
b sync_return
.section .probe_input,"ax"
sub sp,sp,#96
stp x0,x1,[sp,#0]
stp x2,x3,[sp,#16]
stp x29,x30,[sp,#32]
str x25,[sp,#48] // current input Tensor flatbuffer
ldr x5,[x26]
str x5,[sp,#56]
bl original_input_sync
str x0,[sp,#64]
ldr x16,callback_pointer
cbz x16,2f
ldp x0,x1,[sp,#0]
ldp x2,x3,[sp,#16]
ldp x4,x5,[sp,#48]
mov w6,#1
blr x16
2:
ldr x0,[sp,#64]
ldp x29,x30,[sp,#32]
add sp,sp,#96
b input_return
.section .probe_callback,"ax"
callback_pointer: .quad 0
