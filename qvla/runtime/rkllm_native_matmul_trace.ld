original_sync = 0x3406d4; sync_return = 0x332128;
original_input_sync = 0x3407d0; input_return = 0x331cd0;
SECTIONS {
 .probe_matmul 0x713800 : { *(.probe_matmul) }
 .probe_callback 0x7138f0 : { *(.probe_callback) }
 .probe_input 0x713900 : { *(.probe_input) }
}
