mask_return = 0x1cdbdc; context_return = 0x1e36cc;
SECTIONS {
 .probe_mask 0x7136c0 : { *(.probe_mask) }
 .probe_context 0x713740 : { *(.probe_context) }
 .probe_trace 0x7137e0 : { *(.probe_trace) }
 .probe_count 0x7137f0 : { *(.probe_count) }
}
