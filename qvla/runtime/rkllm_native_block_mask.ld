mask_return = 0x1cdbdc; context_return = 0x1e36cc;
SECTIONS { . = 0x7136c0; .probe_mask : { *(.probe_mask) } . = 0x713740; .probe_context : { *(.probe_context) } }
