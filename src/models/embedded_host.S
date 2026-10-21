// VIGENFLOW_HOST_LIBRARY is an absolute, quoted path supplied by Make/CMake.
.section .rodata
.balign 16
.global vgf_host_start
.global vgf_host_end
.hidden vgf_host_start
.hidden vgf_host_end
.type vgf_host_start, @object
vgf_host_start:
.incbin VIGENFLOW_HOST_LIBRARY
vgf_host_end:
.size vgf_host_start, vgf_host_end - vgf_host_start
.section .note.GNU-stack,"",@progbits
