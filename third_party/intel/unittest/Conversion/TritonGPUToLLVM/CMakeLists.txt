add_triton_ut(
	NAME TestpISAAsmFormat
	SRCS pISAAsmFormatTest.cpp
	LIBS TritonGPUToLLVM TritonIntelGPUToLLVM
)

