# COMPILATION SUITE V4
#
# DON'T CHANGE THIS FILE
# this file takes only order in compilation  
#
# this file uses builder.sh script, resolve association host-target
# change setting in builder.sh file for compiler settings

#resolve compilers in makefile
CC_X86_64 := $(shell ./builder.sh x86_64)
CC_RISCV64 := $(shell ./builder.sh riscv64)
CC_RISCV64_EMU := $(shell ./builder.sh riscv64_emu)
CC_RISCV64_14 := gcc

RISCV_OPT = -march=rv64gcv -mabi=lp64d
RISCV_OPT_NOVET = -march=rv64gc -mabi=lp64d

TARGETS = baseline \
          autovect \
          tiling \
		  tiling_v2 \
          tiling_v3 \
          reordered_tiling \
          reordered_tiling_unrolling2 \
		  reordered_tiling_unrolling4 \
		  reordered_tiling_unrolling8 \
		  reordered_tiling_unrolling16 \

# Shared objects paths
UTILS_O_X86    = build/x86_64/utils.o
UTILS_O_QEMU   = build/qemu/utils.o
UTILS_O_RISCV  = build/riscv64/utils.o

all: $(TARGETS)

# Compile utils.o foreach arch
$(UTILS_O_X86):
	@mkdir -p $(dir $@)
	$(CC_X86_64) -c -o $@ utils.c

$(UTILS_O_QEMU):
	@mkdir -p $(dir $@)
	$(CC_RISCV64_EMU) -c -o $@ utils.c $(RISCV_OPT)

$(UTILS_O_RISCV):
	@mkdir -p $(dir $@)
	$(CC_RISCV64) -c -o $@ utils.c $(RISCV_OPT)

# Explicit rule: make utils.o for all arch
utils: $(UTILS_O_X86) $(UTILS_O_QEMU) $(UTILS_O_RISCV)



baseline: $(UTILS_O_X86) $(UTILS_O_QEMU) $(UTILS_O_RISCV)
	$(CC_X86_64) -O3 -fno-tree-vectorize -o build/x86_64/baseline baseline.c $(UTILS_O_X86)
	$(CC_RISCV64_EMU) -O3 -fno-tree-vectorize -o build/qemu/baseline baseline.c $(UTILS_O_QEMU) $(RISCV_OPT_NOVET)
	$(CC_RISCV64) -O3 -fno-tree-vectorize -o build/riscv64/baseline baseline.c $(UTILS_O_RISCV) $(RISCV_OPT_NOVET)

autovect: $(UTILS_O_X86) $(UTILS_O_QEMU) $(UTILS_O_RISCV)
	$(CC_X86_64) -O3 -o build/x86_64/autovect baseline.c $(UTILS_O_X86) -ffast-math
	$(CC_RISCV64_14) -O3 -o build/riscv64/autovect baseline.c $(UTILS_O_RISCV) $(RISCV_OPT) -ffast-math

tiling: $(UTILS_O_X86) $(UTILS_O_QEMU) $(UTILS_O_RISCV)
	$(CC_RISCV64_EMU) -O3 -o build/qemu/tiling tiling.c $(UTILS_O_QEMU) $(RISCV_OPT)
	$(CC_RISCV64) -O3 -o build/riscv64/tiling tiling.c $(UTILS_O_RISCV) $(RISCV_OPT)

tiling_v2: $(UTILS_O_X86) $(UTILS_O_QEMU) $(UTILS_O_RISCV)
	$(CC_RISCV64_EMU) -O3 -o build/qemu/tiling_v2 tiling_v2.c $(UTILS_O_QEMU) $(RISCV_OPT)
	$(CC_RISCV64) -O3 -o build/riscv64/tiling_v2 tiling_v2.c $(UTILS_O_RISCV) $(RISCV_OPT)

tiling_v3: $(UTILS_O_X86) $(UTILS_O_QEMU) $(UTILS_O_RISCV)
	$(CC_RISCV64_EMU) -O3 -o build/qemu/tiling_v3 tiling_v3.c $(UTILS_O_QEMU) $(RISCV_OPT)
	$(CC_RISCV64) -O3 -o build/riscv64/tiling_v3 tiling_v3.c $(UTILS_O_RISCV) $(RISCV_OPT)

reordered_tiling: $(UTILS_O_X86) $(UTILS_O_QEMU) $(UTILS_O_RISCV)
	$(CC_RISCV64_EMU) -O3 -o build/qemu/reordered_tiling reordered_tiling.c $(UTILS_O_QEMU) $(RISCV_OPT)
	$(CC_RISCV64) -O3 -o build/riscv64/reordered_tiling reordered_tiling.c $(UTILS_O_RISCV) $(RISCV_OPT)


# tiling_v3 (UNROLLING) (but not used..)
tiling_unrolling2: $(UTILS_O_X86) $(UTILS_O_QEMU) $(UTILS_O_RISCV)
	$(CC_RISCV64_EMU) -O3 -o build/qemu/tiling_unrolling2 tiling_v3.c $(UTILS_O_QEMU) $(RISCV_OPT) -DUNROLL=2 -fopt-info
	$(CC_RISCV64) -O3 -o build/riscv64/tiling_unrolling2 tiling_v3.c $(UTILS_O_RISCV) $(RISCV_OPT) -DUNROLL=2 -fopt-info

tiling_unrolling4: $(UTILS_O_X86) $(UTILS_O_QEMU) $(UTILS_O_RISCV)
	$(CC_RISCV64_EMU) -O3 -o build/qemu/tiling_unrolling4 tiling_v3.c $(UTILS_O_QEMU) $(RISCV_OPT) -DUNROLL=4 -fopt-info
	$(CC_RISCV64) -O3 -o build/riscv64/tiling_unrolling4 tiling_v3.c $(UTILS_O_RISCV) $(RISCV_OPT) -DUNROLL=4 -fopt-info

tiling_unrolling8: $(UTILS_O_X86) $(UTILS_O_QEMU) $(UTILS_O_RISCV)
	$(CC_RISCV64_EMU) -O3 -o build/qemu/tiling_unrolling8 tiling_v3.c $(UTILS_O_QEMU) $(RISCV_OPT) -DUNROLL=8 -fopt-info
	$(CC_RISCV64) -O3 -o build/riscv64/tiling_unrolling8 tiling_v3.c $(UTILS_O_RISCV) $(RISCV_OPT) -DUNROLL=8 -fopt-info

tiling_unrolling16: $(UTILS_O_X86) $(UTILS_O_QEMU) $(UTILS_O_RISCV)
	$(CC_RISCV64_EMU) -O3 -o build/qemu/tiling_unrolling16 tiling_v3.c $(UTILS_O_QEMU) $(RISCV_OPT) -DUNROLL=16 -fopt-info
	$(CC_RISCV64) -O3 -o build/riscv64/tiling_unrolling16 tiling_v3.c $(UTILS_O_RISCV) $(RISCV_OPT) -DUNROLL=16 -fopt-info


# reordered_tiling (UNROLLING)
reordered_tiling_unrolling2: $(UTILS_O_X86) $(UTILS_O_QEMU) $(UTILS_O_RISCV)
	$(CC_RISCV64_EMU) -O3 -o build/qemu/reordered_tiling_unrolling2 reordered_tiling.c $(UTILS_O_QEMU) $(RISCV_OPT) -DUNROLL=2 -fopt-info
	$(CC_RISCV64) -O3 -o build/riscv64/reordered_tiling_unrolling2 reordered_tiling.c $(UTILS_O_RISCV) $(RISCV_OPT) -DUNROLL=2 -fopt-info

reordered_tiling_unrolling4: $(UTILS_O_X86) $(UTILS_O_QEMU) $(UTILS_O_RISCV)
	$(CC_RISCV64_EMU) -O3 -o build/qemu/reordered_tiling_unrolling4 reordered_tiling.c $(UTILS_O_QEMU) $(RISCV_OPT) -DUNROLL=4 -fopt-info
	$(CC_RISCV64) -O3 -o build/riscv64/reordered_tiling_unrolling4 reordered_tiling.c $(UTILS_O_RISCV) $(RISCV_OPT) -DUNROLL=4 -fopt-info

reordered_tiling_unrolling8: $(UTILS_O_X86) $(UTILS_O_QEMU) $(UTILS_O_RISCV)
	$(CC_RISCV64_EMU) -O3 -o build/qemu/reordered_tiling_unrolling8 reordered_tiling.c $(UTILS_O_QEMU) $(RISCV_OPT) -DUNROLL=8 -fopt-info
	$(CC_RISCV64) -O3 -o build/riscv64/reordered_tiling_unrolling8 reordered_tiling.c $(UTILS_O_RISCV) $(RISCV_OPT) -DUNROLL=8 -fopt-info

reordered_tiling_unrolling16: $(UTILS_O_X86) $(UTILS_O_QEMU) $(UTILS_O_RISCV)
	$(CC_RISCV64_EMU) -O3 -o build/qemu/reordered_tiling_unrolling16 reordered_tiling.c $(UTILS_O_QEMU) $(RISCV_OPT) -DUNROLL=16 -fopt-info
	$(CC_RISCV64) -O3 -o build/riscv64/reordered_tiling_unrolling16 reordered_tiling.c $(UTILS_O_RISCV) $(RISCV_OPT) -DUNROLL=16 -fopt-info



## TEST code

onednn_rvv_smatmul_f32: $(UTILS_O_X86) $(UTILS_O_QEMU) $(UTILS_O_RISCV)
	$(CC_RISCV64_EMU) -O3 -o build/qemu/onednn_rvv_smatmul_f32 test/onednn_rvv_smatmul_f32.c $(UTILS_O_QEMU) $(RISCV_OPT)
	$(CC_RISCV64) -O3 -o build/riscv64/onednn_rvv_smatmul_f32 test/onednn_rvv_smatmul_f32.c $(UTILS_O_RISCV) $(RISCV_OPT)

onednn_rvv_matmul_f32: $(UTILS_O_X86) $(UTILS_O_QEMU) $(UTILS_O_RISCV)
	$(CC_RISCV64_EMU) -O3 -o build/qemu/onednn_rvv_matmul_f32 test/onednn_rvv_matmul_f32.c $(UTILS_O_QEMU) $(RISCV_OPT)
	$(CC_RISCV64) -O3 -o build/riscv64/onednn_rvv_matmul_f32 test/onednn_rvv_matmul_f32.c $(UTILS_O_RISCV) $(RISCV_OPT)

test_onednn_copy: $(UTILS_O_X86) $(UTILS_O_QEMU) $(UTILS_O_RISCV)
	$(CC_RISCV64_EMU) -O3 -o build/qemu/test_onednn_copy test/test_onednn_copy.c $(UTILS_O_QEMU) $(RISCV_OPT)
	$(CC_RISCV64) -O3 -o build/riscv64/test_onednn_copy test/test_onednn_copy.c $(UTILS_O_RISCV) $(RISCV_OPT)

test_fma_vv_sv:
	$(CC_RISCV64_EMU) -O3 -o build/qemu/test_fma_vv_sv test/test_fma_vv_sv.c $(RISCV_OPT)
	$(CC_RISCV64) -O3 -o build/riscv64/test_fma_vv_sv test/test_fma_vv_sv.c $(UTILS_O_RISCV) $(RISCV_OPT)

test_unrolling: $(UTILS_O_X86) $(UTILS_O_QEMU) $(UTILS_O_RISCV)
	$(CC_RISCV64_EMU) -O3 -S -o build/qemu/test_unrolling.s test/test_unrolling.c $(UTILS_O_QEMU) -DUNROLL=2 $(RISCV_OPT) -fopt-info -fopt-info-loop -fopt-info-loop-missed


# assembly analisys
riscv-disass:
	objdump -d build/riscv64/$(EXEC) > build/riscv64/$(EXEC).s 

riscv-disass-func:
	objdump -d --disassemble-symbols=$(FUNC) build/riscv64/$(EXEC) > build/riscv64/$(EXEC).s 

# compile target with -O3 as default
%: %.c $(UTILS_O_X86) $(UTILS_O_QEMU) $(UTILS_O_RISCV)
	$(CC_X86_64) -O3 -o build/x86_64/$@ $< -fopenmp $(UTILS_O_X86)
	$(CC_RISCV64_EMU) -O3 -o build/qemu/$@ $< -fopenmp $(UTILS_O_QEMU) $(RISCV_OPT)
	$(CC_RISCV64) -O3 -o build/riscv64/$@ $< -fopenmp $(UTILS_O_RISCV) $(RISCV_OPT)




# Tables and Charts

table-%:
	python3 script/tables_benchmark.py report/$*/bench*.txt report/$*/table-$*.csv

# %: version
single-plot-%:
	python3 script/plot_benchmarks.py report/$*/table-$*.csv --output-dir report/$* --mode single

# %: current version
# $BASE: previous version
# $OPTIONS: other options (future)
compare-plot-%:
	python3 script/plot_benchmarks.py report/$(BASE)/table-$(BASE).csv --compare-files report/$*/table-$*.csv --output-dir report/$*/compare-$(BASE) --mode compare $(OPTIONS)


# %: current version
# $BASE: previous version
# $CALC_OPT: other options
# $PLOT_OPT: other options
speedup-%:
	python3 script/calc_speedup.py report/$(BASE)/table-$(BASE).csv report/$*/table-$*.csv $(CALC_OPT) --output report/$*/compare-$(BASE)/speedup-$*_$(BASE).csv
	python3 script/plot_speedup.py report/$*/compare-$(BASE)/speedup-$*_$(BASE).csv -o report/$*/compare-$(BASE)/speedup-$*_$(BASE).png $(PLOT_OPT)


clean:
	rm -f build/x86_64/*
	rm -f build/qemu/*
	rm -f build/riscv64/*
