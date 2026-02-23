
# make Tables and Charts

#tables

make table-baseline \
    table-autovect \
    table-tiling_v1 \
    table-tiling_v2 \
    table-tiling_v3 \
    table-reordered_tiling \
    table-reordered_tiling_unrolling



#single

make single-plot-baseline \
    single-plot-autovect \
    single-plot-tiling_v1 \
    single-plot-tiling_v2 \
    single-plot-tiling_v3 \
    single-plot-reordered_tiling \
    single-plot-reordered_tiling_unrolling


#compare

make compare-plot-autovect BASE=baseline

make compare-plot-tiling_v3 BASE=baseline
make compare-plot-tiling_v3 BASE=autovect

make compare-plot-reordered_tiling BASE=baseline
make compare-plot-reordered_tiling BASE=autovect
make compare-plot-reordered_tiling BASE=tiling_v3

make compare-plot-reordered_tiling_unrolling BASE=baseline
make compare-plot-reordered_tiling_unrolling BASE=autovect
make compare-plot-reordered_tiling_unrolling BASE=reordered_tiling


#speedup

make speedup-autovect BASE=baseline CALC_OPT="-m time -k size"

make speedup-tiling_v3 BASE=autovect CALC_OPT="-m time -k size" PLOT_OPT="-g kernel,lmul"
make speedup-tiling_v3 BASE=baseline CALC_OPT="-m time -k size" PLOT_OPT="-g kernel,lmul"

make speedup-reordered_tiling BASE=baseline CALC_OPT="-m time -k size" PLOT_OPT="-g kernel,lmul"
make speedup-reordered_tiling BASE=autovect CALC_OPT="-m time -k size" PLOT_OPT="-g kernel,lmul"
make speedup-reordered_tiling BASE=tiling_v2 CALC_OPT="-m time -k size,kernel,lmul" PLOT_OPT="-g kernel,lmul --exclude-param lmul --exclude-value 1"
make speedup-reordered_tiling BASE=tiling_v3 CALC_OPT="-m time -k size,kernel,lmul" PLOT_OPT="-g kernel,lmul"

make speedup-reordered_tiling_unrolling BASE=baseline CALC_OPT="-m time -k size" PLOT_OPT="-g kernel,lmul,unroll --exclude-param unroll --exclude-value 1"
make speedup-reordered_tiling_unrolling BASE=autovect CALC_OPT="-m time -k size" PLOT_OPT="-g kernel,lmul,unroll --exclude-param unroll --exclude-value 1"
make speedup-reordered_tiling_unrolling BASE=reordered_tiling CALC_OPT="-m time -k size,kernel,lmul" PLOT_OPT="-g kernel,lmul,unroll --exclude-param unroll --exclude-value 1"
