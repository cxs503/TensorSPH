# Reference data for the verification ladder

## `martin_moyce_1952.json`

A copy of `benchmarks/dam_break/martin_moyce_1952.json` (same file, same
provenance block), bundled here so that `tensorsph.vv.dambreak` does not depend
on the benchmark tree.

**Status of this table: WITHHELD as a digitisation.** The repository's own audit
(`benchmarks/dam_break/reference_audit.json`,
`docs/benchmarks/dam-break-reference-audit.md`) found that every abscissa is an
exact multiple of a round step and every ordinate sits on a 0.05 lattice, which
a manual read-off from a 1952 figure would not produce. It is therefore a
*transcription*, quoted because the campaign asks for the comparison, and used
here as a cross-check only -- never as the gate of a rung.

## Why the gate of rung C3 is analytic instead

The Martin & Moyce experiment has `H/a = 2` (a 2:1 column), which is outside
shallow-water theory: the measured front runs 40-50 % behind the Ritter front
(`docs/benchmarks/ritter-shallow-limit.md`) and the deviation is dominated by
the non-hydrostatic collapse, not by the scheme. The analytic front is
therefore only a *bounded* reference in that regime, which the rung reports
explicitly through the front ratio and the absolute displacement error.