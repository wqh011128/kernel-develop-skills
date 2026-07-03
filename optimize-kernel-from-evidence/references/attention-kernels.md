# Attention Kernel Tuning Notes

Use for MHA, MQA, GQA, FlashAttention-style kernels, causal masking, prefix/ring attention, and distributed context-parallel attention.

## Correctness Gates

- Verify logits masking before optimizing softmax or communication.
- Validate `lse` or denominator state separately when using online softmax, blockwise merge, ring attention, or partitioned attention.
- Treat output equality and `lse` equality as separate checks; partitioned merge order can cause small output differences even when normalization is correct.
- Compare against a trusted dense or framework reference before replacing communication or local cores.
- Cover edge cases: short sequence, non-multiple block sizes, causal boundary, grouped heads, and padding.

## Bottleneck Patterns

- Separate local attention compute, softmax merge, reshape/copy, and collective time in profiles.
- Do not assume less communicated data improves wall time; extra loop/control/merge overhead can dominate.
- For distributed attention, profile full device time and collectives, not only the Pallas custom-call.
- For small kernels, prefer device/profile timing over host wall-clock timing.
- For ring or prefix attention, count useful visible shard work separately from invalid/future shard local-core work.
- If MXU utilization is low, check whether the kernel is dominated by online-softmax exp, mask/index arithmetic, vector ALU, scalar ALU, spills, or launch/control overhead before changing matmul tiles.
- For communication overlap, compare collective start/done timing against local-core windows and report whether communication is exposed or hidden.
- For causal CP attention, a lower collective count is not sufficient; verify that `collective-permute-done`, fusion/control, and slice/reshape overhead did not grow.
- Use official XProf roofline/overview-style evidence to classify the kernel before choosing a tactic: compute/MXU-bound, HBM-bound, VMEM-bound, communication-bound, launch/control-bound, or mixed.
- Compare against known high-quality attention implementations or project-local kernels before inventing a new structure. Look for how they handle online softmax state, LSE, masking, layout, and communication boundaries.
- For block-size tuning, measure full latency and XProf component movement. Larger tiles can reduce launch/control overhead but may increase VMEM pressure or reduce occupancy.

## Hypotheses To Test

- Change one of block size, layout, masking strategy, communication pattern, or accumulator precision at a time.
- Tune query and key/value block sizes with correctness fixed; record both custom-call time and full time.
- For ring/prefix attention, measure collective latency, merge overhead, and memory materialization separately.
- Consider fusing merge/update logic only after evidence shows merge/control overhead is material.
- Test invalid/future-shard skipping only when collective order remains identical on every rank.
- Treat rank-specialized branches as suspect until XProf proves custom-call time drops without increasing collective or control overhead.
- When using JAX `lax.cond` or branch functions inside loops, pass loop-derived dynamic scalars such as shard id, global offset, or mask bounds as explicit operands. Do not rely on Python closure capture for correctness-critical branch state.
- Reject rank-specialized or visible-prefix skipping when full latency regresses, even if it reduces invalid shard work or custom-call count.
- If HBM-bound, prioritize avoiding K/V materialization, reducing output/LSE intermediates, and improving reuse before tile micro-tuning.
- If VMEM-bound or spill-heavy, reduce accumulator/state footprint or split state updates.
- For state compression, validate target block sizes and compiler lowering before claiming an HBM win. A smaller output/state tensor can still increase or fail scoped VMEM due scratch shape, tiling, broadcasting, or lowered temporaries.
- If launch/control-bound, reduce Pallas call count and JAX-side dynamic control; avoid large `lax.switch` or per-rank branch duplication unless compile cost is proven acceptable.
- For ring/prefix attention, do not count fewer kernel launches as an optimization if it requires JAX-side K/V materialization, select, gather, or concat. The full latency must include those costs.
- If communication-bound, optimize exposed collective done/sync time, not just start count or payload size.

## Rejection Conditions

- Reject an optimization if `lse` correctness regresses, even when output looks close.
- Reject a communication optimization if collective plus merge overhead erases custom-call gains.
- Reject layout changes that introduce hidden transposes, reshapes, or HBM copies larger than the saved compute.
