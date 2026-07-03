# Reduction And Scan Kernel Tuning Notes

Use for reductions, prefix scans, normalization, statistics, and online merge states.

## Correctness Gates

- Define associativity assumptions and accumulation dtype explicitly.
- Test non-power-of-two sizes, empty or singleton reductions when supported, and padded tails.
- For online algorithms, validate intermediate state semantics, not just final output.

## Bottleneck Patterns

- Distinguish arithmetic work from synchronization, memory traffic, and control overhead.
- Profile whether reduction tree shape, vectorization, or memory layout dominates.
- For distributed reductions, separate local reduction time from collective time.

## Hypotheses To Test

- Tune reduction granularity and tree shape while keeping accumulation semantics fixed.
- Test whether pre-normalization or state compression reduces memory traffic without changing math.
- Compare local-only, blockwise, and collective variants with identical correctness artifacts.

## Rejection Conditions

- Reject changes that rely on non-associative reorderings without documented tolerance.
- Reject faster reductions if numerical stability or documented edge-case behavior regresses.
