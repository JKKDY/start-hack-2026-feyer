Start with docs/README.md and read the relevant documentation before making changes.

Record new findings in the existing document for that topic. Consolidate overlapping
material, correct stale claims, and add evidence or context where it helps. Create
a new document only when it serves a distinct purpose, and link it from the index.

Verify your findings and commit the documentation with the related change, or in a
focused documentation commit when there is no code change. Keep unrelated edits out.

Never modify `benchmark.py`. It must stay identical to the original challenge file,
because the judges run that file. Put what the scored run needs into the
`differometor/` defaults. For a modified benchmark, copy it to a new file such as
`benchmark_<purpose>.py` and edit the copy.

Verify every simulator change with `python tests/check_general.py --suite all`, which
must print ALL PASS, and with `python benchmark.py --cpus 1`, which must print PASS.
See docs/correctness-harness.md.

Document every optimization you keep or try, in docs/performance.md:

- What changed and why it helps, the commit, and whether it was kept or dropped.
- Benchmark times from `benchmark.py`: complete first-run total, compile time and warm
  median, the number of repeats, and whether the process was pinned to one CPU.
- The machine: CPU model, logical CPUs used, RAM, OS (native Windows, WSL or Linux),
  and the Python, JAX and jaxlib versions. Times from different machines are not
  comparable, so never put them in one column without saying so.
- The correctness evidence: the harness result and the worst sensitivity and gradient
  errors.

Keep the docs clean and not fragmented. Add a row to an existing table, or update an
existing section, before you write a new one. Merge duplicate material, and delete or
correct claims a later change made stale. Don't leave notes in scratch files, commit
messages or side documents that the index doesn't link.
