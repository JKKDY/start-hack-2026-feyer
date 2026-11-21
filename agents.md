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
