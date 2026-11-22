"""Attribute optimised-HLO instructions of the benchmark's forward/gradient programs to Python source lines."""
import collections, re, sys
sys.argv = [sys.argv[0], "bench"]
src = open(__file__.replace("prof_attribution.py", "prof_current.py")).read()
src = src.replace('def anatomy(prefix, lowered, compiled):', 'TEXTS = {}\ndef anatomy(prefix, lowered, compiled):\n    TEXTS[prefix] = lowered.as_text(debug_info=True)')
exec(compile(src, "prof_current.py", "exec"))
for prefix, txt in TEXTS.items():
    by_line = collections.Counter()
    # StableHLO: ops reference locations "#locN"; resolve location aliases to file:line
    locs = {}
    for m in re.finditer(r'^(#loc\d+) = loc\((.*)\)$', txt, re.M):
        locs[m.group(1)] = m.group(2)
    def resolve(ref, depth=0):
        body = locs.get(ref, "")
        f = re.findall(r'"([^"]+\.py)":(\d+)', body)
        f = [x for x in f if "/jax/" not in x[0] and "/jaxlib/" not in x[0]]
        if f:
            return f'{f[0][0].split("/")[-1]}:{f[0][1]}'
        for r in re.findall(r'(#loc\d+)', body):
            if depth < 30 and r != ref:
                out = resolve(r, depth + 1)
                if out:
                    return out
        return None
    cache = {}
    for line in txt.splitlines():
        if " = " not in line or line.startswith("#loc"):
            continue
        m = re.search(r'loc\((#loc\d+)\)\s*$', line)
        key = "<none>"
        if m:
            ref = m.group(1)
            if ref not in cache:
                cache[ref] = resolve(ref) or "<jax-internal>"
            key = cache[ref]
        by_line[key] += 1
    total = sum(by_line.values())
    print(f"\n{prefix}: {total} instructions; top source lines (StableHLO ops, before XLA optimisation)")
    for k, v in by_line.most_common(14):
        print(f"  {v:6d} {100*v/total:5.1f}%  {k}")
