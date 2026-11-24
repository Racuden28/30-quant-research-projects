"""Part 2: the same searches on the real data, with out-of-sample results, gross and net of costs."""
from __future__ import annotations

import pandas as pd

from data import RES
from search import World, run_world

if __name__ == "__main__":
    w = World()
    df = pd.DataFrame(run_world(w, None, do_cluster=True, with_oos=True))
    df.to_csv(RES / "real_searches.csv", index=False, float_format="%.6g")
    print(df.shape)
