"""Check that the first columns of esig's depth-4 log-signature equal the depth-2 and depth-3 log-signatures."""
import esig
import numpy as np
import pandas as pd

seg = pd.read_parquet("data/segments.parquet").iloc[0]
m = np.load(seg["mfcc_file"])[seg["row_in_file"]]
t = np.linspace(0.0, 1.0, len(m))[:, None]
path = np.vstack([np.zeros((1, 14)), np.hstack([t, (m - m.mean(0)) / (m.std(0) + 1e-8)])]).astype(np.float64)
d4 = esig.stream2logsig(path, 4)
for depth in (2, 3):
    low = esig.stream2logsig(path, depth)
    n = esig.logsigdim(14, depth)
    print(f"depth {depth}: length {len(low)} (logsigdim {n}); max |difference| vs first {n} columns of depth 4: "
          f"{np.max(np.abs(low - d4[:n])):.3e}")