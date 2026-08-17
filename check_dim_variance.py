import numpy as np

N, L, D = 80000, 36, 1152
CHUNK = 5000

for name in ["binder_alone_mean_hidden.f32", "binder_target_joint_mean_hidden.f32"]:
    arr = np.memmap(f"stageA_mean_pooled/{name}", dtype="float32", mode="r", shape=(N, L, D))

    # Streaming mean/var per (layer, dim) to avoid loading all 13GB into RAM at once.
    sum_ = np.zeros((L, D), dtype=np.float64)
    sumsq = np.zeros((L, D), dtype=np.float64)
    for start in range(0, N, CHUNK):
        chunk = arr[start:start + CHUNK].astype(np.float64)
        sum_ += chunk.sum(axis=0)
        sumsq += (chunk ** 2).sum(axis=0)

    mean = sum_ / N
    var = sumsq / N - mean ** 2  # (L, D)

    print(f"\n=== {name} ===")
    for layer_idx in [0, 1, 17, 18, 34, 35]:
        v = var[layer_idx]
        total = v.sum()
        order = np.argsort(v)[::-1]
        top1_share = v[order[0]] / total
        top5_share = v[order[:5]].sum() / total
        top10_share = v[order[:10]].sum() / total
        print(
            f"layer {layer_idx:2d}: total_var={total:.3g}  "
            f"top1_dim={order[0]:4d} top1_share={top1_share:.1%}  "
            f"top5_share={top5_share:.1%}  top10_share={top10_share:.1%}  "
            f"median_dim_var={np.median(v):.3g}  max/median={v.max() / np.median(v):.1f}x"
        )

    # Which dims are consistently the highest-variance across layers?
    top_dims_per_layer = np.argsort(var, axis=1)[:, -5:]  # top 5 dims for every layer
    from collections import Counter
    counts = Counter(top_dims_per_layer.flatten().tolist())
    most_common = counts.most_common(10)
    print(f"most frequently top-5 dims across all 36 layers: {most_common}")

np.save("stageA_mean_pooled/_tmp_check_done.npy", np.array([1]))
