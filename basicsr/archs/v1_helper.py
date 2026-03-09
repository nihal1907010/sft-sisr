import time
import math
import torch
import ptwt


def _sync_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _check_input(x, levels):
    if x.ndim != 4:
        raise ValueError(f"x must have shape [B, C, H, W], got {tuple(x.shape)}")

    _, _, h, w = x.shape
    req = 2 ** levels

    if h % req != 0 or w % req != 0:
        raise ValueError(
            f"For SWT level={levels}, H and W should be divisible by 2**levels={req}. "
            f"Got H={h}, W={w}."
        )


def _level1_swt2d(x, wavelet):
    """
    One-level separable 2D SWT on x in [B, C, H, W].
    Returns: ll, lh, hl, hh
    """
    # Width transform
    coeff_w = ptwt.swt(x, wavelet, level=1, axis=-1)
    lo_w, hi_w = coeff_w[0], coeff_w[1]

    # Height transform on low-width branch -> LL, HL
    coeff_lo = ptwt.swt(lo_w, wavelet, level=1, axis=-2)
    ll, hl = coeff_lo[0], coeff_lo[1]

    # Height transform on high-width branch -> LH, HH
    coeff_hi = ptwt.swt(hi_w, wavelet, level=1, axis=-2)
    lh, hh = coeff_hi[0], coeff_hi[1]

    return ll, lh, hl, hh


def _level1_iswt2d(ll, lh, hl, hh, wavelet):
    """
    One-level inverse separable 2D SWT.
    Inputs are [B, C, H, W].
    Returns reconstructed x in [B, C, H, W].
    """
    # Invert height transforms
    lo_w = ptwt.iswt([ll, hl], wavelet, axis=-2)
    hi_w = ptwt.iswt([lh, hh], wavelet, axis=-2)

    # Invert width transform
    x = ptwt.iswt([lo_w, hi_w], wavelet, axis=-1)
    return x


def swt2d_forward(x, wavelet="db1", levels=1):
    """
    Multilevel 2D SWT.
    Input:
        x: [B, C, H, W]
    Output:
        coeffs: list of dicts, length=levels
                coeffs[j] = {"ll", "lh", "hl", "hh"} for level j+1
                coeffs[-1]["ll"] is the deepest approximation
    """
    _check_input(x, levels)

    coeffs = []
    current = x

    for _ in range(levels):
        ll, lh, hl, hh = _level1_swt2d(current, wavelet)
        coeffs.append({"ll": ll, "lh": lh, "hl": hl, "hh": hh})
        current = ll

    return coeffs


def swt2d_inverse(coeffs, wavelet="db1"):
    """
    Inverse multilevel 2D SWT.
    Input:
        coeffs: output from swt2d_forward
    Output:
        x_rec: [B, C, H, W]
    """
    if not isinstance(coeffs, list) or len(coeffs) == 0:
        raise ValueError("coeffs must be a non-empty list from swt2d_forward.")

    current = coeffs[-1]["ll"]

    for level in reversed(range(len(coeffs))):
        c = coeffs[level]

        ll = current
        lh = c["lh"]
        hl = c["hl"]
        hh = c["hh"]

        # shape checks
        ref_shape = ll.shape
        for name, t in [("lh", lh), ("hl", hl), ("hh", hh)]:
            if t.shape != ref_shape:
                raise ValueError(
                    f"Shape mismatch at level {level + 1}: "
                    f"ll has shape {tuple(ref_shape)} but {name} has shape {tuple(t.shape)}"
                )

        current = _level1_iswt2d(ll, lh, hl, hh, wavelet)

    return current


def compute_errors(x, x_hat, data_range=None, eps=1e-12):
    diff = x_hat - x

    mse = torch.mean(diff ** 2).item()
    rmse = math.sqrt(mse)
    mae = torch.mean(torch.abs(diff)).item()
    max_abs = torch.max(torch.abs(diff)).item()

    if data_range is None:
        data_range = (x.max() - x.min()).item()
        if data_range == 0:
            data_range = 1.0

    psnr = 20.0 * math.log10(data_range / (rmse + eps))

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "max_abs": max_abs,
        "psnr_db": psnr,
    }


def benchmark_swt2d(
    batch=2,
    channels=3,
    height=256,
    width=256,
    wavelet="db2",
    levels=2,
    dtype=torch.float32,
    device=None,
    warmup=10,
    runs=50,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    x = torch.randn(batch, channels, height, width, dtype=dtype, device=device)
    _check_input(x, levels)

    # warmup
    for _ in range(warmup):
        coeffs = swt2d_forward(x, wavelet=wavelet, levels=levels)
        x_hat = swt2d_inverse(coeffs, wavelet=wavelet)
    _sync_if_needed(device)

    # forward timing
    t0 = time.perf_counter()
    for _ in range(runs):
        coeffs = swt2d_forward(x, wavelet=wavelet, levels=levels)
    _sync_if_needed(device)
    t1 = time.perf_counter()

    # inverse timing
    t2 = time.perf_counter()
    for _ in range(runs):
        x_hat = swt2d_inverse(coeffs, wavelet=wavelet)
    _sync_if_needed(device)
    t3 = time.perf_counter()

    # end-to-end
    coeffs = swt2d_forward(x, wavelet=wavelet, levels=levels)
    x_hat = swt2d_inverse(coeffs, wavelet=wavelet)

    errors = compute_errors(x, x_hat)

    result = {
        "device": str(device),
        "dtype": str(dtype),
        "input_shape": tuple(x.shape),
        "wavelet": wavelet,
        "levels": levels,
        "forward_time_ms": 1000.0 * (t1 - t0) / runs,
        "inverse_time_ms": 1000.0 * (t3 - t2) / runs,
        "total_time_ms": 1000.0 * ((t1 - t0) + (t3 - t2)) / runs,
        **errors,
    }
    return result, coeffs, x, x_hat


if __name__ == "__main__":
    result, coeffs, x, x_hat = benchmark_swt2d(
        batch=2,
        channels=3,
        height=256,
        width=256,
        wavelet="db2",
        levels=2,
        dtype=torch.float32,
        device="cuda" if torch.cuda.is_available() else "cpu",
        warmup=5,
        runs=20,
    )

    print("Benchmark / reconstruction check")
    for k, v in result.items():
        print(f"{k}: {v}")

    print("\nCoefficient shapes")
    for i, c in enumerate(coeffs, start=1):
        print(
            f"level {i}: "
            f"ll={tuple(c['ll'].shape)}, "
            f"lh={tuple(c['lh'].shape)}, "
            f"hl={tuple(c['hl'].shape)}, "
            f"hh={tuple(c['hh'].shape)}"
        )