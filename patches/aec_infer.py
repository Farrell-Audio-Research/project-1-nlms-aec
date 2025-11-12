# metaaf/aec_infer.py
import numpy as np
from datetime import datetime

LOG_PATH = "./results/aec_runs.log"


def _compute_erle_db(mic, err, eps=1e-8):
    """
    ERLE = 10 log10( sum(mic^2) / sum(err^2) )
    mic: original mic / echoed signal
    err: output of AEC (residual)
    """
    num = np.sum(mic ** 2) + eps
    den = np.sum(err ** 2) + eps
    return 10.0 * np.log10(num / den)


def fit_infer(filter_s, filter_p, preprocess_s, postprocess_s, batch, key):
    """
    Stronger hand-written AEC to plug into MetaAF's system.infer().
    Expects JAX-ish arrays in `batch["signals"]`, returns list-of-arrays like the real code.
    """
    d = np.array(batch["signals"]["d"])  # (B, T, 1)  mic
    u = np.array(batch["signals"]["u"])  # (B, T, 1)  ref

    B, T, _ = d.shape

    all_out = []
    erles = []

    for b in range(B):
        mic = d[b, :, 0].astype(np.float32)   # d[n]
        ref = u[b, :, 0].astype(np.float32)   # u[n]

        # --- hyperparams ---
        L = 1800              # filter length (long enough for our synthetic echo)
        mu = 0.6              # step size (strong)
        leak = 1e-1           # leakage to keep weights bounded
        eps = 1e-3
        gate_ratio = 200.0      # if mic power > gate_ratio * echo power -> assume double talk, freeze
        pow_win = 1056         # window for power estimate

        # pre-normalize ref a bit to match mic scale
        if np.max(np.abs(ref)) > 0:
            ref = ref / (np.max(np.abs(ref)) + 1e-6) * min(1.0, np.max(np.abs(mic)) + 1e-6)

        # pad ref so we can grab L samples each time
        ref_pad = np.concatenate([np.zeros(L - 1, dtype=np.float32), ref])
        w = np.zeros(L, dtype=np.float32)
        out = np.zeros_like(mic)

        # moving power (very simple conv)
        kernel = np.ones(pow_win, dtype=np.float32) / pow_win
        ref_pow = np.convolve(ref ** 2, kernel, mode="same") + 1e-4
        mic_pow = np.convolve(mic ** 2, kernel, mode="same") + 1e-4

        for n in range(T):
            # grab most recent L reference samples, reversed to match w
            x = ref_pad[n:n+L][::-1]   # shape (L,)

            y = float(np.dot(w, x))
            e = mic[n] - y
            out[n] = e

            # double-talk gate: only adapt if mic isn't way bigger than echo
            if mic_pow[n] < gate_ratio * ref_pow[n]:
                norm = float(np.dot(x, x)) + eps
                # leakage + NLMS update
                w = (1.0 - leak) * w + (mu * e / norm) * x
            else:
                # only leakage during double talk
                w = (1.0 - leak) * w

        # compute ERLE for this utterance
        erle_db = _compute_erle_db(mic, out)
        erles.append(erle_db)

        # keep shape (T,1)
        all_out.append(out[:, None])

        # log this run's settings + ERLE
        # (wrapped in try so MetaAF doesn't die on permission/path issues)
        try:
            ts = datetime.now().isoformat(timespec="seconds")
            with open(LOG_PATH, "a") as f:
                f.write(
                    f"{ts}  B={b}  L={L}  mu={mu}  leak={leak}  "
                    f"gate_ratio={gate_ratio}  pow_win={pow_win}  ERLE={erle_db:.2f} dB\n"
                )
        except Exception:
            # silently skip logging if folder/file not writable
            pass

    out_arr = np.stack(all_out, axis=0)  # (B, T, 1)

    # MetaAF's infer expects (out, aux)
    # we can also return avg ERLE in aux for debugging
    aux = {"erle_db_mean": float(np.mean(erles))} if erles else {}
    return [out_arr], aux
