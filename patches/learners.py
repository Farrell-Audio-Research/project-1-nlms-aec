# metaaf/learners.py
#
# This is the minimal loader for the published AEC checkpoint.
# It:
#   - finds the AEC run dir under v1.0.1_models/...
#   - builds the MetaAFTrainer via zoo.aec.aec_eval.get_system_ckpt(...)
#   - patches the missing MSFT dataset
#   - looks for a real `fit_infer(...)`

import os
import json
import glob
import pickle as pkl
import numpy as np

# these are in your repo
import zoo.aec.aec as zoo_aec
import zoo.aec.aec_eval as aec_eval


# small helpers
def _repo_root():
    # learners.py lives in .../metaaf/learners.py
    # repo root is one dir up
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(here)


def _models_root():
    return os.path.join(_repo_root(), "v1.0.1_models")


def _listdir_recursive(base):
    out = []
    for root, dirs, files in os.walk(base):
        # include dirs and files so we can show user what we saw
        for d in dirs:
            out.append(os.path.join(root, d))
        for f in files:
            out.append(os.path.join(root, f))
    return out


# the original AEC code expects this dataset to exist and point at a CSV. We just give it a 1-item dummy version.
class _DummyAECDataset:
    def __init__(self, *args, **kwargs):
        # make 1 frame of 1-channel zeros
        T = 16000
        sig = np.zeros((T, 1), dtype=np.float32)
        self._item = {
            "signals": {"u": sig, "d": sig, "e": sig, "s": sig},
            "metadata": {},
        }

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self._item


# try to locate / import some plausible inference entry points
def _try_load_fit_infer():
    """
    Try to find a function with signature:
        fit_infer(filter_s, filter_p, preprocess_s, postprocess_s, batch, key)
    If we can't, return None.
    """
    # 1) maybe someone added metaaf/aec_infer.py
    try:
        from metaaf import aec_infer

        if hasattr(aec_infer, "fit_infer"):
            return aec_infer.fit_infer
    except Exception:
        pass

    # 2) maybe it's in zoo.aec.aec_infer
    try:
        import zoo.aec.aec_infer as aec_infer2

        if hasattr(aec_infer2, "fit_infer"):
            return aec_infer2.fit_infer
    except Exception:
        pass

    # 3) not found
    return None


# main entry point the rest of our code calls
def load_pretrained_model(model_name: str, use_test_init: bool = True):
    """
    Return a *callable* that you can do:

        aec = load_pretrained_model("aec/....")
        y = aec(u, d)

    u, d must be 1-D numpy float arrays at 16 kHz, same length.
    """
    models_root = _models_root()
    model_dir = os.path.join(models_root, model_name)

    # your zip already extracted here; show what we found
    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            f"[learners] expected model dir {model_dir} to exist, but it does not."
        )

    # in your case we saw there was this extra timestamp subdir
    # like 2022_10_19_23_43_22
    subdirs = [
        d
        for d in glob.glob(os.path.join(model_dir, "*"))
        if os.path.isdir(d)
    ]
    if not subdirs:
        raise FileNotFoundError(
            f"[learners] {model_dir} has no run subdir (e.g. 2022_10_19_...). "
            f"Contents I see: {_listdir_recursive(model_dir)}"
        )

    # pick the first one (we only had one)
    run_dir = sorted(subdirs)[0]

    # we know from your log these two exist:
    kwargs_path = os.path.join(run_dir, "all_kwargs.json")
    ckpt_path = os.path.join(run_dir, "epoch_110.pkl")

    if not os.path.exists(kwargs_path):
        raise FileNotFoundError(f"[learners] missing {kwargs_path}")
    if not os.path.exists(ckpt_path):
        # fall back to glob
        pkl_files = glob.glob(os.path.join(run_dir, "epoch_*.pkl"))
        if not pkl_files:
            raise FileNotFoundError(f"[learners] no epoch_*.pkl in {run_dir}")
        ckpt_path = sorted(pkl_files)[-1]

    # BEFORE calling get_system_ckpt
    zoo_aec.MSFTAECDataset = _DummyAECDataset

    # let aec_eval create the MetaAFTrainer
    system, kwargs, outer_learnable = aec_eval.get_system_ckpt(
        run_dir,
        e=110,
        verbose=False,
    )
    system.outer_learnable = outer_learnable

    # now: we need a fit_infer
    fit_infer = _try_load_fit_infer()
    if fit_infer is None:
        # stop here – otherwise people think they ran AEC
        raise RuntimeError(
            "[learners] loaded AEC system, but no 'fit_infer(...)' "
            "could be found in {metaaf.aec_infer, zoo.aec.aec_infer}. "
            "Create one with signature "
            "(filter_s, filter_p, preprocess_s, postprocess_s, batch, key) "
            "and re-run."
        )

    # wrap it in a friendly callable
    def _run(u: np.ndarray, d: np.ndarray) -> np.ndarray:
        # u = far-end / ref
        # d = mic (near + echo)
        if not isinstance(u, np.ndarray):
            u_local = np.asarray(u, dtype=np.float32)
        else:
            u_local = u.astype(np.float32)
        if not isinstance(d, np.ndarray):
            d_local = np.asarray(d, dtype=np.float32)
        else:
            d_local = d.astype(np.float32)

        # match lengths
        L = min(len(u_local), len(d_local))
        u_local = u_local[:L]
        d_local = d_local[:L]

        # shape to (B, T, 1)
        u_b = u_local[None, :, None]
        d_b = d_local[None, :, None]
        z = np.zeros_like(u_b)

        batch = {
            "signals": {"u": u_b, "d": d_b, "e": z, "s": z},
            "metadata": {},
        }

        # MetaAFTrainer.infer(...) wants a key; just give zeros
        # (your JAX is old and CPU-only, this should be fine)
        key = None

        out, aux = system.infer(batch, fit_infer=fit_infer, key=key)

        # out is usually a list with 1 element shaped (B, T, 1)
        if isinstance(out, (list, tuple)):
            out = out[0]

        out = np.array(out)  # (1, T, 1)
        out = out[0, :, 0]   # -> (T,)
        return out

    return _run
