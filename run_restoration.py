from pathlib import Path
from zipfile import ZipFile
import io, json, time, hashlib, importlib.util
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from scipy import ndimage as ndi
from skimage.measure import label, regionprops_table
from skimage.filters import threshold_otsu
from skimage.feature import peak_local_max
from skimage.segmentation import watershed, clear_border, relabel_sequential

ROOT = Path(__file__).resolve().parent
weights = ROOT / "models/deblur_nuclei"
weight = weights


def load_model():
    torch.set_num_threads(4)
    torch.manual_seed(20261010)
    spec = importlib.util.spec_from_file_location(
        "standalone_cpnet", ROOT / "vendor/cellpose/resnet_torch.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    net = module.CPnet([1, 32, 64, 128, 256], 1, 3)
    net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    net.eval()
    return net


def normalize(a):
    lo, hi = np.percentile(a, [1, 99])
    return ((a - lo) / max(hi - lo, 1e-6)).astype(np.float32)


def interior(a):
    return relabel_sequential(clear_border(a))[0]


def props(a):
    return regionprops_table(a, properties=["label", "area", "eccentricity"])


def segment(a, min_distance, min_area):
    x = normalize(a)
    fg = x > threshold_otsu(x)
    fg = ndi.binary_fill_holes(fg)
    dist = ndi.distance_transform_edt(fg)
    peaks = peak_local_max(
        dist,
        min_distance=min_distance,
        threshold_abs=3,
        labels=fg.astype(np.uint8),
        exclude_border=False,
    )
    markers = np.zeros(x.shape, np.int32)
    if len(peaks):
        markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    seg = watershed(-dist, markers, mask=fg)
    areas = np.bincount(seg.ravel())
    small = np.flatnonzero(areas < min_area)
    seg[np.isin(seg, small)] = 0
    return interior(seg)


@torch.inference_mode()
def restore(a, rescale, net):
    x = torch.from_numpy(normalize(a))[None, None]
    h, w = a.shape
    sh, sw = int(round(h * rescale)), int(round(w * rescale))
    x = F.interpolate(x, size=(sh, sw), mode="bilinear", align_corners=False)
    py, px = (-sh) % 16, (-sw) % 16
    x = F.pad(x, (0, px, 0, py), mode="reflect")
    y = net(x)[0][..., :sh, :sw]
    y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
    return y[0, 0].numpy()


def evaluate(gt, pred, ecc_threshold):
    gp, pp = props(gt), props(pred)
    ng, npred = int(gt.max()), int(pred.max())
    # Check that IoU matches are one-to-one.
    joint = np.bincount(
        (gt.astype(np.int64) * (npred + 1) + pred).ravel(), minlength=(ng + 1) * (npred + 1)
    ).reshape(ng + 1, npred + 1)
    ga = np.bincount(gt.ravel(), minlength=ng + 1)
    pa = np.bincount(pred.ravel(), minlength=npred + 1)
    union = ga[1:, None] + pa[None, 1:] - joint[1:, 1:]
    iou = np.divide(joint[1:, 1:], union, out=np.zeros_like(union, dtype=float), where=union > 0)
    gi, pi = np.nonzero(iou >= 0.5)
    assert len(set(gi)) == len(gi) and len(set(pi)) == len(pi)
    rare = np.asarray(gp["eccentricity"]) >= ecc_threshold
    found = np.zeros(ng, bool)
    found[gi] = True
    errors = np.abs(gp["eccentricity"][gi] - pp["eccentricity"][pi])
    errmap = {int(g + 1): float(e) for g, e in zip(gi, errors)}
    return dict(
        gt=ng,
        pred=npred,
        tp=len(gi),
        rare_n=int(rare.sum()),
        rare_tp=int((rare & found).sum()),
        typical_n=int((~rare).sum()),
        typical_tp=int(((~rare) & found).sum()),
        rare_error_sum=float(errors[rare[gi]].sum()),
        rare_error_n=int(rare[gi].sum()),
        typical_error_sum=float(errors[~rare[gi]].sum()),
        typical_error_n=int((~rare[gi]).sum()),
        matched_errors=errmap,
        rare_ids=(np.flatnonzero(rare) + 1).tolist(),
    )


def main():
    net = load_model()
    with ZipFile(ROOT / "bbbc039/metadata.zip") as meta, ZipFile(
        ROOT / "bbbc039/masks.zip"
    ) as masks, ZipFile(ROOT / "bbbc039/images.zip") as images:
        splits = {
            s: meta.read("metadata/" + s + ".txt").decode().splitlines()
            for s in ["training", "validation", "test"]
        }
        training_ecc = []
        training_area = []
        for name in splits["training"]:
            a = np.asarray(Image.open(io.BytesIO(masks.read("masks/" + name))))[:, :, 0]
            p = props(interior(label(a, connectivity=2)))
            training_ecc.extend(p["eccentricity"])
            training_area.extend(p["area"])
        threshold = float(np.quantile(training_ecc, 0.9))
        median_area = float(np.median(training_area))
        diameter = float(np.sqrt(4 * median_area / np.pi))
        min_distance = int(np.ceil(diameter / 4))
        min_area = int(np.floor(median_area / 8))
        config = dict(
            rare_threshold=threshold,
            median_area=median_area,
            diameter=diameter,
            rescale=17 / diameter,
            min_distance=min_distance,
            min_area=min_area,
            blur_sigmas=[1, 2, 3],
            iou_threshold=0.5,
            normalization_percentiles=[1, 99],
            inference="full frame; bilinear resizing; reflect padding to multiple of 16",
            torch_threads=4,
            model="Cellpose3 deblur_nuclei",
            model_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
            data_overlap_with_pretrained_model="not excluded",
            seed=20261010,
        )
        (ROOT / "restoration_config.json").write_text(json.dumps(config, indent=2))
        print(json.dumps(config, indent=2), flush=True)
        rows = []
        start = time.time()
        for k, name in enumerate(splits["test"]):
            gt_raw = np.asarray(Image.open(io.BytesIO(masks.read("masks/" + name))))[:, :, 0]
            gt = interior(label(gt_raw, connectivity=2))
            image = np.asarray(
                Image.open(io.BytesIO(images.read("images/" + Path(name).with_suffix(".tif").name)))
            ).astype(np.float32)
            image = normalize(image)
            base = evaluate(gt, segment(image, min_distance, min_area), threshold)
            rows.append(dict(image=name, condition="original", sigma=0, **base))
            for sigma in config["blur_sigmas"]:
                blurred = ndi.gaussian_filter(image, sigma=sigma, mode="reflect")
                restored = restore(blurred, config["rescale"], net)
                for condition, a in [("blurred", blurred), ("restored", restored)]:
                    rows.append(
                        dict(
                            image=name,
                            condition=condition,
                            sigma=sigma,
                            **evaluate(gt, segment(a, min_distance, min_area), threshold),
                        )
                    )
            (ROOT / "restoration_rows.json").write_text(json.dumps(rows))
            print(f"{k+1}/50 images, {time.time()-start:.1f}s", flush=True)
        assert sum(r["gt"] for r in rows if r["condition"] == "original") == 4816
        assert sum(r["rare_n"] for r in rows if r["condition"] == "original") == 498
    print("Completed test inference.", flush=True)


if __name__ == "__main__":
    main()
