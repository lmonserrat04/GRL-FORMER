import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from nilearn import datasets


def download_single_atlas(pipe: str, roi: str, use_fg: bool, out_dir: str) -> str:
    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 [HILO] Iniciando: Atlas={roi} | Pipeline={pipe} | Carpeta={target_dir}")

    try:
        datasets.fetch_abide_pcp(
            data_dir=str(target_dir),
            pipeline=pipe,
            derivatives=[f"rois_{roi}"],
            band_pass_filtering=use_fg,
            global_signal_regression=False,
            verbose=1
        )
        return f"✅ {roi}: Descargado en {target_dir}"
    except Exception as e:
        return f"❌ {roi}: Error → {str(e)[:150]}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Descarga ABIDE PCP via nilearn")

    parser.add_argument("--pipe",    type=str, default="cpac",
                        choices=["cpac", "dparsf"],
                        help="Pipeline (default: cpac)")
    parser.add_argument("--out_dir", type=str, default="./data/datasets/",
                        help="Directorio de salida (default: ./data/datasets/)")
    parser.add_argument("--atlas",   type=str, default=None,
                        choices=["cc200", "aal", "dosenbach160"],
                        help="Atlas a descargar (default: los tres)")
    parser.add_argument("--fg",    action="store_true",  default=True,
                        help="Band pass filtering (default: True)")
    parser.add_argument("--no-fg", action="store_false", dest="fg",
                        help="Desactivar band pass filtering")

    args = parser.parse_args()
    rois = [args.atlas] if args.atlas else ["cc200", "aal", "dosenbach160"]

    print(f"📂 Directorio de salida : {args.out_dir}")
    print(f"🔧 Pipeline             : {args.pipe}")
    print(f"🔧 Band pass filtering  : {args.fg}")
    print(f"🌐 Atlas                : {rois}")
    print("-" * 40)

    with ThreadPoolExecutor(max_workers=len(rois)) as executor:
        futures = [
            executor.submit(download_single_atlas, args.pipe, roi, args.fg, args.out_dir)
            for roi in rois
        ]
        results = [f.result() for f in futures]

    print("\n--- Estado Final ---")
    for res in results:
        print(res)