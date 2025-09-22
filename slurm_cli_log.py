#!/usr/bin/env python3
# slurm_logs_cli.py
import argparse, datetime, os, sys
from pathlib import Path

def stamp_dir(kind: str) -> str:
    now = datetime.datetime.now()
    if kind == "date":
        return now.strftime("%Y-%m-%d")
    if kind == "datetime":
        return now.strftime("%Y-%m-%d_%H-%M-%S")
    return ""  # none

def ensure_dir(path: Path, create: bool):
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path

def cmd_mkpath(args):
    base = Path(args.root)
    stamped = base / stamp_dir(args.stamp) if args.stamp != "none" else base
    ensure_dir(stamped, not args.no_create)
    print(str(stamped.resolve()))

def build_paths(root: Path, stamp: str, prefix: str, ext_out: str, ext_err: str):
    d = root / (stamp_dir(stamp) if stamp != "none" else "")
    # Keep %j (jobid) in the filename; SLURM expands it at runtime
    ofile = d / f"%j_{prefix}.{ext_out}"
    efile = d / f"%j_{prefix}.{ext_err}"
    return d, ofile, efile

def cmd_sbatch_lines(args):
    d, ofile, efile = build_paths(Path(args.root), args.stamp, args.prefix, args.ext_out, args.ext_err)
    # Make sure the directory exists (so SLURM doesn't fail)
    d.mkdir(parents=True, exist_ok=True)
    print(f"#SBATCH --output={ofile}")
    print(f"#SBATCH --error={efile}")

def cmd_scaffold(args):
    d, ofile, efile = build_paths(Path(args.root), args.stamp, args.prefix, args.ext_out, args.ext_err)
    d.mkdir(parents=True, exist_ok=True)
    script = f"""#!/bin/bash -l
#SLURM SUBMIT SCRIPT
#SBATCH --account=ingenuitylabs
#SBATCH --partition=Northstar
#SBATCH --partition=Sasquatch
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output={ofile}
#SBATCH --error={efile}
#SBATCH --time=100:00:00

# Ensure the log directory exists (idempotent)
mkdir -p "{d}"

echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
 
# Set the environment variable to use the first GPU
export CUDA_VISIBLE_DEVICES=1
 
# Print GPU status using nvidia-smi
nvidia-smi
 
# test inference code
python3 inference.py --model_name ./models/gpt-oss-20b --prompt "Repeat after me: test test!" --use_4bit True

echo "Finished at: $(date)"

"""
    Path(args.script).write_text(script)
    os.chmod(args.script, 0o755)
    print(f"Wrote {args.script}")
    print("Tip: submit with `sbatch {}`".format(args.script))

def main():
    p = argparse.ArgumentParser(
        description="Create dated log folders and generate SBATCH lines for SLURM outputs."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    common = dict(
        root=dict(flags=["-r","--root"], default="slurm_jobs", help="Root folder for logs (default: slurm_jobs)"),
        stamp=dict(flags=["-s","--stamp"], choices=["date","datetime","none"], default="date",
                   help="Folder stamp: date (YYYY-MM-DD), datetime (…_HH-MM-SS), or none"),
        prefix=dict(flags=["-p","--prefix"], default="run",
                    help="Filename prefix after %j_ (default: run)"),
        ext_out=dict(flags=["--ext-out"], default="out", help="Output file extension (default: out)"),
        ext_err=dict(flags=["--ext-err"], default="err", help="Error file extension (default: err)"),
    )

    # mkpath
    mk = sub.add_parser("mkpath", help="Print (and optionally create) the log folder path.")
    mk.add_argument(*common["root"]["flags"], default=common["root"]["default"], help=common["root"]["help"])
    mk.add_argument(*common["stamp"]["flags"], choices=common["stamp"]["choices"], default=common["stamp"]["default"], help=common["stamp"]["help"])
    mk.add_argument("--no-create", action="store_true", help="Do not create the directory, just print the path.")
    mk.set_defaults(func=cmd_mkpath)

    # sbatch-lines
    sl = sub.add_parser("sbatch-lines", help="Print #SBATCH --output/--error lines and ensure folder exists.")
    sl.add_argument(*common["root"]["flags"], default=common["root"]["default"], help=common["root"]["help"])
    sl.add_argument(*common["stamp"]["flags"], choices=common["stamp"]["choices"], default=common["stamp"]["default"], help=common["stamp"]["help"])
    sl.add_argument(*common["prefix"]["flags"], default=common["prefix"]["default"], help=common["prefix"]["help"])
    sl.add_argument(*common["ext_out"]["flags"], default=common["ext_out"]["default"], help=common["ext_out"]["help"])
    sl.add_argument(*common["ext_err"]["flags"], default=common["ext_err"]["default"], help=common["ext_err"]["help"])
    sl.set_defaults(func=cmd_sbatch_lines)

    # scaffold
    sc = sub.add_parser("scaffold", help="Generate a minimal SLURM script that uses the dated log folder.")
    sc.add_argument("-j","--job-name", default="job", help="SLURM job name (default: job)")
    sc.add_argument(*common["root"]["flags"], default=common["root"]["default"], help=common["root"]["help"])
    sc.add_argument(*common["stamp"]["flags"], choices=common["stamp"]["choices"], default=common["stamp"]["default"], help=common["stamp"]["help"])
    sc.add_argument(*common["prefix"]["flags"], default=common["prefix"]["default"], help=common["prefix"]["help"])
    sc.add_argument(*common["ext_out"]["flags"], default=common["ext_out"]["default"], help=common["ext_out"]["help"])
    sc.add_argument(*common["ext_err"]["flags"], default=common["ext_err"]["default"], help=common["ext_err"]["help"])
    sc.add_argument("--script", default="submit.sbatch", help="Output script filename (default: submit.sbatch)")
    sc.set_defaults(func=cmd_scaffold)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
