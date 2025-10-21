#!/usr/bin/env python3
import os
import shutil
import sys

def safe_symlink(src, dest):
    """Create or replace a symlink safely."""
    if os.path.exists(dest) or os.path.islink(dest):
        os.remove(dest)
    os.symlink(os.path.abspath(src), dest)
    print(f"Linked {src} -> {dest}")

def main():
    if len(sys.argv) != 5:
        print("Usage: python3 setup_build_dir.py <path_to_src> <new_dir_name> <model_name> <evolve_name>")
        sys.exit(1)

    src_root = os.path.abspath(sys.argv[1])
    new_dir = sys.argv[2]
    model_name = sys.argv[3]
    evolve_name = sys.argv[4]

    # --- check src directory exists ---
    if not os.path.isdir(src_root):
        print(f"Error: source directory '{src_root}' not found.")
        sys.exit(1)

    # --- create new directory ---
    os.makedirs(new_dir, exist_ok=True)

    # --- fixed files to link ---
    fixed_files = [
        ("./", "run.cu"),
        ("./", "run.h"),
        ("./", "start.cu"),
        ("./", "start.h"),
        ("fft", "fft_utils.cu"),
        ("fft", "fft_utils.h"),
        ("random", "random.cu"),
        ("random", "random.h"),
        ("misc", "misc.cu"),
        ("misc", "misc.h"),
        ("io", "io.h"),
        ("io", "io.cu"),
        ("initcond", "initcond.h"),
        ("initcond", "initcond.cu"),
    ]

    for sub, fname in fixed_files:
        src_file = os.path.join(src_root, sub, fname)
        if not os.path.exists(src_file):
            print(f"Warning: expected file {src_file} not found, skipping.")
            continue
        dest_link = os.path.join(new_dir, fname)
        safe_symlink(src_file, dest_link)

    # --- handle model files ---
    model_dir = os.path.join(src_root, "model")
    model_cu = os.path.join(model_dir, f"{model_name}.cu")
    model_h = os.path.join(model_dir, f"{model_name}.h")

    if not (os.path.exists(model_cu) and os.path.exists(model_h)):
        print(f"Error: model '{model_name}' not found in {model_dir}.")
        print(f"Expected both {model_name}.cu and {model_name}.h.")
        sys.exit(1)

    safe_symlink(model_cu, os.path.join(new_dir, "model.cu"))
    safe_symlink(model_h, os.path.join(new_dir, "model.h"))

    # --- handle evolve files ---
    evolve_dir = os.path.join(src_root, "evolve")
    evolve_cu = os.path.join(evolve_dir, f"{evolve_name}.cu")
    evolve_h = os.path.join(evolve_dir, f"{evolve_name}.h")

    if not (os.path.exists(evolve_cu) and os.path.exists(evolve_h)):
        print(f"Error: evolve '{evolve_name}' not found in {evolve_dir}.")
        print(f"Expected both {evolve_name}.cu and {evolve_name}.h.")
        sys.exit(1)

    safe_symlink(evolve_cu, os.path.join(new_dir, "evolve.cu"))
    safe_symlink(evolve_h, os.path.join(new_dir, "evolve.h"))

    # --- copy Makefile ---
    makefile_src = os.path.join(src_root, "Makefile")
    makefile_dst = os.path.join(new_dir, "Makefile")
    if os.path.exists(makefile_src):
        shutil.copy(makefile_src, makefile_dst)
        print(f"Copied {makefile_src} -> {makefile_dst}")
    else:
        print(f"Warning: Makefile not found in {src_root}")

    print(f"\n✅ Setup complete for model '{model_name}' and evolve '{evolve_name}'.")
    print(f"   You can now build inside '{new_dir}'.")

if __name__ == "__main__":
    main()

