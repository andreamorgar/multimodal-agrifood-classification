"""
Script para añadir datasets en paralelo usando 2 GPUs:
- GPU 0: prompts genéricos
- GPU 1: prompts específicos

Uso:
    python add_dataset_parallel.py --datasets crops
"""

import os
import sys
import json
import torch
import argparse
import subprocess
import time

def run_on_gpu(dataset, ensemble_type, gpu_id, log_file):
    """Ejecuta add_dataset.py con un ensemble_type específico en una GPU."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    cmd = [
        sys.executable, 
        "add_dataset_single.py",
        "--datasets", dataset,
        "--ensemble", ensemble_type
    ]
    
    with open(log_file, 'w') as f:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT
        )
    
    return proc


def main():
    parser = argparse.ArgumentParser(description="Añadir datasets en paralelo (2 GPUs)")
    parser.add_argument("--datasets", nargs="+", required=True,
                       help="Datasets a añadir (ej: --datasets crops)")
    args = parser.parse_args()
    
    print("="*80)
    print("AÑADIR DATASETS EN PARALELO - 2 GPUs")
    print("="*80)
    print(f"\nGPU 0: prompts genéricos")
    print(f"GPU 1: prompts específicos")
    
    processes = []
    
    for dataset in args.datasets:
        print(f"\n→ Lanzando: {dataset}")
        
        # GPU 0: genéricos
        log_generic = f"classification_generic_{dataset}.log"
        proc_generic = run_on_gpu(dataset, "generic", 0, log_generic)
        processes.append((proc_generic, "generic", log_generic))
        print(f"  ✓ Generic en GPU 0 (log: {log_generic})")
        
        # GPU 1: específicos
        log_specific = f"classification_specific_{dataset}.log"
        proc_specific = run_on_gpu(dataset, "specific", 1, log_specific)
        processes.append((proc_specific, "specific", log_specific))
        print(f"  ✓ Specific en GPU 1 (log: {log_specific})")
    
    # Esperar a que terminen todos
    print(f"\n→ Esperando a que terminen los procesos...")
    print(f"  Puedes monitorear con: tail -f classification_*.log")
    
    for proc, etype, logf in processes:
        proc.wait()
        if proc.returncode == 0:
            print(f"  ✓ {etype} completado")
        else:
            print(f"  ✗ {etype} falló (ver {logf})")
    
    print("\n✓ TODOS LOS PROCESOS COMPLETADOS")


if __name__ == "__main__":
    main()
