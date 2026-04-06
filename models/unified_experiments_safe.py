"""
Wrapper seguro para escritura paralela en JSONL.
Añade locks a append_samples_to_jsonl para evitar corrupción.
"""

import json
import fcntl
import os


def append_samples_to_jsonl_safe(sample_rows, output_jsonl):
    """Guarda samples en formato JSONL con lock para escritura segura en paralelo."""
    lock_file = output_jsonl + ".lock"
    
    # Crear archivo si no existe
    if not os.path.exists(output_jsonl):
        open(output_jsonl, 'a').close()
    
    # Lock exclusivo para escritura
    with open(lock_file, 'w') as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        
        try:
            with open(output_jsonl, "a", encoding="utf-8") as f:
                for row in sample_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        finally:
            fcntl.flock(lockf, fcntl.LOCK_UN)
