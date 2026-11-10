#!/usr/bin/env bash
# Centinela de recursos para entrenamientos GNN.
#
# Uso interno (lo lanza scripts/train/run_gnn.py):
#   resource_guard.sh PID RAM_LIMIT GPU_MEM_LIMIT GPU_UTIL_LIMIT INTERVAL STATUS_FILE
#
# - Muestrea cada INTERVAL segundos (0.5 por defecto).
# - No imprime muestras: solo escribe STATUS_FILE si se supera un límite.
# - Si se supera RAM o VRAM, termina el grupo de procesos del entrenamiento
#   y sale con código 42.

set -u

pid="${1:?falta PID}"
ram_limit="${2:-80}"
gpu_mem_limit="${3:-80}"
gpu_util_limit="${4:-}"
interval="${5:-0.5}"
status_file="${6:?falta STATUS_FILE}"

sample_ram() {
    awk '
        /^MemTotal:/ { total = $2 }
        /^MemAvailable:/ { available = $2 }
        END {
            if (total > 0) {
                printf "%.2f", 100.0 * (1.0 - available / total)
            }
        }
    ' /proc/meminfo 2>/dev/null
}

sample_gpu() {
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    nvidia-smi \
        --query-gpu=memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null |
    awk -F',' '
        {
            gsub(/^[ \t]+|[ \t]+$/, "", $1)
            gsub(/^[ \t]+|[ \t]+$/, "", $2)
            gsub(/^[ \t]+|[ \t]+$/, "", $3)
            used = $1 + 0
            total = $2 + 0
            util = $3 + 0
            if (total > 0) {
                pct = 100.0 * used / total
                if (pct > max_pct) {
                    max_pct = pct
                    max_used = used
                    max_total = total
                    max_util = util
                }
            }
        }
        END {
            if (max_pct >= 0) {
                printf "%.2f %.0f %.0f %.0f\n",
                    max_pct, max_used, max_total, max_util
            }
        }
    '
}

write_status() {
    printf '%s\n' "$1" > "$status_file" 2>/dev/null || true
}

kill_training() {
    # El entrenamiento se lanza en su propia sesión POSIX.
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    sleep 1
    kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
}

while kill -0 "$pid" 2>/dev/null; do
    ram="$(sample_ram)"
    if [[ -n "$ram" ]] && awk -v v="$ram" -v l="$ram_limit" \
        'BEGIN { exit !(v >= l) }'; then
        reason="RAM ${ram}% >= ${ram_limit}%"
        write_status "$reason"
        kill_training
        exit 42
    fi

    gpu="$(sample_gpu)"
    if [[ -n "$gpu" ]]; then
        read -r gpu_mem gpu_used gpu_total gpu_util <<< "$gpu"

        if awk -v v="$gpu_mem" -v l="$gpu_mem_limit" \
            'BEGIN { exit !(v >= l) }'; then
            reason="VRAM ${gpu_mem}% (${gpu_used}/${gpu_total} MB) >= ${gpu_mem_limit}%"
            write_status "$reason"
            kill_training
            exit 42
        fi

        if [[ -n "$gpu_util_limit" ]] && \
            awk -v v="$gpu_util" -v l="$gpu_util_limit" \
                'BEGIN { exit !(v >= l) }'; then
            reason="GPU util ${gpu_util}% >= ${gpu_util_limit}%"
            write_status "$reason"
            kill_training
            exit 42
        fi
    fi

    sleep "$interval"
done

exit 0
