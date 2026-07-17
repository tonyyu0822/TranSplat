#!/usr/bin/env bash
# TranSplat driver: train / relight / render / diagnose, or run the whole
# pipeline end to end. Every stage prints a banner and an explicit OK/FAILED
# result; `pipeline` stops immediately at the first failing stage and names
# it, instead of silently continuing.
#
# Usage:
#   scripts/run_transplat.sh <command> [options] [-- <extra args passed to the underlying python script>]
#
# Commands:
#   train       Train a model from a dataset
#   relight     Relight a trained model's point cloud (radiance_transfer_TranSplat.py)
#   render      Render images / extract a mesh from a trained (or relit) point cloud
#   visibility  Render the RGB directional-visibility diagnostic
#   demo        Render a rotating-envmap relighting demo video
#   pipeline    Run train -> relight -> render for one scene
#
# Run any command with -h/--help for its options.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_NAME="$(basename "$0")"

# ---------------------------------------------------------------------------
# Stage runner: prints a banner, runs the command, and reports OK/FAILED.
# On failure it prints which stage failed and returns that stage's exit code
# to the caller (callers that chain stages, e.g. `pipeline`, check this and
# stop instead of proceeding to the next stage).
# ---------------------------------------------------------------------------
run_stage() {
    local name="$1"
    shift
    echo ""
    echo "================================================================"
    echo "  STAGE: ${name}"
    echo "================================================================"
    echo "  + $*"
    echo ""
    if "$@"; then
        echo ""
        echo "  [OK] STAGE ${name} succeeded"
        return 0
    else
        local code=$?
        echo ""
        echo "  [FAILED] STAGE ${name} exited with code ${code}" >&2
        return "${code}"
    fi
}

check_conda_env() {
    local expected="${1:-surfel_splatting}"
    if [[ "${CONDA_DEFAULT_ENV:-}" != "${expected}" ]]; then
        echo "  [WARN] Expected conda env '${expected}', but CONDA_DEFAULT_ENV='${CONDA_DEFAULT_ENV:-<none>}'." >&2
        echo "         Run 'conda activate ${expected}' first if this was unintentional." >&2
    fi
}

usage() {
    sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
}

# Splits "$@" into named-flag args (before a literal "--") and passthrough
# args (after it). Result left in NAMED_ARGS / EXTRA_ARGS arrays.
split_extra_args() {
    NAMED_ARGS=()
    EXTRA_ARGS=()
    local seen_separator=false
    for arg in "$@"; do
        if [[ "${seen_separator}" == false && "${arg}" == "--" ]]; then
            seen_separator=true
            continue
        fi
        if [[ "${seen_separator}" == true ]]; then
            EXTRA_ARGS+=("${arg}")
        else
            NAMED_ARGS+=("${arg}")
        fi
    done
}

require_arg() {
    local flag="$1" value="${2:-}"
    if [[ -z "${value}" ]]; then
        echo "Missing required argument: ${flag}" >&2
        exit 2
    fi
}

# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
cmd_train() {
    split_extra_args "$@"
    if [[ ${#NAMED_ARGS[@]} -gt 0 ]]; then
        set -- "${NAMED_ARGS[@]}"
    else
        set --
    fi

    local source_path="" model_path="" gpu="0"
    local sh_degree="" sh_degree_up_interval="" lambda_normal="" lambda_dist=""
    while [[ "$#" -gt 0 ]]; do
        case "$1" in
            -s|--source-path) source_path="$2"; shift 2 ;;
            -m|--model-path)  model_path="$2"; shift 2 ;;
            --gpu)            gpu="$2"; shift 2 ;;
            --sh-degree)      sh_degree="$2"; shift 2 ;;
            --sh-degree-up-interval) sh_degree_up_interval="$2"; shift 2 ;;
            --lambda-normal)  lambda_normal="$2"; shift 2 ;;
            --lambda-dist)    lambda_dist="$2"; shift 2 ;;
            -h|--help)
                cat <<'EOF'
Usage: run_transplat.sh train -s <dataset> -m <output dir> [options] [-- <extra train.py args>]

  -s, --source-path PATH        Dataset path (COLMAP or NeRF Synthetic)         [required]
  -m, --model-path PATH         Output model directory                         [required]
      --gpu N                   CUDA_VISIBLE_DEVICES value (default: 0)
      --sh-degree N             Max SH degree
      --sh-degree-up-interval N Increase active SH degree every N iters (default: 1000)
      --lambda-normal F         Normal consistency loss weight
      --lambda-dist F           Depth distortion loss weight
  -- ...                        Extra args forwarded verbatim to train.py
EOF
                return 0 ;;
            *) echo "Unknown train option: $1" >&2; return 2 ;;
        esac
    done
    require_arg "-s/--source-path" "${source_path}"
    require_arg "-m/--model-path" "${model_path}"

    local args=(-s "${source_path}" -m "${model_path}")
    [[ -n "${sh_degree}" ]] && args+=(--sh_degree "${sh_degree}")
    [[ -n "${sh_degree_up_interval}" ]] && args+=(--sh_degree_up_interval "${sh_degree_up_interval}")
    [[ -n "${lambda_normal}" ]] && args+=(--lambda_normal "${lambda_normal}")
    [[ -n "${lambda_dist}" ]] && args+=(--lambda_dist "${lambda_dist}")
    [[ ${#EXTRA_ARGS[@]} -gt 0 ]] && args+=("${EXTRA_ARGS[@]}")

    check_conda_env
    CUDA_VISIBLE_DEVICES="${gpu}" python "${REPO_DIR}/train.py" "${args[@]}"
}

# ---------------------------------------------------------------------------
# relight
# ---------------------------------------------------------------------------
cmd_relight() {
    split_extra_args "$@"
    if [[ ${#NAMED_ARGS[@]} -gt 0 ]]; then
        set -- "${NAMED_ARGS[@]}"
    else
        set --
    fi

    local ply="" source_env="" target_env="" output="" gpu="0"
    local floor_alpha="" tau_max="" specular_boost="" num_samples="" shadow_res=""
    while [[ "$#" -gt 0 ]]; do
        case "$1" in
            --ply)             ply="$2"; shift 2 ;;
            --source-env)      source_env="$2"; shift 2 ;;
            --target-env)      target_env="$2"; shift 2 ;;
            -o|--output)       output="$2"; shift 2 ;;
            --gpu)             gpu="$2"; shift 2 ;;
            --floor-alpha)     floor_alpha="$2"; shift 2 ;;
            --tau-max)         tau_max="$2"; shift 2 ;;
            --specular-boost)  specular_boost="$2"; shift 2 ;;
            --num-samples)     num_samples="$2"; shift 2 ;;
            --shadow-res)      shadow_res="$2"; shift 2 ;;
            -h|--help)
                cat <<'EOF'
Usage: run_transplat.sh relight --ply <input.ply> --source-env <src.hdr> --target-env <tgt.hdr> -o <output.ply> [options]

      --ply PATH          Trained Gaussian point cloud to relight  [required]
      --source-env PATH   Source HDR environment map               [required]
      --target-env PATH   Target HDR environment map                [required]
  -o, --output PATH       Where to write the relit .ply             [required]
      --gpu N             CUDA_VISIBLE_DEVICES value (default: 0)
      --floor-alpha F     Diffuse denominator floor (default: 0.05)
      --tau-max F         Diffuse transfer clamp (default: 3.0)
      --specular-boost F  Specular SH rest-band scale (default: 1.0)
      --num-samples N     Fibonacci SH samples (default: 1200)
      --shadow-res N      Shadow-map bake resolution (default: 1024)
EOF
                return 0 ;;
            *) echo "Unknown relight option: $1" >&2; return 2 ;;
        esac
    done
    require_arg "--ply" "${ply}"
    require_arg "--source-env" "${source_env}"
    require_arg "--target-env" "${target_env}"
    require_arg "-o/--output" "${output}"

    local args=(--source "${source_env}" --target "${target_env}" --ply "${ply}" --output "${output}")
    [[ -n "${floor_alpha}" ]] && args+=(--floor_alpha "${floor_alpha}")
    [[ -n "${tau_max}" ]] && args+=(--tau_max "${tau_max}")
    [[ -n "${specular_boost}" ]] && args+=(--specular_boost "${specular_boost}")
    [[ -n "${num_samples}" ]] && args+=(--num_samples "${num_samples}")
    [[ -n "${shadow_res}" ]] && args+=(--shadow_res "${shadow_res}")
    [[ ${#EXTRA_ARGS[@]} -gt 0 ]] && args+=("${EXTRA_ARGS[@]}")

    check_conda_env
    CUDA_VISIBLE_DEVICES="${gpu}" python "${REPO_DIR}/radiance_transfer_TranSplat.py" "${args[@]}"
}

# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------
cmd_render() {
    split_extra_args "$@"
    if [[ ${#NAMED_ARGS[@]} -gt 0 ]]; then
        set -- "${NAMED_ARGS[@]}"
    else
        set --
    fi

    local source_path="" model_path="" ply="" iteration="" gpu="0"
    local skip_train=false skip_test=false skip_mesh=false
    while [[ "$#" -gt 0 ]]; do
        case "$1" in
            -s|--source-path) source_path="$2"; shift 2 ;;
            -m|--model-path)  model_path="$2"; shift 2 ;;
            --ply)            ply="$2"; shift 2 ;;
            --iteration)      iteration="$2"; shift 2 ;;
            --gpu)            gpu="$2"; shift 2 ;;
            --skip-train)     skip_train=true; shift ;;
            --skip-test)      skip_test=true; shift ;;
            --skip-mesh)      skip_mesh=true; shift ;;
            -h|--help)
                cat <<'EOF'
Usage: run_transplat.sh render -s <dataset> -m <model dir> [--ply <point_cloud.ply>] [options]

  -s, --source-path PATH   Dataset path                                    [required]
  -m, --model-path PATH    Trained model directory                         [required]
      --ply PATH           Render this point cloud instead of the trained
                            iteration's point_cloud.ply (e.g. a relit .ply)
      --iteration N        Iteration to load when --ply is not given (default: latest)
      --gpu N              CUDA_VISIBLE_DEVICES value (default: 0)
      --skip-train         Skip exporting training-view renders
      --skip-test          Skip exporting test-view renders
      --skip-mesh          Skip mesh extraction
EOF
                return 0 ;;
            *) echo "Unknown render option: $1" >&2; return 2 ;;
        esac
    done
    require_arg "-s/--source-path" "${source_path}"
    require_arg "-m/--model-path" "${model_path}"

    local args=(-s "${source_path}" -m "${model_path}")
    [[ -n "${ply}" ]] && args+=(--ply "${ply}")
    [[ -n "${iteration}" ]] && args+=(--iteration "${iteration}")
    [[ "${skip_train}" == true ]] && args+=(--skip_train)
    [[ "${skip_test}" == true ]] && args+=(--skip_test)
    [[ "${skip_mesh}" == true ]] && args+=(--skip_mesh)
    [[ ${#EXTRA_ARGS[@]} -gt 0 ]] && args+=("${EXTRA_ARGS[@]}")

    check_conda_env
    CUDA_VISIBLE_DEVICES="${gpu}" python "${REPO_DIR}/render.py" "${args[@]}"
}

# ---------------------------------------------------------------------------
# visibility
# ---------------------------------------------------------------------------
cmd_visibility() {
    split_extra_args "$@"
    if [[ ${#NAMED_ARGS[@]} -gt 0 ]]; then
        set -- "${NAMED_ARGS[@]}"
    else
        set --
    fi

    local dataset="" model="" iteration="" train_view="" output_dir="" gpu="0"
    while [[ "$#" -gt 0 ]]; do
        case "$1" in
            --dataset)    dataset="$2"; shift 2 ;;
            --model)      model="$2"; shift 2 ;;
            --iteration)  iteration="$2"; shift 2 ;;
            --train-view) train_view="$2"; shift 2 ;;
            --output-dir) output_dir="$2"; shift 2 ;;
            --gpu)        gpu="$2"; shift 2 ;;
            -h|--help)
                cat <<'EOF'
Usage: run_transplat.sh visibility --dataset <path> --model <model dir> [options]

      --dataset PATH    Dataset used to train --model     [required]
      --model PATH      Trained model directory            [required]
      --iteration N     Checkpoint iteration (default: 30000)
      --train-view N    Zero-based training camera index (default: 0)
      --output-dir PATH Defaults to <model>/directional_visibility/train_view_<view>
      --gpu N           CUDA_VISIBLE_DEVICES value (default: 0)
EOF
                return 0 ;;
            *) echo "Unknown visibility option: $1" >&2; return 2 ;;
        esac
    done
    require_arg "--dataset" "${dataset}"
    require_arg "--model" "${model}"

    local args=(--dataset "${dataset}" --model "${model}" --gpu "${gpu}")
    [[ -n "${iteration}" ]] && args+=(--iteration "${iteration}")
    [[ -n "${train_view}" ]] && args+=(--train-view "${train_view}")
    [[ -n "${output_dir}" ]] && args+=(--output-dir "${output_dir}")
    [[ ${#EXTRA_ARGS[@]} -gt 0 ]] && args+=("${EXTRA_ARGS[@]}")

    check_conda_env
    python "${REPO_DIR}/scripts/render_directional_visibility.py" "${args[@]}"
}

# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------
cmd_demo() {
    split_extra_args "$@"
    if [[ ${#NAMED_ARGS[@]} -gt 0 ]]; then
        set -- "${NAMED_ARGS[@]}"
    else
        set --
    fi

    local dataset="" model="" source_env="" target_env="" iteration="" output_dir="" gpu="0"
    while [[ "$#" -gt 0 ]]; do
        case "$1" in
            --dataset)    dataset="$2"; shift 2 ;;
            --model)      model="$2"; shift 2 ;;
            --source-env) source_env="$2"; shift 2 ;;
            --target-env) target_env="$2"; shift 2 ;;
            --iteration)  iteration="$2"; shift 2 ;;
            --output-dir) output_dir="$2"; shift 2 ;;
            --gpu)        gpu="$2"; shift 2 ;;
            -h|--help)
                cat <<'EOF'
Usage: run_transplat.sh demo --dataset <path> --model <model dir> --source-env <src.hdr> --target-env <tgt.hdr> [options]

      --dataset PATH    Dataset used to train --model     [required]
      --model PATH      Trained model directory            [required]
      --source-env PATH Source HDR environment map          [required]
      --target-env PATH Target HDR environment map to rotate around the object [required]
      --iteration N     Checkpoint iteration (default: 30000)
      --output-dir PATH Defaults to <model>/rotating_light_demo/<target-env stem>
      --gpu N           CUDA_VISIBLE_DEVICES value (default: 0)
EOF
                return 0 ;;
            *) echo "Unknown demo option: $1" >&2; return 2 ;;
        esac
    done
    require_arg "--dataset" "${dataset}"
    require_arg "--model" "${model}"
    require_arg "--source-env" "${source_env}"
    require_arg "--target-env" "${target_env}"

    local args=(--dataset "${dataset}" --model "${model}" --source-env "${source_env}" --target-env "${target_env}" --gpu "${gpu}")
    [[ -n "${iteration}" ]] && args+=(--iteration "${iteration}")
    [[ -n "${output_dir}" ]] && args+=(--output-dir "${output_dir}")
    [[ ${#EXTRA_ARGS[@]} -gt 0 ]] && args+=("${EXTRA_ARGS[@]}")

    check_conda_env
    python "${REPO_DIR}/scripts/make_rotating_light_demo.py" "${args[@]}"
}

# ---------------------------------------------------------------------------
# pipeline: train -> relight -> render, stopping and naming the failed stage
# ---------------------------------------------------------------------------
cmd_pipeline() {
    local source_path="" model_path="" source_env="" target_env="" iteration="30000" gpu="0"
    local skip_train_stage=false
    while [[ "$#" -gt 0 ]]; do
        case "$1" in
            -s|--source-path)   source_path="$2"; shift 2 ;;
            -m|--model-path)    model_path="$2"; shift 2 ;;
            --source-env)       source_env="$2"; shift 2 ;;
            --target-env)       target_env="$2"; shift 2 ;;
            --iteration)        iteration="$2"; shift 2 ;;
            --gpu)              gpu="$2"; shift 2 ;;
            --skip-train-stage) skip_train_stage=true; shift ;;
            -h|--help)
                cat <<'EOF'
Usage: run_transplat.sh pipeline -s <dataset> -m <model dir> --source-env <src.hdr> --target-env <tgt.hdr> [options]

Runs train -> relight -> render for one scene. Stops immediately (with the
stage clearly named) if any stage fails.

  -s, --source-path PATH  Dataset path                                     [required]
  -m, --model-path PATH   Output/model directory                            [required]
      --source-env PATH   Source HDR environment map                       [required]
      --target-env PATH   Target HDR environment map                        [required]
      --iteration N       Checkpoint iteration (default: 30000)
      --gpu N             CUDA_VISIBLE_DEVICES value (default: 0)
      --skip-train-stage  Reuse an existing checkpoint at --model-path instead of training
EOF
                return 0 ;;
            *) echo "Unknown pipeline option: $1" >&2; return 2 ;;
        esac
    done
    require_arg "-s/--source-path" "${source_path}"
    require_arg "-m/--model-path" "${model_path}"
    require_arg "--source-env" "${source_env}"
    require_arg "--target-env" "${target_env}"

    local ply_dir="${model_path}/point_cloud/iteration_${iteration}"
    local trained_ply="${ply_dir}/point_cloud.ply"
    local target_stem
    target_stem="$(basename "${target_env}")"
    target_stem="${target_stem%.*}"
    local relit_ply="${ply_dir}/relit_${target_stem}.ply"

    if [[ "${skip_train_stage}" == true ]]; then
        echo "Skipping TRAIN stage (--skip-train-stage); expecting checkpoint at ${trained_ply}"
    else
        run_stage "TRAIN" cmd_train -s "${source_path}" -m "${model_path}" --gpu "${gpu}" || return $?
    fi

    run_stage "RELIGHT" cmd_relight --ply "${trained_ply}" --source-env "${source_env}" \
        --target-env "${target_env}" --output "${relit_ply}" --gpu "${gpu}" || return $?

    run_stage "RENDER" cmd_render -s "${source_path}" -m "${model_path}" --ply "${relit_ply}" \
        --skip-mesh --gpu "${gpu}" || return $?

    echo ""
    echo "================================================================"
    echo "  PIPELINE COMPLETE"
    echo "================================================================"
    echo "  Trained checkpoint : ${trained_ply}"
    echo "  Relit point cloud  : ${relit_ply}"
    echo "  Renders            : ${model_path}/train, ${model_path}/test"
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
if [[ "$#" -lt 1 ]]; then
    usage
    exit 1
fi

command="$1"
shift

case "${command}" in
    train)      cmd_train "$@" ;;
    relight)    cmd_relight "$@" ;;
    render)     cmd_render "$@" ;;
    visibility) cmd_visibility "$@" ;;
    demo)       cmd_demo "$@" ;;
    pipeline)   cmd_pipeline "$@" ;;
    -h|--help)  usage; exit 0 ;;
    *)
        echo "Unknown command: ${command}" >&2
        usage
        exit 1
        ;;
esac
