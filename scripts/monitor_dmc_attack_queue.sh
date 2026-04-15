#!/bin/bash

set -euo pipefail

ROOT_DIR="/share/guozhix/WMAuotoattack"
TASK_FILE="/share/guozhix/sheeprl/dreamerv3_dmc_tasks_from_paper.txt"
TASK_ROOT="/share/guozhix/sheeprl/logs/runs/dreamer_v3"
SUBMIT_SCRIPT="${ROOT_DIR}/sgpu_dmc_attack_generic.sub"
STATE_DIR="${ROOT_DIR}/.queue_state"
STATE_FILE="${STATE_DIR}/dmc_attack_next_position.state"
SUBMIT_LOG="${STATE_DIR}/dmc_attack_submissions.log"
MONITOR_LOG="${STATE_DIR}/dmc_attack_monitor.log"
USER_NAME="${USER:-guozhix}"
START_POSITION="${START_POSITION:-1}"
MAX_ACTIVE_JOBS="${MAX_ACTIVE_JOBS:-2}"
POLL_SECONDS="${POLL_SECONDS:-60}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-ckpt_250000_0.ckpt}"
ATTACKS="${ATTACKS:-apgd_ce,apgd_dlr,fab,two_stage,square}"
LATENT_SEED_STORE="${LATENT_SEED_STORE:-/share/guozhix/WMAuotoattack/logs/attack_search/probe_scale_20_tasks_experience_2265.jsonl}"
ATTACKS_B64="$(printf '%s' "${ATTACKS}" | base64 -w0)"

mkdir -p "${STATE_DIR}"
touch "${SUBMIT_LOG}" "${MONITOR_LOG}"

if [[ ! -f "${STATE_FILE}" ]]; then
    echo "${START_POSITION}" > "${STATE_FILE}"
fi

read_next_position() {
    cat "${STATE_FILE}"
}

write_next_position() {
    printf '%s\n' "$1" > "${STATE_FILE}"
}

task_count() {
    awk '/^[0-9]+\./ {count++} END {print count+0}' "${TASK_FILE}"
}

task_line_by_index() {
    local index="$1"
    awk -v idx="${index}" '$1 == idx "." {print $2}' "${TASK_FILE}"
}

task_folder_name() {
    local spec="$1"
    case "${spec}" in
        cup_catch)
            echo "ball_in_cup_catch"
            ;;
        *)
            echo "${spec}"
            ;;
    esac
}

job_label() {
    local spec="$1"
    local label
    label="$(echo "${spec}" | tr '[:lower:]' '[:upper:]')"
    label="${label//ACROBOT_SWINGUP/Acrobot}"
    label="${label//CARTPOLE_BALANCE_SPARSE/CartBalSpr}"
    label="${label//CARTPOLE_BALANCE/CartBalance}"
    label="${label//CARTPOLE_SWINGUP_SPARSE/CartSwSp}"
    label="${label//CARTPOLE_SWINGUP/CartSwing}"
    label="${label//CHEETAH_RUN/CheetahRun}"
    label="${label//CUP_CATCH/CupCatch}"
    label="${label//FINGER_SPIN/FingerSpin}"
    label="${label//FINGER_TURN_EASY/FingTurnE}"
    label="${label//FINGER_TURN_HARD/FingTurnH}"
    label="${label//HOPPER_HOP/HopperHop}"
    label="${label//HOPPER_STAND/HopperStand}"
    label="${label//PENDULUM_SWINGUP/PendSwing}"
    label="${label//QUADRUPED_RUN/QuadRun}"
    label="${label//QUADRUPED_WALK/QuadWalk}"
    label="${label//REACHER_EASY/ReachEasy}"
    label="${label//REACHER_HARD/ReachHard}"
    label="${label//WALKER_RUN/WalkerRun}"
    label="${label//WALKER_STAND/WalkerStand}"
    label="${label//WALKER_WALK/WalkerWalk}"
    echo "${label}"
}

position_task_index() {
    local position="$1"
    echo $(( (position + 1) / 2 ))
}

position_mode() {
    local position="$1"
    if (( position % 2 == 1 )); then
        echo "random"
    else
        echo "latent"
    fi
}

active_job_count() {
    squeue -u "${USER_NAME}" -h -o "%j %T" | awk '$1 ~ /^WMA_DMC_/ && ($2 == "RUNNING" || $2 == "PENDING") {count++} END {print count+0}'
}

resolve_checkpoint() {
    local folder="$1"
    if [[ ! -d "${TASK_ROOT}/${folder}" ]]; then
        return 0
    fi
    find "${TASK_ROOT}/${folder}" -name "${CHECKPOINT_NAME}" | sort | head -n 1
}

submit_position() {
    local position="$1"
    local task_index spec folder mode checkpoint_path label job_name job_id

    task_index="$(position_task_index "${position}")"
    mode="$(position_mode "${position}")"
    spec="$(task_line_by_index "${task_index}")"
    folder="$(task_folder_name "${spec}")"
    checkpoint_path="$(resolve_checkpoint "${folder}")"

    if [[ -z "${checkpoint_path}" ]]; then
        printf '[%s] skipped position=%s task_index=%s spec=%s mode=%s reason=missing_checkpoint\n' \
            "$(date -Iseconds)" "${position}" "${task_index}" "${spec}" "${mode}" >> "${MONITOR_LOG}"
        return 0
    fi

    label="$(job_label "${spec}")"
    job_name="WMA_DMC_${label}_$(echo "${mode}" | tr '[:lower:]' '[:upper:]')"
    job_id="$(
        sbatch --parsable \
            -J "${job_name}" \
            --export=ALL,TASK_ROOT="${TASK_ROOT}",TASK_LABEL="${spec}",TASK_FOLDER="${folder}",CHECKPOINT_NAME="${CHECKPOINT_NAME}",MODE="${mode}",ATTACKS_B64="${ATTACKS_B64}",LATENT_SEED_STORE="${LATENT_SEED_STORE}" \
            "${SUBMIT_SCRIPT}"
    )"

    printf '[%s] submitted position=%s task_index=%s spec=%s mode=%s job_id=%s job_name=%s checkpoint=%s\n' \
        "$(date -Iseconds)" "${position}" "${task_index}" "${spec}" "${mode}" "${job_id}" "${job_name}" "${checkpoint_path}" \
        | tee -a "${SUBMIT_LOG}" >> "${MONITOR_LOG}"
}

TOTAL_TASKS="$(task_count)"
TOTAL_POSITIONS="$((TOTAL_TASKS * 2))"
printf '[%s] monitor started: start_position=%s total_tasks=%s total_positions=%s max_active_jobs=%s poll=%ss\n' \
    "$(date -Iseconds)" "${START_POSITION}" "${TOTAL_TASKS}" "${TOTAL_POSITIONS}" "${MAX_ACTIVE_JOBS}" "${POLL_SECONDS}" >> "${MONITOR_LOG}"

while true; do
    next_position="$(read_next_position)"
    active_jobs="$(active_job_count)"

    while [[ "${active_jobs}" -lt "${MAX_ACTIVE_JOBS}" && "${next_position}" -le "${TOTAL_POSITIONS}" ]]; do
        submit_position "${next_position}"
        next_position="$((next_position + 1))"
        write_next_position "${next_position}"
        active_jobs="$(active_job_count)"
    done

    if [[ "${next_position}" -gt "${TOTAL_POSITIONS}" && "${active_jobs}" -eq 0 ]]; then
        printf '[%s] queue completed: all positions processed and no active jobs remain\n' "$(date -Iseconds)" >> "${MONITOR_LOG}"
        exit 0
    fi

    sleep "${POLL_SECONDS}"
done
