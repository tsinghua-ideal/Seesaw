#!/bin/bash
# shellcheck disable=SC2120
function run_proxylessnas() {
  model=$1
  lossType=$2
  wid=$3
  count=$5
  batch=$6
  echo start "${model}" "${lossType}" "$wid" "${count}"
  dir=./checkpoints/replace/"${model}"/"${count}"/"${lossType}"
  #  search
  mkdir -p "${dir}"
  python main.py  \
  --net "${model}" \
  --dataset imagenet \
  --data_path ~/data/ \
  --grad_reg_loss_type "${lossType}" \
  --worker_id "$wid" \
  --pretrained \
  --epochs 50 \
  --train_batch_size "${batch}" \
  --expect_latency_rate "${count}" \
  --checkpoint_path "${dir}"/arch_path.pt \
  --exported_arch_path "${dir}"/checkpoint2.json \
  --train_mode "$4" \
  --kd_teacher_path ~/projects/nonlinearNAS/checkpoints/teacher/d2_224_85.2.pth.tar
}
function run_spatial() {
  model=$1
  lossType=$2
  wid=$3
  count=$5
  batch=$6
  echo start "${model}" "${lossType}" "$wid" "${count}"
  dir=./checkpoints/replace/"${model}"/"${count}"/"${lossType}"
  #  search
  mkdir -p "${dir}"
  python main.py  \
  --net "${model}" \
  --dataset imagenet \
  --data_path ~/data/ \
  --grad_reg_loss_type "${lossType}" \
  --worker_id "$wid" \
  --pretrained \
  --epochs 50 \
  --train_batch_size "${batch}" \
  --expect_latency_rate "${count}" \
  --checkpoint_path "${dir}"/arch_path.pt \
  --exported_arch_path "${dir}"/checkpoint2.json \
  --train_mode "$4" \
  --spatial \
  --kd_teacher_path ~/projects/nonlinearNAS/checkpoints/teacher/d2_224_85.2.pth.tar
}

# Parse script arguments
for arg in "$@"
do
    if [ "$arg" == "--spatial" ]
    then
        # Run run_spatial if --spatial is among the arguments
        run_spatial "$1" add#linear 0,1,2,3,4,5,6,7 "$2" "$3" "$4"
        exit 0
    fi
done

# Run run_proxylessnas if --spatial is not among the arguments
run_proxylessnas "$1" add#linear 0 "$2" "$3" "$4"
