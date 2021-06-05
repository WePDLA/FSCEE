#!/usr/bin/env bash
echo usage: pass gpu id list as param, split with ,
echo eg: source run_main.sh 3 snips OR source run_main.sh 3 ner

echo log file path  /home/feng/code/MetaEE-master/zh_trigger_result/


gpu_list=$1

# Comment one of follow 2 to switch debugging status
#do_debug=--do_debug
do_debug=

# ======= dataset setting ======
dataset_lst=(zh_trigger_5 zh_trigger_1)
support_shots_lst=(5 1)  # 5-shot
# data_batch_size=20
# word_piece_data=True


# Cross evaluation's data
cross_data_id_lst=(1)  # for debug


# ====== train & test setting ======
seed_lst=(10150 10151 10152 10153 10154)
# seed_lst=(10150 10151 10152 10153 10154 10155 10156 10157 10158 10159)

# lr_lst=(0.000001 0.000005 0.00005)
lr_lst=(0.00003)  # learning_rate

clip_grad=5 # 梯度截断防止梯度爆炸 , set < 0 to disable this

decay_lr_lst=(0.5)
#decay_lr_lst=(-1)

# upper_lr_lst=(0.005 0.0005 0.0001)
upper_lr_lst=(0.0005)

# Fix embedding for first x epochs.[abandon]
fix_embd_epoch_lst=(-1)
# fix_embd_epoch_lst=(1 2)

warmup_epoch=-1
# warmup_epoch=-1


train_batch_size_lst=(4)
test_batch_size=1
#grad_acc=2
grad_acc=4  # if the GPU-memory is not enough, use bigger gradient accumulate  
epoch=50

# ==== model setting =========
# ---- encoder setting ----- 
embedder_lst=(bert sep_bert)
# embedder=sep_bert  # 会报错

# emission_lst=(mnet)
# emission_lst=(tapnet)
# emission_lst=(proto_with_label)
# emission_lst=(proto proto_feature)
# emission_lst=(tapnet)
emission_lst=(proto proto_feature tapnet mnet)
inter_loss_lst=(0 0.1 0.2 0.3)
gcns_lst=(0)
# '''相似计算  'cosine', 'dot', 'bi-affine', 'l2' '''
# dot似乎最好
similarity=(dot cosine)

# normalize emission into 1-0
emission_normalizer=(none)
# emission_normalizer=softmax
#emission_normalizer=norm

# method to scale emission and transition into 1-0'''
# emission_scaler=none
#emission_scaler=fix
emission_scaler=learn 
#emission_scaler=relu
#emission_scaler=exp

# (For MNet) Divide emission by each tag's token num in support set
# if true, model will div each types emission by its slot type num
# do_div_emission=-dbt
do_div_emission=

# 使用lable语义信息
# use_schema=--use_schema
# use_schema=

ems_scale_rate_lst=(1)
# ems_scale_rate_lst=(0.01 0.02 0.05 0.005)

# Method to represent label'''

# label_reps=sep   # represent each label independently
# label_reps = cat  # represent labels by concat all all labels
label_reps=(sep)

ple_normalizer=none  
# normalize scaled label embedding into 1-0

ple_scaler=fix  
# method to scale label embedding into 1-0

ple_scale_r=0.3  # proto with lable  embedding   (1 - self.scaler) * prototype_reps + self.scaler * label_reps
#ple_scale_r=1
#ple_scale_r=0.01

# tap_random_init=--tap_random_init   # Set random init for label reps in tap-net
tap_random_init=
tap_random_init_r=0.3 # Set random init rate for label reps in tap-net   label_reps = (1 - self.random_init_r) * label_reps + self.random_init_r * random_label_reps
tap_mlp=                        #Set MLP in tap-net
emb_log=                # Save embedding log in all emission step

# ------ decoder setting -------
#decoder_lst=(rule)
#decoder_lst=(sms)
decoder_lst=(crf sms)
#decoder_lst=(crf sms)

# transition for target domain
transition=(learn)
# transition=learn

trans_init_lst=(fix)
# trans_init_lst=(rand)

# mask_trans=-mk_tr    # Block out-of domain transitions. 阻止域外转换。
# mask_trans=-mk_tr

trans_scaler=fix
#trans_scale_rate_lst=(10)
trans_scale_rate_lst=(1)

trans_rate=(1)
#trans_rate=0.8

trans_normalizer=none
# trans_normalizer=softmax
#trans_normalizer=norm

# trans_scaler=none
#trans_scaler=fix
trans_scaler=none
#trans_scaler=relu
#trans_scaler=exp

#label_trans_scaler=none
#label_trans_scaler=fix
label_trans_scaler=learn  # transition matrix FROM LABEL scaler, such as re-scale the value to non-negative

label_trans_normalizer=none
# label_trans_normalizer=softmax
#label_trans_normalizer=norm


# ======= default path (for quick distribution) ==========
bert_base_uncased=/home/feng/model/bert-base-chinese/
bert_base_uncased_vocab=/home/feng/model/bert-base-chinese/vocab.txt
base_data_dir=/home/feng/code/MetaEE-master/ACE2005data/ # acl20 data


echo [START] set jobs on dataset [ ${dataset_lst[@]} ] on gpu [ ${gpu_list} ]
# === Loop for all case and run ===
for seed in ${seed_lst[@]}
do
  for dataset in ${dataset_lst[@]}
  do
    for support_shots in ${support_shots_lst[@]}
    do
        for train_batch_size in ${train_batch_size_lst[@]}
        do
              for decay_lr in ${decay_lr_lst[@]}
              do
                  for fix_embd_epoch in ${fix_embd_epoch_lst[@]}
                  do
                      for lr in ${lr_lst[@]}
                      do
                          for upper_lr in ${upper_lr_lst[@]}
                          do
                                for trans_init in ${trans_init_lst[@]}
                                do
                                    for ems_scale_r in ${ems_scale_rate_lst[@]}
                                    do
                                        for trans_scale_r in ${trans_scale_rate_lst[@]}
                                        do
                                            for emission in ${emission_lst[@]}
                                            do
                                                for decoder in ${decoder_lst[@]}
                                                do
                                                    for cross_data_id in ${cross_data_id_lst[@]}
                                                    do
                                                        for inter_loss in ${inter_loss_lst[@]}
                                                        do
                                                            for embedder in ${embedder_lst[@]}
                                                            do
                                                                for gcn in ${gcns_lst[@]}
                                                                do
                                                                    # model names
                                                                    model_name=${dataset}_${support_shots}.dec_${decoder}.gcn_${gcn}.enc_${embedder}.ems_${emission}.inter_loss_${inter_loss}.random_${tap_random_init_r}.proto_${tap_proto_r}.sim_${similarity}.lr_${lr}.up_lr_${upper_lr}.bs_${train_batch_size}_${test_batch_size}.sp_b_${grad_acc}.w_ep_${warmup_epoch}.ep_${epoch}${do_debug}

                                                                    data_dir=${base_data_dir}xval_${dataset}_shot_${support_shots}/
                                                                    file_mark=${dataset}.shots_${support_shots}.cross_id_${cross_data_id}.m_seed_${seed}
                                                                    train_file_name=${dataset}-train-${cross_data_id}-shot-${support_shots}.json
                                                                    dev_file_name=${dataset}-valid-${cross_data_id}-shot-${support_shots}.json
                                                                    test_file_name=${dataset}-test-${cross_data_id}-shot-${support_shots}.json
                                                                    trained_model_path=${data_dir}${model_name}.DATA.${file_mark}/model.path


                                                                    echo [CLI]
                                                                    echo Model: ${model_name}
                                                                    echo Task:  ${file_mark}
                                                                    echo [CLI]
                                                                    export OMP_NUM_THREADS=1  # threads num for each task
                                                                    CUDA_VISIBLE_DEVICES=${gpu_list} python main.py ${do_debug} \
                                                                        --seed ${seed} \
                                                                        --task sl \
                                                                        --gcns ${gcn} \
                                                                        --do_train \
                                                                        --do_predict \
                                                                        --train_path ${data_dir}${train_file_name} \
                                                                        --dev_path ${data_dir}${dev_file_name} \
                                                                        --test_path ${data_dir}${test_file_name} \
                                                                        --output_dir ${data_dir}${model_name}.DATA.${file_mark} \
                                                                        --bert_path ${bert_base_uncased} \
                                                                        --bert_vocab ${bert_base_uncased_vocab} \
                                                                        --train_batch_size ${train_batch_size} \
                                                                        --cpt_per_epoch 1 \
                                                                        --delete_checkpoint \
                                                                        --gradient_accumulation_steps ${grad_acc} \
                                                                        --num_train_epochs ${epoch} \
                                                                        --learning_rate ${lr} \
                                                                        --inter_loss ${inter_loss} \
                                                                        --decay_lr ${decay_lr} \
                                                                        --upper_lr ${upper_lr} \
                                                                        --clip_grad ${clip_grad} \
                                                                        --fix_embed_epoch ${fix_embd_epoch} \
                                                                        --warmup_epoch ${warmup_epoch} \
                                                                        --test_batch_size ${test_batch_size} \
                                                                        --context_emb ${embedder} \
                                                                        --label_reps ${label_reps} \
                                                                        --projection_layer none \
                                                                        --emission ${emission} \
                                                                        --similarity ${similarity} \
                                                                        -e_nm ${emission_normalizer} \
                                                                        -e_scl ${emission_scaler} \
                                                                        --ems_scale_r ${ems_scale_r} \
                                                                        -ple_nm ${ple_normalizer} \
                                                                        -ple_scl ${ple_scaler} \
                                                                        --ple_scale_r ${ple_scale_r} \
                                                                        ${tap_random_init} \
                                                                        --tap_random_init_r ${tap_random_init_r} \
                                                                        ${tap_mlp} \
                                                                        ${emb_log} \
                                                                        ${do_div_emission} \
                                                                        --decoder ${decoder} \
                                                                        --transition ${transition} \
                                                                        --backoff_init ${trans_init} \
                                                                        --trans_r ${trans_rate} \
                                                                        -t_nm ${trans_normalizer} \
                                                                        -t_scl ${trans_scaler} \
                                                                        --trans_scale_r ${trans_scale_r} \
                                                                        -lt_scl ${label_trans_scaler} \
                                                                        --label_trans_scale_r ${trans_scale_r} \
                                                                        -lt_nm ${label_trans_normalizer} \
                                                                        ${mask_trans}  >  /home/feng/code/MetaEE-master/zh_trigger_result/${model_name}.DATA.${file_mark}.log
                                                                    echo [CLI]
                                                                    echo Model: ${model_name}
                                                                    echo Task:  ${file_mark}
                                                                    echo [CLI]
                                                                done
                                                            done
                                                        done
                                                    done
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
      done
	done
done


echo [FINISH] set jobs on dataset [ ${dataset_lst[@]} ] on gpu [ ${gpu_list} ]
