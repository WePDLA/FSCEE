# FSCEE
The main code comes from:
# Meta Dialog Platform (MDP)
```
@article{hou2020fewjoint,
	title={FewJoint: A Few-shot Learning Benchmark for Joint Language Understanding},
	author={Yutai Hou, Jiafeng Mao, Yongkui Lai, Cheng Chen, Wanxiang Che, Zhigang Chen, Ting Liu},
	journal={arXiv preprint},
	year={2020}
}
```
Meta Dialog Platform is  a toolkit platform for **NLP Few-Shot Learning** tasks of:
- Text Classification
- Sequence Labeling

It also provides the baselines for:
- [Track-1 of SMP2020: Few-shot dialog language understanding](https://smp2020.aconf.cn/smp.html#3).
- [Benchmark Paper: "FewJoint: A Few-shot Learning Benchmark for Joint Language Understanding"]("https://arxiv.org/abs/2009.08138")

## Get Started

### Environment Requirement
```
python>=3.6
torch>=1.2.0
transformers>=2.9.0
numpy>=1.17.0
tqdm>=4.31.1
allennlp>=0.8.4
pytorch-nlp
```

### Example for Event extraction (also for Sequence Labeling)


#### Step1: Prepare pre-trained embedding
- Download the pytorch bert model, or convert tensorflow param by yourself with [scripts](https://github.com/huggingface/transformers/blob/master/src/transformers/convert_bert_original_tf_checkpoint_to_pytorch.py).
- Set BERT path in the `./scripts/your_script.sh` to your setting, for example:

```bash
bert_base_uncased=/your_dir/uncased_L-12_H-768_A-12/
bert_base_uncased_vocab=/your_dir/uncased_L-12_H-768_A-12/vocab.txt
```

#### Step2: Prepare data
- Prepare the few-shot data set. 

##### few-shot/meta-episode style data example

```json
{
    [
        {"support":
            {"seq_ins": ["…明年实行破产…"], 
            "seq_outs": [[…, "O", "O", "O", "O", "B-Business-Declare-Bankruptcy", "I-Business-Declare-Bankruptcy", …]], 
            "labels": ["Business:Declare-Bankruptcy"], 
            "adj": ["ROOT/dep=20/gov=-1", "dep/dep=0/gov=6",...]}, 
        "batch": 
            {"seq_ins": ["…上个月申请破产…"], 
            "seq_outs": [["O", "O", "O", "O","O","B-Business-Declare-Bankruptcy", "I-Business-Declare-Bankruptcy",…]], 
            "labels": ["Business:Declare-Bankruptcy"], 
            "adj": [["ROOT/dep=10/gov=-1", "compound:nn/dep=0/gov=2", ...]]}
        }
    ], 
  ...
}

```


- Set test, train, dev data file path in `./scripts/your_script.sh` to your setting.
  
> For simplicity, your only need to set the root path for data as follow:
```bash
base_data_dir=/your_dir/
```
#### Step3: select the model
You can set the loss weight by :
```
inter_loss_lst=(0 0.1 0.2 0.3)
```
if want to use GCNs , do following:
```
gcns_lst=(1)
```
if want to use label semantic information , do following:
```
use_schema=--use_schema
```

#### Step4: Train and test the main model
- Build a folder to collect running log
```bash
mkdir result
```

- Execute cross-evaluation script with two params: -[gpu id] -[dataset name]

##### Example for 5-shot trigger extraction:
```bash
source ./scripts/zh_ACE_run_5_shot_slot_tagging.sh 0
```  

### Other detailed functions and options:
You can experiment freely by passing parameters to `main.py` to choose different model architectures, hyperparameters, etc.

To view detailed options and corresponding descriptions, run commandline: 
```bash
python main.py --h
```

We provide scripts for general few-shot classification and sequence labeling task respectively:

## Few-shot Data Construction Tool
We also provide a generation tool for converting normal data into few-shot/meta-episode style. 
The tool is included at path: `scripts/other_tool/meta_dataset_generator.py`. 

Run following commandline to view detailed interface:
```bash
python generate_meta_dataset.py --h
```

For simplicity, we provide an example script to help generate few-shot data: `./scripts/gen_meta_data.sh`.

The following are some key params for you to control the generation process:
- `input_dir`: raw data path
- `output_dir`: output data path
- `episode_num`: the number of episode which you want to generate
- `support_shots_lst`: to specified the support shot size in each episode, we can specified multiple number to generate at the same time.
- `query_shot`: to specified the query shot size in each episode
- `seed_lst`: random seed list to control random generation
- `use_fix_support`:  set the fix support in dev dataset
- `dataset_lst`: specified the dataset type which our tool can handle, there are some choices: `stanford` & `SLU` & `TourSG` & `SMP`. 

> If you want to handle other type of dataset, 
> you can add your code for load raw dataset in `meta_dataset_generator/raw_data_loader.py`.



## Acknowledgment

The platform is developed by [HIT-SCIR](http://ir.hit.edu.cn/). 
