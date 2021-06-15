# Syntactic Enhanced Projection Network for Few-shot Chinese Event Extraction 
This code is for KSEM 2021 paper "Syntactic Enhanced Projection Network for Few-shot Chinese Event Extraction".

# Overview
![FSCEE framework](https://github.com/WeSIG/FSCEE/blob/main/FECEE.pdf)

In this work, we explore the Chinese event extraction with limited labeled data and reformulate it as a few-shot sequence tagging task. 
To this end, we propose a novel and practical few-shot syntactic enhanced projection network(SEPN), which exploits a syntactic learner to not only integrate the semantics of the sentence, but also make the extracted feature more discriminative.  
Considering the corpus in this task is limited and the performance perturbation, we propose an adaptive max-margin loss. 
In future work, we will further improve the transition module under the few-shot setting to better capture the dependency between tags. 

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

#### Step2: Prepare the few-shot data set. 
You can get the dependency analysis of the sentence through the StanfordNLP tool. 
Make sure that:
- There are no crossover categories in the training set and the test set. 
- Each type of label appears at least K times in each episode. 
- To get enough training data for the 5-shot query setting, we delete the class which contains less than 10 samples. 
	
##### few-shot/meta-episode style data example

```json
{
    [
        {
	"support":
            {
	    "seq_ins": ["…明年实行破产…","…"], 
	    "seq_outs": [[…, "O", "O", "O", "O", "B-Business-Declare-Bankruptcy", "I-Business-Declare-Bankruptcy", …],[]],
	    "labels": ["Business:Declare-Bankruptcy","…"], 
	    "adj": [["ROOT/dep=20/gov=-1", "dep/dep=0/gov=6",…],[]]
	    }, 
        "batch": 
            {
	    "seq_ins": ["…上个月申请破产…","…"], 
            "seq_outs": [["O", "O", "O", "O","O","B-Business-Declare-Bankruptcy", "I-Business-Declare-Bankruptcy",…],[]], 
            "labels": ["Business:Declare-Bankruptcy","…"], 
            "adj": [["ROOT/dep=10/gov=-1", "compound:nn/dep=0/gov=2", ...],[]]
	    }
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
#### Step3: select the model parameter
You can set the loss weight by :
```
inter_loss_lst=(0 0.1 0.2 0.3)
```
if want to add GCNs embedding, do following:
```
gcns_lst=(1)
```
if want to use label semantic information , do following:
```
use_schema=--use_schema
```
You can experiment freely by passing parameters to `main.py` to choose different model architectures, hyperparameters, etc.

To view detailed options and corresponding descriptions, run commandline: 
```bash
python main.py --h
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

## Few-shot Data Construction Tool
You can use a generation tool for converting normal data into few-shot/meta-episode style. 
The tool is included at path: `scripts/other_tool/meta_dataset_generator.py`. 

Run following commandline to view detailed interface:
```bash
python generate_meta_dataset.py --h
```

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
Our code based on Meta Dialog Platform (MDP). The platform is developed by [HIT-SCIR](http://ir.hit.edu.cn/). 

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

