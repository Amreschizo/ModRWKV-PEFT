########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
import torch.nn.functional as F

import json
import math
import random
import os
import sys
import numpy as np
import torch
import lightning as L
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from lightning_utilities.core.rank_zero import rank_zero_info
from infer.rwkv.utils import PIPELINE
pipeline = PIPELINE('rwkv', "rwkv_vocab_v20230424")
from PIL import Image

import pandas as pd
import librosa
import io
import soundfile as sf
# 读取parquet文件
from torchvision import transforms

# import PIL.PngImagePlugin
# # 增加MAX_TEXT_CHUNK的大小，默认是1MB，可以设置为更大的值，例如10MB
# PIL.PngImagePlugin.MAX_TEXT_CHUNK = 10 * 1024 * 1024

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()  # 将图像转换为张量
])

def process_tokens(conversations):
    # conversation_text = f"\x16"
    inputs = []
    labels = []
    for conv in conversations:
        role = conv.get('from', '').lower()
        content = conv.get('value', '')
        
        if role in ['user','human']:
            question = f"\x16User: {content}\x17"
            input = torch.tensor(pipeline.encode(question))
            label = torch.full_like(input, -100)
        elif role in ['assistant', 'gpt']:
            answer = f"\x16Assistant: {content}\x17"
            input= torch.tensor(pipeline.encode(answer))
            label = input
        inputs.append(input)
        labels.append(label)
    inputs =torch.cat(inputs)
    labels =torch.cat(labels)
    return inputs, labels

def bytes_to_audio(audio_bytes):
    with io.BytesIO(audio_bytes) as buf:
        # 使用 soundfile 读取音频数据
        audio_array, sr = sf.read(buf)
        
        # 确保是单声道
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)
        
        # 确保是 float32 类型
        audio_array = audio_array.astype(np.float32)
        
        return {
            'array': audio_array,
            'sampling_rate': sr
        }



def get_data_by_l_version(trainer: L.Trainer, args):
    if L.__version__[0] == '2':
        train_data = MyDataModule(args)
    else:
        raise ValueError(f"Unsupported PyTorch Lightning version: {L.__version__}")
    return train_data

class GlobalIndexManager:
    def __init__(self, rank=0, device_num=1, shuffle=True):
        self.current_idx = 0
        self.rank = rank
        self.device_num = device_num
        self.shuffle = shuffle
        
    def get_next_idx(self, idx_t):
        if self.shuffle:
            idx = idx_t
        else:
            idx = self.current_idx * self.device_num + self.rank 
            self.current_idx += 1
        # #region agent log
        with open('/home/amreschizo/Documents/.cursor/debug.log', 'a') as f:
            f.write(f'{{"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"dataset.py:get_next_idx","message":"Index manager generated index","data":{{"input_idx_t":{idx_t},"output_idx":{idx},"shuffle":{self.shuffle},"current_idx":{self.current_idx},"device_num":{self.device_num},"rank":{self.rank}}},"timestamp":{int(__import__("time").time()*1000)}}}\n')
        # #endregion
        return idx

class MyDataModule(L.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_data = None
        
        
    def setup(self, stage=None):
        self.train_data = MyDataset(self.args)
        self.args.vocab_size = self.train_data.vocab_size
        self.train_data.real_epoch = self.trainer.current_epoch
        self.train_data.rank = self.trainer.global_rank
        self.train_data.world_size = self.trainer.world_size
        self.train_data.setup(self.trainer.global_rank, self.trainer.world_size, 
                              int(self.args.devices), self.args.data_shuffle)
        
    def train_dataloader(self):
        # must set shuffle=False, persistent_workers=False (because worker is in another thread)
        return DataLoader(
            self.train_data,
            shuffle=self.args.data_shuffle,
            pin_memory=True,
            batch_size=self.args.micro_bsz,
            num_workers=1,
            persistent_workers=False,
            drop_last=True
        )

class WorldDataset(Dataset):
    def __init__(self, args, emb=None):
        self.args = args
        self.rank = 0
        self.real_epoch = 0
        self.world_size = 0
        self.index_manager = None
        self.emb = emb

        if args.data_type =='wav':
            import jsonlines

            # 打开并读取 JSON 文件
            #with open(f'{args.data_file}/answer.jsonl', 'r') as file:
            with jsonlines.open(f'{args.data_file}/answer.jsonl') as file:
                self.data = list(file)
        elif args.data_type=='visual':
            import jsonlines
            # Support both chat.json (JSON array) and chat.jsonl (JSONL format)
            chat_json_path = f'{args.data_file}/chat.json'
            chat_jsonl_path = f'{args.data_file}/chat.jsonl'
            
            if os.path.exists(chat_json_path):
                # Load JSON array format (original LLaVA-CC3M format)
                with open(chat_json_path, 'r', encoding='utf-8') as file:
                    self.data = json.load(file)
            elif os.path.exists(chat_jsonl_path):
                # Load JSONL format (one JSON object per line)
                with jsonlines.open(chat_jsonl_path) as file:
                    self.data = list(file)
            else:
                raise FileNotFoundError(f"Neither chat.json nor chat.jsonl found in {args.data_file}")
            # #region agent log
            with open('/home/amreschizo/Documents/.cursor/debug.log', 'a') as f:
                data_len = len(self.data) if self.data is not None else 0
                f.write(f'{{"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"dataset.py:__init__:visual","message":"Data loaded for visual type","data":{{"data_len":{data_len},"data_file":"{args.data_file}","is_list":{isinstance(self.data, list)},"is_empty":{data_len == 0}}},"timestamp":{int(__import__("time").time()*1000)}}}\n')
            # #endregion
        elif args.data_type == 'arrow':
            from datasets import load_from_disk, concatenate_datasets, load_dataset
            
            # 获取主目录下所有子目录
            subdirs = [os.path.join(args.data_file, d) for d in os.listdir(args.data_file) 
                    if os.path.isdir(os.path.join(args.data_file, d))]
            
            if subdirs:
                # 加载每个子目录的数据集
                datasets = [load_from_disk(subdir) for subdir in subdirs]
                # 连接所有数据集
                self.data = concatenate_datasets(datasets)
                print(f"已连接{len(datasets)}个子目录的数据集，总大小: {len(self.data)}")
            else:
                # 如果没有子目录，直接加载主目录
                self.data = load_from_disk(args.data_file)
                print(self.data[0])
                print(f"从单一目录加载数据集，大小: {len(self.data)}")
        elif args.data_type =='hf' or args.data_type =='qa' or args.data_type =='cnqa' or args.data_type =='cnasr' or args.data_type =='tts':
            from datasets import load_dataset, concatenate_datasets

            def list_subdirectories(base_path):
                return [
                    name for name in os.listdir(base_path)
                    if os.path.isdir(os.path.join(base_path, name)) and not name.startswith('.')
                ]

            datasets = []
            files = list_subdirectories(args.data_file)
            if not files:
                datasets = load_dataset(args.data_file, split="train")
            else:
                for file in files:
                    dataset = load_dataset(f'{args.data_file}/{file}', split="train")
                    datasets.append(dataset)
                datasets = concatenate_datasets(datasets)
            self.data = datasets
            print(len(datasets))
            
        elif args.data_type == "jsonl":
            import jsonlines

            with jsonlines.open(args.data_file) as file:
                self.data = list(file)

        else:
            self.data = pd.read_parquet(args.data_file)

        

    def setup(self, rank, world_size, devices, shuffle):
        self.rank = rank
        self.world_size = world_size
        self.index_manager = GlobalIndexManager(rank=rank, device_num=devices, shuffle=shuffle)
        # #region agent log
        with open('/home/amreschizo/Documents/.cursor/debug.log', 'a') as f:
            actual_data_len = len(self.data) if hasattr(self, 'data') and self.data is not None else 0
            f.write(f'{{"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"dataset.py:setup","message":"Dataset setup complete","data":{{"rank":{rank},"world_size":{world_size},"devices":{devices},"shuffle":{shuffle},"actual_data_len":{actual_data_len}}},"timestamp":{int(__import__("time").time()*1000)}}}\n')
        # #endregion
    
    def __len__(self):
        # #region agent log
        with open('/home/amreschizo/Documents/.cursor/debug.log', 'a') as f:
            actual_data_len = len(self.data) if hasattr(self, 'data') and self.data is not None else 0
            computed_len = self.args.epoch_steps * self.args.micro_bsz
            f.write(f'{{"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"dataset.py:__len__","message":"Dataset length calculation","data":{{"actual_data_len":{actual_data_len},"computed_len":{computed_len},"epoch_steps":{self.args.epoch_steps},"micro_bsz":{self.args.micro_bsz}}},"timestamp":{int(__import__("time").time()*1000)}}}\n')
        # #endregion
        return self.args.epoch_steps * self.args.micro_bsz


    def __getitem__(self, idx):
        # #region agent log
        with open('/home/amreschizo/Documents/.cursor/debug.log', 'a') as f:
            actual_data_len = len(self.data) if hasattr(self, 'data') and self.data is not None else 0
            f.write(f'{{"sessionId":"debug-session","runId":"post-fix","hypothesisId":"A,B,C","location":"dataset.py:__getitem__:entry","message":"__getitem__ called","data":{{"requested_idx":{idx},"actual_data_len":{actual_data_len},"has_index_manager":{self.index_manager is not None},"data_type":"{self.args.data_type}"}},"timestamp":{int(__import__("time").time()*1000)}}}\n')
        # #endregion
        idx = self.index_manager.get_next_idx(idx_t=idx) if self.index_manager else idx
        
        # Wrap index to handle cases where __len__() returns more than actual data size
        # This allows the dataset to repeat data across epochs
        actual_data_len = len(self.data) if hasattr(self, 'data') and self.data is not None else 0
        if actual_data_len > 0:
            idx = idx % actual_data_len
        
        # #region agent log
        with open('/home/amreschizo/Documents/.cursor/debug.log', 'a') as f:
            f.write(f'{{"sessionId":"debug-session","runId":"post-fix","hypothesisId":"B,C","location":"dataset.py:__getitem__:after_wrap","message":"Index after wrapping","data":{{"final_idx":{idx},"actual_data_len":{actual_data_len},"idx_in_bounds":{idx < actual_data_len if actual_data_len > 0 else False}}},"timestamp":{int(__import__("time").time()*1000)}}}\n')
        # #endregion
        args = self.args

        if args.data_type =='hf':
            sample = self.data[idx]
            audio = sample['audio']
            data_answer = sample['text'] #####caption
            audio = librosa.resample(audio['array'],orig_sr= audio['sampling_rate'],target_sr= 16000)  # sr=None 保持原采样率
            sign = audio
            text_tokens = torch.tensor(pipeline.encode(f'\x16Assistant: {data_answer}\x17'))
            text_labels = text_tokens
        elif args.data_type == 'visual':
            # #region agent log
            with open('/home/amreschizo/Documents/.cursor/debug.log', 'a') as f:
                actual_data_len = len(self.data) if hasattr(self, 'data') and self.data is not None else 0
                f.write(f'{{"sessionId":"debug-session","runId":"post-fix","hypothesisId":"A","location":"dataset.py:__getitem__:before_access","message":"About to access self.data[idx]","data":{{"idx":{idx},"actual_data_len":{actual_data_len},"will_fail":{idx >= actual_data_len if actual_data_len > 0 else True}}},"timestamp":{int(__import__("time").time()*1000)}}}\n')
            # #endregion

            img_name = self.data[idx]['image']
            conversation_text = self.data[idx]['conversations']

            mod_path = f'{args.data_file}/{img_name}' 
            image = Image.open(mod_path).convert('RGB')
            sign = image
            text_tokens, text_labels = process_tokens(conversation_text)
        elif args.data_type == 'arrow':

            sample = self.data[idx]
            image = sample['image']
            conversation_text = sample['conversations']
            sign = image.convert('RGB')
            text_tokens, text_labels = process_tokens(conversation_text)


        return sign, text_tokens, text_labels
