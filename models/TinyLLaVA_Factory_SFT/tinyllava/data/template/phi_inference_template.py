from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

from .formatter import EmptyFormatter, StringFormatter
from .base import Template
from .formatter import Formatter
from . import register_template

from ...utils.constants import *

from transformers import PreTrainedTokenizer
import torch
    
# system = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. You are assistant and you will find a question in the given image and you need to understand it and answer the quesiton based on the entire image"
system = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."

@register_template('phi_inference')
@dataclass
class PhiInferenceTemplate(Template):
    format_image_token: "Formatter" = StringFormatter(slot="<image>\n{{content}}")
    format_user: "Formatter" = StringFormatter(slot="USER" + ": " + "<image>" + "{{content}}" + " ")
    format_assistant: "Formatter" = StringFormatter(slot="")
    system: "Formatter" = EmptyFormatter(slot=system+" ")
    separator: "Formatter" = EmptyFormatter(slot=['', '<|endoftext|>'])

    def _prompt(
        self,
        question_list, answer_list,
    ):
        msg = ""
        for i, (question, answer) in enumerate(zip(question_list, answer_list)):
            if i == 0:
                msg += self.system.apply()
            if DEFAULT_IMAGE_TOKEN in question:
                question = question.replace(DEFAULT_IMAGE_TOKEN, '').strip()
                question = self.format_image_token.apply(content=question).strip()
            msg += self.format_user.apply(content=question)
            # msg += self.format_assistant.apply(content=answer)
        return msg


    def _make_masks(self, labels, tokenizer, sep, eos_token_length, rounds):
        """
        重写基类中的_make_masks函数,用于生成训练时的标签掩码
        与基类不同的是,问题部分也作为label的一部分进行训练
        
        Args:
            labels: 输入的标签张量
            tokenizer: 分词器 
            sep: 分隔符
            eos_token_length: 结束符的长度
            rounds: 对话轮次
            
        Returns:
            labels: 处理后的标签张量
            cur_len: 当前处理的长度
        """
        cur_len = 0
        for rou in rounds:
            if rou == "":
                break
                
            # 按分隔符分割当前轮次为问题和回答
            parts = rou.split(sep)
            if len(parts) != 2:
                break
                
            # 计算当前轮次长度
            round_len = len(self.tokenizer_image_token(rou, tokenizer)) + eos_token_length
            
            # 更新当前长度
            cur_len += round_len
            
        # 将剩余部分的标签设为IGNORE_INDEX    
        labels[cur_len:] = IGNORE_INDEX
        
        return labels, cur_len




