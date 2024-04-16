import os
import re
import json
import tqdm
import torch
import time
import argparse
import transformers
from transformers import AutoTokenizer
import openai
from transformers import StoppingCriteria, StoppingCriteriaList
import tiktoken
import sys
sys.path.append("src")
from utils import RetrievalSystem
from template import *

if openai.api_key is None:
    from config import config
    openai.api_type = config["api_type"]
    openai.api_base = config["api_base"] 
    openai.api_version = config["api_version"]
    openai.api_key = config["api_key"]


class MedRAG:

    def __init__(self, llm_name="OpenAI/gpt-3.5-turbo-16k", rag=True, retriever_name="MedCPT", corpus_name="Textbooks", db_dir="./corpus", cache_dir=None):
        self.llm_name = llm_name
        self.rag = rag
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        self.db_dir = db_dir
        self.cache_dir = cache_dir
        if rag:
            self.retrieval_system = RetrievalSystem(self.retriever_name, self.corpus_name, self.db_dir)
        else:
            self.retrieval_system = None
        self.templates = {"cot_system": general_cot_system, "cot_prompt": general_cot,
                    "medrag_system": general_medrag_system, "medrag_prompt": general_medrag}
        if self.llm_name.split('/')[0].lower() == "openai":
            self.model = self.llm_name.split('/')[-1]
            if "gpt-3.5" in self.model or "gpt-35" in self.model:
                self.max_length = 16384
                self.context_length = 15000
            elif "gpt-4" in self.model:
                self.max_length = 32768
                self.context_length = 30000
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name, cache_dir=self.cache_dir)
            if "mixtral" in llm_name.lower():
                self.tokenizer.chat_template = open('./templates/mistral-instruct.jinja').read().replace('    ', '').replace('\n', '')
                self.max_length = 32768
                self.context_length = 30000
            elif "llama-2" in llm_name.lower():
                self.max_length = 4096
                self.context_length = 3072
            elif "meditron-70b" in llm_name.lower():
                self.tokenizer.chat_template = open('./templates/meditron.jinja').read().replace('    ', '').replace('\n', '')
                self.max_length = 4096
                self.context_length = 3072
                self.templates["cot_prompt"] = meditron_cot
                self.templates["medrag_prompt"] = meditron_medrag
            elif "pmc_llama" in llm_name.lower():
                self.tokenizer.chat_template = open('./templates/pmc_llama.jinja').read().replace('    ', '').replace('\n', '')
                self.max_length = 2048
                self.context_length = 1024
            self.model = transformers.pipeline(
                "text-generation",
                model=self.llm_name,
                torch_dtype=torch.float16,
                device_map="auto",
                model_kwargs={"cache_dir":self.cache_dir},
            )

    def answer(self, question, options=None, k=32, rrf_k=100, save_dir = None):
        '''
        question (str): question to be answered
        options (Dict[str, str]): options to be chosen from
        k (int): number of snippets to retrieve
        save_dir (str): directory to save the results
        '''

        if options is not None:
            options = '\n'.join([key+". "+options[key] for key in sorted(options.keys())])
        else:
            options = '' # double check this later!!!!!! See if new prompt tempates are needed.

        # retrieve relevant snippets
        if self.rag:
            retrieved_snippets, scores = self.retrieval_system.retrieve(question, k=k, rrf_k=rrf_k)
            contexts = ["Document [{:d}] (Title: {:s}) {:s}".format(idx, retrieved_snippets[idx]["title"], retrieved_snippets[idx]["content"]) for idx in range(len(retrieved_snippets))]
            if len(contexts) == 0:
                contexts = [""]
            if "openai" in self.llm_name.lower():
                contexts = [self.tokenizer.decode(self.tokenizer.encode("\n".join(contexts))[:self.context_length])]
            else:
                contexts = [self.tokenizer.decode(self.tokenizer.encode("\n".join(contexts), add_special_tokens=False)[:self.context_length])]
        else:
            retrieved_snippets = []
            scores = []
            contexts = []

        if save_dir is not None and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # generate answers
        answers = []
        if not self.rag:
            prompt_cot = self.templates["cot_prompt"].render(question=question, options=options)
            messages = [
                {"role": "system", "content": self.templates["cot_system"]},
                {"role": "user", "content": prompt_cot}
            ]
            ans = self.generate(messages)
            answers.append(re.sub("\s+", " ", ans))
        else:
            for context in contexts:
                prompt_medrag = self.templates["medrag_prompt"].render(context=context, question=question, options=options)
                messages=[
                        {"role": "system", "content": self.templates["medrag_system"]},
                        {"role": "user", "content": prompt_medrag}
                ]
                ans = self.generate(messages)
                answers.append(re.sub("\s+", " ", ans))
        
        if save_dir is not None:
            with open(os.path.join(save_dir, "snippets.json"), 'w') as f:
                json.dump(retrieved_snippets, f, indent=4)
            with open(os.path.join(save_dir, "response.json"), 'w') as f:
                json.dump(answers, f, indent=4)
        
        return answers[0] if len(answers)==1 else answers, retrieved_snippets, scores
            
    @staticmethod
    class CustomStoppingCriteria(StoppingCriteria):
        def __init__(self, stop_words, tokenizer, input_len=0):
            super().__init__()
            self.tokenizer = tokenizer
            self.stops_words = stop_words
            self.input_len = input_len
            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
                tokens = self.tokenizer.decode(input_ids[0][self.input_len:])
                return any(stop in tokens for stop in self.stops_words)

    def custom_stop(self, stop_str, input_len=0):
        stopping_criteria = StoppingCriteriaList([CustomStoppingCriteria(stop_str, self.tokenizer, input_len)])
        return stopping_criteria

    def generate(self, messages):
        '''
        generate response given messages
        '''
        if "openai" in self.llm_name.lower():
            if openai.api_type == "azure":
                response = openai.ChatCompletion.create(
                    engine=self.model,
                    messages=messages,
                    temperature=0.0,
                )
            else:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0,
                )
            ans = response["choices"][0]["message"]["content"]
        else:
            stopping_criteria = None
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if "meditron" in self.llm_name.lower():
                # stopping_criteria = custom_stop(["###", "User:", "\n\n\n"], self.tokenizer, input_len=len(self.tokenizer.encode(prompt_cot, add_special_tokens=True)))
                stopping_criteria = self.custom_stop(["###", "User:", "\n\n\n"], input_len=len(self.tokenizer.encode(prompt_cot, add_special_tokens=True)))
            response = self.model(
                prompt,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                max_length=self.max_length,
                truncation=True,
                stopping_criteria=stopping_criteria
            )
            ans = response[0]["generated_text"]
        return ans
