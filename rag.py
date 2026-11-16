'''Retrieval-augmented generation engine of the wine study assistant.

The same pipeline developed in the notebook, organised as a module so that the Streamlit
interface (app.py) can import it: retrieval with MMR, a relevance threshold, grounded
prompting, short-term memory used for query rewriting, and structured logging.
'''
import os

# PyTorch and FAISS each ship their own OpenMP runtime. On macOS, loading both in the same process and
# running them from Streamlit's worker thread can crash Python, so both are limited to a single thread
# before they are imported. On the Linux hosting platform these settings are harmless.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import json
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer  # imported before faiss on purpose
import faiss

faiss.omp_set_num_threads(1)

BASE_DIR = Path(__file__).resolve().parent
CITATION_PATTERN = re.compile(r'\s*\[\d+\]')
# Phrasings of the refusal that the system prompt asks for ("My sources do not cover this question")
REFUSAL_PATTERN = re.compile(r'sources do not (cover|contain|include|mention)|not covered (by|in) my sources', re.IGNORECASE)


def load_config(path=BASE_DIR / 'config.yaml'):
    '''Read the central configuration file.'''
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


class ChatSession:
    '''Short-term conversational memory: a rolling window of recent turns with a token cap.'''

    def __init__(self, max_turns, max_tokens):
        self.turns = []  # (question, answer) pairs, oldest first
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.session_id = uuid.uuid4().hex  # random identifier that groups the turns of one conversation, not a person

    def add(self, question, answer):
        '''Store a turn, removing citations that only make sense for that turn's passages.'''
        self.turns.append((question, CITATION_PATTERN.sub('', answer)))

    def history(self, count_tokens):
        '''Most recent turns, as chat messages, that fit within the window and the token cap.'''
        messages, used = [], 0
        for question, answer in reversed(self.turns[-self.max_turns:]):
            pair = [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer}]
            cost = sum(count_tokens(m['content']) for m in pair)
            if used + cost > self.max_tokens:
                break  # older turns are dropped first
            messages = pair + messages
            used += cost
        return messages


class WineAssistant:
    '''Loads the knowledge base once and answers questions turn by turn.'''

    def __init__(self, config, groq_api_key=None):
        self.config = config
        paths = config['paths']

        # Knowledge base: chunk metadata and the FAISS index built in the notebook (row i <-> vector i)
        self.chunks = pd.read_json(BASE_DIR / paths['chunks'], lines=True)
        self.index = faiss.read_index(str(BASE_DIR / paths['index']))
        if self.index.ntotal != len(self.chunks):
            raise ValueError('The index and the chunk file are out of sync.')
        self.chunk_embeddings = self.index.reconstruct_n(0, self.index.ntotal)  # vectors used by MMR

        # Embedding model: must be the same model that built the index
        self.embedder = SentenceTransformer(config['embeddings']['model'], device='cpu')
        self.embedder.tokenizer.model_max_length = 1_000_000  # used only to count tokens

        self.log_path = BASE_DIR / paths['log']
        self.provider = config['llm']['provider']
        if self.provider == 'groq':
            from groq import Groq
            if not groq_api_key:
                raise ValueError('GROQ_API_KEY is missing: add it to .streamlit/secrets.toml.')
            self.client = Groq(api_key=groq_api_key)
        elif self.provider == 'local':
            self._load_local_model(config['llm']['local_model'])
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    # --- Language model ---------------------------------------------------------------

    def _load_local_model(self, model_name):
        '''Load the open LLM on the development machine (Apple Silicon GPU when available).'''
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16).to(self.device)
        self.llm.eval()

    def generate(self, messages, max_new_tokens):
        '''Generate a reply with the configured provider.'''
        llm = self.config['llm']
        if self.provider == 'groq':
            response = self.client.chat.completions.create(
                model=llm['groq_model'],
                messages=messages,
                temperature=llm['temperature'],
                max_completion_tokens=max_new_tokens + llm['groq_reasoning_budget'],
                reasoning_effort=llm['groq_reasoning_effort'],
            )
            return (response.choices[0].message.content or '').strip()

        prompt = self.llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.llm_tokenizer(prompt, return_tensors='pt').to(self.device)
        temperature = llm['temperature']
        sampling = {'do_sample': True, 'temperature': temperature, 'top_p': 0.9} if temperature > 0 else {'do_sample': False}
        with self.torch.no_grad():
            output = self.llm.generate(**inputs, max_new_tokens=max_new_tokens, **sampling)
        new_tokens = output[0][inputs['input_ids'].shape[1]:]
        return self.llm_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # --- Retrieval ----------------------------------------------------------------------

    def count_tokens(self, text):
        '''Approximate token count, used for the memory cap.'''
        return len(self.embedder.tokenizer.encode(text, add_special_tokens=False))

    def retrieve(self, query):
        '''Retrieve passages relevant to the query but diverse among themselves (Maximal Marginal Relevance).'''
        settings = self.config['retrieval']
        instruction = self.config['embeddings']['query_instruction']
        query_vector = self.embedder.encode([instruction + query], normalize_embeddings=True).astype('float32')
        scores, positions = self.index.search(query_vector, settings['candidate_pool'])
        scores, positions = scores[0], positions[0]
        candidates = self.chunk_embeddings[positions]

        selected = []
        while len(selected) < min(settings['top_k'], len(positions)):
            best, best_value = None, -np.inf
            for i in range(len(positions)):
                if i in selected:
                    continue
                redundancy = max((candidates[i] @ candidates[j] for j in selected), default=0.0)
                value = settings['mmr_lambda'] * scores[i] - (1 - settings['mmr_lambda']) * redundancy
                if value > best_value:
                    best, best_value = i, value
            selected.append(best)

        passages = self.chunks.iloc[positions[selected]].copy()
        passages.insert(0, 'score', scores[selected])
        return passages

    # --- Prompting ----------------------------------------------------------------------

    def build_messages(self, question, passages):
        '''Keep the system rules, the retrieved context and the question in separate places.'''
        blocks = []
        for n, row in enumerate(passages.itertuples(), start=1):
            body = row.text.split('\n', 1)[1]  # drop the contextual header, shown in the label instead
            blocks.append(f'[{n}] {row.title} > {row.section}\n{body}')
        user_content = 'Context passages:\n\n' + '\n\n'.join(blocks) + f'\n\nQuestion: {question}'
        return [{'role': 'system', 'content': self.config['prompts']['system']},
                {'role': 'user', 'content': user_content}]

    def rewrite_query(self, question, session):
        '''Turn a follow-up such as "And in Austria?" into a standalone question for retrieval.'''
        history = session.history(self.count_tokens)
        if not history:
            return question
        conversation = '\n'.join(f"{m['role'].capitalize()}: {m['content']}" for m in history)
        messages = [{'role': 'system', 'content': self.config['prompts']['rewrite']},
                    {'role': 'user', 'content': f'Conversation:\n{conversation}\n\nFollow-up question: {question}'}]
        rewritten = self.generate(messages, self.config['llm']['rewrite_max_tokens'])
        return rewritten or question

    @staticmethod
    def format_sources(answer, passages):
        '''One line per cited article, listing every passage number cited from it.'''
        cited = sorted({int(n) for n in re.findall(r'\[(\d+)\]', answer) if 1 <= int(n) <= len(passages)})
        grouped = {}
        for n in cited:
            row = passages.iloc[n - 1]
            grouped.setdefault((row['title'], row['url']), []).append(n)
        return [(''.join(f'[{n}]' for n in numbers), title, url) for (title, url), numbers in grouped.items()]

    @staticmethod
    def related_topics(passages, n=3):
        '''Titles of the closest articles, offered when the question cannot be answered.'''
        return list(passages['title'].unique()[:n])

    # --- One conversational turn ----------------------------------------------------------

    def chat(self, question, session):
        '''Rewrite with memory, retrieve, answer from the sources, update memory and log the turn.'''
        start = time.time()
        standalone = self.rewrite_query(question, session)
        passages = self.retrieve(standalone)
        settings = self.config['retrieval']

        if passages['score'].max() < settings['relevance_threshold']:  # first defense
            outcome = 'refused_by_threshold'
            reply = 'My sources do not cover this question.'
        else:
            reply = self.generate(self.build_messages(standalone, passages), self.config['llm']['max_new_tokens'])
            # Second defense: the model declines. A reply with citations is an answer; one without citations is
            # a refusal only if it says so, otherwise it is an answer that failed to cite its sources.
            if CITATION_PATTERN.search(reply):
                outcome = 'answered'
            elif REFUSAL_PATTERN.search(reply):
                outcome = 'refused_by_model'
            else:
                outcome = 'answered_uncited'

        session.add(question, reply)
        result = {
            'question': question,
            'standalone_query': standalone,
            'reply': reply,
            'outcome': outcome,
            'sources': self.format_sources(reply, passages) if outcome == 'answered' else [],
            'related_topics': self.related_topics(passages) if outcome.startswith('refused') else [],
            'passages': passages,
            'latency_s': round(time.time() - start, 2),
        }
        self.log_interaction(session, result)
        return result

    # --- Logging --------------------------------------------------------------------------

    def log_interaction(self, session, result):
        '''Append the turn to the JSON Lines log and print it, so it also reaches the hosting platform logs.
        Records carry no personal identifiers.'''
        llm = self.config['llm']
        record = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'session_id': session.session_id,
            'turn': len(session.turns),
            'question': result['question'],
            'standalone_query': result['standalone_query'],
            'top_score': round(float(result['passages']['score'].max()), 4),
            'retrieved': [{'title': row.title, 'section': row.section, 'score': round(float(row.score), 4)}
                          for row in result['passages'].itertuples()],
            'outcome': result['outcome'],
            'answer': result['reply'],
            'latency_s': result['latency_s'],
            'provider': self.provider,
            'model': llm['groq_model'] if self.provider == 'groq' else llm['local_model'],
            'top_k': self.config['retrieval']['top_k'],
            'threshold': self.config['retrieval']['relevance_threshold'],
            'mmr_lambda': self.config['retrieval']['mmr_lambda'],
        }
        line = json.dumps(record, ensure_ascii=False)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
        print(line, flush=True)
