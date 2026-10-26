import time
import pandas as pd
import numpy as np

t0 = time.time()
print("Loading 1,000 samples to verify feature extraction speed...")
cols = ['id', 'prompt', 'response_a', 'response_b', 'winner_model_a', 'winner_model_b', 'winner_tie']
df = pd.read_csv('/root/llm_classification_finetuning/train.csv', nrows=1000, usecols=cols)

def extract_features(df):
    feats = pd.DataFrame(index=df.index)
    
    # Text strings
    prompt = df['prompt'].fillna('').astype(str)
    ra = df['response_a'].fillna('').astype(str)
    rb = df['response_b'].fillna('').astype(str)
    
    # 1. Length features
    len_p = prompt.str.len()
    len_a = ra.str.len()
    len_b = rb.str.len()
    
    feats['len_p'] = len_p
    feats['len_a'] = len_a
    feats['len_b'] = len_b
    feats['len_diff'] = len_a - len_b
    feats['len_ratio'] = (len_a - len_b) / (len_a + len_b + 1.0)
    feats['len_ratio_p_a'] = len_a / (len_p + 1.0)
    feats['len_ratio_p_b'] = len_b / (len_p + 1.0)
    
    # 2. Word count features
    words_p = prompt.str.split().str.len()
    words_a = ra.str.split().str.len()
    words_b = rb.str.split().str.len()
    
    feats['words_p'] = words_p
    feats['words_a'] = words_a
    feats['words_b'] = words_b
    feats['words_diff'] = words_a - words_b
    feats['words_ratio'] = (words_a - words_b) / (words_a + words_b + 1.0)
    
    # Mean word length
    feats['mean_word_len_a'] = len_a / (words_a + 1.0)
    feats['mean_word_len_b'] = len_b / (words_b + 1.0)
    feats['mean_word_len_diff'] = feats['mean_word_len_a'] - feats['mean_word_len_b']
    
    # 3. Structural & Formatting signals
    # Code blocks
    has_code_a = ra.str.contains('```', regex=False).astype(float)
    has_code_b = rb.str.contains('```', regex=False).astype(float)
    feats['has_code_a'] = has_code_a
    feats['has_code_b'] = has_code_b
    feats['code_diff'] = has_code_a - has_code_b
    
    count_code_a = ra.str.count('```')
    count_code_b = rb.str.count('```')
    feats['code_count_diff'] = count_code_a - count_code_b
    
    # Markdown lists (*, -, 1.)
    feats['list_count_a'] = ra.str.count(r'(?m)^[\s]*[-*•]\s')
    feats['list_count_b'] = rb.str.count(r'(?m)^[\s]*[-*•]\s')
    feats['list_count_diff'] = feats['list_count_a'] - feats['list_count_b']
    
    feats['num_list_count_a'] = ra.str.count(r'(?m)^[\s]*\d+\.\s')
    feats['num_list_count_b'] = rb.str.count(r'(?m)^[\s]*\d+\.\s')
    feats['num_list_count_diff'] = feats['num_list_count_a'] - feats['num_list_count_b']
    
    # Headers (#, ##)
    feats['header_count_a'] = ra.str.count(r'(?m)^#{1,4}\s')
    feats['header_count_b'] = rb.str.count(r'(?m)^#{1,4}\s')
    feats['header_count_diff'] = feats['header_count_a'] - feats['header_count_b']
    
    # Bold emphasis (**text**)
    feats['bold_count_a'] = ra.str.count(r'\*\*[^*]+\*\*')
    feats['bold_count_b'] = rb.str.count(r'\*\*[^*]+\*\*')
    feats['bold_count_diff'] = feats['bold_count_a'] - feats['bold_count_b']
    
    # Newlines / paragraphs
    feats['newline_a'] = ra.str.count('\n')
    feats['newline_b'] = rb.str.count('\n')
    feats['newline_diff'] = feats['newline_a'] - feats['newline_b']
    
    # 4. Punctuation & Tone
    feats['excl_a'] = ra.str.count('!')
    feats['excl_b'] = rb.str.count('!')
    feats['excl_diff'] = feats['excl_a'] - feats['excl_b']
    
    feats['quest_a'] = ra.str.count(r'\?')
    feats['quest_b'] = rb.str.count(r'\?')
    feats['quest_diff'] = feats['quest_a'] - feats['quest_b']
    
    # Politeness / opening markers
    feats['sorry_a'] = ra.str.lower().str.contains(r'sorry|apologize|apologies', regex=True).astype(float)
    feats['sorry_b'] = rb.str.lower().str.contains(r'sorry|apologize|apologies', regex=True).astype(float)
    feats['sorry_diff'] = feats['sorry_a'] - feats['sorry_b']
    
    feats['sure_a'] = ra.str.lower().str.contains(r'sure|certainly|gladly|of course|here is|here are', regex=True).astype(float)
    feats['sure_b'] = rb.str.lower().str.contains(r'sure|certainly|gladly|of course|here is|here are', regex=True).astype(float)
    feats['sure_diff'] = feats['sure_a'] - feats['sure_b']
    
    return feats

feats = extract_features(df)
print(f"Extraction completed in {time.time() - t0:.2f}s!")
print(f"Generated {feats.shape[1]} engineered features for {len(feats)} rows:")
print(feats.head(2))
