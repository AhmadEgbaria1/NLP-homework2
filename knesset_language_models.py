import json
import math
import re
import pandas as pd
import os
import sys
import random

from collections import Counter, defaultdict


START_0 = "<s_0>"
START_1 = "<s_1>"
UNK = "<UNK>"


class LM_Trigram:
    def __init__(self, jsonl_path, remove_punct=False,
                 lambdas=(0.6, 0.3, 0.1), alpha=1.0):
        """
        jsonl_path: the path for the jsonl file from the last homework (with sentece_text field)
        remove_punct:  if True -  building a model on the no punctuation corpus
        lambdas: interpolation factors (λ3, λ2, λ1) - must sum to 1
        alpha: Laplace smoothing parameter (Add-alpha, usually 1.0)
        """
        self.jsonl_path = jsonl_path
        self.remove_punct = remove_punct
        self.l3, self.l2, self.l1 = lambdas
        if abs(self.l1 + self.l2 + self.l3 - 1.0) > 1e-8:
            raise ValueError("lambdas must sum to 1.0")
        self.alpha = alpha

        self.unigram_counts = Counter()   # get the frequency of single every word
        self.bigram_counts = Counter()    # get the frequency of every 2 consecutive words
        self.trigram_counts = Counter()   # get the frequency every 3 consecutive words

        self.vocab = set()
        self.total_tokens = 0  # including dummy tokens
        self.V = 0             # vocabulary size

        # Loading sentences and building the model
        self.sentences_tokens = self._load_and_preprocess_sentences()
        self._build_counts(self.sentences_tokens)

    # ---------- Loading the corpus  ----------

    def _load_and_preprocess_sentences(self):
        """Reads the JSONL and returns a list of sentences, where every sentence is a list of tokens:
        [[tok1, tok2, ...], [tok1, tok2, ...], ...]
        """
        sentences_tokens = []

        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                text = obj.get("sentence_text", "").strip()
                if not text:
                    continue

                tokens = self._tokenize(text)

                if self.remove_punct:
                    tokens = self._remove_punctuation(tokens)

                if tokens:
                    sentences_tokens.append(tokens)

        return sentences_tokens

    def _tokenize(self, sentence):
        """
        This function is responsible for splitting a raw text string into a list of individual meaningful tokens.
        Instead of just splitting by spaces,this function uses the Regex from HW1 below, counting dates,numbers,words,punctuation,etc. as a single unit.
        """
        pattern = (
            r'\d{1,2}:\d{2}|\d{1,2}\.\d{1,2}\.\d{4}|'
            r'\d+\.\d+|\d+(?:,\d{3})*(?:\.\d+)?%?|'
            r'[\dא-ת]\.|[\w״"\'א-ת]+|'
            r'[:.,!?;%–()]|'
            r'\d+\.(?=\s*[\w״"\'א-ת]|[:.,!?;%–()])|'
            r'\.{3}'
        )
        tokens = re.findall(pattern, sentence)
        return tokens

    def _remove_punctuation(self, tokens):
        """
       Removes tokens that are pure punctuation like (",", "?!", "...", "(", ..) and returns the list of sentences cleaned from punctuation (to be used in the no punctuation model).
        """
        punct_pattern = r'^[:.,!?;%–()]+$'
        cleaned = [t for t in tokens if not re.match(punct_pattern, t)]
        return cleaned

    # ---------- Building counts ----------

    def _build_counts(self, sentences_tokens):
        for sent in sentences_tokens:
            # adding dummy tokens ("artificial history" for first token in Trigram calculation)
            tokens = [START_0, START_1] + sent

            # Unigram
            for w in tokens:
                self.unigram_counts[w] += 1
                self.vocab.add(w)
                self.total_tokens += 1

            # Bigram
            for i in range(1, len(tokens)):
                bigram = (tokens[i - 1], tokens[i])
                self.bigram_counts[bigram] += 1

            # Trigram
            for i in range(2, len(tokens)):
                trigram = (tokens[i - 2], tokens[i - 1], tokens[i])
                self.trigram_counts[trigram] += 1

        # Adding UNK to vocabulary
        self.vocab.add(UNK)
        self.V = len(self.vocab)

    # maps unknown words to the token <UNK>
    def _map_unk(self, w):
        return w if w in self.vocab else UNK

    # ---------- Calculate Probabilities of sentences using Laplace ----------

    def _p_unigram(self, w):
        w = self._map_unk(w)
        return (self.unigram_counts[w] + self.alpha) / (self.total_tokens + self.alpha * self.V)

    def _p_bigram(self, w_prev, w):
        w_prev = self._map_unk(w_prev)
        w = self._map_unk(w)
        bigram = (w_prev, w)
        count_bigram = self.bigram_counts[bigram]
        count_prev = self.unigram_counts[w_prev]
        return (count_bigram + self.alpha) / (count_prev + self.alpha * self.V)

    def _p_trigram(self, w_prev2, w_prev1, w):
        w_prev2 = self._map_unk(w_prev2)
        w_prev1 = self._map_unk(w_prev1)
        w = self._map_unk(w)
        trigram = (w_prev2, w_prev1, w)
        history = (w_prev2, w_prev1)
        count_trigram = self.trigram_counts[trigram]
        count_hist = self.bigram_counts[history]
        return (count_trigram + self.alpha) / (count_hist + self.alpha * self.V)

    def _p_interpolated(self, w_prev2, w_prev1, w):
        """
        P(w | w_prev2, w_prev1) = λ3 * P_tri + λ2 * P_bi + λ1 * P_uni
        """
        p_uni = self._p_unigram(w)
        p_bi = self._p_bigram(w_prev1, w)
        p_tri = self._p_trigram(w_prev2, w_prev1, w)
        return self.l3 * p_tri + self.l2 * p_bi + self.l1 * p_uni

    # ----------  ----------

    def calculate_prob_of_sentence(self, sentence_str):
        """
        sentence_str: the sentence as a stringb(without the starting tokens).
        returns: a float number - log(Probability(sentence)).
        """
        tokens = sentence_str.strip().split()
        # Adding start tokens as required
        tokens = [START_0, START_1] + tokens

        log_prob = 0.0
        for i in range(2, len(tokens)):
            w_prev2, w_prev1, w = tokens[i - 2], tokens[i - 1], tokens[i]
            p = self._p_interpolated(w_prev2, w_prev1, w)
            if p <= 0.0:
                p = 1e-12
            log_prob += math.log(p)

        return log_prob

    def generate_next_token(self, prefix_str):
        """
        prefix_str: Sequence of tokens (separated by spaces)
        returns: (best_token, log_prob)
        """
        tokens = prefix_str.strip().split()
        if len(tokens) == 0:
            w_prev2, w_prev1 = START_0, START_1
        elif len(tokens) == 1:
            w_prev2, w_prev1 = START_0, tokens[-1]
        else:
            w_prev2, w_prev1 = tokens[-2], tokens[-1]

        best_token = None
        best_p = -1.0

        for w in self.vocab:
            # we don't want to predict start tokens or UNK
            if w in {START_0, START_1, UNK}:
                continue
            p = self._p_interpolated(w_prev2, w_prev1, w)
            if p > best_p:
                best_p = p
                best_token = w

        if best_p <= 0.0:
            best_p = 1e-12
        return best_token, math.log(best_p)


'''************************************************************************************************
*****************************part2*******************************************************'''


def get_k_n_t_collocations(k, n, t, corpus_df, type="frequency"):
    """
    k: number of collocations to return
    n: length of collocation (n-gram)
    t: threshold – minimum number of appearances
    corpus_df: dataframe with columns:
        - 'protocol_name' : protocol ID
        - 'sentence_text' : Text(seperate by white space)
    type: "frequency" or "tfidf"/"tf-idf"
    """

    # --- Pass 1: go once over all documents (protocols) and n-grams ---
    total_docs = 0

    global_counts = Counter()          # How many times does each n-gram appear in total
    df_counts = Counter()              #In how many documents (protocols) does it appear
    sum_tf = defaultdict(float)        #The sum of the TF of each n-gram across all documents
    doc_counts = Counter()             #In how many documents did it appear (for TF average)

    # Each group represents one protocol = document
    for protocol_name, group in corpus_df.groupby("protocol_name"):
        #Combine all the sentences of the protocol into one document
        text = " ".join(group["sentence_text"])
        tokens = text.split()
        if len(tokens) < n:
            continue

        total_docs += 1
        ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        counts = Counter(ngrams)
        total_ngrams_in_doc = len(ngrams)

        for ng, c in counts.items():
            global_counts[ng] += c
            df_counts[ng] += 1
            tf = c / total_ngrams_in_doc
            sum_tf[ng] += tf
            doc_counts[ng] += 1

    # --- filter by threshold t ---
    filtered = {ng for ng, c in global_counts.items() if c >= t}

    # ---------- Frequency ----------
    if type.lower() == "frequency":
        freq_items = [(ng, global_counts[ng]) for ng in filtered]
        return sorted(freq_items, key=lambda x: x[1], reverse=True)[:k]

    # ---------- TF-IDF ----------
    elif type.lower() in ["tfidf", "tf-idf"]:
        tfidf_scores = {}
        for ng in filtered:
            if doc_counts[ng] == 0:
                continue
            mean_tf = sum_tf[ng] / doc_counts[ng]
            idf = math.log(total_docs / (df_counts[ng] + 1))
            tfidf_scores[ng] = mean_tf * idf

        return sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:k]

    else:
        raise ValueError(f"Unknown type for collocations: {type}")


def save_collocations_file(df_full, df_no_punct, output_path):
    with open(output_path, "w", encoding="utf-8") as f:

        #  two n-grams types: 2-gram and 4-gram
        for n in [2, 4]:
            title = "Two-gram collocations:" if n == 2 else "Four-gram collocations:"
            f.write(f"{title}\n")

            #  two metrics: Frequency and TF-IDF
            for metric in ["Frequency", "TF-IDF"]:
                f.write(f"{metric}:\n")

                # --- Full corpus ---
                f.write("Full corpus:\n")
                res = get_k_n_t_collocations(
                    k=10,
                    n=n,
                    t=10 if n == 2 else 5,
                    corpus_df=df_full,
                    type=metric.lower()
                )
                for colloc, _ in res:
                    # colloc is a tuple of words -> written as a sequence of words
                    f.write(" ".join(colloc) + "\n")
                f.write("\n")  # empty line

                # --- No punctuation corpus ---
                f.write("No punctuation corpus:\n")
                res = get_k_n_t_collocations(
                    k=10,
                    n=n,
                    t=10 if n == 2 else 5,
                    corpus_df=df_no_punct,
                    type=metric.lower()
                )
                for colloc, _ in res:
                    f.write(" ".join(colloc) + "\n")
                f.write("\n")  # empty line


def load_corpus_df(jsonl_path):
    """
    loads JSONL file and returns DataFrame with:
    - protocol_name
    - protocol_type
    - sentence_text
    (line for each sentence)
    """
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj.get("sentence_text", "").strip()
            if not text:
                continue
            rows.append({
                "protocol_name": obj.get("protocol_name"),
                "protocol_type": obj.get("protocol_type"),
                "sentence_text": text
            })
    return pd.DataFrame(rows)


def remove_punct_from_df(df):
    """
     gets DF with:
    - protocol_name
    - sentence_text
    returns new clean from punctuation DF .
    split can do the work, we split the sentence into tokens and clean the tokens match the pattern
    """
    punct_pattern = r'^[:.,!?;%–()]+$'
    rows = []
    for _, row in df.iterrows():
        tokens = row["sentence_text"].split()
        cleaned = [t for t in tokens if not re.match(punct_pattern, t)]
        rows.append({
            "protocol_name": row["protocol_name"],
            "sentence_text": " ".join(cleaned)
        })
    return pd.DataFrame(rows)


'''**************************************************************************************************
****************************************part3*****************************************************'''
############################################ question 3.1 ################################
# Masks X percent of the tokens in each sentence and returns:
# 1. list of masked sentences
# 2. list of groups of indexes masked in every sentence
def mask_tokens_in_sentences(sentences, x):
    masked_list = []
    masked_positions = []

    for sentence in sentences:
        tokens = sentence.split()

        # Remove dummies if any (in our model they shouldn't be here, but just to be safe)
        if len(tokens) >= 2 and tokens[0] == "<s_0>" and tokens[1] == "<s_1>":
            tokens = tokens[2:]

        # at least on token is masked
        num_to_mask = max(1, int(len(tokens) * (x / 100.0)))
        masked_indices = set()

        while len(masked_indices) < num_to_mask:
            idx = random.randint(0, len(tokens) - 1)
            if tokens[idx] != "[*]":
                masked_indices.add(idx)
                tokens[idx] = "[*]"

        masked_list.append(" ".join(tokens))
        masked_positions.append(masked_indices)

    return masked_list, masked_positions


########################################### 3.2 #######################
#  Selects 10 random sentences with at least 5 tokens from the model (committee)
def pick_ten_sentences(model):
    valid_sentences = []

    # sentences_tokens: list of tokens lists.
    for tokens in model.sentences_tokens:
        #  delete dummies from start
        if len(tokens) >= 2 and tokens[0] == "<s_0>" and tokens[1] == "<s_1>":
            tokens = tokens[2:]

        if len(tokens) >= 5:
            valid_sentences.append(" ".join(tokens))

    if len(valid_sentences) < 10:
        raise ValueError("Not enough valid sentences in corpus")

    return random.sample(valid_sentences, 10)


# completes masked sentence using the LM (plenary/committee)
def complete_masked_sentence(lm, masked_sentence):
    tokens = masked_sentence.split()
    # adding dummies for the context
    tokens = ["<s_0>", "<s_1>"] + tokens

    filled_tokens = []

    while "[*]" in tokens:
        idx = tokens.index("[*]")

        # if a masked token is the first one change to -UNK-
        if idx < 2:
            tokens[idx] = "<UNK>"
            continue

        prefix = " ".join(tokens[:idx])
        next_token, _ = lm.generate_next_token(prefix)
        tokens[idx] = next_token
        filled_tokens.append(next_token)

    completed_sentence = " ".join(tokens[2:])
    return completed_sentence, filled_tokens





def process_masked_sentences(full_lm, no_punct_lm, output_dir):
    # 3.2 – select 10 random sentences (with ≥5 tokens)
    original_sentences = pick_ten_sentences(full_lm)

    # Save original sentences
    with open(os.path.join(output_dir, "original_sampled_sents.txt"), "w", encoding="utf-8") as f:
        for sentence in original_sentences:
            f.write(sentence + "\n")

    # Mask 10% of tokens
    masked_sentences, _ = mask_tokens_in_sentences(original_sentences, 10)

    # Save masked sentences
    with open(os.path.join(output_dir, "masked_sampled_sents.txt"), "w", encoding="utf-8") as f:
        for sentence in masked_sentences:
            f.write(sentence + "\n")

    # 3.3 – completion + probability evaluation
    with open(os.path.join(output_dir, "sampled_sents_results.txt"), "w", encoding="utf-8") as f:
        for original, masked in zip(original_sentences, masked_sentences):

            # Complete using FULL LM
            full_sentence, full_tokens = complete_masked_sentence(full_lm, masked)

            # Complete using NO-PUNCT LM
            no_punct_sentence, no_punct_tokens = complete_masked_sentence(no_punct_lm, masked)

            # Four required probabilities
            p_full_in_full = round(full_lm.calculate_prob_of_sentence(full_sentence), 2)
            p_full_in_no   = round(no_punct_lm.calculate_prob_of_sentence(full_sentence), 2)

            p_no_in_full   = round(full_lm.calculate_prob_of_sentence(no_punct_sentence), 2)
            p_no_in_no     = round(no_punct_lm.calculate_prob_of_sentence(no_punct_sentence), 2)

            # Write to file
            f.write(f"original_sentence: {original}\n")
            f.write(f"masked_sentence: {masked}\n")

            f.write(f"full_sentence: {full_sentence}\n")
            f.write(f"full_tokens: {', '.join(full_tokens)}\n")

            f.write(f"no_punc_sentence: {no_punct_sentence}\n")
            f.write(f"no_punc_tokens: {', '.join(no_punct_tokens)}\n")

            f.write(f"probability of full sentence in full corpus: {p_full_in_full}\n")
            f.write(f"probability of full sentence in no_punc corpus: {p_full_in_no}\n")
            f.write(f"probability of no_punc sentence in full corpus: {p_no_in_full}\n")
            f.write(f"probability of no_punc sentence in no_punc corpus: {p_no_in_no}\n\n")



if __name__ == "__main__":
    try:
        # python knesset_language_models.py <path/to/corpus_file_name.jsonl> <path/to/output_dir>
        if len(sys.argv) != 3:
            print("Usage: python knesset_language_models.py <path/to/corpus_file_name.jsonl> <path/to/output_dir>")
            sys.exit(1)

        corpus_path = sys.argv[1]
        output_dir = sys.argv[2]

        os.makedirs(output_dir, exist_ok=True)

        # -------- Part 1 models (for use in 2+3) --------

        lm_full = LM_Trigram(corpus_path, remove_punct=False)


        lm_no_punct = LM_Trigram(corpus_path, remove_punct=True)

        # -------- Part 2: collocations --------


        # using the original document in order to get protocol_name + sentence_text
        corpus_df = load_corpus_df(corpus_path)
        # full corpus: with punctuation
        df_full = corpus_df[["protocol_name", "sentence_text"]].copy()
        # no punctuation corpus: Removing simple punctuation marks
        df_no_punct = remove_punct_from_df(df_full)

        colloc_path = os.path.join(output_dir, "knesset_collocations.txt")
        save_collocations_file(df_full, df_no_punct, colloc_path)

        # -------- Part 3: masking, completion, perplexity --------
       # We will use the model "committee_lm" = lm_full, and "plenary_lm" = lm_full as well
       # (If you really want to separate into committee/plenary, we will add filtering by protocol_type)


        # full_lm  = lm_full        # a model with punctuation
        # no_punct_lm = lm_no_punct # a model without punctuation
        process_masked_sentences(lm_full, lm_no_punct, output_dir)



    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
