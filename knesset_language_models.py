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
        jsonl_path: נתיב לקובץ JSONL מהתרגיל הקודם (עם השדה sentence_text)
        remove_punct: אם True – בונה מודל על קורפוס בלי סימני פיסוק
        lambdas: מקדמי אינטרפולציה (λ3, λ2, λ1) – צריכים לסכם ל-1
        alpha: פרמטר החלקת לפלס (Add-alpha, בד"כ 1.0)
        """
        self.jsonl_path = jsonl_path
        self.remove_punct = remove_punct
        self.l3, self.l2, self.l1 = lambdas
        if abs(self.l1 + self.l2 + self.l3 - 1.0) > 1e-8:
            raise ValueError("lambdas must sum to 1.0")
        self.alpha = alpha

        self.unigram_counts = Counter()   # count how much every word
        self.bigram_counts = Counter()    # count how much every consecutive words appeared
        self.trigram_counts = Counter()   # count how much every 3 consecutive words appear

        self.vocab = set()
        self.total_tokens = 0  # כולל דמי
        self.V = 0             # גודל אוצר מילים

        # טוענים משפטים ומבנים את המודל
        self.sentences_tokens = self._load_and_preprocess_sentences()
        self._build_counts(self.sentences_tokens)

    # ---------- טעינת הקורפוס ----------

    def _load_and_preprocess_sentences(self):
        """
        קורא את ה-JSONL, מחזיר רשימה של משפטים מטוקננים:
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
        לוקח רעיון מה-regex שלך ב-DataCorpus.tokenization:
        מפרק תאריכים, מספרים, מילים, סימני פיסוק וכו'
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
        מסיר טוקנים שהם רק סימני פיסוק.
        (רעיון דומה למה שעשית בהרבה מקומות – לנקות לפני קורפוס)
        """
        punct_pattern = r'^[:.,!?;%–()]+$'
        cleaned = [t for t in tokens if not re.match(punct_pattern, t)]
        return cleaned

    # ---------- בניית הספירות ----------

    def _build_counts(self, sentences_tokens):
        for sent in sentences_tokens:
            # מוסיפים טוקני התחלה (כמו בקוד שלך)
            tokens = [START_0, START_1] + sent

            # יוניגרם
            for w in tokens:
                self.unigram_counts[w] += 1
                self.vocab.add(w)
                self.total_tokens += 1

            # ביגרם
            for i in range(1, len(tokens)):
                bigram = (tokens[i - 1], tokens[i])
                self.bigram_counts[bigram] += 1

            # טריגרם
            for i in range(2, len(tokens)):
                trigram = (tokens[i - 2], tokens[i - 1], tokens[i])
                self.trigram_counts[trigram] += 1

        # מוסיפים UNK לאוצר מילים
        self.vocab.add(UNK)
        self.V = len(self.vocab)

    # maps unknown words to the token <UNK>
    def _map_unk(self, w):
        return w if w in self.vocab else UNK

    # ---------- הסתברויות עם Laplace ----------

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

    # ---------- פונקציות רשמיות של שאלה 1 ----------

    def calculate_prob_of_sentence(self, sentence_str):
        """
        sentence_str: משפט כמחרוזת (בלי טוקני התחלה).
        מחזירה: לוג-הסתברות של המשפט לפי המודל.
        """
        tokens = sentence_str.strip().split()
        # מוסיפים טוקני התחלה לפי הדרישה
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
        prefix_str: רצף טוקנים (מופרד ברווחים).
        מחזירה: (best_token, log_prob)
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
            # לא נרצה לנבא טוקני התחלה או UNK
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
        - 'protocol_name' : מזהה מסמך (פרוטוקול)
        - 'sentence_text' : טקסט (מופרד ברווחים)
    type: "frequency" or "tfidf"/"tf-idf"
    """

    # --- Pass 1: go once over all documents (protocols) and n-grams ---
    total_docs = 0

    global_counts = Counter()          # כמה פעמים כל n-gram מופיע סה"כ
    df_counts = Counter()              # בכמה מסמכים (פרוטוקולים) הוא מופיע
    sum_tf = defaultdict(float)        # סכום ה-TF של כל n-gram על פני כל המסמכים
    doc_counts = Counter()             # בכמה מסמכים הופיע (לממוצע TF)

    # כל group מייצג פרוטוקול אחד = document
    for protocol_name, group in corpus_df.groupby("protocol_name"):
        # מאחדים את כל המשפטים של הפרוטוקול למסמך אחד
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

        # שני סוגי n-grams: 2-gram ו-4-gram
        for n in [2, 4]:
            title = "Two-gram collocations:" if n == 2 else "Four-gram collocations:"
            f.write(f"{title}\n")

            # שני מדדים: Frequency ו-TF-IDF
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
                    # colloc הוא tuple של המילים -> נכתוב כרצף מילים
                    f.write(" ".join(colloc) + "\n")
                f.write("\n")  # שורה ריקה

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
                f.write("\n")  # שורה ריקה


def load_corpus_df(jsonl_path):
    """
    טוען את הקובץ JSONL ומחזיר DataFrame עם:
    - protocol_name
    - protocol_type
    - sentence_text
    (שורה לכל משפט)
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
    מקבל DF עם:
    - protocol_name
    - sentence_text
    ומחזיר DF חדש שבו סימני פיסוק "טהורים" הוסרו.
    כאן אנחנו מסתמכים על split פשוט – זה מספיק למטלה.
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
# ממסך אחוז X מהטוקנים בכל משפט ומחזיר:
# 1. רשימת משפטים ממוסכים
# 2. רשימת קבוצות של אינדקסים שמוסכו בכל משפט
def mask_tokens_in_sentences(sentences, x):
    masked_list = []
    masked_positions = []

    for sentence in sentences:
        tokens = sentence.split()

        # מסירים דמיים אם יש (במודל שלנו הם לא אמורים להיות פה, אבל ליתר ביטחון)
        if len(tokens) >= 2 and tokens[0] == "<s_0>" and tokens[1] == "<s_1>":
            tokens = tokens[2:]

        # לפחות טוקן אחד ממוסך
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
# בוחר 10 משפטים רנדומליים עם לפחות 5 טוקנים מהמודל (committee)
def pick_ten_sentences(model):
    valid_sentences = []

    # sentences_tokens: רשימה של רשימות טוקנים
    for tokens in model.sentences_tokens:
        # מוחקים דמיים להתחלה
        if len(tokens) >= 2 and tokens[0] == "<s_0>" and tokens[1] == "<s_1>":
            tokens = tokens[2:]

        if len(tokens) >= 5:
            valid_sentences.append(" ".join(tokens))

    if len(valid_sentences) < 10:
        raise ValueError("Not enough valid sentences in corpus")

    return random.sample(valid_sentences, 10)


# משלים משפט ממוסך בעזרת מודל השפה (plenary/committee)
def complete_masked_sentence(lm, masked_sentence):
    tokens = masked_sentence.split()
    # נוסיף דמיים בשביל ההקשר
    tokens = ["<s_0>", "<s_1>"] + tokens

    filled_tokens = []

    while "[*]" in tokens:
        idx = tokens.index("[*]")

        # אם מסכה נמצאת ממש בהתחלה – נחליף ל-UNK
        if idx < 2:
            tokens[idx] = "<UNK>"
            continue

        prefix = " ".join(tokens[:idx])
        next_token, _ = lm.generate_next_token(prefix)
        tokens[idx] = next_token
        filled_tokens.append(next_token)

    completed_sentence = " ".join(tokens[2:])
    return completed_sentence, filled_tokens


################################### q 3.4 ###################
# perplexity סטנדרטי של מודל על רשימת משפטים
def calculate_perplexity(model, sentences):
    """
    sentences: רשימת משפטים (בלי טוקני התחלה).
    מחשב perplexity של המודל על כל הטוקנים בכל המשפטים.
    """
    total_log_prob = 0.0
    total_tokens = 0

    for sentence in sentences:
        # מוסיפים טוקני התחלה
        sent = "<s_0> <s_1> " + sentence
        tokens = sent.split()

        for i in range(2, len(tokens)):
            w_prev2, w_prev1, w = tokens[i - 2], tokens[i - 1], tokens[i]
            p = model._p_interpolated(w_prev2, w_prev1, w)
            if p <= 0.0:
                p = 1e-12
            # לוג בסיס 2 (log2) – מקובל ל-perplexity
            total_log_prob += math.log2(p)
            total_tokens += 1

    if total_tokens == 0:
        return float("inf")

    cross_entropy = - total_log_prob / total_tokens
    perplexity = 2 ** cross_entropy
    return perplexity


# שומר את תוצאות סעיפים 3.2, 3.3, 3.4 בקבצים
def process_masked_sentences(committee_lm, plenary_lm, output_dir):
    # 3.2 – בוחרים 10 משפטים רנדומליים מה-committee
    original_sentences = pick_ten_sentences(committee_lm)

    # שומרים את המשפטים המקוריים
    with open(os.path.join(output_dir, "original_sampled_sents.txt"), "w", encoding="utf-8") as f:
        for sentence in original_sentences:
            f.write(f"{sentence}\n")

    # ממסכים 10% מהטוקנים
    masked_sentences, positions = mask_tokens_in_sentences(original_sentences, 10)

    # שומרים את המשפטים הממוסכים
    with open(os.path.join(output_dir, "masked_sampled_sents.txt"), "w", encoding="utf-8") as f:
        for sentence in masked_sentences:
            f.write(f"{sentence}\n")

    # 3.3 – השלמה + חישוב הסתברויות
    with open(os.path.join(output_dir, "sampled_sents_results.txt"), "w", encoding="utf-8") as f:
        for original, masked in zip(original_sentences, masked_sentences):

            # משלים בעזרת מודל המליאה
            plenary_completed, plenary_tokens = complete_masked_sentence(plenary_lm, masked)

            plenary_prob_in_plenary = round(plenary_lm.calculate_prob_of_sentence(plenary_completed), 2)
            plenary_prob_in_committee = round(committee_lm.calculate_prob_of_sentence(plenary_completed), 2)

            f.write(f"original_sentence: {original}\n")
            f.write(f"masked_sentence: {masked}\n")
            f.write(f"plenary_sentence: {plenary_completed}\n")
            f.write(f"plenary_tokens: {', '.join(plenary_tokens)}\n")
            f.write(f"probability of plenary sentence in plenary corpus: {plenary_prob_in_plenary}\n")
            f.write(f"probability of plenary sentence in committee corpus: {plenary_prob_in_committee}\n\n")

    # 3.4 – perplexity של מודל המליאה על המשפטים המקוריים (committee)
    with open(os.path.join(output_dir, "perplexity_result.txt"), "w", encoding="utf-8") as f:
        perplexity = calculate_perplexity(plenary_lm, original_sentences)
        f.write(f"{perplexity:.2f}\n")


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
        print("Building full-corpus model (with punctuation)...")
        lm_full = LM_Trigram(corpus_path, remove_punct=False)

        print("Building no-punctuation model...")
        lm_no_punct = LM_Trigram(corpus_path, remove_punct=True)

        # -------- Part 2: collocations --------
        print("Saving collocations...")

        # נשתמש בקובץ המקורי כדי לקבל protocol_name + sentence_text
        corpus_df = load_corpus_df(corpus_path)
        # full corpus: עם פיסוק
        df_full = corpus_df[["protocol_name", "sentence_text"]].copy()
        # no punctuation corpus: מסירים סימני פיסוק פשוטים
        df_no_punct = remove_punct_from_df(df_full)

        colloc_path = os.path.join(output_dir, "knesset_collocations.txt")
        save_collocations_file(df_full, df_no_punct, colloc_path)

        # -------- Part 3: masking, completion, perplexity --------
        # נשתמש במודל "committee_lm" = lm_full, ו-"plenary_lm" = lm_full גם כן
        # (אם תרצה באמת להפריד לוועדה/מליאה, נוסיף סינון לפי protocol_type)
        print("Processing masked sentences and perplexity...")
        committee_lm = lm_full
        plenary_lm = lm_full

        process_masked_sentences(committee_lm, plenary_lm, output_dir)

        print("Done. All outputs written to:", output_dir)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
