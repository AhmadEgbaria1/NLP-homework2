import json
import math
import re
from collections import Counter

START_0 = "<s_0>"
START_1 = "<s_1>"
UNK = "<UNK>"


class TrigramLM:
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


        self.unigram_counts = Counter()  #count how much every word
        self.bigram_counts = Counter()  # count how much every consecutive words appeared
        self.trigram_counts = Counter() #count how much evert 3 consucyeve words appear

        self.vocab = set()
        self.total_tokens = 0  # כולל דמי
        self.V = 0             # גודל אוצר מילים

        # טוענים משפטים ומבנים את המודל
        sentences_tokens = self._load_and_preprocess_sentences()
        self._build_counts(sentences_tokens)

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
        pattern = r'\d{1,2}:\d{2}|\d{1,2}\.\d{1,2}\.\d{4}|\d+\.\d+|\d+(?:,\d{3})*(?:\.\d+)?%?|[\dא-ת]\.|[\w״"\'א-ת]+|[:.,!?;%–()]|\d+\.(?=\s*[\w״"\'א-ת]|[:.,!?;%–()])|\.{3}'
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
                bigram = (tokens[i-1], tokens[i])
                self.bigram_counts[bigram] += 1

            # טריגרם
            for i in range(2, len(tokens)):
                trigram = (tokens[i-2], tokens[i-1], tokens[i])
                self.trigram_counts[trigram] += 1

        # מוסיפים UNK לאוצר מילים
        self.vocab.add(UNK)
        self.V = len(self.vocab)

  #maps unknown words to the token <UNK>
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
            w_prev2, w_prev1, w = tokens[i-2], tokens[i-1], tokens[i]
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


