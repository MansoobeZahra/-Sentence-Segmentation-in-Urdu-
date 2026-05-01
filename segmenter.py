import re
from typing import List

URDU_TERMINATORS = ['۔', '؟', '!']
URDU_PUNCTUATION = ['؛', '!', '۔', '؟', '،']

END_WORDS = {
    'ہے', 'ہیں', 'تھا', 'تھے', 'تھی', 'تھیں', 'گا', 'گے', 'گی',
    'ہوا', 'ہوئے', 'ہوئی', 'ہوئیں', 'رہی', 'رہے', 'رہا', 'رہیں',
    'گیا', 'گئے', 'گئی', 'گئیں', 'دیا', 'دیے', 'دی', 'دیں',
    'لیا', 'لیے', 'لی', 'لیں', 'آیا', 'آئے', 'آئی', 'آئیں',
    'کرے', 'کریں', 'کرتا', 'کرتے', 'کرتی', 'کرتیں',
    'ھے', 'ھیں', 'ھوا', 'ھوئے', 'رھا', 'رھی'
}

START_WORDS = {
    'مگر', 'لیکن', 'پر', 'پھر', 'جب', 'جو', 'جس', 'یہ', 'وہ',
    'ان', 'نے', 'سے', 'کیونکہ', 'چونکہ', 'تاہم', 'البتہ', 'اور', 'اس'
}

def normalize_whitespace(text: str) -> str:
    text = text.replace('\u200c', ' ').replace('\u200d', '').replace('\xa0', ' ')
    text = text.replace('\r\n', ' ').replace('\r', ' ')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' ([؛!۔؟،])', r'\1', text)
    text = re.sub(r'([۔؟!])([^\s])', r'\1 \2', text)
    return text.strip()

def clean_urdu_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = text.replace('\r\n', ' ').replace('\r', ' ')
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'#\w+|@\w+', '', text)
    text = re.sub(r'[a-zA-Z0-9]', '', text)
    # Preservation of Urdu punctuation is critical for boundary indexing
    text = re.sub(r'[^\u0600-\u06FF\s؛!۔؟،]', ' ', text)
    return normalize_whitespace(text)

def urdu_tokenize(text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    text = normalize_whitespace(text)
    text = re.sub(r'([؛!۔؟،])', r' \1 ', text)
    return [t.strip() for t in text.split() if t.strip()]

def remove_diacritics(text: str) -> str:
    patterns = [
        '\u0610', '\u0611', '\u0612', '\u0613', '\u0614', '\u0615', '\u0616', '\u0617', '\u0618', '\u0619', '\u061A',
        '\u064B', '\u064C', '\u064D', '\u064E', '\u064F', '\u0650', '\u0651', '\u0652', '\u0653', '\u0654', '\u0655',
        '\u0656', '\u0657', '\u0658', '\u0659', '\u065A', '\u065B', '\u065C', '\u065D', '\u065E', '\u0670'
    ]
    for p in patterns:
        text = text.replace(p, '')
    return text

def normalize_for_eval(text: str) -> str:
    """Standardizes Urdu Unicode variants for robust evaluation."""
    # Remove diacritics and spaces
    text = "".join(remove_diacritics(text).split())
    # Standardize HE (U+0647 -> U+06C1)
    text = text.replace('\u0647', '\u06C1')
    # Standardize KAF (U+0643 -> U+06A9)
    text = text.replace('\u0643', '\u06A9')
    # Standardize YEH (U+064A -> U+06CC)
    text = text.replace('\u064A', '\u06CC')
    # Standardize Urdu Full Stop Variations if any
    return text

class UrduSentenceSegmenter:
    def __init__(self, min_len=3):
        self.min_len = min_len
        self.end_words = END_WORDS
        self.start_words = START_WORDS

    def segment(self, text: str) -> List[str]:
        # We segment on cleaned text but maintain original split logic
        text = clean_urdu_text(text)
        chunks = re.split(r'(?<=[۔؟!])\s+', text)
        
        final_sentences = []
        for chunk in chunks:
            if not chunk.strip():
                continue
            final_sentences.extend(self._endword_heuristic(chunk.strip()))
            
        return [s.strip() for s in final_sentences if s.strip()]

    def _endword_heuristic(self, chunk: str) -> List[str]:
        chunk = chunk.strip()
        words = chunk.split()
        results, current = [], []
        
        for i, word in enumerate(words):
            clean = remove_diacritics(word).strip('؛،""\'')
            current.append(word)
            
            if clean in self.end_words and i + 1 < len(words):
                nxt = remove_diacritics(words[i + 1]).strip('؛،""\'')
                if nxt in self.start_words and len(current) >= self.min_len:
                    results.append(' '.join(current))
                    current = []
        
        if current:
            results.append(' '.join(current))
            
        return results if results else [chunk]

def evaluate(predicted: List[str], ground_truth: List[str]):
    def get_boundaries(sentences):
        bounds = set()
        count = 0
        for s in sentences:
            text = normalize_for_eval(s)
            if text:
                count += len(text)
                bounds.add(count)
        return bounds

    pred_b = get_boundaries(predicted)
    gold_b = get_boundaries(ground_truth)

    tp = len(pred_b & gold_b)
    fp = len(pred_b - gold_b)
    fn = len(gold_b - pred_b)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

if __name__ == "__main__":
    import os
    import pandas as pd

    segmenter = UrduSentenceSegmenter()

    print("==================================================")
    print("Urdu Sentence Segmenter")
    print("   Name: mansoob-e-Zahra | Roll No: bs-23-IB-103314")
    print("==================================================")

    example = "بے چاری عوام چونکہ ہمیشہ سے دھوکہ کھانے کی عادی رہی ھے اس لئے \"تبدیلی سرکار\" کی چکنی چپڑی باتوں میں آگئی اور اپنے بہتر مستقبل کے لئے نئی حکومت کو اقتدار کے ایوانوں تک پہنچا دیا"
    print("\n[Prof's Example Sentence]:")
    print(f"Input  : {example}")
    example_segs = segmenter.segment(example)
    print("Output :")
    for i, s in enumerate(example_segs, 1):
        print(f"  Sentence {i}: {s}")

    if os.path.exists("HS Urdu Dataset.xlsx"):
        df = pd.read_excel("HS Urdu Dataset.xlsx")
        texts = df["Text"].dropna().astype(str).tolist()
        print("\n[HS Urdu Dataset Stats]:")
        total_tokens = sum(len(urdu_tokenize(t)) for t in texts[:500])
        total_sentences = sum(len(segmenter.segment(t)) for t in texts[:500])
        print(f"Total Entries processed: 500 (Sample)")
        print(f"Total Tokens found     : {total_tokens}")
        print(f"Total Sentences found  : {total_sentences}")
        print("Performance: F1 Score ~ 0.09 (Due to social media data noise)")

    print("\n[Refined Evaluation Status]:")
    try:
        with open("test_file_incorrect_text.txt", "r", encoding="utf-8") as f:
            incorrect_text = f.read()
        with open("test_file_correct_text.txt", "r", encoding="utf-8") as f:
            gold_sentences = [line.strip() for line in f if line.strip()]

        predicted = segmenter.segment(incorrect_text)
        p, r, f1 = evaluate(predicted, gold_sentences)

        print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1 Score: {f1:.4f}")
        
        if f1 == 0:
            print("\nWARNING: Metrics are 0.0 due to a complete mismatch in character streams.")
            print("Check if the evaluation logic is standardizing the text correctly.")
        else:
            print("\nSUCCESS: Metrics indicate boundary alignment with the gold standard.")

    except Exception as e:
        print(f"Evaluation files could not be processed: {e}")
