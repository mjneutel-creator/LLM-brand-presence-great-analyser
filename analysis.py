import re
import math
from collections import Counter

STOPWORDS_TEXT = '''
about above after again against all am an and any are aren't as at be because been before being below
between both but by can't cannot could couldn't did didn't do does doesn't doing don't down during
each few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here here's
hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself
let's me more most mustn't my myself no nor not of off on once only or other ought our ours ourselves
out over own same shan't she she'd she'll she's should shouldn't so some such than that that's the
their theirs them themselves then there there's these they they'd they'll they're they've this those
through to too under until up very was wasn't we we'd we'll we're we've were weren't what what's when
when's where where's which while who who's whom why why's with won't would wouldn't you you'd you'll
you're you've your yours yourself yourselves
'''

STOPWORDS = set(STOPWORDS_TEXT.split())

POSITIVE = set("strong leading trusted reliable stable positive progress committed responsible credible resilient improved".split())
NEGATIVE = set("risk criticism critical legacy greenwashing slow cautious concern controversy complaint scrutiny".split())


def count_brand_mentions(text: str, brand: str) -> int:
    return len(re.findall(re.escape(brand), text, flags=re.IGNORECASE))


def sentiment_score(text: str) -> float:
    # Lightweight lexicon score: (pos - neg) / sqrt(n_tokens)
    tokens = re.findall(r"[A-Za-z']+", text.lower())
    if not tokens:
        return 0.0
    pos = sum(1 for t in tokens if t in POSITIVE)
    neg = sum(1 for t in tokens if t in NEGATIVE)
    return (pos - neg) / math.sqrt(len(tokens))


def extract_themes(texts: list[str], top_k: int = 10) -> list[tuple[str, float]]:
    # Simple corpus tf-idf without external deps.
    docs = []
    for t in texts:
        tokens = [w.lower() for w in re.findall(r"[A-Za-z']+", t) if w.lower() not in STOPWORDS and len(w) > 2]
        docs.append(tokens)

    df = Counter()
    for tokens in docs:
        for w in set(tokens):
            df[w] += 1

    N = len(docs) or 1
    tfidf = Counter()
    for tokens in docs:
        tf = Counter(tokens)
        for w, c in tf.items():
            idf = math.log((N + 1) / (df[w] + 1)) + 1
            tfidf[w] += c * idf

    return tfidf.most_common(top_k)


def classify_tone(score: float) -> str:
    if score >= 0.15:
        return "Positive"
    if score <= -0.15:
        return "Critical"
    return "Neutral"
