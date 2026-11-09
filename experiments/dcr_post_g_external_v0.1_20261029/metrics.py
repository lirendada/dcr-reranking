"""IFIR ranking metrics: known positives, raw rel gain, full-qrels denominators."""
import math


def evaluate(order, judgments):
    positive={d for d,r in judgments.items() if r>0}
    first=next((i for i,d in enumerate(order[:20],1) if d in positive),0)
    gain=sum(judgments.get(d,0)/math.log2(i+1) for i,d in enumerate(order[:20],1))
    ideal=sum(r/math.log2(i+1) for i,r in enumerate(sorted((r for r in judgments.values() if r>0),reverse=True)[:20],1))
    return {'P1':float(bool(order) and order[0] in positive),'nDCG20':gain/ideal if ideal else 0.,
        'MRR20':1/first if first else 0.,'R10':len(set(order[:10])&positive)/len(positive) if positive else 0.}


def coverage(order, judgments, k):
    positive={d for d,r in judgments.items() if r>0};hit=len(set(order[:k])&positive)
    return {'positive_document_recall':hit/len(positive) if positive else 0.,'at_least_one_positive':float(hit>0),
            'positive_documents_retrieved':hit,'known_positive_documents':len(positive)}
