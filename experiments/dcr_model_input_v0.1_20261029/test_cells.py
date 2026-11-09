"""Small field/length/cutoff fixtures only; no tokenizer, weights, reference or model calls."""
import unittest
from prepare_inputs import validate_historical_pair,length_statistics


class CellTests(unittest.TestCase):
    def pair(self):return {'query_id':'q','document_id':'d','doc_prefix_end_char':3,'score':.5,'input_tokens':10}

    def test_exact_prefix_and_empty(self):
        a=self.pair();b=self.pair();end,digest=validate_historical_pair(a,b,'abcd')
        self.assertEqual(end,3)
        import hashlib
        self.assertEqual(digest,hashlib.sha256(b'abc').hexdigest())
        a['doc_prefix_end_char']=b['doc_prefix_end_char']=0
        self.assertEqual(validate_historical_pair(a,b,'')[1],hashlib.sha256(b'').hexdigest())

    def test_mismatch_refuses_new_cutoff(self):
        a=self.pair();b=self.pair();b['doc_prefix_end_char']=2
        with self.assertRaises(AssertionError):validate_historical_pair(a,b,'abcd')
        b=self.pair();a['doc_prefix_end_char']=9;b['doc_prefix_end_char']=9
        with self.assertRaises(AssertionError):validate_historical_pair(a,b,'abcd')
        a=self.pair();b=self.pair();b['query_id']='other'
        with self.assertRaises(AssertionError):validate_historical_pair(a,b,'abcd')

    def test_nonfinite_and_lengths(self):
        a=self.pair();b=self.pair();b['score']=float('nan')
        with self.assertRaises(AssertionError):validate_historical_pair(a,b,'abcd')
        self.assertEqual(length_statistics([3,1,2])['total'],6)
        self.assertEqual(length_statistics([3,1,2])['min'],1)
        self.assertEqual(length_statistics([3,1,2])['max'],3)


if __name__=='__main__':unittest.main()
