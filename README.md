# Multiclass Naive Bayes SVM (NB-SVM)
Luis Rei 
luis.rei@ijs.si 
@lmrei
http://luisrei.com


Learns a multiclass classifier (OneVsRest) based on word ngrams.

Uses scikit learn. Reads input from TSV files. Text is expected to already be tokenized.

Original paper (binary classifier): Sida Wang and Christopher D. Manning: Baselines and Bigrams: Simple, Good Sentiment and Topic Classification; ACL 2012. [http://nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf](http://nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf)

Based on a work at [https://github.com/mesnilgr/nbsvm](https://github.com/mesnilgr/nbsvm):
Naive Bayes SVM by Grégoire Mesnil

### Notes:
 - I modified the code from Mesnil for multiclass because I had neutral sentiment.
 - This does not use the MNB + SVM ensemble from the original paper, just the SVM part (same as Mesnil)


### License
<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">Multiclass Naive Bayes SVM (NB-SVM)</span> by <a xmlns:cc="http://creativecommons.org/ns#" href="http://luisrei.com/" property="cc:attributionName" rel="cc:attributionURL">Luis Rei</a> is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.<br />Based on a work at <a xmlns:dct="http://purl.org/dc/terms/" href="https://github.com/mesnilgr/nbsvm" rel="dct:source">https://github.com/mesnilgr/nbsvm</a>.
