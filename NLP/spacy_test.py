# coding=utf-8

text = "The sequel, Yes, Prime Minister, ran from 1986 to 1988. In total there were 38 episodes, of which all but one lasted half an hour. " \
       "Almost all episodes ended with a variation of the title of the series spoken as the answer to a question posed by the same character, " \
       "Jim Hacker. Several episodes were adapted for BBC Radio, and a stage play was produced in 2010, the latter leading to a new television series on UKTV Gold in 2013."

import spacy
nlp = spacy.load('en')
doc = nlp(text)
# for token in doc:
# #     print('"' + token.text + '"')

# for token in doc[:10]:
#     print("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}".format(
#         token.text,
#         token.idx,
#         token.lemma_,
#         token.is_punct,
#         token.is_space,
#         token.shape_,
#         token.pos_,
#         token.tag_
#     ))

# for ent in doc.ents:
#     print(ent.text, ent.label_)

from spacy import displacy
displacy.render(doc, style='ent', jupyter=True)