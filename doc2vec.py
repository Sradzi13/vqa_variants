from collections import defaultdict
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
from nltk.tokenize import word_tokenize
#nltk.download('punkt')

"""
# prefix declarations
PREFIX foo: <http://example.com/resources/>
...
# dataset definition
FROM ...
# result clause
SELECT ...
# query pattern
WHERE {
    ...
}
# query modifiers
ORDER BY ...


PREFIX foaf:  <http://dbpedia.org>
SELECT ?name
WHERE {
    ?person foaf:name ?name .
}


select distinct ?Concept where {[] a ?Concept} LIMIT 100
"""
def prepare_data():
    data = ["I love machine learning. Its awesome.",
        "I love coding in python",
        "I love building chatbots",
        "they chat amagingly well"]

    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]
    return tagged_data

def train(tagged_data):
    max_epochs = 100
    vec_size = 20
    alpha = 0.025

    model = Doc2Vec(size=vec_size,
                    alpha=alpha,
                    min_alpha=0.00025,
                    min_count=1,
                    dm=1)

    model.build_vocab(tagged_data)

    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha

    model.save("d2v.model")
    print("Model Saved")

def model_manipulation():
    model = Doc2Vec.load("d2v.model")
    # to find the vector of a document which is not in training data
    test_data = word_tokenize("I love chatbots".lower())
    v1 = model.infer_vector(test_data)
    print("V1_infer", v1)

    # to find most similar doc using tags
    similar_doc = model.docvecs.most_similar('1')
    print(similar_doc)

    # to find vector of doc in training data using tags or in other words, printing the vector of document at index 1 in training data
    print(model.docvecs['1'])


def main():
    tagged_data = prepare_data()
    print(tagged_data)
    train(tagged_data)
    model_manipulation()

if __name__ == "__main__":
    main()