from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
import time

app = FastAPI()

# The sentence we like to encode
#sentences = ["what is a bank transit number"]

sentences = ["Download article as a PDF. An employer identification number"]

# Load or create a SentenceTransformer model.
# Replace the model name with 'sentence-transformers/msmarco-distilbert-base-dot-prod-v3' to map the query to
# a 768 dimensional dense vector space.
model = SentenceTransformer('../../models/bert-multilingual-passage-reranking-msmarco')


@app.get("/embeddingsentence")
def main(sentence):
    print("sentence: %s" % sentence)
    #sentence = ["Download article as a PDF. An employer identification number"]
    #sentence = "Download article as a PDF. An employer identification number"

    #sentence = "[" +sentence +"]"
    #sentence = [sentence]
    print("sentence: %s" % sentence)
    try:
        initial_time = time.time()
        result = model.encode(sentence)
        print("result: %s" % result)
        #vector = list(result)
        finish_time = time.time()
        #print("vector: %s" % vector)
        print('Vectors created in {:f} seconds\n'.format(finish_time - initial_time))

        #for v in result:
        #list=result.tolist()
            #print("v: %s" % v)
        joinedvector = ",".join([str(i) for i in result.tolist()])
        print("joinedvector: %s" % joinedvector)

        return {"vec": "["+joinedvector+"]"}
    except Exception as e:
        print("an error occured %s" % str(e))
        return {"error": str(e)}

# Compute sentence embeddings.
embeddings = model.encode(sentences)
print(embeddings)

# Creates a list object, comma separated.
vector = list(embeddings)
print(vector)
