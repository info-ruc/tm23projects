import sys
import time
import random
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

BATCH_SIZE = 1000

# Elastic configuration.
ELASTIC_ADDRESS = "http://elastic:Zhld,123@10.230.107.101:29200"
INDEX_NAME = "renda_dense_index768"
# Uncomment the following lines if start ES with SECURITY ENABLED.
#ELASTIC_ADDRESS = "https://localhost:9200"
#ELASTIC_USER = "elastic"
#ELASTIC_PASSWORD = "admin123"
#CA_CERTS_PATH = "path/to/http_ca.crt"

def index_documents(article_doi_filename,question_en_documents_filename,answer_en_documents_filename,question_en_vectors_filename,question_cn_documents_filename,answer_cn_documents_filename,question_cn_vectors_filename,index_name,client):
    # Open the file containing text.
    with open(article_doi_filename) as article_doi_file,open(question_en_documents_filename, "r") as question_en_documents_file,open(answer_en_documents_filename, "r") as answer_en_documents_file,open(question_cn_documents_filename,"r") as question_cn_documents_file,open(answer_cn_documents_filename,"r") as answer_cn_documents_file:
        # Open the file containing vectors.
        with open(question_en_vectors_filename, "r") as question_en_vectors_file,open(question_cn_vectors_filename) as question_cn_vectors_file:
            documents=[]

            # For each document creates a JSON document including both text and related vector.
            for index, (article_doi,question_en_document,answer_en_document,question_en_vector,question_cn_document,answer_cn_document,question_cn_vector) in enumerate(
                    zip(article_doi_file,question_en_documents_file,answer_en_documents_file,question_en_vectors_file,question_cn_documents_file,answer_cn_documents_file,question_cn_vectors_file)):

                question_en_vector_str=[float(w) for w in question_en_vector.split(",")]
                question_cn_vector_str=[float(w) for w in question_cn_vector.split(",")]

                doc = {
                        "doi": article_doi,
                        "faq_question_en": question_en_document,
                        "faq_answer_en": answer_en_document,
                        "faq_question_cn": question_cn_document,
                        "faq_answer_cn": answer_cn_document,
                        "faq_question_en_vector": question_en_vector_str,
                        "faq_question_cn_vector": question_cn_vector_str,
                }
         #       print("doc=%s" % doc)
                # Append JSON document to a list.
                documents.append(doc)

                # To index batches of documents at a time.
                if index % BATCH_SIZE == 0 and index != 0:
                    # How you'd index data to Elastic.
                    indexing = bulk(client, documents, index=index_name)
                    documents = []
                    print("Success - %s , Failed - %s" % (indexing[0], len(indexing[1])))
            # To index the rest, when 'documents' list < BATCH_SIZE.
            if documents:
                bulk(client, documents, index=index_name)
            print("Finished")

def main():
    article_doi_filename = sys.argv[1]

    question_en_documents_filename = sys.argv[2]
    answer_en_documents_filename = sys.argv[3]
    question_en_vectors_filename = sys.argv[4]

    question_cn_documents_filename = sys.argv[5]
    answer_cn_documents_filename = sys.argv[6]
    question_cn_vectors_filename = sys.argv[7]

    # Declare a client instance of the Python Elasticsearch library.
    client = Elasticsearch(hosts=[ELASTIC_ADDRESS],request_timeout=60, max_retries=10, retry_on_timeout=True)
    # Use this instead, IF using SECURITY ENABLED.
    # client = Elasticsearch(hosts=[ELASTIC_ADDRESS], ca_certs=CA_CERTS_PATH, basic_auth=("elastic", ELASTIC_PASSWORD))

    initial_time = time.time()
    index_documents(article_doi_filename,question_en_documents_filename,answer_en_documents_filename,question_en_vectors_filename,question_cn_documents_filename,answer_cn_documents_filename,question_cn_vectors_filename,INDEX_NAME, client)
    finish_time = time.time()
    print('Documents indexed in {:f} seconds\n'.format(finish_time - initial_time))


if __name__ == "__main__":
    main()
