"""
This script is for building the embeddings index used by the bot.

As input, it takes the JSON files output by Science Parse, with the documents 
split into sections and paragraphs. The documents are listed with some metadata 
in document-manifest.csv.

This script doesn't take any command-line arguments, but the parameters are coded into
this script, including
- connection parameters for the vector index (as weaviate_client)
- manifest file for the documents to include: document-manifest.csv 
- folder of documents to include, as the JSON output by Science parse.

(You can use this to build the index, but the source code also 
has an backup that you can use to restore the index instead. If you want to
add or change the documents, put them in the documents-json folder and
edit document-manifest.csv.)
"""


import json
import csv
import time

import weaviate

import openai
from openai_key import OPENAI_KEY
openai.api_key = OPENAI_KEY





# For token counting, get the encoding used by the GPT and text-embedding-ada models.
import tiktoken
oiaEncoding = tiktoken.get_encoding('cl100k_base')
def countTokens(string):
    """Returns the number of tokens in a text string."""
    return len(oiaEncoding.encode(string))




def weaviate_setup(resetDatabase=False):
    """
    Sets up and returns the weaviate client object. 
    
    Args:
        resetDatabase (boolean, optional): If this is true, it wipes the database and re-initializes it with a new schema.

    Returns:
        Weaviate client object: The weaviate client
    """

    # Connect to Weaviate
    weaviate_client = weaviate.Client(
        url="http://localhost:8080/",
        additional_headers={
            "X-OpenAI-Api-Key": openai.api_key
        }
    )
    assert weaviate_client.is_ready()


    # The schema specification for the Weaviate embedding index
    document_schema = {
        "class": "DocumentChunk",
        "description": "A collection of text chunks (mostly paragraphs), from documents",
        "vectorizer": "text2vec-openai",
        "moduleConfig": {
            "text2vec-openai": {
            "model": "ada",
            "modelVersion": "002",
            "type": "text"
            }
        },
        "properties": [{
            "name": "documentId",
            "description": "ID of the document",
            "dataType": ["string"],
            "moduleConfig": { "text2vec-openai": { "skip": True } }
        },
        {
            "name": "documentTitle",
            "description": "Title of the document",
            "dataType": ["string"]
        },
        {
            "name": "chunkNumber",
            "description": "Sequential chunk number within a document",
            "dataType": ["int"],
            "moduleConfig": { "text2vec-openai": { "skip": True } }
        },
        {
            "name": "header",
            "description": "Header for this chunk.",
            "dataType": ["string"]
        },
        {
            "name": "content",
            "description": "Contents of the chunk",
            "dataType": ["text"]
        }]
    }
    
    
    # if we're resetting the database
    if resetDatabase:
        # Delete all of the exiting schemas
        weaviate_client.schema.delete_all()
        # weaviate_client.schema.get()
        
        # Create a new collection with the new schema
        weaviate_client.schema.create_class(document_schema)

    assert weaviate_client.schema.get() 
    
    return weaviate_client






def getChunks(spJsonFileName):
    """Loads a JSON file output from ScienceParse, break/merge them into chunks

    Args:
        spJsonFileName (string): filename to open, with JSON output from Science Parse.
        
    Returns:
        ({string 'header', string 'content'}): an iterator over the chunks, with their heading and content in a dict.
    """
    
    # Grab the "sections" output by Science Parse, from Science Parse's outputted JSON.
    with open(spJsonFileName, 'r') as f:
        data = json.load(f)
    sections = data['metadata']['sections']

    # Keep this external to the loop since we sometimes merge paragraphs together into a single chunk.
    chunk_in_progress = ''
    
    # Loop over all of the sections (as identified by Science Parse.)
    for section in sections:
        
        header = section['heading']
        content = section['text']
        
        # if the section is less than 200 tokens, just return it as a chunk
        if countTokens(content) < 200:
            yield {'header':header,'content':content}
            
        # else, split it into paragraphs/newlines, and return each of them as its own chunk. Except to try to lump togeather really short chunks, combine adjacent chunks that are less than 100 tokens. (This is mostly because the documents have lots of bulleted lists and SP identifies each item as a paragraph; we want to merge adjacent ones together to keep them in context, to some extent.)
        else:
            chunk_in_progress = ''
            for paragraph in content.split('\n'):
                chunk_in_progress += '\n' + paragraph
                if countTokens(chunk_in_progress) > 100:
                    yield {'header':header,'content':chunk_in_progress.strip()}
                    chunk_in_progress = ''
                    
            yield {'header':header,'content':chunk_in_progress.strip()}
            
            



def indexChunks(filename, documentId, documentTitle, weaviate_client):
    """For one document, this function separates it into chunks and loads them into the embeddings index.

    Args:
        filename (string): Name of the JSON file, output from ScienceParse.
        documentId (string): ID for the document, to store in the vector index
        documentTitle (string): Title to store in the index
        weaviate_client: Weaviate client object
    """
    
    weaviate_client.batch.configure(
        batch_size=10, 
        dynamic=True,
        timeout_retries=3,
    #   callback=None,
    )
    
    
    print("Importing Document")

    counter=0

    # Separate the document into chunks, and insert each chunk into the vector index
    for chunk in getChunks(filename):
        if (counter %100 == 0):
            print(f"Import {counter} ") 

        # record to indert, matching the index schema
        properties = {
            "documentId": documentId,
            "documentTitle": documentTitle,
            "chunkNumber": counter,
            "header": chunk['header'],
            "content": chunk['content'],
        }
        
        # insert into the index     
        with weaviate_client.batch as batch:
           batch.add_data_object(properties, "DocumentChunk")
        
        # Sleep so we don't do too many openAI requests (since we have to call their API to compute an embedding for each chunk)        
        time.sleep(60/2000)
        
        counter = counter+1

    print("Importing Document complete")   





def query_weaviate(query, weaviate_client, collection_name="DocumentChunk"):
    """This function isn't actually called in building the index, but I used it to play around with querying and make sure the index was being built properly.

    Args:
        query (string): Query to find nearest passages in the embedding index
        weaviate_client: Weaviate client object
        collection_name (str, optional): The collection/database name. Defaults to "DocumentChunk".

    Raises:
        Exception: if the API call gives us an error, probably for exceeding the rate liit.

    Returns: the object from Weaviate with the search results.
    """
    
    # Set up components of the query
    nearText = {
        "concepts": [query],
        "distance": 0.7,
    }
    properties = [
        "documentId", "documentTitle", "chunkNumber", "header", "content",
        "_additional {certainty distance}"
    ]

    # Run the query on the weaviate client
    result = (
        weaviate_client.query
        .get(collection_name, properties)
        .with_near_text(nearText)
        .with_limit(10)
        .do()
    )
    
    # Check for errors
    if ("errors" in result):
        print ("\033[91mYou probably have run out of OpenAI API calls for the current minute – the limit is set at 60 per minute.")
        raise Exception(result["errors"][0]['message'])
    
    return result["data"]["Get"][collection_name]




def load_document_information(filename='document-manifest.csv'):
    """
    Open a CSV file that lists information about each document, and then yield them one document at a time.
    """
    with open(filename) as f:
        reader = csv.DictReader(f)
        for line in reader:
            yield line



# If we're running this is a command, then load all the documents from document-manifest.csv and add them to the index.
if __name__ == '__main__':
    
    # Set up the database connection    
    weaviate_client = weaviate_setup()
    
    # Iterate over all the documents in document-manifest.csv
    for document in load_document_information():
        #print(document)
        
        # Chunk it up and add it to the index
        indexChunks(
            filename='documents-json/'+document['document_id']+'.pdf.json',
            documentId=document['document_id'],
            documentTitle=document['document_name'],
            weaviate_client=weaviate_client
        )
        
