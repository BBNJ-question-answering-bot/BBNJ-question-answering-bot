""" 
This script has all the code for running the BBNJ Question-Answering Bot App.

It includes: 
- the web server, via Gradio 
- user interface 
- routines for searching the embeddings index for relevant passages 
- OpenAI API calls, along with prompt templates 

When run via the command line, it will run the web server provided by Gradio.
"""


import gradio as gr
import openai_key
import weaviate
import json
import os 



# Grabs the OpenAI API key from a separate file
import openai
openai.api_key = openai_key.OPENAI_KEY #openai_key.OPENAI_KEY_gpt4

# These parameters are set in docker-compsoe
WEAVIATE_HOST = os.environ['WEAVIATE_HOST']
LOGGING_PATH = os.environ['LOGGING_PATH']

# Max tokens for the query to send to GPT. This should be a bit under the model's context size limit, to make sure we have room for the response.
MAX_TOKENS = 3000 #000

# Model name
MODEL_NAME = 'gpt-3.5-turbo' #'gpt-4'



# Set up logging
import logging
import logging.handlers
log_handler = logging.handlers.RotatingFileHandler(os.path.join(LOGGING_PATH,'log'), maxBytes=1000000,backupCount=10)
log_formatter = logging.Formatter('{asctime}\t{levelname:8s}\t{message}', style='{')
log_handler.setFormatter(log_formatter)
logger = logging.getLogger()
logger.setLevel('INFO')
logger.addHandler(log_handler)




# For token counting, get the encoding used by the GPT and text-embedding-ada models.
import tiktoken
oiaEncoding = tiktoken.get_encoding('cl100k_base')
def countTokens(string):
    """Returns the number of tokens in a text string."""
    return len(oiaEncoding.encode(string))


# Define the groups of documents turned on/off by the checkboxes in the interface. These are the labels for the checkboxes, paired with the corresponding document IDs in the vector index.
document_groups = [
    ('BBNJ final draft agreement', [0]),
    ('Prior 5th-session draft agreements', [1,5,45]),
    ('Small group work outcomes', [2]),
    ('Delegates\' submitted proposals',[6]),
    ('President\'s statement on suspension',[46]),
    ('Party statements', [7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]),
    ('Earth Negotiations Bulletin Reports: initial 5th session', [35,36,37,38,39,40,41,42,43,44]),
    ('Earth Negotiations Bulletin Reports: resumed 5th session', [25,26,227,28,29,30,31,32,33,34]),
]
document_checkboxes=[d[0] for d in document_groups] # just get the list of names to pass to Gradio

def checkboxIndicestoDocumentIds(checkboxIndices):
    """Given the list of checkboxes that are ticked, return all of the document indices to include in the embedding search.

    Args:
        checkboxIndices [integer]: A list with the indices of the selected checkboxes in the UI, corresponding to their order in 'document_groups'. This is generated by gradio.

    Returns:
        [integer]: a list with all of the document indices to include in the embedding search.
    """
    documentIds = []
    for i in checkboxIndices:
        documentIds.extend(document_groups[i][1])
    return documentIds






def fetchchunks(query, documentIds, chunkcount=100, collection_name="DocumentChunk"):
    """This retrieves the most relevant text chunks (or "passages") from the embeddings index, and returns them.

    Args:
        query (string): The user's question; we will search the embeddings index for the embeddings most relevant to this text string.
        documentIds (list of integers): A list with the ID's of documents to include in the search.
        chunkcount (int, optional): Maximum number of chunks to return. Defaults to 100.
        collection_name (str, optional): Collection to search in the Weaviate index. (a.k.a. the database name.) Defaults to "DocumentChunk".

    Raises:
        Exception: if the Weaviate client gives us any error connecting to the database or doing the search.

    Returns:
        list: The collection of Weaviate vector objects returned by the query, ordered by relevance (descending).
    """
    
    # Connect to the vector index
    weaviate_client = weaviate.Client(
        url="http://"+WEAVIATE_HOST+":8080/",
        additional_headers={
            "X-OpenAI-Api-Key": openai_key.OPENAI_KEY
        }
    )
    assert weaviate_client.is_ready()    
    
    
    # Form pieces of the query to run on Weaviate
    nearText = {
        "concepts": [query],
        "distance": 0.6,
    }
    properties = [
        "documentId", "documentTitle", "chunkNumber", "header", "content",
        "_additional {certainty distance}"
    ]
    documentId_filter = {
        'operator': 'Or',
        'operands': [
            {
                "path": ["documentId"],
                "operator": "Equal",
                "valueString": str(documentId),
            }
            for documentId in documentIds
        ]
    }
    

    # Run the query on Weaviate, putting together all the pieces above.
    result = (
        weaviate_client.query
        .get(collection_name, properties)
        .with_where(documentId_filter)
        .with_near_text(nearText)
        .with_limit(chunkcount)
        .do()
    )
    
    # Check for errors
    if ("errors" in result):
        print ("\033[91mYou probably have run out of OpenAI API calls for the current minute – the limit is set at 60 per minute.")
        raise Exception(result["errors"][0]['message'])
    
    
    # Return the ordered list of vectors found in the search.
    return result["data"]["Get"][collection_name]





class DocumentStringBuilder:
    """Used for dumping all of the relevant chunks into a continuous string to include in an LLM prompt, grouping chunks by their document name, and keeping track of the number of tokens. This helps us put together the prompt to send to the LLM, grouping together chunks from the same document.
    """
    
    def __init__(self, maxTokens=3500):
        """Create a new empty DocumentStringBuilder, passing the maximum number of tokens.

        Args:
            maxTokens (int, optional): Maximum number of tokens to allow. Defaults to 3500.
        """
        self.maxTokens = maxTokens 
        
        # number of tokens included so far
        self.tokenCount = 0
        
        
        """
        Store documents in a dictionary shaped like this:
        {documentId: {
                        "documentTitle": documentTitle,
                        "insertOrder": int
                        "headers": {
                            "header": "header string",
                            "chunks": [ # list of chunks
                                {
                                    "content": "content",
                                    "chunkNumber": chunkNumber
                                }
                            ]
                        }
                     }
            
        }
        """
        self.documents = {} 
    
    
    
    def addChunk(self, documentId, documentTitle, header, content, chunkNumber):
        """Add the chunk, throws an OverflowError if it's too big to fit within the token limit. (This is a rough approximation because I'm too lazy to properly count tokens in all the document titles and whitespace and stuff, it might end up slightly over the limit which is fine.)

        Args: Fields from weaviate database
        """
    
        chunkSize = countTokens(content)
                
        # Raise an OverflowError if this would give us too many tokens
        if chunkSize + self.tokenCount > self.maxTokens:
            raise OverflowError
        
        # Add this new document if it's not already in our collection.
        if documentId not in self.documents:
            self.documents[documentId] = {
                "documentTitle": documentTitle,
                "insertOrder": len(self.documents),
                "headers": {}
            }
            
            # Add the token count of the document title, because we'll need to print it
            self.tokenCount += countTokens(documentTitle)    
            
        
        # For some reason, some of the headers are coming back as None. Maybe weaviate is storing them that way? That's breaking things, so I'm just going to convert None headers to an empty string.
        if header is None:
            header = ''
        
        
        # Add this header if it's not already in the collection.    
        if header not in self.documents[documentId]['headers']:
            self.documents[documentId]['headers'][header] = {
                "header": header,
                "chunks": []
            }
            
            # Count the tokens for the header
            try:
                self.tokenCount += countTokens(header)
            except TypeError:
                raise TypeError('The problem objects are:', documentId, documentTitle, header, content, chunkNumber)
            
        
        # add the chunk contents
        self.documents[documentId]['headers'][header]['chunks'].append({
            "content": content,
            "chunkNumber": chunkNumber
        })    
        self.tokenCount += countTokens(content)
            
            
            
            
        
    def __str__(self):
        """Output the string representation of these chunks, suitable to insert into an LLM query.
        """
        
        lines = []
        
        for documentId, document in sorted(self.documents.items(), key=lambda doc: doc[1]['insertOrder']):
            
            # This is a terribly lazy hack to end the word "final" to the title of the final draft, because I don't want to bother updating the index right now.
            extra_hacky_final_word = 'FINAL ' if documentId =='0' else '' 
            
            # print document title
            lines.append('From document "' + extra_hacky_final_word + document['documentTitle'].upper() + '":')
            
            # try to sort headers in order of their appearance in the document, grabbing a number from one of its chunks as the sort key. (This will break when there are multiple headers with the same text. I kinda screwed this up.)
            for throwaway, header in sorted(document['headers'].items(), key=lambda h: h[1]['chunks'][0]['chunkNumber']):
                if header != '':
                    lines.append(header['header'] + ':')
                
                previousChunkNumber = 0 # keeping track of whether chunks are consecutive or not, to 
                for chunk in sorted(header['chunks'], key=lambda chunk: chunk['chunkNumber']):
                    if previousChunkNumber != 0 and chunk['chunkNumber'] != previousChunkNumber+1:
                        lines.append('...')
                    lines.append(chunk['content'])
                    previousChunkNumber = chunk['chunkNumber']
            
            # add an extra newline after each document
            lines.append('')
                         
        return '\n'.join(lines)    
                

        
        
    
    


def chunksToText(chunks, maxTokens=MAX_TOKENS):
    """Takes a list of chunks output from the vector database, and turns them into a string that can be sent in an LLM query.

    Args:
        chunks (dict): Dictionary with fields from the vector database schema (documentId, documentTitle, header, content, chunkNumber)
        maxTokens (int, optional): Keep adding documents until we reach this size. Defaults to 3500. This is a rough maximum -- might be a few over due to possibly not counting the last header or document title, or whitespace, etc.

    Returns:
        string: The most-relevant text chunks  x
    """
    
    
    builder = DocumentStringBuilder(maxTokens)
    
    for chunk in chunks:
        try:
            builder.addChunk(
                documentId=chunk['documentId'],
                documentTitle=chunk['documentTitle'],
                header=chunk['header'],
                content=chunk['content'],
                chunkNumber=chunk['chunkNumber']
            )   
        except OverflowError:
            break
        
    return str(builder) 







def fetchOpenaiCompletion(question, chunksString, temperature=0.8):
    """This function actually makes the API call to GPT, and puts together the query from a template.

    Args:
        question (string): The user's question
        chunksString (string): Relevant document passages, formatted into a string suitable to include in the prompt.
        temperature (float, optional): Temperature parameter to pass to GPT. Defaults to 0.8.

    Returns:
        _type_: _description_
    """

    response = openai.ChatCompletion.create(
        # model="gpt-3.5-turbo",
        model=MODEL_NAME,
        messages=[
                {"role": "system", "content": "You are a helpful policy analyst working to understand the UN Biodiversity Beyond National Borders draft agreement."},
                {"role": "user", "content": "You are a helpful policy analyst working to understand the UN Biodiversity Beyond National Borders draft agreement."},
                {"role": "user", "content": "Below are some paragraphs to consider from various documents in the UN negotiations process, including drafts of the agreement, news bulletings about the negotiations, and statements by various parties:\n\n{chunks}".format(chunks=chunksString)},
                {"role": "user", "content": "###\nFrom information in the preceding paragraphs, please try to answer the following question. There are several drafts of the agreement leading up to the final verison; please assume the question refers to the final draft unless otherwise specified.\n\nQuestion: {question}\n\nAnswer:".format(question=question)},
            ],
        temperature=temperature
    )
    
    return response




def runquery(question, documentCheckboxIndices, temperature, request: gr.Request):
    """This function is called when the user presses the "submit" button. It processes data from the input widgets, searches for relevant document passages, makes the OpenAI AIP call, and then returns results for the user to see.

    Args:
        question (string): User input from the "question" box
        documentCheckboxIndices [integer]: List of turned-on document-selection checkboxes, indexed by their order.
        temperature (integer): Temperature parameter to pass to GPT.
        request (gr.Request): Request object from Gradio, so we can get the IP address for logging.

    Returns (as a 2-tuple):
        completion (string): The answer returned from GPT
        chunksString (string): The source passages found in the search step, formatted as a string to show to the user.
    """

    # Log the question
    logger.info('Query:' + question)
    
    
    # Wrap this whole thing in a try block, so we can log any errors.
    try:
        # Get the ID's of documents to include in the embeddings search, from the list of turned-on checkboxes
        documentIds = checkboxIndicestoDocumentIds(documentCheckboxIndices)
    
        # Search the embeddings index for passages most-relevant to the user's question.
        chunks = fetchChunks(question, documentIds)

        # Format the passages into a string to include in the query (and also display this to the user.)
        chunksString = chunksToText(chunks)

        # Fetch the 
        completion = fetchOpenaiCompletion(question,chunksString,temperature)['choices'][0]['message']['content']

        logger.info('{ip}\t{question}\t{documentCheckboxIndices}\t{temperature}\t{completion}'.format(
            ip=request.client.host,
            question=question,
            documentCheckboxIndices=repr(documentCheckboxIndices),
            temperature=temperature,
            completion=completion
            ))

        return completion, chunksString

    # If an exception occurs, make sure we log it.
    except Exception as e:
        logger.exception('exception happened')
        logger.exception('Exception occurred\t{ip}\t{question}\t{documentCheckboxIndices}\t{temperature}'.format(
            ip=request.client.host,
            question=question,
            documentCheckboxIndices=repr(documentCheckboxIndices),
            temperature=temperature,
            ))



# This section sets up the user interface using Gradio

try: # Put the whole thing in a try block so we can log errors
    with gr.Blocks(
            title='BBNJ Question-Answering Bot',
            # analytics=False,
            # theme='Tshackelton/IBMPlex-DenseReadable'
        ) as demo:
        
        # Text at the top
        gr.Markdown("""
            # Experimental BBNJ Question-Answering Bot  
            
            Experimental! Using ChatGPT 3.5, this bot attempts to answer questions from documents in the BBNJ 5th session. 
            When you ask a question, this program first tries to find the most relevant passages from the selected documents (visible under "souce passages"), and then sends them to ChatGPT 3.5 to try to answer your question. You can also record feedback on its answers using this page.          
            """)
        
        with gr.Row():
            
            # Left column for input widgets
            with gr.Column():
                
                # "Question" box
                question = gr.inputs.Textbox(lines=5, label="Question")
                
                # Checkboxes to include/exclude specific documents
                documents = gr.CheckboxGroup(
                    label="Documents",
                    info="Select documents to use in the search",
                    choices=document_checkboxes,
                    value=document_checkboxes,
                    type='index',
                )
                
                # Temperature slider
                temperature = gr.Slider(0,1,0.3, label="ChatGPT Temperature / Randomness")
                
                # Submit button
                btn_run = gr.Button("Submit", variant='primary')

            # Right column for output widgets
            with gr.Column():
                
                # Answer box
                answer = gr.inputs.Textbox(lines=5, label="Answer")
                
                # Collabsible box to see the source passages used.
                with gr.Accordion("Source passages used for the answer", open=False):
                    sourcechunks = gr.inputs.Textbox(lines=10, label="This text was provided to GPT to analyze for its response. We can currently only put about 2500 words here, so a pre-processing step searches for the most relevant chunks in the BBNJ documents.")

                # Collabsible box to record feedback
                with gr.Accordion("Record feedback on this answer", open=False):
                    gr.Markdown('Here you can record feedback for this question/answer from the bot. When you click "save feedback," your feedback will get saved on the server along with the question/answer and settings.')
                    feedback_tags = gr.CheckboxGroup(
                        label="Tags",
                        info="",
                        choices=['Good', 'Wrong', 'Biased', 'Unhelpful'],
                        type='value',
                    )
                    feedback_comment = gr.inputs.Textbox(lines=5, label="Comments/notes on this answer")
                    feedback_name = gr.inputs.Textbox(lines=1, label="Your name")
                    btn_flag = gr.Button('Save Feedback')
                    feedback_message = gr.Markdown("Your feedback is recorded!", visible=False)
                    

        def runbutton_click_helper(question, documentCheckboxIndices, temperature, request: gr.Request):
            """This is in a helper function just to hide the "your feedback has been submitted" message when you submit a new question.
            
                Args: takes the arguments supplied by Gradio on the button click, representing the state of the input widgets. We just pass all of them to runquery()
                
                Returns: A dictionary which sets the state of all the UI widgets after the query is run.
            """
            answer_value, sourcechunks_value = runquery(question, documentCheckboxIndices, temperature, request)
            return {
                answer: answer_value,
                sourcechunks: sourcechunks_value,
                feedback_message: gr.update(value='Your feedback is recorded!', visible=False),
                feedback_comment: '',
                feedback_tags:None,
            }

        # Set up the "submit" button-click action
        btn_run.click(fn=runbutton_click_helper, 
            inputs=[question,documents,temperature], 
            outputs=[answer,sourcechunks,feedback_message,feedback_comment,feedback_tags],
            api_name='compute-answer')
        
        
        # Set up a seperate logger for Gradio's flagging feature, to record user feedback
        flagger = gr.CSVLogger()
        flagger.setup([question, answer, documents, temperature, feedback_tags, feedback_comment, feedback_name], flagging_dir=os.path.join(LOGGING_PATH,'flags'))
        
        # Set up the "record feedback" button-click action
        def btn_flag_click_helper(*args):
            flagger.flag(args)
            return {feedback_message: gr.update(visible=True)}
        btn_flag.click(
            fn=btn_flag_click_helper, 
            inputs=[question, answer, documents, temperature, feedback_tags, feedback_comment, feedback_name],
            outputs=[feedback_message],
            preprocess=False,
            api_name='record-feedback'
            )       
    
# Log any errors we get from the GUI
except Exception as e:
    logger.exception('exception')


# Run the application
demo.launch(server_name="0.0.0.0") 
