"""
Little script for backing up / restoring the Weaviate index.

Positional arguments:
    1. either 'backup' or 'restore'.
    2. a backup ID. defaults to '1'.

Call like:  python weaviate_backup_restore backup 2 
"""


import weaviate
import os 

WEAVIATE_HOST = os.environ['WEAVIATE_HOST']
LOGGING_PATH = os.environ['LOGGING_PATH']
DEFAULT_BACKUP_ID = '1' # We need to give backups an ID


weaviate_client = weaviate.Client(
        url="http://"+WEAVIATE_HOST+":8080/",
        # additional_headers={
        #     "X-OpenAI-Api-Key": openai_key.OPENAI_KEY
        # }
    )
assert weaviate_client.is_ready()    




if __name__ == '__main__':
    from sys import argv
    
    # Get backup id if it's provided as the 2nd command-line argument
    backup_id = argv[2] if len(argv) > 2 else DEFAULT_BACKUP_ID
    
    if argv[1] == 'backup':
        # do backup
        result = weaviate_client.backup.create(
            backup_id=backup_id,
            backend="filesystem",
            # include_classes=["DocumentChunk"], # it should include all classes if we don't provide this.
            wait_for_completion=True,
        )
        print(result)
    
    
    
    elif argv[1] == 'restore':
        # clear out everything from Weaviate
        weaviate_client.schema.delete_all()
        
        # do restore
        result = weaviate_client.backup.restore(
            backup_id=backup_id,
            backend="filesystem",
            # exclude_classes="Article",
            wait_for_completion=True,
        )
        print(result)
        
    
    else:
        print('Please give an argument either "backup" or "restore," (followed optionally by a backup ID).')