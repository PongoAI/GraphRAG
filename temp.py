from interfaces.vector_dbs.datastax_db import DatastaxDB
import os

# Initialize DatastaxDB instance
db = DatastaxDB()

# Assuming we have a collection named 'us_people' and it's already created
collection_name = 'hotpot_qa'

results = db.search(collection_name, 'Who was in the us?', k=5)
print(results)


