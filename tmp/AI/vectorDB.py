from pymilvus import connections, CollectionSchema, FieldSchema, DataType, Collection
from pymilvus import utility


connections.connect(
    alias="default",
    user="username",
    password="password",
    host="localhost",
    port="19530",
)

def create_collection():
    document_id = FieldSchema(
        name="document_id",
        dtype=DataType.INT64,
        is_primary=True,
    )
    document_type = FieldSchema(
        name="document_type",
        dtype=DataType.INT64,
    )
    document_identifier = FieldSchema(
        name="document_identifier",
        dtype=DataType.INT64,
    )

    document_vector = FieldSchema(
      name="document_vector",
      dtype=DataType.FLOAT_VECTOR,
      dim=2048
    )

    schema = CollectionSchema(
        fields=[document_id, document_type, document_identifier,document_vector],
        description="document lookup table ",
        enable_dynamic_field=True,
    )
    collection_name = "documents"
    collection = Collection(
        name=collection_name, schema=schema, using="default", shards_num=2
    )

def lookup_document( query_vector):
    collection = Collection("documents")  
    print("Searching for similar documents.")
    try:
        search_params = {
            "metric_type": "L2", 
            "offset": 0, 
            "ignore_growing": False, 
            "params": {"nprobe": 10}
        }
        results = collection.search(
            data=[query_vector],
            anns_field="document_vector",
            param=search_params,
            limit=1,
            expr=None,
            consistency_level="Strong"
        )
        return results
    except Exception as e:
        print("Error with search:", e)
        return None

def add_document(document_type, document_identifier, document_vector):
    collection = Collection("documents")  
    collection.insert(
        [
            [document_type, document_identifier, document_vector],
        ]
    )
    print("Document added successfully.")
create_collection()