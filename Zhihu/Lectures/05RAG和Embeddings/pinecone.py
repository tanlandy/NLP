from pinecone import Pinecone

pc = Pinecone(api_key="b189cf1c-f2ec-4251-85b2-02ca6a3638ae")

from openai import OpenAI
client = OpenAI()

res = client.embeddings.create(
  model="text-embedding-ada-002",
  input="The food was delicious and the waiter...",
  encoding_format="float"
)

emb_res = res.data[0].embedding

index = pc.Index("grammardb")

index.upsert(
    vectors=[
        {
            "id": "vec1", 
            "values": emb_res, 

        }, 
    ],
    namespace= "ns1"
)

