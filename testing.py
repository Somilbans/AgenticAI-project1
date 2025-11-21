import chromadb
client = chromadb.PersistentClient(path="mini_chroma")
# list all collections
print(client.list_collections())
collection = client.get_collection("availablepositions")
sample = collection.peek()
# print(sample["documents"])
# print(sample["metadatas"])

rows = collection.get(include=["documents", "metadatas"], limit=10)
for doc, meta in zip(rows["documents"], rows["metadatas"]):
   print(doc)
   print(meta)
   print("---\n\n\n")