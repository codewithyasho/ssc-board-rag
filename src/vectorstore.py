'''
vectorstore.py, contains functions to create vectorstore, load vectorstore and add new documents to existing vectorstore
'''

# VECTORSTORE RELATED IMPORTS
from langchain_classic.vectorstores import FAISS


# 1. creating a new vectorstore from scratch

def create_vectorstore(documents, embeddings):
    """
    Create a new FAISS vectorstore from documents and embeddings.
    """
    try:
        vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=embeddings,
            distance_strategy='COSINE'  # Better for normalized embeddings
        )

        print(f"\n[INFO] Vector dimension: {vectorstore.index.d}")

        print(
            f"[INFO] Total Vectors in the store: <{vectorstore.index.ntotal}>")
        print("=" * 50)

        # Save
        vectorstore.save_local("faiss_index")
        print("\n✅✅ Successfully CREATED and SAVED Vectorstore to disk.")

        return vectorstore

    except Exception as e:
        print(f"❌ Error during embedding and storing: {e}")


# 2. loading existing vectorstore from disk

def load_vectorstore(embeddings, vectorstore_path="faiss_index"):
    """
    Load an existing FAISS vectorstore.
    """

    try:
        vectorstore = FAISS.load_local(
            vectorstore_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )

        print(f"\n[INFO] Vector dimension: {vectorstore.index.d}")

        print(
            f"[INFO] Total Vectors in the store: <{vectorstore.index.ntotal}>")
        print("=" * 50)

        print("\n✅✅ Successfully LOADED Vectorstore.")

        return vectorstore

    except Exception as e:
        print(f"❌ Error during LOADING: {e}")


# ============================================================================
## END OF FILE ##
