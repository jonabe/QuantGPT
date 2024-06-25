import os
import logging
from typing import Iterator
from langchain.chat_models import ChatOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage, BaseCallbackHandler, get_response_synthesizer
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuantSimpleVectorStorage:
    """
    A class that represents a simple vector storage for a GPT model.

    Attributes:
            persist_dir (str): The directory where the index and other data will be persisted.
            gpt_model (str): The name of the GPT model to use for predictions.
            gpt_temperature (float): The temperature to use for GPT predictions.
            source_folder (str): The folder containing the markdown files to use for indexing.
    """


class QuantSimpleVectorStorage:
    def __init__(self, persist_dir: str, gpt_model: str, gpt_temperature: float, source_folder: str):
        # collect arguments
        self.persist_dir = persist_dir
        self.gpt_model = gpt_model
        self.gpt_temperature = gpt_temperature
        self.source_folder = source_folder

        Settings.llm = OpenAI(model=self.gpt_model, temperature=self.gpt_temperature, max_tokens=2048, streaming=True) 
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

        node_parser = SimpleNodeParser.from_defaults(separator="\n## ", chunk_size=1024, chunk_overlap=0)

        Settings.node_parser = node_parser
        Settings.num_output = 512
        Settings.context_window = 3900


        # initialize attributes
        self.index = None
        
        # setup index
        self.setup_index()

    def list_sources(self) -> Iterator[str]:
        """
        Returns an iterator over the paths of all markdown files in the source folder, excluding certain files.

        Excluded files:
        - api.md
        - all_pages.md
        - unknown.md
        - chainlit.md
        """
        exclude_files = {"api.md", "all_pages.md", "unknown.md", "chainlit.md"}
        for root, _, files in os.walk(self.source_folder):
            for file in files:
                if file.endswith(".md") and file not in exclude_files:
                    yield os.path.join(root, file)

    def load_index_nodes(self):
        logger.info('Loading documents...')

        node_parser = SimpleNodeParser.from_defaults(separator="\n## ", chunk_size=1024, chunk_overlap=0)
        documents = SimpleDirectoryReader(
            input_files=self.list_sources()).load_data()

        index_nodes = node_parser.get_nodes_from_documents(
            documents, show_progress=True)

        return index_nodes

    def create_index(self):
        logger.info('Building index...')

        index = VectorStoreIndex(
            nodes=self.load_index_nodes(),
            show_progress=True,
        )

        return index

    def setup_index(self):
        """
        Sets up the index for the vector store. If the index is already present in the storage context, it is loaded
        from there. Otherwise, a new index is built from the markdown files in the input directory and saved to the
        storage context for future use.
        """
        try:
            logger.info('Loading index...')
            storage_context = StorageContext.from_defaults(
                persist_dir=self.persist_dir)
            self.index = load_index_from_storage(storage_context)
        except Exception as e:
            logger.info('Persisted Index not found, building new one.')

            # create index
            self.index = self.create_index()

            logger.info('Saving index...')
            self.index.storage_context.persist(persist_dir=self.persist_dir)

    def create_query_engine(self, callback_handler: BaseCallbackHandler = None) -> RetrieverQueryEngine:
        """
        Creates a RetrieverQueryEngine object with the configured VectorIndexRetriever and response synthesizer.

        Returns:
                RetrieverQueryEngine: The created RetrieverQueryEngine object.
        """
       
        # Configure retriever within the service context
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=20,
            vector_store_query_mode="default",
            
        )

        # Configure response synthesizer within the service context
        response_synthesizer = get_response_synthesizer(
            response_mode="tree_summarize", use_async=True)

        # Assemble query engine
        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            # didn't get this to work, no error, query will not return anything when enabled
            # node_postprocessors=[
            #     SimilarityPostprocessor(similarity_cutoff=0.73)
            # ]
        )
        return query_engine
