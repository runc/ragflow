import unittest
from unittest.mock import patch, MagicMock, ANY

from rag.utils.milvus_conn import MilvusConnection
from rag.utils.doc_store_conn import DocStoreConnection, MatchDenseExpr # Assuming MatchDenseExpr is relevant
from rag.settings import settings # For potential config override, though less used with heavy mocking

# Define a test tenant ID
TEST_TENANT_ID = "test_milvus_tenant"
TEST_INDEX_NAME = "test_idx"
TEST_KB_ID = "test_kb_123"
TEST_COLLECTION_NAME = f"{TEST_INDEX_NAME}_{TEST_KB_ID}".replace("-", "_")
DEFAULT_VECTOR_SIZE = 128 # Example vector size

class TestMilvusConnection(unittest.TestCase):

    def setUp(self):
        """
        Set up for each test.
        This will involve patching pymilvus to avoid actual Milvus calls.
        """
        # Patch pymilvus.connections.connect
        self.mock_connections = patch('pymilvus.connections').start()
        self.mock_connect = self.mock_connections.connect
        self.mock_connect.return_value = MagicMock() # Mock connection object

        # Patch pymilvus.utility
        self.mock_utility = patch('pymilvus.utility').start()
        self.mock_utility.has_database.return_value = True # Assume DB exists or is created
        self.mock_utility.list_database.return_value = [f"rag_{TEST_TENANT_ID}"] # For older check
        self.mock_utility.has_collection.return_value = False # Default: collection doesn't exist
        self.mock_utility.drop_collection.return_value = None

        # Patch pymilvus.Collection
        self.mock_collection_class = patch('pymilvus.Collection').start()
        self.mock_collection_instance = MagicMock()
        self.mock_collection_class.return_value = self.mock_collection_instance

        # Mock schema details for Collection instance
        self.mock_collection_instance.schema = MagicMock()
        mock_id_field = MagicMock(name='id', dtype=pymilvus.DataType.VARCHAR, is_primary=True)
        mock_kb_id_field = MagicMock(name='kb_id', dtype=pymilvus.DataType.VARCHAR)
        mock_vec_field = MagicMock(name=f'q_{DEFAULT_VECTOR_SIZE}_vec', dtype=pymilvus.DataType.FLOAT_VECTOR, dim=DEFAULT_VECTOR_SIZE)
        self.mock_collection_instance.schema.fields = [mock_id_field, mock_kb_id_field, mock_vec_field]

        self.mock_collection_instance.name = TEST_COLLECTION_NAME
        self.mock_collection_instance.description = "Mocked Collection"
        self.mock_collection_instance.is_empty = True # Or num_entities
        self.mock_collection_instance.num_entities = 0
        self.mock_collection_instance.primary_field = mock_id_field

        # Mock collection methods
        self.mock_collection_instance.load.return_value = None
        self.mock_collection_instance.release.return_value = None
        self.mock_collection_instance.insert.return_value = MagicMock(insert_count=1, primary_keys=["test_id_1"], err_count=0)
        self.mock_collection_instance.delete.return_value = MagicMock(delete_count=1)
        self.mock_collection_instance.query.return_value = [] # Default empty query result
        self.mock_collection_instance.search.return_value = [[]] # Default empty search result (list of lists of hits)
        self.mock_collection_instance.flush.return_value = None
        self.mock_collection_instance.create_index.return_value = None


        # Instantiate MilvusConnection with the test tenant
        # This needs to happen *after* patches are started if __init__ makes Milvus calls.
        self.milvus_conn = MilvusConnection(tenant=TEST_TENANT_ID)
        # Ensure the connection alias used by MilvusConnection matches what might be expected by mocks if they check 'using'
        self.milvus_conn.db_name = f"rag_{TEST_TENANT_ID}".replace("-","_")


    def tearDown(self):
        """
        Clean up after each test.
        Stop all patches.
        """
        patch.stopall()

    def test_connection_and_health(self):
        """Test MilvusConnection instantiation and health check."""
        self.assertIsNotNone(self.milvus_conn)
        self.assertIsNotNone(self.milvus_conn.conn) # Check if connection object was set up

        # Mock the specific utility call for health if not covered by broader setUp mocks
        self.mock_utility.list_collections.return_value = ["some_collection"] # Simulate successful listing

        is_healthy = self.milvus_conn.health()
        self.assertTrue(is_healthy)
        self.mock_connections.connect.assert_called_with(
            alias=f"default_{TEST_TENANT_ID}",
            host=settings.MILVUS.get("host","localhost"), # Assuming settings are available
            port=settings.MILVUS.get("port","19530")
        )
        self.mock_utility.list_collections.assert_called_with(
            using=f"default_{TEST_TENANT_ID}",
            db_name=self.milvus_conn.db_name
        )

    def test_create_delete_index_exist(self):
        """Test creating, checking existence, and deleting an index (collection)."""
        # createIdx
        # 1. Initial state: collection does not exist
        self.mock_utility.has_collection.return_value = False
        self.milvus_conn.createIdx(TEST_INDEX_NAME, TEST_KB_ID, DEFAULT_VECTOR_SIZE)

        self.mock_collection_class.assert_called_with(
            name=TEST_COLLECTION_NAME,
            schema=ANY, # Check schema fields more specifically if needed
            using=f"default_{TEST_TENANT_ID}",
            db_name=self.milvus_conn.db_name
        )
        self.mock_collection_instance.create_index.assert_called_once()
        self.mock_collection_instance.load.assert_called_once()

        # indexExist - after creation
        self.mock_utility.has_collection.return_value = True # Simulate collection now exists
        exists = self.milvus_conn.indexExist(TEST_INDEX_NAME, TEST_KB_ID)
        self.assertTrue(exists)
        self.mock_utility.has_collection.assert_called_with(
            collection_name=TEST_COLLECTION_NAME,
            using=f"default_{TEST_TENANT_ID}",
            db_name=self.milvus_conn.db_name
        )

        # deleteIdx
        self.milvus_conn.deleteIdx(TEST_INDEX_NAME, TEST_KB_ID)
        self.mock_utility.drop_collection.assert_called_with(
            collection_name=TEST_COLLECTION_NAME,
            using=f"default_{TEST_TENANT_ID}",
            db_name=self.milvus_conn.db_name
        )

        # indexExist - after deletion
        self.mock_utility.has_collection.return_value = False # Simulate collection is deleted
        exists_after_delete = self.milvus_conn.indexExist(TEST_INDEX_NAME, TEST_KB_ID)
        self.assertFalse(exists_after_delete)

    def test_insert_and_get_data(self):
        """Test inserting data, getting total count, getting a document, and getting chunk IDs."""
        # Assume collection exists for this test (or call createIdx and mock its internals again)
        self.mock_utility.has_collection.return_value = True
        self.milvus_conn.createIdx(TEST_INDEX_NAME, TEST_KB_ID, DEFAULT_VECTOR_SIZE) # Call to ensure schema setup on mock_collection_instance

        sample_doc_id = "doc1_chunk1"
        sample_vector = [0.1] * DEFAULT_VECTOR_SIZE
        sample_docs = [{
            "id": sample_doc_id,
            "kb_id": TEST_KB_ID,
            "doc_id": "doc1",
            "content_ltks": "This is a test document.",
            # The MilvusConnection.insert expects 'embedding' or the actual vector field name
            "embedding": sample_vector
        }]
        vector_field_name = f'q_{DEFAULT_VECTOR_SIZE}_vec'

        # Mock insert
        mock_mutation_result = MagicMock(insert_count=len(sample_docs), primary_keys=[d["id"] for d in sample_docs], err_count=0)
        self.mock_collection_instance.insert.return_value = mock_mutation_result

        insert_errors = self.milvus_conn.insert(sample_docs, TEST_INDEX_NAME, TEST_KB_ID)
        self.assertEqual(insert_errors, []) # Expect empty list on success

        # Prepare data for Milvus insert call check (vector field name is mapped)
        expected_insert_payload = []
        for doc in sample_docs:
            payload_doc = doc.copy()
            payload_doc[vector_field_name] = payload_doc.pop("embedding")
            expected_insert_payload.append(payload_doc)
        self.mock_collection_instance.insert.assert_called_with(expected_insert_payload)
        self.mock_collection_instance.flush.assert_called()

        # Test getTotal
        self.mock_collection_instance.num_entities = len(sample_docs)
        total = self.milvus_conn.getTotal(TEST_INDEX_NAME, TEST_KB_ID)
        self.assertEqual(total, len(sample_docs))
        # flush is called by getTotal
        self.assertEqual(self.mock_collection_instance.flush.call_count, 2) # Once for insert, once for getTotal

        # Test get
        # Mock query result for get()
        # The actual document stored would have the vector under vector_field_name
        mock_retrieved_doc = {
            "id": sample_doc_id, "kb_id": TEST_KB_ID, "doc_id": "doc1",
            "content_ltks": "This is a test document.",
            vector_field_name: sample_vector
        }
        self.mock_collection_instance.query.return_value = [mock_retrieved_doc]

        retrieved_doc = self.milvus_conn.get(sample_doc_id, TEST_INDEX_NAME, [TEST_KB_ID])
        self.assertIsNotNone(retrieved_doc)
        self.assertEqual(retrieved_doc["id"], sample_doc_id)
        self.assertEqual(retrieved_doc["content_ltks"], sample_docs[0]["content_ltks"])
        # Ensure all schema fields were requested by get()
        # self.get_schema_field_names is not directly mocked, it uses collection.schema
        expected_query_expr = f"id == '{sample_doc_id}'"
        self.mock_collection_instance.query.assert_called_with(
            expr=expected_query_expr,
            output_fields=[f.name for f in self.mock_collection_instance.schema.fields],
            limit=1
        )

        # Test getChunkIds
        # Mock query result for getChunkIds()
        self.mock_collection_instance.query.reset_mock() # Reset call count etc.
        mock_chunk_ids_payload = [{"id": sample_doc_id}]
        self.mock_collection_instance.query.return_value = mock_chunk_ids_payload

        chunk_ids = self.milvus_conn.getChunkIds(TEST_INDEX_NAME, TEST_KB_ID)
        self.assertIn(sample_doc_id, chunk_ids)
        self.assertEqual(len(chunk_ids), 1)
        self.mock_collection_instance.query.assert_called_with(
            expr="id != ''",
            output_fields=["id"],
            limit=ANY, # batch_size for pagination
            offset=0
        )

    def test_delete_data(self):
        """Test deleting data by condition."""
        self.mock_utility.has_collection.return_value = True
        self.milvus_conn.createIdx(TEST_INDEX_NAME, TEST_KB_ID, DEFAULT_VECTOR_SIZE)

        sample_doc_id = "doc_to_delete"
        sample_docs = [{"id": sample_doc_id, "kb_id": TEST_KB_ID, "embedding": [0.2] * DEFAULT_VECTOR_SIZE}]

        # Mock insert
        mock_insert_res = MagicMock(insert_count=1, primary_keys=[sample_doc_id], err_count=0)
        self.mock_collection_instance.insert.return_value = mock_insert_res
        self.milvus_conn.insert(sample_docs, TEST_INDEX_NAME, TEST_KB_ID)

        # Mock delete operation
        self.mock_collection_instance.delete.return_value = MagicMock(delete_count=1)

        delete_condition = {"id": sample_doc_id}
        deleted_count = self.milvus_conn.delete(delete_condition, TEST_INDEX_NAME, TEST_KB_ID)
        self.assertEqual(deleted_count, 1)

        expected_delete_expr = f"id == '{sample_doc_id}'" # Assumes only id in condition for this test
        self.mock_collection_instance.delete.assert_called_with(expected_delete_expr)
        self.mock_collection_instance.flush.assert_called() # delete also calls flush

        # Verify by trying to get the document (should not be found)
        self.mock_collection_instance.query.return_value = [] # Simulate doc not found
        retrieved_doc = self.milvus_conn.get(sample_doc_id, TEST_INDEX_NAME, [TEST_KB_ID])
        self.assertIsNone(retrieved_doc)

        # Verify count if getTotal is called
        self.mock_collection_instance.num_entities = 0
        total_after_delete = self.milvus_conn.getTotal(TEST_INDEX_NAME, TEST_KB_ID)
        self.assertEqual(total_after_delete, 0)

    def test_search_data_dense(self):
        """Test dense vector search."""
        self.mock_utility.has_collection.return_value = True
        # Ensure schema is "created" for the mock collection instance
        self.milvus_conn.createIdx(TEST_INDEX_NAME, TEST_KB_ID, DEFAULT_VECTOR_SIZE)

        query_vector = [0.3] * DEFAULT_VECTOR_SIZE
        vector_field_name = f'q_{DEFAULT_VECTOR_SIZE}_vec' # From createIdx logic

        # Mock search results
        mock_hit1_entity_dict = {"id": "hit1", "content_ltks": "found this one", vector_field_name: [0.31]*DEFAULT_VECTOR_SIZE}
        mock_hit1_entity = MagicMock()
        mock_hit1_entity.to_dict.return_value = mock_hit1_entity_dict
        mock_hit1 = MagicMock(entity=mock_hit1_entity, distance=0.1)

        mock_hit2_entity_dict = {"id": "hit2", "content_ltks": "another find", vector_field_name: [0.32]*DEFAULT_VECTOR_SIZE}
        mock_hit2_entity = MagicMock()
        mock_hit2_entity.to_dict.return_value = mock_hit2_entity_dict
        mock_hit2 = MagicMock(entity=mock_hit2_entity, distance=0.2)

        # Milvus search returns a list of lists of hits (one list per query vector)
        self.mock_collection_instance.search.return_value = [[mock_hit1, mock_hit2]]

        match_expr = MatchDenseExpr(data=query_vector, field="embedding") # 'field' in MatchDenseExpr is conceptual here

        search_results = self.milvus_conn.search(
            selectFields=["id", "content_ltks"],
            highlightFields=[],
            condition={}, # Empty condition for this test
            matchExprs=[match_expr],
            orderBy=None,
            offset=0,
            limit=2,
            indexNames=[TEST_INDEX_NAME], # List of index names
            knowledgebaseIds=[TEST_KB_ID]
        )

        self.assertEqual(len(search_results), 2)
        self.assertEqual(search_results[0]["id"], "hit1")
        self.assertEqual(search_results[0]["score"], 0.1)
        self.assertIn("content_ltks", search_results[0])
        self.assertNotIn(vector_field_name, search_results[0]) # Vector field should be removed by default if not in selectFields

        self.mock_collection_instance.search.assert_called_once_with(
            data=[query_vector],
            anns_field=vector_field_name,
            param=ANY, # Search params like nprobe
            limit=2, # limit (2) + offset (0)
            expr="", # Empty filter expression
            output_fields=['id', 'content_ltks'], # id is auto-added if not present
            consistency_level=ANY
        )

        # Test with offset and limit
        self.mock_collection_instance.search.reset_mock()
        self.mock_collection_instance.search.return_value = [[mock_hit1, mock_hit2]] # Assume backend returns enough

        search_results_offset = self.milvus_conn.search(
            selectFields=["id"],
            highlightFields=[], condition={}, matchExprs=[match_expr], orderBy=None,
            offset=1, limit=1, indexNames=TEST_INDEX_NAME, knowledgebaseIds=[TEST_KB_ID] # Name as str
        )
        self.assertEqual(len(search_results_offset), 1)
        self.assertEqual(search_results_offset[0]["id"], "hit2") # mock_hit2 was second

        self.mock_collection_instance.search.assert_called_once_with(
            data=[query_vector],
            anns_field=vector_field_name,
            param=ANY,
            limit=2, # limit (1) + offset (1)
            expr="",
            output_fields=['id'],
            consistency_level=ANY
        )

    def test_update_data(self):
        """Test updating an existing document."""
        self.mock_utility.has_collection.return_value = True
        self.milvus_conn.createIdx(TEST_INDEX_NAME, TEST_KB_ID, DEFAULT_VECTOR_SIZE)

        doc_id_to_update = "doc_to_update"
        original_vector = [0.4] * DEFAULT_VECTOR_SIZE
        vector_field_name = f'q_{DEFAULT_VECTOR_SIZE}_vec'

        original_doc_data = {
            "id": doc_id_to_update, "kb_id": TEST_KB_ID, "doc_id": "orig_doc",
            "content_ltks": "Original content",
            vector_field_name: original_vector,
            "other_field": "original_value"
        }

        # Mock query to return the original document for the fetch step in update
        self.mock_collection_instance.query.return_value = [original_doc_data]
        # Mock delete success for the deletion step
        self.mock_collection_instance.delete.return_value = MagicMock(delete_count=1)
        # Mock insert success for the re-insertion step
        mock_update_insert_res = MagicMock(insert_count=1, primary_keys=[doc_id_to_update], err_count=0)
        self.mock_collection_instance.insert.return_value = mock_update_insert_res

        update_values = {
            "content_ltks": "Updated content",
            "other_field": "updated_value"
            # Not updating vector in this test case, but could add "embedding": new_vector
        }
        update_condition = {"id": doc_id_to_update, "kb_id": TEST_KB_ID} # kb_id for more specific query in update

        update_successful = self.milvus_conn.update(update_condition, update_values, TEST_INDEX_NAME, TEST_KB_ID)
        self.assertTrue(update_successful)

        # Verify query call for fetching the doc
        expected_query_expr = f"id == '{doc_id_to_update}' and kb_id == '{TEST_KB_ID}'"
        self.mock_collection_instance.query.assert_called_once_with(
            expr=expected_query_expr,
            output_fields=ANY, # Schema fields
            limit=1
        )
        # Verify delete call
        expected_delete_expr_update = f"id == '{doc_id_to_update}' and kb_id == '{TEST_KB_ID}'"
        self.mock_collection_instance.delete.assert_called_once_with(expected_delete_expr_update)

        # Verify insert call (the important part is checking the payload)
        # The payload to insert will be original_doc_data merged with update_values
        expected_inserted_doc = original_doc_data.copy()
        expected_inserted_doc.update(update_values)

        self.mock_collection_instance.insert.assert_called_once_with([expected_inserted_doc])
        # Ensure flush was called (once for delete, once for insert - could be one combined flush)
        self.assertTrue(self.mock_collection_instance.flush.call_count >= 1) # Called by insert and delete

    def test_get_fields_helper(self):
        """Test the getFields helper method for processing search results."""
        simulated_search_results = [
            {"id": "doc1", "title": "Title 1", "content_ltks": "Content 1", "other": "misc1"},
            {"id": "doc2", "title": "Title 2", "content_ltks": "Content 2", "vector_field": [0.1]}
        ]
        fields_to_extract = ["id", "title", "non_existent_field"]

        extracted = self.milvus_conn.getFields(simulated_search_results, fields_to_extract)

        expected_output = {
            "doc1": {"id": "doc1", "title": "Title 1"},
            "doc2": {"id": "doc2", "title": "Title 2"}
        }
        self.assertEqual(extracted, expected_output)

        # Test with empty fields list
        self.assertEqual(self.milvus_conn.getFields(simulated_search_results, []), {})
        # Test with empty results list
        self.assertEqual(self.milvus_conn.getFields([], fields_to_extract), {})

    def test_unsupported_features(self):
        """Test methods for features not supported by Milvus."""
        # getHighlight
        self.assertEqual(self.milvus_conn.getHighlight([], [], ""), {})

        # getAggregation
        self.assertEqual(self.milvus_conn.getAggregation([], ""), [])

        # sql
        with self.assertRaises(NotImplementedError):
            self.milvus_conn.sql("SELECT * FROM dummy", 0, "dataframe")


if __name__ == '__main__':
    unittest.main()
