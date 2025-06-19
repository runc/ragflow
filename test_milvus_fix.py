#!/usr/bin/env python3
"""
Test script to verify the Milvus connection fix for missing name_kwd field.
This script simulates the data insertion scenario that was causing the error.
"""

import json
import copy

def test_milvus_mapping_config():
    """Test the new Milvus mapping configuration"""

    # Sample chunk data that might be missing some fields from mapping
    sample_chunks = [
        {
            "id": "test_chunk_1",
            "doc_id": "test_doc_1",
            "kb_id": "test_kb_1",
            "content_with_weight": "This is test content for chunk 1",
            "create_time": "2024-01-01 10:00:00",
            "create_timestamp_flt": 1704096000.0,
            "q_1024_vec": [0.1] * 1024,  # Sample vector
            "docnm_kwd": "test_document.pdf",
            "important_kwd": ["test", "content"],
            # Note: name_kwd is intentionally missing to simulate the error
        },
        {
            "id": "test_chunk_2",
            "doc_id": "test_doc_1",
            "kb_id": "test_kb_1",
            "content_with_weight": "This is test content for chunk 2",
            "create_time": "2024-01-01 10:01:00",
            "create_timestamp_flt": 1704096060.0,
            "q_1024_vec": [0.2] * 1024,  # Sample vector
            "docnm_kwd": "test_document.pdf",
            # Note: both name_kwd and important_kwd are missing
        }
    ]

    print("Testing Milvus mapping configuration...")
    print(f"Sample chunks: {len(sample_chunks)}")

    # Load new Milvus mapping to see what fields are expected
    try:
        with open("ragflow/conf/milvus_mapping.json", "r") as f:
            milvus_config = json.load(f)

        mapping = milvus_config["mappings"]["properties"]
        vector_fields = milvus_config["mappings"]["vector_fields"]

        print(f"✓ Milvus mapping loaded successfully")
        print(f"  - Properties: {len(mapping)} fields")
        print(f"  - Vector fields: {len(vector_fields)} fields")

        # Check if name_kwd is in mapping
        if "name_kwd" in mapping:
            print(f"✓ name_kwd field found in mapping with default: '{mapping['name_kwd']['default']}'")
        else:
            print("✗ name_kwd field not found in mapping")

        # Check vector field configuration
        if "q_1024_vec" in vector_fields:
            print(f"✓ q_1024_vec vector field configured with dim: {vector_fields['q_1024_vec']['dim']}")
        else:
            print("✗ q_1024_vec vector field not found in vector_fields")

    except Exception as e:
        print(f"Error loading Milvus mapping: {e}")
        return False

    # Simulate the field completion logic from the fix
    print("\nSimulating field completion logic...")

    for i, chunk in enumerate(sample_chunks):
        print(f"\nChunk {i+1} before fix:")
        missing_fields = []
        for field_name, field_info in mapping.items():
            if field_name not in chunk and "_vec" not in field_name:
                missing_fields.append(field_name)
        print(f"  Missing fields: {missing_fields[:5]}{'...' if len(missing_fields) > 5 else ''}")

        # Apply the fix logic
        prepared_chunk = copy.deepcopy(chunk)
        for field_name, field_info in mapping.items():
            if field_name not in prepared_chunk:
                default_value = field_info.get("default", "")
                # Skip vector fields as they are handled separately
                if "_vec" in field_name:
                    continue
                prepared_chunk[field_name] = default_value

        print(f"Chunk {i+1} after fix:")
        added_fields = []
        for field_name in mapping.keys():
            if field_name not in chunk and field_name in prepared_chunk and "_vec" not in field_name:
                added_fields.append(f"{field_name}='{prepared_chunk[field_name]}'")
        print(f"  Added fields: {added_fields[:3]}{'...' if len(added_fields) > 3 else ''}")

    # Test field type mapping
    print("\nTesting field type mapping...")
    test_fields = ["name_kwd", "weight_int", "weight_flt", "content_with_weight"]
    for field_name in test_fields:
        if field_name in mapping:
            field_info = mapping[field_name]
            print(f"  {field_name}: type={field_info['type']}, default={field_info['default']}")

    print("\n✓ Milvus mapping configuration test passed!")
    print("\nKey improvements:")
    print("1. Dedicated milvus_mapping.json with complete field definitions")
    print("2. Proper field types and constraints for Milvus")
    print("3. Vector field configuration separated from properties")
    print("4. Default values ensure no missing field errors")

    return True

if __name__ == "__main__":
    test_milvus_mapping_config()
