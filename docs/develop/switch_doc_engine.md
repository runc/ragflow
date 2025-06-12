---
sidebar_position: 3
slug: /switch_doc_engine
---

# Switch document engine

Switch your doc engine from Elasticsearch to Infinity.

---

RAGFlow uses Elasticsearch by default for storing full text and vectors. To switch to [Infinity](https://github.com/infiniflow/infinity/), follow these steps:

:::caution WARNING
Switching to Infinity on a Linux/arm64 machine is not yet officially supported.
:::

1. Stop all running containers:

   ```bash
   $ docker compose -f docker/docker-compose.yml down -v
   ```

:::caution WARNING
`-v` will delete the docker container volumes, and the existing data will be cleared.
:::

2. Set `DOC_ENGINE` in **docker/.env** to `infinity`.

3. Start the containers:

   ```bash
   $ docker compose -f docker-compose.yml up -d
   ```

---

## Switching to Milvus

This section explains how to configure RAGFlow to use [Milvus](https://milvus.io/) as its document engine. Milvus is a highly scalable open-source vector database.

:::caution Prerequisites
- You need a running instance of Milvus 2.5.x (or a compatible version). Ensure it is accessible from your RAGFlow environment.
- For installation instructions, please refer to the [official Milvus installation guide](https://milvus.io/docs/install_standalone-docker.md) (Docker Standalone version linked, choose the one appropriate for your setup).
:::

:::danger Data Migration
Switching document engines will **not** automatically migrate your existing data. If you have data in another engine (e.g., Elasticsearch, Infinity), it will not be transferred to Milvus. Please back up any critical data before switching. You will need to re-index your documents into Milvus after the switch.
:::

Follow these steps to switch to Milvus:

1. **Stop RAGFlow services**:
   If RAGFlow is currently running, stop all containers:
   ```bash
   docker compose -f docker/docker-compose.yml down -v
   ```
   As noted above, using `-v` will delete existing Docker container volumes, clearing any previous data. If you intend to switch back or preserve other data, consider your volume management strategy.

2. **Configure RAGFlow for Milvus**:
   - Open the `docker/.env` file in your RAGFlow project's root directory.
   - Set the `DOC_ENGINE` environment variable to `milvus`:
     ```env
     DOC_ENGINE=milvus
     ```

3. **Configure Milvus Connection Details**:
   - Milvus connection parameters (like host, port, username, password) are managed in the `conf/service_conf.yaml` file.
   - This file is typically generated from `docker/service_conf.yaml.template` using environment variables set in `docker/.env`, or can be edited directly if you are not using environment variable substitution for these settings.
   - Refer to the [Service Configurations documentation](../configurations.md#milvus) for details on the specific Milvus settings (e.g., `MILVUS_HOST`, `MILVUS_PORT`).

4. **Start RAGFlow Services**:
   Start the RAGFlow containers with the new configuration:
   ```bash
   docker compose -f docker/docker-compose.yml up -d
   ```
   RAGFlow will now attempt to connect to and use Milvus as its document engine. Ensure your Milvus instance is running and accessible.