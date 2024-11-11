# langchain_setup

# build and run langchain docker container


create .env file with following content:
```
DOCKER_VOLUME_DIRECTORY=c:/Users/I567905/projects/docker_volumes
OLLAMA_USE_GPU=all
# OLLAMA_USE_GPU=
LANGFLOW_AUTO_LOGIN=True

USER_AGENT=langchain_setup/0.1
EMBEDDING_MODEL_NAME=sentence-transformers/all-mpnet-base-v2
LLM_MODEL_NAME=llama3.2:1b
```
# build and run containers
```bash
docker-compose up -d
```

# install llm model ito ollama container
examples:
```bash
docker exec ollama ollama run llama3.2:3b
```
or
```bash
docker exec ollama ollama run llama3.2:1b
```

# insert data into milvus db
```bash
python -m ./milvus_insert.py
```

# run two queries using langchain and milvus vector db
```bash
python -m ./langchain_query.py
```

# run two queries usin langchain directly connected to llm
no context from vector db
```bash
python -m ./langchain_query_no_vector.py
```

# stop all containers
```bash
docker-compose down
```




