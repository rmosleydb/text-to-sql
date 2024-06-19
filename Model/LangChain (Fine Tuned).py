# Databricks notebook source
# MAGIC %pip install langchain_community langchain_openai
# MAGIC %pip install databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
workspaceUrl = 'e2-demo-field-eng.cloud.databricks.com'#spark.conf.get('spark.databricks.workspaceUrl')
workspaceUrl = f"https://{workspaceUrl}"
os.environ['DATABRICKS_HOST'] = dbutils.secrets.get("agent_studio", "databricks_host")
os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get("robert_mosley", "databricks_token")

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain_community.embeddings import DatabricksEmbeddings

# Test embedding Langchain model
#NOTE: your question embedding model must match the one used in the chunk in the previous model 
#embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")
#print(f"Test embeddings: {embedding_model.embed_query('What is Apache Spark?')[:20]}...")

def get_retriever(persist_dir: str = None):
    #os.environ["DATABRICKS_HOST"] = host
    #Get the vector search index
    vsc = VectorSearchClient(workspace_url=os.environ["DATABRICKS_HOST"], personal_access_token=os.environ["DATABRICKS_TOKEN"])
    vs_index = vsc.get_index(
        endpoint_name="one-env-shared-endpoint-7",
        index_name="robert_mosley.text_to_sql.query_history_vs_index"
    )

    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column="generated_question", columns=["generated_question", "statement_text", "manual_summarization", "table_list", "schema_list", "catalog_list", "statement_id"]
    )
    return vectorstore.as_retriever()




# test our retriever
retriever = get_retriever()
similar_documents = retriever.get_relevant_documents("How do I track my Databricks Billing?")
print(f"Relevant documents: {similar_documents[0]}")

# COMMAND ----------

retriever.vectorstore.similarity_search('test', filters={"table_list": ['mfg_sales_demo.northwind.regions', 'mfg_sales_demo.northwind.orders']})

# COMMAND ----------

# DBTITLE 1,Simple RAG - no filters

from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
#from langchain_openai import OpenAIEmbeddings
import json, requests
import pandas as pd

from langchain_community.chat_models import ChatDatabricks
from langchain_community.llms import Databricks
#dbrx_model = ChatDatabricks(endpoint="databricks-dbrx-instruct")
def transform_output(response):
    return response["candidates"][0]["text"]
llama_ft_model = Databricks(endpoint_name="t2s_llama3_8b_ft", transform_output_fn=transform_output)
mistral_ft_model = Databricks(endpoint_name="t2s_mistral_7b_ft")
mpt30 = Databricks(endpoint_name="databricks-mpt-30b-instruct")



def mine_queries(queries):
    return {'sample_queries':[{'question':q.page_content, 'sql':q.metadata['statement_text']} for q in queries], 'tables':list(set([table for q in queries for table in q.metadata['table_list']]))}

def table_definitions(input):
    print(input)
    workspaceUrl = os.environ['DATABRICKS_HOST']
    token = os.environ['DATABRICKS_TOKEN']
    tables = list(set(input["tables"]))
    #data = {'inputs': {"full_table": [[table] for table in tables]}}

    df = pd.DataFrame({'full_table': tables})
    data = {'dataframe_split': df.to_dict(orient='split')}
    
    data_json = json.dumps(data, allow_nan=True)
    url = f"{workspaceUrl}/serving-endpoints/rm_t2s_table_definition/invocations"
    headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
    #print(data_json)
    response = requests.request(method='POST', headers=headers, url=url, data=data_json)
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    input["table_response"] = response.json()['outputs']
    return input

def build_prompt(input):
    workspaceUrl = os.environ['DATABRICKS_HOST']
    token = os.environ['DATABRICKS_TOKEN']

    enriched = input["enriched_data"]
    question = input["prompt"]
    tables = enriched["table_response"]
    definitions = [t["table_definition"] for t in tables]
    samples = enriched["sample_queries"]


    df = pd.DataFrame({'question': [question], 'table_definitions': [definitions], 'sample_queries': [samples]})
    data = {'dataframe_split': df.to_dict(orient='split')}
    
    data_json = json.dumps(data, allow_nan=True)

    url = f"{workspaceUrl}/serving-endpoints/rm_t2s_build_prompt/invocations"
    headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
    response = requests.request(method='POST', headers=headers, url=url, data=data_json)
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    input["full_prompt"] = response.json()['outputs'][0]['prompt']
    print(input)
    return input




prompt = PromptTemplate.from_template("{full_prompt}")
output_parser = StrOutputParser()


# COMMAND ----------

# DBTITLE 1,RAG with Filters
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from operator import itemgetter
from langchain_core.messages.human import HumanMessage

# Create a new RunnablePassthrough object with param2 as the context
param2_passthrough = RunnablePassthrough.assign(filters=itemgetter("key2"))

def get_matches(input):
  query = input["prompt"]
  schema_filter = input.get("schema_filter", None) 
  table_filter = input.get("table_filter", None)
  filter = {}
  if schema_filter:
    filter["schema_list"] = schema_filter
  elif table_filter:
    if not isinstance(table_filter, list):
      table_filter = [table_filter]
    filter["table_list"] = list(set(table_filter))

  docs = retriever.vectorstore.similarity_search(query, filters=filter)
  samples = [{'question':q.page_content, 'sql':q.metadata['statement_text']} for q in docs]
  if "table_list" in filter:
    tables = filter["table_list"]
  else:
    tables = list(set([table for q in docs for table in q.metadata['table_list']]))

  return {'sample_queries': samples, 'tables': tables}

setup_and_retrieval = RunnablePassthrough.assign(enriched_data = RunnableLambda(get_matches) | table_definitions)

chain = setup_and_retrieval | build_prompt | prompt | llama_ft_model | output_parser

chain.invoke({"prompt":"How can I track usage on my individual workspaces?", "schema_filter":["system.billing"]})

# COMMAND ----------

# langchain.debug = True #uncomment to see the chain details and the full prompt being sent
question = {"prompt":"what customer ordered the most items?", "schema_filter":["mfg_sales_demo.northwind"]}
answer = chain.invoke(question)
question_signature = {"prompt":"what customer ordered the most items?", "schema_filter":["mfg_sales_demo.northwind" ,None], "table_filter":["mfg_sales_demo.northwind.customer" , None]}
print(answer)

# COMMAND ----------

from mlflow.models import infer_signature
import mlflow
import langchain_core
import langchain_community

mlflow.set_registry_uri("databricks-uc")
model_name = f"robert_mosley.text_to_sql.llama3_8b_ft_chain"

with mlflow.start_run(run_name="text_to_sql") as run:
    signature = infer_signature(question_signature, answer)
    model_info = mlflow.langchain.log_model(
        chain,
        loader_fn=get_retriever,  # Load the retriever with DATABRICKS_TOKEN env as secret (for authentication).
        artifact_path="chain",
        registered_model_name=model_name,
        pip_requirements=[
            "mlflow==" + mlflow.__version__,
            "langchain-core==" + langchain_core.__version__,
            "langchain-community==" + langchain_community.__version__,
            "langchain-openai",
            "databricks-vectorsearch",
        ],
        input_example=question,
        signature=signature
    )

# COMMAND ----------

# DBTITLE 1,Test Model by Loading it and running it
model = mlflow.langchain.load_model(model_info.model_uri)
question = {"prompt":"who ordered the most items?", "schema_filter":["mfg_sales_demo.northwind"]}
model.invoke(question)

# COMMAND ----------

# Create or update serving endpoint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput, ServedModelInputWorkloadSize

serving_endpoint_name = "llama3_8b_ft_chain"
latest_model_version = 1

w = WorkspaceClient()
endpoint_config = EndpointCoreConfigInput(
    name=serving_endpoint_name,
    served_models=[
        ServedModelInput(
            model_name=model_name,
            model_version=latest_model_version,
            workload_size=ServedModelInputWorkloadSize.SMALL,
            scale_to_zero_enabled=True,
            environment_vars={
                "DATABRICKS_HOST": "{{secrets/agent_studio/databricks_host}}",
                "DATABRICKS_TOKEN": "{{secrets/robert_mosley/databricks_token}}"# <scope>/<secret> that contains an access token
            }
        )
    ]
)

existing_endpoint = next(
    (e for e in w.serving_endpoints.list() if e.name == serving_endpoint_name), None
)
serving_endpoint_url = f"{workspaceUrl}/ml/endpoints/{serving_endpoint_name}"
if existing_endpoint == None:
    print(f"Creating the endpoint {serving_endpoint_url}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.create_and_wait(name=serving_endpoint_name, config=endpoint_config)
else:
    print(f"Updating the endpoint {serving_endpoint_url} to version {latest_model_version}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.update_config_and_wait(served_models=endpoint_config.served_models, name=serving_endpoint_name)
    
displayHTML(f'Your Model Endpoint Serving is now available. Open the <a href="/ml/endpoints/{serving_endpoint_name}">Model Serving Endpoint page</a> for more details.')